"""Deep tabular base learners (PyTorch).

MLP, ResNet and FT-Transformer use the reference implementations of
Gorishniy et al. (2021, ``rtdl_revisiting_models``); TabNet uses
``pytorch-tabnet`` (Arik & Pfister, 2021). CNN and LSTM treat the encoded
feature vector as a length-F sequence with one channel. All models share the
same preprocessing (median imputation and quantile-normal transformation of
numeric features, training-fitted categorical vocabularies with a reserved
index for missing/unseen levels, standardised target). All except TabNet are
trained by the same early-stopped AdamW loop; TabNet uses the training loop of
``pytorch-tabnet`` with the same batch size, epoch budget and patience. In this
way differences between the models reflect architecture rather than training
protocol.
"""
from __future__ import annotations

import contextlib
import copy
import io
import os
import warnings

import numpy as np
import pandas as pd

from .models import Learner

# Values used for settings missing from tuning.deep.
DEEP_DEFAULTS = {"device": "auto", "batch_size": 1024, "eval_batch_size": 8192, "max_epochs": 100, "patience": 8}


def _torch():
    """Import PyTorch, with an installation hint when the optional dependency is missing."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Deep learners require the optional dependencies: pip install 'stackxai[deep]'") from exc
    return torch


class _DeepLearner(Learner):
    """Preprocessing shared by the deep learners: median imputation and quantile-normal transformation of numeric
    features, and integer codes for categorical features in which 0 marks a missing or unseen level.
    """
    iterative = True

    @property
    def deep(self) -> dict:
        """Settings of ``tuning.deep``, completed with ``DEEP_DEFAULTS``."""
        return {**DEEP_DEFAULTS, **(self.cfg.get("deep") or {})}

    def _limit_threads(self):
        """Honour run.n_jobs so that a GPU stage can share the machine with a CPU stage."""
        if self.n_jobs and self.n_jobs > 0:
            _torch().set_num_threads(self.n_jobs)

    def _device(self):
        """Torch device from ``tuning.deep.device``; ``auto`` chooses CUDA when it is available."""
        torch = _torch()
        d = self.deep["device"]
        return torch.device(("cuda" if torch.cuda.is_available() else "cpu") if d == "auto" else d)

    def _fit_transform(self, X: pd.DataFrame):
        """Fit the imputer, the quantile transformer and the embedding sizes on the training rows ``X`` and return
        the arrays of ``X``.
        """
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import QuantileTransformer
        Z = self.encoder.transform(X)
        num = Z[self.encoder.numeric].to_numpy(dtype=float)
        self.imputer = self.qt = None
        if num.shape[1]:
            self.imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(num)
            self.qt = QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, len(num)),
                                          subsample=100_000, random_state=self.seed).fit(self.imputer.transform(num))
        self.cardinalities = [len(self.encoder.categories[c]) + 1 for c in self.encoder.categorical]
        return self._arrays(X)

    def _arrays(self, X: pd.DataFrame):
        """Numeric matrix (float32) and categorical code matrix (int64) of ``X`` under the fitted preprocessing."""
        Z = self.encoder.transform(X)
        num = self.qt.transform(self.imputer.transform(Z[self.encoder.numeric].to_numpy(dtype=float))) \
            if self.qt is not None else np.zeros((len(Z), 0))
        cat = np.column_stack([Z[c].cat.codes.to_numpy().astype("int64") + 1 for c in self.encoder.categorical]) \
            if self.encoder.categorical else np.zeros((len(Z), 0), dtype="int64")
        if cat.size:                                   # fail with a clear message rather than a CUDA device assert
            too_large = cat.max(axis=0) >= np.asarray(self.cardinalities)
            if too_large.any():
                bad = [c for c, b in zip(self.encoder.categorical, too_large) if b]
                raise ValueError(f"categorical codes exceed the embedding size for {bad}")
        return num.astype("float32"), cat.astype("int64")

    def set_inference_threads(self, n: int):
        """Number of PyTorch threads used for prediction; ``n <= 0`` uses every core."""
        _torch().set_num_threads(n if n and n > 0 else (os.cpu_count() or 1))


class _TorchNet(_DeepLearner):
    """Shared training loop: AdamW, MSE on the standardised target, early stopping on validation RMSE."""

    def _module(self, n_num: int):
        """Build the network for ``n_num`` numeric inputs; implemented by every architecture."""
        raise NotImplementedError

    def _parameter_groups(self, module):
        """Parameters passed to the optimiser; the FT-Transformer overrides this with its own weight-decay groups."""
        return module.parameters()

    def fit(self, X, y, X_val=None, y_val=None):
        """Train on ``X`` and ``y``.

        With validation rows, training stops when the validation RMSE has not improved for ``patience`` epochs, and
        the best weights are restored. Without them, the model is trained for ``n_estimators`` epochs, the count
        found by early stopping during tuning, or ``max_epochs`` if none is given.
        """
        torch = _torch()
        self._limit_threads()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.encoder.fit(X)
        num, cat = self._fit_transform(X)
        y = np.asarray(y, float)
        self.y_mu, self.y_sd = float(y.mean()), float(y.std() + 1e-9)
        dev = self._device()
        self.net = self._module(num.shape[1]).to(dev)
        opt = torch.optim.AdamW(self._parameter_groups(self.net), lr=self.params.get("learning_rate", 1e-3),
                                weight_decay=self.params.get("weight_decay", 1e-5))
        tn, tc = torch.as_tensor(num, device=dev), torch.as_tensor(cat, device=dev)
        ty = torch.as_tensor(((y - self.y_mu) / self.y_sd).astype("float32"), device=dev)
        has_val = X_val is not None
        if has_val:
            val_arrays, yv = self._arrays(X_val), np.asarray(y_val, float)
        epochs = self.deep["max_epochs"] if has_val else int(self.params.get("n_estimators", self.deep["max_epochs"]))
        best, best_state, wait, bs = np.inf, None, 0, self.deep["batch_size"]
        for epoch in range(epochs):
            self.net.train()
            for idx in np.array_split(rng.permutation(len(ty)), max(1, len(ty) // bs)):
                i = torch.as_tensor(idx, device=dev)
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(self.net(tn[i], tc[i]).squeeze(-1), ty[i])
                loss.backward()
                opt.step()
            if has_val:
                score = float(np.sqrt(np.mean((self._forward(*val_arrays) - yv) ** 2)))
                if score < best - 1e-6:
                    best, best_state, wait, self.best_iteration = score, copy.deepcopy(self.net.state_dict()), 0, epoch + 1
                else:
                    wait += 1
                    if wait >= self.deep["patience"]:
                        break
        if has_val and best_state is not None:
            self.net.load_state_dict(best_state)
        else:
            self.best_iteration = epochs
        return self

    def __getstate__(self):
        """Pickle the network on the CPU, so that a model trained on a GPU can be loaded on any machine."""
        state = self.__dict__.copy()
        if state.get("net") is not None:
            state["net"] = copy.deepcopy(state["net"]).cpu()
        return state

    def predict(self, X):
        return self._forward(*self._arrays(X))

    def _forward(self, num, cat):
        """Predictions in batches of ``eval_batch_size``, transformed back from the standardised target."""
        torch = _torch()
        dev = next(self.net.parameters()).device
        self.net.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(num), self.deep["eval_batch_size"]):
                e = s + self.deep["eval_batch_size"]
                out.append(self.net(torch.as_tensor(num[s:e], device=dev), torch.as_tensor(cat[s:e], device=dev))
                           .squeeze(-1).cpu().numpy())
        return np.concatenate(out) * self.y_sd + self.y_mu if out else np.empty(0)


def _modules():
    """Module with the network definitions, imported only when a deep learner is used."""
    _torch()
    from . import deep_modules
    return deep_modules


def _d_in(n_num: int, cardinalities: list[int], d_embedding: int) -> int:
    """Input width of a backbone: the numeric features plus one embedding of size ``d_embedding`` per categorical
    feature.
    """
    return n_num + d_embedding * len(cardinalities)


class MLPLearner(_TorchNet):
    """Multilayer perceptron on numeric features and categorical embeddings (Gorishniy et al., 2021)."""
    name = "MLP"

    def _module(self, n_num):
        dm, p = _modules(), self.params
        backbone = dm.rtdl.MLP(d_in=_d_in(n_num, self.cardinalities, 8), d_out=1, n_blocks=p.get("n_blocks", 2),
                               d_block=p.get("d_block", 256), dropout=p.get("dropout", 0.1))
        return dm.EmbeddingNet(self.cardinalities, n_num, 8, backbone)


class ResNetLearner(_TorchNet):
    """Residual network for tabular data (Gorishniy et al., 2021)."""
    name = "RESNET"

    def _module(self, n_num):
        dm, p = _modules(), self.params
        backbone = dm.rtdl.ResNet(d_in=_d_in(n_num, self.cardinalities, 8), d_out=1, n_blocks=p.get("n_blocks", 2),
                                  d_block=p.get("d_block", 192), d_hidden=None, d_hidden_multiplier=2.0,
                                  dropout1=p.get("dropout", 0.1), dropout2=0.0)
        return dm.EmbeddingNet(self.cardinalities, n_num, 8, backbone)


class CNNLearner(_TorchNet):
    """One-dimensional convolutional network that reads the encoded feature vector as a one-channel sequence."""
    name = "CNN"

    def _module(self, n_num):
        dm = _modules()
        backbone = dm.CNNBackbone(_d_in(n_num, self.cardinalities, 1), self.params.get("dropout", 0.2))
        return dm.EmbeddingNet(self.cardinalities, n_num, 1, backbone)


class LSTMLearner(_TorchNet):
    """LSTM network that reads the encoded feature vector one element per step."""
    name = "LSTM"

    def _module(self, n_num):
        dm = _modules()
        return dm.EmbeddingNet(self.cardinalities, n_num, 1, dm.LSTMBackbone(self.params.get("dropout", 0.2)))


class FTTransformerLearner(_TorchNet):
    """FT-Transformer (Gorishniy et al., 2021)."""
    name = "FTT"

    def _module(self, n_num):
        p = self.params
        return _modules().FTTNet(n_num, self.cardinalities, n_blocks=p.get("n_blocks", 3),
                                 d_block=p.get("d_block", 192), attention_n_heads=8,
                                 attention_dropout=p.get("attention_dropout", 0.2), ffn_d_hidden=None,
                                 ffn_d_hidden_multiplier=4 / 3, ffn_dropout=p.get("ffn_dropout", 0.1),
                                 residual_dropout=0.0)

    def _parameter_groups(self, module):
        return module.make_parameter_groups()


class TabNetLearner(_DeepLearner):
    """TabNet (Arik & Pfister, 2021), trained by the loop of ``pytorch-tabnet``."""
    name = "TABNET"

    def fit(self, X, y, X_val=None, y_val=None):
        """Train on ``X`` and ``y``. With validation rows the number of epochs is set by early stopping; without
        them ``n_estimators`` epochs are run, or ``max_epochs`` if none is given.
        """
        torch = _torch()
        self._limit_threads()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from pytorch_tabnet.tab_model import TabNetRegressor
        self.encoder.fit(X)
        num, cat = self._fit_transform(X)
        y = np.asarray(y, float)
        self.y_mu, self.y_sd = float(y.mean()), float(y.std() + 1e-9)
        n_num, p = num.shape[1], self.params
        n_d = int(p.get("n_d", 16))
        self.model = TabNetRegressor(n_d=n_d, n_a=n_d, n_steps=int(p.get("n_steps", 4)), gamma=p.get("gamma", 1.3),
                                     lambda_sparse=p.get("lambda_sparse", 1e-4),
                                     cat_idxs=list(range(n_num, n_num + cat.shape[1])),
                                     cat_dims=self.cardinalities, cat_emb_dim=4, seed=self.seed, verbose=0,
                                     optimizer_fn=torch.optim.Adam,
                                     optimizer_params={"lr": p.get("learning_rate", 2e-2)},
                                     device_name=str(self._device()))
        A = np.hstack([num, cat]).astype("float32")
        t = ((y - self.y_mu) / self.y_sd).reshape(-1, 1)
        kw = {"batch_size": self.deep["batch_size"], "virtual_batch_size": 256, "drop_last": False,
              "compute_importance": False}
        if X_val is not None:
            nv, cv = self._arrays(X_val)
            tv = ((np.asarray(y_val, float) - self.y_mu) / self.y_sd).reshape(-1, 1)
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(A, t, eval_set=[(np.hstack([nv, cv]).astype("float32"), tv)], eval_metric=["rmse"],
                               max_epochs=self.deep["max_epochs"], patience=self.deep["patience"], **kw)
            self.best_iteration = int(self.model.best_epoch) + 1
        else:
            epochs = int(p.get("n_estimators", self.deep["max_epochs"]))
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(A, t, max_epochs=epochs, patience=0, **kw)
            self.best_iteration = epochs
        return self

    def predict(self, X):
        num, cat = self._arrays(X)
        return self.model.predict(np.hstack([num, cat]).astype("float32")).ravel() * self.y_sd + self.y_mu
