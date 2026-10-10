"""Explainable out-of-fold stacking for travel- and delivery-time prediction."""
__version__ = "1.4.2"
# Identifier of the checkpoint format. It enters the fingerprint of every checkpointed computation (base
# learners, sensitivity analysis); change it only when a code change alters how those results are computed,
# so that checkpoints computed by the earlier code are not reused.
__cache_version__ = "1.1.0"
