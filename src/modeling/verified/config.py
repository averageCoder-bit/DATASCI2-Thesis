"""Recorded configuration for the checkpoint-compatible frozen-inference track.

Do NOT apply these configuration values to a retraining claim without new provenance.
"""
FEATURE_COLUMNS = (
    "log_return", "range_pct", "body_pct", "upper_wick_pct",
    "lower_wick_pct", "rolling_volatility", "rolling_mean_return",
)
SEQUENCE_LENGTH = 60
FEATURE_COUNT = 7
HIDDEN_UNITS = 128
LATENT_DIM = 32
DROPOUT = 0.10
BATCH_SIZE = 64
GMM_COMPONENTS = (2, 3, 4, 5)
GMM_KWARGS = {"covariance_type": "full", "random_state": 42, "n_init": 10}
PROTECTED_ARCHITECTURE_SHA256 = "ffbb34e036856c44ccb78c738420008106c46cbd9f05d8f0f6250b2b877b6773"
CHECKPOINT_EXP04_SHA256 = "013cceafec71c9a0af14bc73d3f52c75a1b6856f804c8a69ae711a579123a785"
VERIFIED_THRESHOLD = -0.27292780574931297
EXPECTED_COUNTS = {"train": 101620, "validation": 21702, "test": 21749}
