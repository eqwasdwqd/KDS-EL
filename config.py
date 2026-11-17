"""Dataset loading settings used by utils.py."""

SPECIAL_COL = {"CNS": 3, "MLL": 3}
SRBCT_KEEP = {"BL", "EWS", "NB", "RMS"}
MAT_DATASETS = {
    "BASEHOCK", "COIL20", "PCMAC", "RELATHE", "ISOLET",
    "USPS", "Pancancer", "MultipleFeaturesDigit", "madelon",
}

# Optimizer seeds used in the revised nested experiments, not a claim about all historical runs.
OPTIMIZER_SEEDS = [7, 30, 1, 42, 100, 2024, 999, 1234, 555, 888]

# Random state used to generate the revised nested CV partitions.
CV_RANDOM_STATE = 42

# Original experiments: run r (zero-indexed) uses optimizer seed
# SEED_BASE + 37 * r, with SEED_BASE = 42.
ORIGINAL_SEED_BASE = 42
ORIGINAL_SEED_STRIDE = 37
