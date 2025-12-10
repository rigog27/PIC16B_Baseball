import torch

# device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parameters used in training
BATCH_SIZE = 32
NUM_EPOCHS = 22
LEARNING_RATE = 1e-3
GRAD_CLIP = 1.0


# parameters for the model
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0

# columns wanted
NUMERIC_COLS = [
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "is_top_inning",
    "on_1b_flag",
    "on_2b_flag",
    "on_3b_flag",
    "score_diff_pov",
    "pitcher_ahead_flag",
    "hitter_ahead_flag",
    "putaway_count_flag",
    "platoon_adv",
    "fastballs_in_pa",
    "last_two_fastballs_flag",
    "risp_flag",
    "high_leverage_flag",
    "prev_release_speed",
    "prev_pfx_x",
    "prev_pfx_z",
    "prev_speed_minus_ff_mean",
]