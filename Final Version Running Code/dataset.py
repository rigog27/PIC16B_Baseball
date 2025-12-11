import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class YamaPADataset(Dataset):
    """
    Dataset for pitch-by-pitch sequences within plate appearances (PAs).
    Each item is a full PA: a sequence of pitches.
    """
    def __init__(self, df, pa_keys, numeric_cols):
        """
        df: dataframe of all pitches
        pa_keys: list of (game_pk, at_bat_number) pairs representing each PA
        numeric_cols: list of numeric feature columns to extract per pitch
        """
        self.df = df
        self.pa_keys = pa_keys
        self.numeric_cols = numeric_cols

    def __len__(self):
        return len(self.pa_keys)

    def __getitem__(self, idx):
        """
        Returns all sequences for a single PA.
        """
        game_pk, at_bat = self.pa_keys[idx]

        sub = self.df[
            (self.df["game_pk"] == game_pk) &
            (self.df["at_bat_number"] == at_bat)
        ].sort_values("pitch_in_pa")

        # Categorical feature tensors
        prev_pitch_idx = torch.tensor(sub["prev_pitch_idx"].values, dtype=torch.long)
        batter_hand_idx = torch.tensor(sub["batter_hand_idx"].values, dtype=torch.long)
        prev_result_idx = torch.tensor(sub["prev_pitch_result_idx"].values, dtype=torch.long)

        # Numeric features (N_pitches × N_numeric_features)
        numeric_feats = torch.tensor(
            sub[self.numeric_cols].values,
            dtype=torch.float32
        )

        labels = torch.tensor(sub["pitch_type_idx"].values, dtype=torch.long)

        return prev_pitch_idx, batter_hand_idx, prev_result_idx, numeric_feats, labels

def yama_collate(batch):
    """
    Collate function to pad sequences within a batch.

    batch: list of tuples returned by __getitem__
    Returns:
        - padded feature tensors
        - padded labels (with ignore index)
        - mask indicating real (non-padded) timesteps
    """
    # Unzip elements from each PA tuple
    prev_list, hand_list, prev_res_list, num_list, label_list = zip(*batch)

    # Pad each categorical and numeric sequence to max length in batch
    prev_padded = pad_sequence(prev_list, batch_first=True, padding_value=0)
    hand_padded = pad_sequence(hand_list, batch_first=True, padding_value=0)
    prev_res_padded = pad_sequence(prev_res_list, batch_first=True, padding_value=0)

    num_padded = pad_sequence(num_list, batch_first=True, padding_value=0.0)
    labels_padded = pad_sequence(label_list, batch_first=True, padding_value=-100)

    lengths = torch.tensor([len(x) for x in label_list], dtype=torch.long)
    max_len = labels_padded.size(1)
    mask = (torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1))

    return prev_padded, hand_padded, prev_res_padded, num_padded, labels_padded, mask