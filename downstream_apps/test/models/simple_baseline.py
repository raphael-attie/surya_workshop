"""
A baseline for CH segmentation with learnable intensity threshold taking advantage of the z-score / standardization of the AIA data. 
"""

import torch
import torch.nn as nn
from einops import rearrange


class ThresholdCHModel(nn.Module):
    def __init__(self, channel_index: int = 3, starting_threshold: float= -0.2, temperature: float = 0.05):
        """
        Initializes the model.

        Args:
            input_dim (int): The size of the input vector after channel and time dimensions are flattened.

        Note:
            This model expects 'ts' in the batch dict to already be in **signum-log** space
            (channel z-scores undone, log compression retained). Use
            destandardize_channels() to pre-process normalized SDO inputs before passing
            them here (e.g., via the preprocess_fn argument of FlareLightningModule).
        """
        super().__init__()
        self.channel_index = channel_index
        self.threshold = nn.Parameter(torch.tensor(float(starting_threshold)))
        self.temperature = temperature  # fixed; controls sigmoid sharpness, not learned

    
    def forward(self, x: dict) -> torch.Tensor:
        """
        Performs a forward pass through the model.

        Args:
            x (dict): Batch dict with 'ts' of shape (B, C, T, H, W) in signum-log space.
            starting_threshold (float): Threshold for the CH detection (in units of input image)

        B - Batch size
        C - Channels
        T - Time steps
        H - Height
        W - Width
        """
        # We only work with the channel of AIA 193
        x = x["ts"][:, self.channel_index, ...]
        x = x.squeeze(1) 

        # CH pixels are low-intensity: (threshold - x) is large/positive below threshold, 
        # so sigmoid of it approaches 1 there — same direction as the original `x < threshold`.
        logits = (self.threshold - x) / self.temperature                                         

        return logits