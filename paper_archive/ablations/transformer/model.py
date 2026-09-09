import json
import re
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float


class TransformerEncoder(nn.Module):
    """
    Self-attention morphology encoder, the latent encoding is a prepended class token.

    Positional information is supplied by a sinusoidal encoding rather than a learned one, so that sequences longer
    than those seen during training remain encoded consistently.
    """

    def __init__(self, dim_encoding: int, num_layers: int, drop_prob: float, num_heads: int, dim_feedforward: int,
                 max_seq: int = 64):
        """
        Initialise the encoder.

        Args:
            dim_encoding: The dimension of the latent morphology encoding.
            num_layers: Number of self-attention blocks.
            drop_prob: Dropout probability.
            num_heads: Number of attention heads, has to divide dim_encoding.
            dim_feedforward: Hidden dimension of the pointwise feedforward blocks.
            max_seq: Largest number of links the positional encoding is precomputed for.
        """

        super().__init__()
        assert dim_encoding % num_heads == 0, f"dim_encoding ({dim_encoding}) must be divisible by num_heads ({num_heads})"
        assert dim_encoding % 2 == 0, f"dim_encoding ({dim_encoding}) must be even"

        self.projection = nn.Linear(3, dim_encoding, bias=False)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim_encoding))
        layer = nn.TransformerEncoderLayer(dim_encoding, num_heads, dim_feedforward, drop_prob, batch_first=True,
                                           norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers)

        position = torch.arange(max_seq).unsqueeze(1)
        frequency = torch.exp(torch.arange(0, dim_encoding, 2) * (-torch.log(torch.tensor(10000.0)) / dim_encoding))
        encoding = torch.zeros(1, max_seq, dim_encoding)
        encoding[0, :, 0::2] = torch.sin(position * frequency)
        encoding[0, :, 1::2] = torch.cos(position * frequency)
        self.register_buffer("positional_encoding", encoding, persistent=False)

    def forward(self, morph: Float[Tensor, "batch seq 3"]) -> Float[Tensor, "batch dim_encoding"]:
        """
        Encode the morphology, padded links are masked out of the attention.

        Args:
            morph: Morphology description, zero padded to the maximal number of links.
        Returns:
            Latent morphology encoding.
        """
        valid = (morph != 0).any(dim=-1)
        tokens = self.projection(morph) + self.positional_encoding[:, :morph.shape[1]]
        tokens = torch.cat([self.cls.expand(morph.shape[0], -1, -1), tokens], dim=1)
        valid = torch.cat([torch.ones_like(valid[:, :1]), valid], dim=1)
        return self.transformer(tokens, src_key_padding_mask=~valid)[:, 0]


class Model(nn.Module):
    """
    RAM model that predicts reachability from a modified Denavit-Hartenberg morphology parametrisation and a pose.
    """

    @classmethod
    def from_id(cls, model_id: int):
        """
        Instantiate a model from its wandb ID.

        Args:
            model_id: Wandb ID of the model.

        Returns:
            model: Instantiated model.
        """

        model_dir = Path(__file__).parent.parent / "data" / "trained_models"
        pattern = rf"{model_id}-[a-z]+-[a-z]+"
        folder = next((f for f in model_dir.iterdir() if re.match(pattern, f.name)), None)
        metadata_path = model_dir / folder / 'metadata.json'
        metadata = json.load(open(metadata_path, 'r'))

        model = cls(**metadata["hyperparameter"])
        model_folder = Path(str(model_dir / folder))
        prime = model_folder / "model.pth"
        model.load_state_dict(torch.load(prime if prime.exists() else model_folder / "checkpoint.pth"))
        return model

    def __init__(self,
                 dim_encoding: int = 128,
                 num_encoder_layers: int = 1,
                 drop_prob: float = 0.0,
                 dim_decoder: int = 1792,
                 num_decoder_layer: int = 8,
                 num_heads: int = 4,
                 dim_feedforward: int = 512):
        """
        Initialise the model.

        Args:
            dim_encoding: The dimension of the latent morphology encoding.
            num_encoder_layers: Number of encoder layers.
            drop_prob: Dropout probability of the encoder.
            dim_decoder: Hidden dimension of the MLP.
            num_decoder_layer: Number of layers of the MLP.
            num_heads: Number of attention heads, transformer only.
            dim_feedforward: Hidden dimension of the pointwise feedforward blocks, transformer only.
        """

        super().__init__()
        self.encoder = TransformerEncoder(dim_encoding, num_encoder_layers, drop_prob, num_heads, dim_feedforward)
        self.decoder = nn.Sequential(
            nn.Linear(9 + dim_encoding, dim_decoder),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(dim_decoder, dim_decoder), nn.ReLU())
              for _ in range(num_decoder_layer)],
            nn.Linear(dim_decoder, 1)
        )

    def forward(self, morph: Float[Tensor, "batch seq 3"], pose: Float[Tensor, "batch 9"]) -> Float[Tensor, "batch"]:
        """
        Predict reachability.

        Args:
            morph: Morphology description.
            pose: Pose as vector encoded.
        Returns:
            Reachability logit.
        """
        latent = self.encoder(morph)
        logit = self.decoder(torch.cat([pose, latent], dim=-1)).squeeze(-1)
        return logit

    @torch.inference_mode()
    def predict(self, *args, **kwargs):
        """
        forward with inference mode
        """
        return self.forward(*args, **kwargs)
