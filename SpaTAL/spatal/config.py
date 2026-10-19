"""Effective defaults for the DLPFC experiment."""

from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 2023
    top_genes: int = 2000
    k_cutoff: int = 23
    nps: int = 30
    gradient_clipping: float = 5.0
    learning_rate: float = 0.001
    weight_decay: float = 2e-4
    max_epoch: int = 1650
