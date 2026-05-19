from dataclasses import dataclass
from typing import Sequence

import pytest

from omegaconf import OmegaConf, ValidationError


@dataclass
class SequenceConfig:
    seq_items: Sequence[int]
    mixed: Sequence[int] = (1, 2, 3)


def test_sequence_creation():
    cfg = OmegaConf.structured(SequenceConfig(seq_items=[1, 2, 3]))
    assert cfg.seq_items == [1, 2, 3]
    assert cfg.mixed == [1, 2, 3]  # Tuple converted to List in OmegaConf


def test_sequence_assignment_valid():
    cfg = OmegaConf.structured(SequenceConfig(seq_items=[1, 2]))
    cfg.seq_items = [3, 4]
    assert cfg.seq_items == [3, 4]

    cfg.seq_items = (5, 6)
    assert cfg.seq_items == [5, 6]


def test_sequence_assignment_invalid():
    cfg = OmegaConf.structured(SequenceConfig(seq_items=[1, 2]))
    with pytest.raises(ValidationError):
        cfg.seq_items = ["a", "b"]


def test_sequence_merge():
    cfg = OmegaConf.structured(SequenceConfig(seq_items=[1, 2]))
    merge_cfg = OmegaConf.create({"seq_items": [3, 4]})
    res = OmegaConf.merge(cfg, merge_cfg)
    assert res.seq_items == [3, 4]


def test_sequence_merge_invalid():
    cfg = OmegaConf.structured(SequenceConfig(seq_items=[1, 2]))
    merge_cfg = OmegaConf.create({"seq_items": ["a", "b"]})
    with pytest.raises(ValidationError):
        OmegaConf.merge(cfg, merge_cfg)
