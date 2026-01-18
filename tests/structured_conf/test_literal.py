from dataclasses import dataclass, field
from typing import Optional

from pytest import raises

from omegaconf import OmegaConf, ValidationError

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


@dataclass
class LiteralConfig:
    int_lit: Literal[1, 2, 3] = 1
    str_lit: Literal["a", "b", "c"] = "a"
    bool_lit: Literal[True, False] = True
    mixed_lit: Literal[1, "a", True] = 1
    optional_lit: Optional[Literal["foo", "bar"]] = None


@dataclass
class NestedLiteralConfig:
    cfg: LiteralConfig = field(default_factory=LiteralConfig)


class TestLiteral:
    def test_literal_int(self) -> None:
        cfg = OmegaConf.structured(LiteralConfig)
        assert cfg.int_lit == 1
        cfg.int_lit = 2
        assert cfg.int_lit == 2

        with raises(ValidationError):
            cfg.int_lit = 4

    def test_literal_str(self) -> None:
        cfg = OmegaConf.structured(LiteralConfig)
        assert cfg.str_lit == "a"
        cfg.str_lit = "b"
        assert cfg.str_lit == "b"

        with raises(ValidationError):
            cfg.str_lit = "d"

    def test_literal_bool(self) -> None:
        cfg = OmegaConf.structured(LiteralConfig)
        assert cfg.bool_lit is True
        cfg.bool_lit = False
        assert cfg.bool_lit is False

        with raises(ValidationError):
            cfg.bool_lit = 100

    def test_mixed_literal(self) -> None:
        cfg = OmegaConf.structured(LiteralConfig)
        cfg.mixed_lit = "a"
        assert cfg.mixed_lit == "a"
        cfg.mixed_lit = True
        assert cfg.mixed_lit is True

        with raises(ValidationError):
            cfg.mixed_lit = 2.5

    def test_optional_literal(self) -> None:
        cfg = OmegaConf.structured(LiteralConfig)
        assert cfg.optional_lit is None
        cfg.optional_lit = "foo"
        assert cfg.optional_lit == "foo"
        cfg.optional_lit = None
        assert cfg.optional_lit is None

        with raises(ValidationError):
            cfg.optional_lit = "baz"

    def test_nested_literal(self) -> None:
        cfg = OmegaConf.structured(NestedLiteralConfig)
        assert cfg.cfg.int_lit == 1
        cfg.cfg.int_lit = 3
        assert cfg.cfg.int_lit == 3

        with raises(ValidationError):
            cfg.cfg.int_lit = 10

    def test_literal_merge(self) -> None:
        cfg = OmegaConf.structured(LiteralConfig)
        merge_cfg = OmegaConf.create({"int_lit": 2})
        res = OmegaConf.merge(cfg, merge_cfg)
        assert res.int_lit == 2

        invalid_merge = OmegaConf.create({"int_lit": 10})
        with raises(ValidationError):
            OmegaConf.merge(cfg, invalid_merge)

    def test_literal_instantiation_failure(self) -> None:
        with raises(ValidationError):
            OmegaConf.structured(LiteralConfig(int_lit=10))  # type: ignore
