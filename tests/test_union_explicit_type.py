from dataclasses import dataclass, field
from typing import Union

from omegaconf import MISSING, DictConfig, OmegaConf


def test_union_explicit_type_serialization():
    """
    Test that explicit type information (_type_) is serialized for Union fields,
    and used during deserialization to restore the correct type.
    """

    @dataclass
    class RSNNConfig:
        base_current: float = 0.0

    @dataclass
    class TCNConfig:
        input_weight_scaling: float = 1.0

    @dataclass
    class Config:
        backbone: Union[RSNNConfig, TCNConfig] = field(default_factory=RSNNConfig)

    cfg = OmegaConf.structured(Config)

    # 1. Switch to TCNConfig
    cfg.backbone = TCNConfig(input_weight_scaling=5.0)

    # 2. Serialize to YAML
    yaml_dump = OmegaConf.to_yaml(cfg)

    # Verify _type_ is present
    # We expect _type_: tests.test_union_explicit_type.TCNConfig (or similar)
    assert "_type_" in yaml_dump
    assert "TCNConfig" in yaml_dump

    # 3. Deserialize
    cfg_new = OmegaConf.structured(Config)
    loaded = OmegaConf.create(yaml_dump)

    # Merge loaded yaml (which has _type_) into new config
    cfg_new.merge_with(loaded)

    # Verify type switch
    assert isinstance(cfg_new.backbone, DictConfig)
    # Check if content matches TCNConfig schema
    assert "input_weight_scaling" in cfg_new.backbone
    assert cfg_new.backbone.input_weight_scaling == 5.0

    # Verify _type_ key is NOT present in the final object (it should be consumed/skipped)
    # Actually, structured config doesn't allow extra keys, so it MUST be absent.
    assert "_type_" not in cfg_new.backbone


def test_union_explicit_type_conflict():
    """
    Test that if a Dataclass explicitly defines a '_type_' field,
    OmegaConf respects it as a data field and does NOT use it for
    Union type switching magic (deserialization).
    Also ensures serializer doesn't overwrite it.
    """

    @dataclass
    class TypeHaver:
        _type_: str = "original"
        data: int = 0

    @dataclass
    class Other:
        data: int = 1

    @dataclass
    class Config:
        item: Union[TypeHaver, Other] = field(default_factory=TypeHaver)

    cfg = OmegaConf.structured(Config)

    # 1. Merge that looks like a type switch but hits a valid field
    # "Other" matches the class name of the matching Union candidate.
    # But TypeHaver has a field _type_.
    # Expectation: "bail out" -> Do NOT switch to Other class. Set _type_ field value to "Other".
    cfg.merge_with({"item": {"_type_": "Other"}})

    # Verify it stayed as TypeHaver
    assert isinstance(cfg.item, DictConfig)
    # OmegaConf stores object type metadata for structured configs
    assert cfg.item._metadata.object_type == TypeHaver

    # Verify the field was updated
    assert cfg.item._type_ == "Other"

    # 2. Serialization check
    # If TypeHaver has _type_="special", serializer should NOT overwrite it with "tests.test_...TypeHaver"
    cfg.item._type_ = "special"
    dump = OmegaConf.to_yaml(cfg)

    assert "_type_: special" in dump
    # We should NOT see the auto-generated type info for TypeHaver here because it's overridden/conflicted
    assert (
        "TypeHaver" not in dump
    )  # Class name shouldn't appear in _type_ value if we set "special"


@dataclass
class UnionTypeA:
    x: int = 10
    kind: str = "A"


@dataclass
class UnionTypeB:
    b: int = 20
    kind: str = "B"


@dataclass
class UnionTypeConfig:
    u: Union[UnionTypeA, UnionTypeB] = MISSING


def test_union_merge_with_type_field():
    """
    Test that we can use the `_type_` field to tell OmegaConf which Union subclass
    to use when merging a dict.
    """
    cfg = OmegaConf.structured(UnionTypeConfig)

    # 1. Merge B using fully qualified name in _type_
    # Note: For this to work with `_type_` pointing to a class, the class needs to be importable.
    # We will use the full name of B here.
    # Since these classes are now at module level, they should be importable.
    # Note: qualname might still need to include the module prefix if we construct it manually,
    # but for simple module level classes, __module__ + . + __qualname__ works.

    full_name_B = f"{UnionTypeB.__module__}.{UnionTypeB.__qualname__}"

    # Test with _type_ (explicit selection)
    data = {"u": {"_type_": full_name_B, "b": 100}}

    merged = OmegaConf.merge(cfg, data)

    assert isinstance(merged.u, DictConfig)
    # Check if we successfully switched to B (by checking field 'b')
    assert merged.u.b == 100

    # If duck typing works, object_type should be UnionTypeB.
    # If _type_ works, object_type should be UnionTypeB.
    # If _type_ causes fallback to dict, that's the bug/behavior we see.

    # If this fails, we know it's not promoting to B.
    # We'll assert object type matches B
    assert merged.u._metadata.object_type == UnionTypeB

    assert merged.u.kind == "B"
    # Verify the underlying type metadata
    assert merged.u._metadata.object_type == UnionTypeB

    # 2. Verify we can switch back to A via merge
    full_name_A = f"{UnionTypeA.__module__}.{UnionTypeA.__qualname__}"
    data_a = {"u": {"_type_": full_name_A, "x": 999}}
    merged_a = OmegaConf.merge(merged, data_a)

    assert merged_a.u.x == 999
    assert merged_a.u.kind == "A"
    assert merged_a.u._metadata.object_type == UnionTypeA


def test_union_merge_bad_list():
    """
    Regression test for: KeyValidationError: ListConfig indices must be integers or slices, not str
    When merging a list into a UnionNode, it should raise ValidationError (incompatible type),
    NOT KeyValidationError (crash when checking _type_).
    """
    cfg = OmegaConf.structured(UnionTypeConfig)
    data = {"u": [123]}

    import pytest

    from omegaconf.errors import ValidationError

    with pytest.raises(ValidationError):
        OmegaConf.merge(cfg, data)
