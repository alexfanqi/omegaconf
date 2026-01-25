from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from omegaconf import MISSING, DictConfig, OmegaConf


@dataclass
class _AmbigA:
    # Intentionally overlaps with _AmbigB to make duck-typing ambiguous.
    # Only explicit `_type_` should disambiguate.
    val: int = 0
    kind: str = "A"


@dataclass
class _AmbigB:
    val: int = 0
    kind: str = "B"


@dataclass
class _ListOfUnionCfg:
    lst: List[Union[_AmbigA, _AmbigB]] = field(default_factory=list)


@dataclass
class _DictOfUnionCfg:
    mp: Dict[str, Union[_AmbigA, _AmbigB]] = field(default_factory=dict)


@dataclass
class _OptionalListOfUnionCfg:
    lst: List[Optional[Union[_AmbigA, _AmbigB]]] = field(default_factory=list)


@dataclass
class _NestedContainerUnionCfg:
    lst: List[Dict[str, Union[_AmbigA, _AmbigB]]] = field(default_factory=list)


def test_union_explicit_type_in_list_roundtrip() -> None:
    cfg = OmegaConf.structured(_ListOfUnionCfg)

    full_name_b = f"{_AmbigB.__module__}.{_AmbigB.__qualname__}"
    cfg.merge_with({"lst": [{"_type_": full_name_b, "val": 123}]})

    # Ensure per-element `_type_` is emitted.
    dump = OmegaConf.to_yaml(cfg)
    assert "_type_" in dump
    assert "_AmbigB" in dump

    # Round-trip via YAML.
    loaded = OmegaConf.create(dump)
    cfg2 = OmegaConf.structured(_ListOfUnionCfg)
    cfg2.merge_with(loaded)

    assert isinstance(cfg2.lst[0], DictConfig)
    assert OmegaConf.get_type(cfg2.lst[0]) is _AmbigB
    assert cfg2.lst[0].kind == "B"
    assert cfg2.lst[0].val == 123
    assert "_type_" not in cfg2.lst[0]


def test_union_explicit_type_in_dict_roundtrip() -> None:
    cfg = OmegaConf.structured(_DictOfUnionCfg)

    full_name_b = f"{_AmbigB.__module__}.{_AmbigB.__qualname__}"
    cfg.merge_with({"mp": {"k": {"_type_": full_name_b, "val": 123}}})

    dump = OmegaConf.to_yaml(cfg)
    assert "_type_" in dump
    assert "_AmbigB" in dump

    loaded = OmegaConf.create(dump)
    cfg2 = OmegaConf.structured(_DictOfUnionCfg)
    cfg2.merge_with(loaded)

    assert isinstance(cfg2.mp["k"], DictConfig)
    assert OmegaConf.get_type(cfg2.mp["k"]) is _AmbigB
    assert cfg2.mp["k"].kind == "B"
    assert cfg2.mp["k"].val == 123
    assert "_type_" not in cfg2.mp["k"]


def test_union_in_container_append_dict_and_selection_string_roundtrip() -> None:
    # Append dict with explicit _type_
    cfg = OmegaConf.structured(_ListOfUnionCfg)
    full_name_b = f"{_AmbigB.__module__}.{_AmbigB.__qualname__}"
    cfg.lst.append({"_type_": full_name_b, "val": 7})
    assert OmegaConf.get_type(cfg.lst[0]) is _AmbigB

    d = OmegaConf.to_container(cfg, resolve=False)
    assert isinstance(d, dict)
    assert d["lst"][0]["_type_"].endswith("._AmbigB")

    cfg2 = OmegaConf.structured(_ListOfUnionCfg)
    cfg2.merge_with(d)
    assert OmegaConf.get_type(cfg2.lst[0]) is _AmbigB
    assert cfg2.lst[0].val == 7

    # Append selection string (only safe because str is not a Union member here)
    cfg3 = OmegaConf.structured(_ListOfUnionCfg)
    cfg3.lst.append("_AmbigB")
    cfg3.lst[0].val = 11
    assert OmegaConf.get_type(cfg3.lst[0]) is _AmbigB
    cfg4 = OmegaConf.structured(_ListOfUnionCfg)
    cfg4.merge_with(OmegaConf.create(OmegaConf.to_yaml(cfg3)))
    assert OmegaConf.get_type(cfg4.lst[0]) is _AmbigB
    assert cfg4.lst[0].val == 11


def test_union_in_container_without_type_is_ambiguous() -> None:
    cfg = OmegaConf.structured(_ListOfUnionCfg)
    # Both candidates accept {"val": ...} so duck-typing should pick the first.
    cfg.merge_with({"lst": [{"val": 1}]})
    assert OmegaConf.get_type(cfg.lst[0]) is _AmbigA


def test_optional_union_in_container_roundtrip() -> None:
    cfg = OmegaConf.structured(_OptionalListOfUnionCfg)
    full_name_b = f"{_AmbigB.__module__}.{_AmbigB.__qualname__}"
    cfg.merge_with({"lst": [None, {"_type_": full_name_b, "val": 3}]})
    assert cfg.lst[0] is None
    assert OmegaConf.get_type(cfg.lst[1]) is _AmbigB

    cfg2 = OmegaConf.structured(_OptionalListOfUnionCfg)
    cfg2.merge_with(OmegaConf.create(OmegaConf.to_yaml(cfg)))
    assert cfg2.lst[0] is None
    assert OmegaConf.get_type(cfg2.lst[1]) is _AmbigB
    assert cfg2.lst[1].val == 3


def test_nested_container_with_union_values_roundtrip() -> None:
    cfg = OmegaConf.structured(_NestedContainerUnionCfg)
    full_name_b = f"{_AmbigB.__module__}.{_AmbigB.__qualname__}"
    cfg.merge_with({"lst": [{"k": {"_type_": full_name_b, "val": 9}}]})
    assert OmegaConf.get_type(cfg.lst[0]["k"]) is _AmbigB

    cfg2 = OmegaConf.structured(_NestedContainerUnionCfg)
    cfg2.merge_with(OmegaConf.create(OmegaConf.to_yaml(cfg)))
    assert OmegaConf.get_type(cfg2.lst[0]["k"]) is _AmbigB
    assert cfg2.lst[0]["k"].val == 9


def test_union_in_container_type_field_conflict_is_data() -> None:
    @dataclass
    class TypeHaver:
        _type_: str = "original"
        data: int = 0

    @dataclass
    class Other:
        data: int = 1

    @dataclass
    class Cfg:
        lst: List[Union[TypeHaver, Other]] = field(
            default_factory=lambda: [TypeHaver()]
        )

    cfg = OmegaConf.structured(Cfg)

    # Assign dict containing `_type_` should update field, not switch union member.
    cfg.lst[0] = {"_type_": "Other", "data": 5}
    assert OmegaConf.get_type(cfg.lst[0]) is TypeHaver
    assert cfg.lst[0]._type_ == "Other"
    assert cfg.lst[0].data == 5

    # Serializer must not overwrite the user-controlled `_type_` field.
    dump = OmegaConf.to_yaml(cfg)
    assert "_type_: Other" in dump


def test_union_explicit_type_serialization() -> None:
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


def test_union_explicit_type_conflict() -> None:
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


def test_union_merge_with_type_field() -> None:
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


def test_union_merge_bad_list() -> None:
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
