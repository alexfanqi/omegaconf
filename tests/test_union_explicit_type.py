from dataclasses import dataclass, field
from typing import Union
from omegaconf import OmegaConf, DictConfig


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
