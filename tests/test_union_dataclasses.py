from dataclasses import dataclass, field
from typing import Union, Optional, Any
import pytest
from omegaconf import OmegaConf, ValidationError, DictConfig


@dataclass
class User:
    name: str = "???"
    age: int = "???"


@dataclass
class Admin(User):
    permissions: str = "all"


@dataclass
class Guest(User):
    temporary: bool = True


@dataclass
class Config:
    user: Union[Admin, Guest] = "Guest"


def test_union_dataclass_basic():
    cfg = OmegaConf.structured(Config)
    cfg.user.name = "GuestUser"
    assert cfg.user.name == "GuestUser"
    assert cfg.user.temporary is True

    cfg.user = "Admin"
    cfg.user.name = "AdminUser"
    assert cfg.user.permissions == "all"
    assert cfg.user.name == "AdminUser"


def test_union_dataclass_merge():
    cfg = OmegaConf.structured(Config)
    cfg.user = "Guest"
    OmegaConf.set_struct(cfg, False)
    OmegaConf.update(cfg, "user.name", "John")
    assert cfg.user.name == "John"

    # Selection switch via update
    OmegaConf.update(cfg, "user", "Admin")
    OmegaConf.update(cfg, "user.name", "Jane")
    assert isinstance(cfg.user, DictConfig)
    assert cfg.user.permissions == "all"
    assert cfg.user.name == "Jane"


def test_union_dataclass_invalid_selection():
    cfg = OmegaConf.structured(Config)
    with pytest.raises(ValidationError):
        cfg.user = "Invalid"


def test_union_with_primitive():
    @dataclass
    class Simple:
        val: Union[int, Admin] = 10

    cfg = OmegaConf.structured(Simple)
    assert cfg.val == 10

    cfg.val = "Admin"
    assert cfg.val.permissions == "all"

    cfg.val = 20
    assert cfg.val == 20


def test_optional_union_dataclass():
    @dataclass
    class OptionalConfig:
        user: Optional[Union[Admin, Guest]] = None

    cfg = OmegaConf.structured(OptionalConfig)
    assert cfg.user is None

    cfg.user = "Admin"
    assert cfg.user.permissions == "all"

    cfg.user = None
    assert cfg.user is None


def test_nested_union_dataclass():
    @dataclass
    class Sub:
        x: int = 1

    @dataclass
    class Parent:
        child: Union[Sub, int] = "Sub"

    @dataclass
    class Root:
        node: Parent = field(default_factory=Parent)

    cfg = OmegaConf.structured(Root)
    assert cfg.node.child.x == 1

    cfg.node.child = 10
    assert cfg.node.child == 10

    cfg.node.child = "Sub"
    assert cfg.node.child.x == 1


def test_union_dataclass_from_dict():
    cfg = OmegaConf.structured(Config)
    # This should work if it matches Admin uniquely (duck typing still works if not selecting via string?)
    cfg.user = {"permissions": "some", "name": "AdminUser"}
    assert cfg.user.permissions == "some"
    assert cfg.user.name == "AdminUser"


def test_union_dataclass_duck_typing_ambiguity():
    @dataclass
    class A:
        x: int = 1

    @dataclass
    class B:
        x: int = 2

    @dataclass
    class C:
        val: Union[A, B] = "A"

    cfg = OmegaConf.structured(C)
    assert cfg.val.x == 1

    # Matching x uniquely. If both have x, the first one (A) might be picked by duck typing
    cfg.val = {"x": 10}
    # Current OmegaConf duck typing picks the first candidate that works.
    assert cfg.val.x == 10

    # Explicit switch
    cfg.val = "B"
    assert cfg.val.x == 2


def test_hierarchical_union_dataclass():
    @dataclass
    class InnerA:
        a: int = 1

    @dataclass
    class InnerB:
        b: int = 2

    @dataclass
    class OuterA:
        inner: Union[InnerA, InnerB] = "InnerA"

    @dataclass
    class OuterB:
        val: int = 10

    @dataclass
    class Root:
        outer: Union[OuterA, OuterB] = "OuterA"

    cfg = OmegaConf.structured(Root)
    assert cfg.outer.inner.a == 1

    # Switch inner
    cfg.outer.inner = "InnerB"
    assert cfg.outer.inner.b == 2

    # Switch outer
    cfg.outer = "OuterB"
    assert cfg.outer.val == 10

    # Switch outer back to A
    cfg.outer = "OuterA"
    assert cfg.outer.inner.a == 1


def test_union_with_any():
    @dataclass
    class A:
        x: int = 1

    @dataclass
    class ConfigWithAny:
        val: Union[A, Any] = 10

    cfg = OmegaConf.structured(ConfigWithAny)
    assert cfg.val == 10

    cfg.val = "A"
    assert cfg.val.x == 1

    cfg.val = {"x": 20}
    assert cfg.val.x == 20

    cfg.val = "hello"
    assert cfg.val == "hello"


def test_union_mandatory_missing():
    @dataclass
    class A:
        x: int = "???"

    @dataclass
    class B:
        y: int = 2

    @dataclass
    class Config:
        val: Union[A, B] = "A"

    cfg = OmegaConf.structured(Config)
    from omegaconf.errors import MissingMandatoryValue

    with pytest.raises(MissingMandatoryValue):
        _ = cfg.val.x

    cfg.val.x = 10
    assert cfg.val.x == 10


def test_union_readonly():
    @dataclass
    class A:
        x: int = 1

    @dataclass
    class B:
        y: int = 2

    @dataclass
    class Config:
        val: Union[A, B] = "A"

    cfg = OmegaConf.structured(Config)
    OmegaConf.set_readonly(cfg, True)

    from omegaconf.errors import ReadonlyConfigError

    with pytest.raises(ReadonlyConfigError):
        cfg.val = "B"

    with pytest.raises(ReadonlyConfigError):
        cfg.val.x = 10


def test_union_interpolation():
    @dataclass
    class A:
        x: int = 1

    @dataclass
    class B:
        y: int = 2

    @dataclass
    class Config:
        val: Union[A, B] = "A"
        target: str = "B"
        proxy: Union[A, B] = "${val}"

    cfg = OmegaConf.structured(Config)
    assert cfg.proxy.x == 1

    cfg.val = "B"
    assert cfg.proxy.y == 2

    # Interpolation to selection string
    cfg.val = "${target}"
    assert cfg.val.y == 2
    assert cfg.proxy.y == 2


def test_union_none_handling():
    @dataclass
    class A:
        x: int = 1

    @dataclass
    class Config:
        val: Optional[Union[A, int]] = "A"

    cfg = OmegaConf.structured(Config)
    assert cfg.val.x == 1

    cfg.val = None
    assert cfg.val is None

    cfg.val = 10
    assert cfg.val == 10

    cfg.val = "A"
    assert cfg.val.x == 1


def test_union_dataclass_complex_merge():
    @dataclass
    class Inner1:
        v1: int = 1

    @dataclass
    class Inner2:
        v2: int = 2

    @dataclass
    class Middle:
        inner: Union[Inner1, Inner2] = "Inner1"
        m: int = 0

    @dataclass
    class Root:
        middle: Middle = field(default_factory=Middle)

    cfg = OmegaConf.structured(Root)
    assert cfg.middle.inner.v1 == 1

    # Merge a dict that changes middle.m and middle.inner.v1
    cfg.merge_with({"middle": {"m": 10, "inner": {"v1": 100}}})
    assert cfg.middle.m == 10
    assert cfg.middle.inner.v1 == 100

    # Merge a dict that switches middle.inner to Inner2 and sets v2
    # This tests if UnionNode handles nested updates that include a selection string
    cfg.merge_with({"middle": {"inner": "Inner2"}})
    assert cfg.middle.inner.v2 == 2

    # Merge and switch simultaneously
    cfg.merge_with({"middle": {"inner": {"v2": 200}}})  # Duck typing should keep Inner2
    assert cfg.middle.inner.v2 == 200

    # Test merging two structured configs
    over = {"middle": {"inner": "Inner1", "m": 50}}
    cfg.merge_with(over)
    assert cfg.middle.inner.v1 == 1
    assert cfg.middle.m == 50


def test_union_merge_into_missing():
    @dataclass
    class A:
        x: int = 1

    @dataclass
    class B:
        y: int = 2

    @dataclass
    class Config:
        val: Union[A, B] = "???"

    cfg = OmegaConf.structured(Config)

    # Merging "selection string" into missing
    cfg.merge_with({"val": "A"})
    assert cfg.val.x == 1

    # Merging dict (duck typing) into missing
    cfg = OmegaConf.structured(Config)
    cfg.merge_with({"val": {"y": 10}})
    assert cfg.val.y == 10


def test_union_or_operator_syntax():
    @dataclass
    class A:
        x: int = 1

    @dataclass
    class B:
        y: int = 2

    @dataclass
    class Config:
        # Use | syntax (PEP 604)
        val: A | B = field(default_factory=A)

    cfg = OmegaConf.structured(Config)
    assert cfg.val.x == 1


def test_union_merge_string_selection():
    @dataclass
    class RSNNConfig:
        foo: int = 1

    @dataclass
    class TCNConfig:
        bar: int = 2

    @dataclass
    class Config:
        backbone: Union[RSNNConfig, TCNConfig] = field(default_factory=RSNNConfig)

    cfg = OmegaConf.structured(Config)

    # Test merging string matching the current type (no-op effectively, but ensures validity)
    cfg.merge_with({"backbone": "RSNNConfig"})
    assert cfg.backbone.foo == 1
    assert isinstance(cfg.backbone, DictConfig)

    # Test switching type via merge with string
    cfg.merge_with({"backbone": "TCNConfig"})
    assert cfg.backbone.bar == 2
    assert "foo" not in cfg.backbone


def test_union_with_str_skip_selection():
    @dataclass
    class A:
        x: int = 1

    @dataclass
    class Config:
        val: Union[str, A] = field(default_factory=A)

    cfg = OmegaConf.structured(Config)

    # "A" matches class name A. But str is in Union.
    # So it should NOT be converted to A().
    # It should result in Unsupported value type (because DictConfig can't be str)
    # or validated as a string if OmegaConf supports that transition (it currently errors).
    with pytest.raises(ValidationError, match="Unsupported value type"):
        cfg.val = "A"
