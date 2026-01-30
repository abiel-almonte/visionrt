import sys
from typing import Any
from types import ModuleType

from ._visionrt import set_verbose

use_custom: bool = True  # use custom optimize_fx to modify the fx graph before inductor
use_inductor: bool = True  # use the inductor backend in `torch.compile`
debug: bool = False  # print fx graph to terminal
verbose: bool = False  # print the visionrt logs to the terminal
cudagraphs: bool = False  # capture cudagraph in visionrt backend


class _optims:
    # constant folding
    fold_conv_bn: bool = False

    # kernel fusing
    fuse_add_relu: bool = False

    def __setattr__(self, name: str, value: Any) -> None:
        global use_custom

        super().__setattr__(name, value)
        use_custom = any(getattr(self, attr) for attr in self.__class__.__annotations__)

    def clear(self):
        for attr in self.__class__.__annotations__:
            setattr(self, attr, False)

    def all(self):
        for attr in self.__class__.__annotations__:
            setattr(self, attr, True)


class _module(ModuleType):
    def __setattr__(self, name, value):
        if name == "verbose":
            set_verbose(value)  # set verbose in _visionrt (csrc) logging

        super().__setattr__(name, value)


optims = _optims()
sys.modules[__name__].__class__ = _module
