"""Lightweight public package surface for True DNA.

Heavy model/backbone imports can initialize optional CUDA/Mamba extensions.
Keep package import cheap and load those symbols only when requested.
"""

__all__ = [
    "DnaConfig",
    "DnaModel",
    "DnaTokenizer",
    "BaseTokenizer",
    "build_tokenizer",
    "DatasetImpl",
    "evaluate",
    "get_scheduler",
]


def __getattr__(name):
    if name == "DnaConfig":
        from .config import DnaConfig

        return DnaConfig
    if name == "DnaTokenizer":
        from .tokenizer import DnaTokenizer

        return DnaTokenizer
    if name in {"BaseTokenizer", "build_tokenizer"}:
        from .tokenizer import BaseTokenizer, build_tokenizer

        return {"BaseTokenizer": BaseTokenizer, "build_tokenizer": build_tokenizer}[name]
    if name == "DatasetImpl":
        from .dataset import DatasetImpl

        return DatasetImpl
    if name == "DnaModel":
        from .model import DnaModel

        return DnaModel
    if name in {"get_scheduler", "evaluate"}:
        from .utils import evaluate, get_scheduler

        return {"evaluate": evaluate, "get_scheduler": get_scheduler}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
