import runpy
import types
from copy import deepcopy
from pathlib import Path

CONFIG_ROOT = Path(__file__).resolve().parent


class Config(dict):

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self[name] = _wrap(value)


def _wrap(obj):
    if isinstance(obj, dict):
        return Config({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return type(obj)(_wrap(v) for v in obj)
    return obj


def _merge(base, override):
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def resolve(path):
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    in_package = CONFIG_ROOT / path
    return (in_package if in_package.exists() else path).resolve()


def _load(path):
    path = resolve(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{path}")

    namespace = runpy.run_path(str(path))
    cfg = {}

    #_base_ 里的路径相对当前配置文件所在目录
    for base in namespace.get("_base_", []):
        cfg = _merge(cfg, _load(path.parent / base))

    own = {
        key: value
        for key, value in namespace.items()
        if not key.startswith("_")
        and not callable(value)
        and not isinstance(value, types.ModuleType)
    }

    return _merge(cfg, own)


def load_config(path):
    merged = _load(path)
    if "name" not in merged.get("model", {}):
        raise ValueError(f"{Path(path).name} 里没有指定 model=dict(name=...)")

    return _wrap(merged)


def to_plain(cfg):
    """写 json 用"""
    if isinstance(cfg, dict):
        return {k: to_plain(v) for k, v in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return [to_plain(v) for v in cfg]
    return cfg
