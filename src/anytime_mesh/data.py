"""Dataset configuration: map part ids ('<source>/<relative path>') to local CAD files."""
import os
import tomllib


def load_config(path: str) -> dict:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def source_roots(cfg: dict) -> dict:
    """{source: root dir} for every source configured with an existing directory."""
    return {k: v for k, v in cfg.get("sources", {}).items() if v and os.path.isdir(v)}


def resolve(part_id: str, cfg: dict) -> str:
    src, rel = part_id.split("/", 1)
    root = cfg.get("sources", {}).get(src)
    if not root:
        raise KeyError(f"no root configured for source '{src}' (configs/data.toml)")
    return os.path.join(root, rel)
