"""Create a pinned vLLM overlay; never patch the installed/reference package."""

import hashlib
import json
from pathlib import Path


def patched_sources(source):
    source = Path(source)
    recipe = json.loads(Path(__file__).with_suffix(".json").read_text())
    contents = {}
    for name, expected in recipe["preimages"].items():
        data = (source / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"lazy GDN patch requires its pinned preimage: {name}")
        contents[name] = data.decode()
    for item in recipe["patches"]:
        name, old = item["file"], item["before"]
        if contents[name].count(old) != 1:
            raise ValueError(f"lazy GDN patch anchor changed: {item['purpose']}")
        contents[name] = contents[name].replace(old, item["after"])
    for name, text in contents.items():
        compile(text, name, "exec")
        if hashlib.sha256(text.encode()).hexdigest() != recipe["postimages"][name]:
            raise ValueError(f"lazy GDN patch postimage changed: {name}")
    return contents, recipe


def overlay(source, destination):
    source, destination = Path(source).resolve(), Path(destination)
    contents, recipe = patched_sources(source)  # validate EVERYTHING before creating files
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(mode=0o700)

    def copy(directory, relative=Path()):
        for child in directory.iterdir():
            rel = relative / child.name
            target = destination / rel
            key = rel.as_posix()
            if key in contents:
                target.write_text(contents[key])
            elif child.is_dir() and any(p.startswith(key + "/") for p in contents):
                target.mkdir()
                copy(child, rel)
            elif child.name != "__pycache__":
                target.symlink_to(child.resolve(), target_is_directory=child.is_dir())

    copy(source)
    return {
        "source": str(source),
        "path": str(destination),
        "preimages": recipe["preimages"],
        "postimages": recipe["postimages"],
        "upstream": recipe["upstream"],
    }
