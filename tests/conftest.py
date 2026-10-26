from __future__ import annotations

import io
import json
import zlib
from pathlib import Path

import nbtlib
import pytest
from nbtlib import Compound, Int, List, String

SECTOR = 4096
DATA_VERSION = 4900  # inside the java-26.2 range


def chunk_nbt(cx: int, cz: int, data_version: int = DATA_VERSION, block_entities=()) -> bytes:
    root = nbtlib.File({
        "DataVersion": Int(data_version),
        "xPos": Int(cx),
        "zPos": Int(cz),
        "block_entities": List[Compound]([
            Compound({"id": String(be_id), "x": Int(x), "y": Int(y), "z": Int(z)})
            for be_id, x, y, z in block_entities
        ]),
    })
    buf = io.BytesIO()
    root.write(buf)
    return buf.getvalue()


def write_region(path: Path, chunks: dict[tuple[int, int], bytes]) -> None:
    """Write an Anvil region file; ``chunks`` maps absolute chunk pos to raw NBT."""
    header = bytearray(2 * SECTOR)
    body = bytearray()
    sector = 2
    for (cx, cz), raw in chunks.items():
        payload = zlib.compress(raw)
        data = (len(payload) + 1).to_bytes(4, "big") + b"\x02" + payload
        data += b"\x00" * (-len(data) % SECTOR)
        count = len(data) // SECTOR
        index = (cx % 32) + (cz % 32) * 32
        header[index * 4 : index * 4 + 4] = ((sector << 8) | count).to_bytes(4, "big")
        body += data
        sector += count
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(header) + bytes(body))


@pytest.fixture
def world(tmp_path: Path) -> Path:
    """World copy with two overworld regions in the 26.x folder layout.

    r.0.0 holds chunks (0,0) with a chest, (1,0) and (0,1).
    r.-1.0 holds chunk (-1,0) and a second chunk (-32,0) at its far edge.
    """
    region_dir = tmp_path / "world" / "dimensions" / "minecraft" / "overworld" / "region"
    write_region(region_dir / "r.0.0.mca", {
        (0, 0): chunk_nbt(0, 0, block_entities=[("minecraft:chest", 3, 64, 4),
                                                ("minecraft:chest", 5, 64, 4),
                                                ("minecraft:sign", 1, 65, 1)]),
        (1, 0): chunk_nbt(1, 0),
        (0, 1): chunk_nbt(0, 1),
    })
    write_region(region_dir / "r.-1.0.mca", {
        (-1, 0): chunk_nbt(-1, 0),
        (-32, 0): chunk_nbt(-32, 0),
    })
    return tmp_path / "world"


def write_config(workdir: Path, worlds: dict[str, tuple[Path, str]], extra: str = "") -> Path:
    """Working directory whose config maps world name to (path, role); returns the directory."""
    entries = "".join(
        f"[[worlds]]\nname = {json.dumps(name)}\npath = {json.dumps(p.as_posix())}\nrole = {json.dumps(role)}\n"
        for name, (p, role) in worlds.items()
    )
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "chunktriager.toml").write_text(f"{extra}\n{entries}", encoding="utf-8")
    return workdir


@pytest.fixture
def config(tmp_path: Path, world: Path) -> Path:
    return write_config(tmp_path / "setup", {"w": (world, "train")})


def pack(indices, bits: int):
    """Pack palette indices into the long-array layout used since 1.16."""
    per_long = 64 // bits
    longs = []
    for start in range(0, len(indices), per_long):
        v = 0
        for j, idx in enumerate(indices[start:start + per_long]):
            v |= int(idx) << (j * bits)
        longs.append(v - (1 << 64) if v >= 1 << 63 else v)
    return nbtlib.LongArray(longs)


def section_nbt(y: int, palette: list[str], indices=None, biome: str = "minecraft:plains"):
    """A section compound; ``indices`` is 4096 palette indices in y/z/x order."""
    from nbtlib import Byte, LongArray  # noqa: F401

    states = {"palette": List[Compound]([Compound({"Name": String(n)}) for n in palette])}
    if indices is not None and len(palette) > 1:
        states["data"] = pack(indices, max(4, (len(palette) - 1).bit_length()))
    return Compound({
        "Y": nbtlib.Byte(y),
        "block_states": Compound(states),
        "biomes": Compound({"palette": List[String]([String(biome)])}),
    })


def full_chunk_nbt(cx: int, cz: int, sections, block_entities=(), structures=()) -> bytes:
    root = nbtlib.File({
        "DataVersion": Int(DATA_VERSION),
        "xPos": Int(cx), "zPos": Int(cz), "yPos": Int(-4),
        "Status": String("minecraft:full"),
        "sections": List[Compound](sections),
        "block_entities": List[Compound]([
            Compound({"id": String(i), "x": Int(0), "y": Int(y), "z": Int(0), **extra})
            for i, y, extra in block_entities
        ]),
        "structures": Compound({
            "starts": Compound({}),
            "References": Compound({s: nbtlib.LongArray([0]) for s in structures}),
        }),
    })
    buf = io.BytesIO()
    root.write(buf)
    return buf.getvalue()
