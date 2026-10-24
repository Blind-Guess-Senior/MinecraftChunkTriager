from __future__ import annotations

import io
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


@pytest.fixture
def workspace(tmp_path: Path, world: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "triage.toml").write_text(
        f"""
world = {str(world)!r}
game_version = "java-26.2"

[mcaselector]
command = ["java", "-jar", "mcaselector.jar"]

[[layers]]
name = "surface"
title = "Surface"

[export]
buffer = 1
""",
        encoding="utf-8",
    )
    return ws
