"""Generate a version vocabulary (block and biome ids) from MCA Selector's mappings.

Usage: uv run python scripts/gen_vocab.py <mcaselector repo> <mapping version> <out.json>
e.g.   uv run python scripts/gen_vocab.py ../mcaselector 26.2-snapshot-1 \
           src/chunktriager/versions/data/java-26.2.json

The colour table lists every non-air block and every biome of the version.
Air variants are added explicitly. The surface block set comes from the
matching heightmap table. The output order is sorted and must not
change for a released feature schema, since column positions depend on it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

AIR = ["minecraft:air", "minecraft:cave_air", "minecraft:void_air"]


def main() -> None:
    repo, version, reg_dir, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
    rel = f"src/main/resources/mapping/java_26/colors_{version}.json"
    colors = json.loads((repo / rel).read_text(encoding="utf-8"))
    heightmaps = json.loads((repo / rel.replace("colors_", "heightmaps_")).read_text(encoding="utf-8"))
    commit = subprocess.run(["git", "-C", str(repo), "log", "-1", "--format=%H", "--", rel],
                            capture_output=True, text=True, check=True).stdout.strip()
    blocks = sorted(set(colors["colors"]) | set(AIR))
    biomes = sorted(colors["tints"])
    # Ground surface: blocks that stop the ocean-floor heightmap, minus
    # leaves, so water and trees do not count as ground.
    surface = sorted(set(heightmaps["ocean_floor"]) - set(heightmaps["leaves"]))
    registry = lambda name: sorted(
        "minecraft:" + x for x in json.loads((reg_dir / f"{name}.json").read_text(encoding="utf-8")))
    body = {
        "blocks": blocks, "biomes": biomes,
        "entities": registry("entity_type"), "block_entities": registry("block_entity_type"),
        "structures": registry("structure"),
        "surface_blocks": surface, "air_blocks": AIR,
    }
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
    doc = {
        "source": [f"Querz/mcaselector {rel} @ {commit}", "misode/mcmeta <version>-registries"],
        "hash": digest, **body,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(f"{len(blocks)} blocks, {len(biomes)} biomes, {len(body['entities'])} entities, "
          f"{len(body['block_entities'])} block entities, hash {digest} -> {out}")


if __name__ == "__main__":
    main()
