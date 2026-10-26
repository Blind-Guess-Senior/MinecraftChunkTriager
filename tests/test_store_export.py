from pathlib import Path

import pytest

from chunktriager.export import write_selection_csv
from chunktriager.store import Store, StoreError

DIM = "minecraft:overworld"


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store(tmp_path / "labels.sqlite3")
    s.replace_chunks(DIM, [(x, z) for x in range(5) for z in range(5)])
    yield s
    s.close()


def test_set_labels_ignores_unknown_chunks(store: Store):
    assert store.set_labels(DIM, [(0, 0), (99, 99)], "keep", "single") == 1
    assert store.labels(DIM) == {(0, 0): ("keep", "single")}


def test_relabel_counts_only_real_changes(store: Store):
    store.set_labels(DIM, [(0, 0), (1, 0)], "keep", "area")
    assert store.set_labels(DIM, [(0, 0), (1, 0)], "keep", "area") == 0
    assert store.set_labels(DIM, [(0, 0)], "delete", "single") == 1
    assert store.label_counts(DIM) == {"keep": 1, "delete": 1}


def test_undo_restores_previous_batch(store: Store):
    store.set_labels(DIM, [(0, 0)], "keep", "single")
    store.set_labels(DIM, [(0, 0), (1, 1)], "delete", "area")
    assert store.undo(DIM) == 2
    assert store.labels(DIM) == {(0, 0): ("keep", "single")}
    store.set_labels(DIM, [(0, 0)], None, "single")
    assert store.labels(DIM) == {}
    store.undo(DIM)
    assert store.labels(DIM) == {(0, 0): ("keep", "single")}


def test_invalid_label_rejected(store: Store):
    with pytest.raises(StoreError):
        store.set_labels(DIM, [(0, 0)], "maybe", "single")


def test_selection_csv_matches_mcaselector_format(tmp_path: Path):
    path = tmp_path / "out.csv"
    write_selection_csv(path, [(0, 0), (-1, 0), (-33, 31), (32, -1)])
    assert path.read_bytes() == b"0;0;0;0\n-1;0;-1;0\n-2;0;-33;31\n1;-1;32;-1\n"


def test_concurrent_access_from_threads(store: Store):
    # The server shares one Store across its thread pool.
    from concurrent.futures import ThreadPoolExecutor

    def work(i: int):
        store.set_labels(DIM, [(i % 5, i // 5 % 5)], "keep" if i % 2 else "delete", "single")
        return len(store.labels(DIM, (0, 0, 4, 4)))

    with ThreadPoolExecutor(16) as pool:
        results = list(pool.map(work, range(400)))
    assert all(0 < n <= 25 for n in results)
