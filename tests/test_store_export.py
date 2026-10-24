from pathlib import Path

import pytest

from chunktriager.export import dilate, plan_export, write_selection_csv
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


def test_dilate_is_chebyshev():
    assert dilate([(0, 0)], 1) == {(x, z) for x in (-1, 0, 1) for z in (-1, 0, 1)}


def test_plan_export_buffer_and_unlabeled():
    chunks = {(x, 0) for x in range(6)}
    labels = {(0, 0): "keep", (1, 0): "delete", (2, 0): "delete", (3, 0): "delete"}
    plan = plan_export(chunks, labels, buffer=1)
    # (1,0) is next to a kept chunk; (4,0) and (5,0) are unlabeled.
    assert plan.delete == [(2, 0), (3, 0)]
    assert (plan.kept_by_label, plan.kept_by_buffer, plan.kept_unlabeled) == (1, 1, 2)


def test_plan_export_delete_unlabeled_still_respects_buffer():
    chunks = {(x, 0) for x in range(4)}
    plan = plan_export(chunks, {(0, 0): "keep"}, buffer=1, delete_unlabeled=True)
    assert plan.delete == [(2, 0), (3, 0)]


def test_plan_export_ignores_labels_outside_index():
    plan = plan_export({(0, 0)}, {(0, 0): "delete", (9, 9): "keep"}, buffer=0)
    assert plan.delete == [(0, 0)]
    assert plan.kept_by_label == 0


def test_selection_csv_matches_mcaselector_format(tmp_path: Path):
    path = tmp_path / "out.csv"
    write_selection_csv(path, [(0, 0), (-1, 0), (-33, 31), (32, -1)])
    assert path.read_bytes() == b"0;0;0;0\n-1;0;-1;0\n-2;0;-33;31\n1;-1;32;-1\n"
