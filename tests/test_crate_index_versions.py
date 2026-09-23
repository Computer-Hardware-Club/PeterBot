"""Offline Cargo cache preserves more than one pinned version per crate."""
import hashlib
import json

import pytest

from peterbot.hermes_worker import _stage_artifact


def crate(version: str, *, extra: bool = False):
    data = ("crate-" + version).encode()
    digest = hashlib.sha256(data).hexdigest()
    line = {"name": "itoa", "vers": version, "cksum": digest,
            "deps": [], "features": {}, "yanked": False}
    if extra:
        line["features2"] = {"unexpected": []}
    return {"registry": "cratesio", "name": "itoa", "version": version,
            "filename": f"itoa-{version}.crate", "sha256": digest,
            "size": len(data), "index_line": json.dumps(line, separators=(",", ":"))}, data


def test_two_versions_keep_both_index_lines_and_repeated_stage_is_idempotent(tmp_path):
    workspace, cargo_home = tmp_path / "workspace", tmp_path / "cargo"
    first, first_bytes = crate("1.0.14")
    second, second_bytes = crate("1.0.15")
    first_result = _stage_artifact(first, first_bytes, workspace, cargo_home)
    second_result = _stage_artifact(second, second_bytes, workspace, cargo_home)
    index = tmp_path / "cargo" / "registry" / "index" / "it" / "oa" / "itoa"
    assert first_result["index"] == second_result["index"] == str(index)
    assert index.read_text().splitlines() == [first["index_line"], second["index_line"]]
    _stage_artifact(first, first_bytes, workspace, cargo_home)
    assert index.read_text().splitlines() == [first["index_line"], second["index_line"]]

    conflicting, data = crate("1.0.14", extra=True)
    with pytest.raises(ValueError, match="conflicting version"):
        _stage_artifact(conflicting, data, workspace, cargo_home)
    assert index.read_text().splitlines() == [first["index_line"], second["index_line"]]
