"""Meaningful tests for the trusted project store (PETER-14).

Proves: durable round-trip across reopen, audience/ownership denial, forged
task-reference rejection, archive/traversal/bomb/size rejection, hash and
manifest integrity, partial checkpoints, quota/retention, and restrictive
blob permissions.
"""
import base64
import hashlib
import io
import os
import shutil
import stat
import tarfile
import zipfile

import pytest

from peterbot.agent_policy import Principal
from peterbot.project_store import (
    ProjectDenied,
    ProjectIntegrityError,
    ProjectQuota,
    ProjectSettings,
    ProjectStore,
    ProjectViolation,
)

MEMBER = Principal(10, 1, 20)
OTHER = Principal(10, 2, 20)
OTHER_CH = Principal(10, 1, 21)
OTHER_GUILD = Principal(11, 1, 20)

CARGO = b"""[package]
name = "edigits"
version = "0.1.0"
edition = "2021"

[dependencies]
"""
MAIN_V1 = b"""fn digit_sum(n: u32) -> u32 {
    n.to_string().chars().map(|c| c.to_digit(10).unwrap()).sum()
}

fn main() {
    println!("{}", digit_sum(12345));
}
"""
MAIN_V2 = MAIN_V1.replace(b"digit_sum(12345)", b"digit_sum(987654)")
TESTS_RS = b"""#[cfg(test)]
mod tests {
    use super::digit_sum;

    #[test]
    fn sums_digits() {
        assert_eq!(digit_sum(12345), 15);
    }
}
"""


def rust_project():
    """A small generated Rust project as a validated file mapping."""
    return {"Cargo.toml": CARGO, "src/main.rs": MAIN_V1, "src/tests.rs": TESTS_RS}


def make(tmp_path, **overrides):
    clock = getattr(make, "clock", None)
    store = ProjectStore(tmp_path / "projects",
                         settings=ProjectSettings(**overrides),
                         clock=clock)
    return store


@pytest.fixture
def store(tmp_path):
    s = make(tmp_path)
    yield s
    s.close()


@pytest.fixture
def project(store):
    return store.create_project(MEMBER, name="edigits", task_id="task-a1")


def digest(data):
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------- durability core

def test_rust_project_survives_reopen_exact_bytes_then_modified_version(store, tmp_path, project):
    files = rust_project()
    saved = store.save(MEMBER, project["id"], task_id="task-a1", files=files,
                       provenance="task task-a1 generated the edigits calculator",
                       dependency_instructions="cargo --offline build")
    assert saved["state"] == "verified" and saved["version"] == 1

    # Simulated gateway restart: close, reopen the same root.
    store.close()
    reopened = make(tmp_path)
    try:
        restored = reopened.restore(MEMBER, project["id"], task_id="task-b2")
        assert dict(restored["files"]) == files
        assert restored["state"] == "verified"
        assert restored["dependency_instructions"] == "cargo --offline build"
        assert "task-a1" in restored["provenance"]

        # Continuation edits the restored source and saves a new version.
        modified = dict(files)
        modified["src/main.rs"] = MAIN_V2
        second = reopened.save(MEMBER, project["id"], task_id="task-b2", files=modified,
                               provenance="task task-b2 changed the demo number",
                               verified=True)
        assert second["version"] == 2
        assert dict(reopened.restore(MEMBER, project["id"], task_id="task-b2")["files"]) == modified
        # Older version stays addressable byte-for-byte.
        old = reopened.list_files(MEMBER, project["id"], version=1)
        assert {f["name"] for f in old["files"]} == set(files)
        assert reopened.read_file(MEMBER, project["id"], "src/main.rs", version=1) == MAIN_V1
        assert reopened.verify(MEMBER, project["id"])["ok"] is True
    finally:
        reopened.close()


def test_blob_permissions_are_owner_only(store, project):
    store.save(MEMBER, project["id"], task_id="task-a1", files=rust_project(),
               provenance="task-a1 output", verified=True)
    assert stat.S_IMODE(os.stat(store.root).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(store.blob_dir).st_mode) == 0o700
    blobs = [p for sub in store.blob_dir.iterdir() for p in sub.iterdir()]
    assert blobs
    for blob in blobs:
        assert stat.S_IMODE(os.stat(blob).st_mode) == 0o600


def test_worker_payload_is_base64_and_round_trips(store, project):
    store.save(MEMBER, project["id"], task_id="task-a1", files=rust_project(),
               provenance="task-a1", verified=True)
    payload = store.worker_payload(MEMBER, project["id"], task_id="task-c3")
    assert payload["version"] == 1
    assert payload["state"] == "verified" and payload["provenance"] == "task-a1"
    staged = {f["name"]: base64.b64decode(f["data_base64"]) for f in payload["files"]}
    assert staged == rust_project()
    assert all(f["sha256"] == digest(b) for f in payload["files"]
               for b in [staged[f["name"]]])


def test_files_may_be_pair_sequence(store, project):
    saved = store.save(MEMBER, project["id"], task_id="task-a1",
                       files=[("main.rs", b"fn main() {}")], provenance="x", verified=True)
    assert saved["file_count"] == 1


# ------------------------------------------------------------------ audience

def test_other_users_channels_and_guilds_are_denied_everything(store, project):
    store.save(MEMBER, project["id"], task_id="task-a1", files=rust_project(),
               provenance="private work", verified=True)
    for stranger, method in [
        (OTHER, store.list_projects), (OTHER_CH, store.list_projects),
        (OTHER_GUILD, store.list_projects),
    ]:
        assert method(stranger) == []
    for stranger in (OTHER, OTHER_CH, OTHER_GUILD):
        with pytest.raises(ProjectDenied):
            store.check_access(stranger, project["id"])
        with pytest.raises(ProjectDenied):
            store.describe(stranger, project["id"])
        with pytest.raises(ProjectDenied):
            store.list_files(stranger, project["id"])
        with pytest.raises(ProjectDenied):
            store.read_file(stranger, project["id"], "src/main.rs")
        with pytest.raises(ProjectDenied):
            store.restore(stranger, project["id"], task_id="sneaky")
        with pytest.raises(ProjectDenied):
            store.save(stranger, project["id"], task_id="sneaky",
                       files={"evil.rs": b"x"}, provenance="forged", verified=True)
        with pytest.raises(ProjectDenied):
            store.delete_project(stranger, project["id"])
        with pytest.raises(ProjectDenied):
            store.share(stranger, project["id"], user_ids=[9])
    # The denial gives no existence oracle.
    with pytest.raises(ProjectDenied) as missing:
        store.describe(OTHER, "0" * 32)
    assert str(missing.value)  # same generic class as a hidden private project


def test_unknown_project_id_shape_is_rejected(store):
    with pytest.raises(ProjectViolation):
        store.check_access(MEMBER, "../projects")
    with pytest.raises(ProjectViolation):
        store.describe(MEMBER, "ABCD")


def test_explicit_share_and_immediate_revocation(store, project):
    store.save(MEMBER, project["id"], task_id="task-a1", files=rust_project(),
               provenance="collab seed", verified=True)
    with pytest.raises(ProjectDenied):
        store.share(OTHER, project["id"], user_ids=[1])  # non-owner cannot grant
    store.share(MEMBER, project["id"], user_ids=[2])
    assert store.check_access(OTHER, project["id"])["audience"] == "shared"
    assert dict(store.restore(OTHER, project["id"], task_id="collab-t1")["files"]) == rust_project()
    store.revoke(MEMBER, project["id"], user_id=2)
    with pytest.raises(ProjectDenied):
        store.check_access(OTHER, project["id"])
    with pytest.raises(ProjectDenied):
        store.restore(OTHER, project["id"], task_id="collab-t1")
    with pytest.raises(ProjectDenied):
        store.list_files(OTHER, project["id"])


def test_relocate_changes_the_only_audience_channel_immediately(store, project):
    store.save(MEMBER, project["id"], task_id="task-a1", files=rust_project(),
               provenance="moved work", verified=True)
    store.relocate(MEMBER, project["id"], channel_id=99)
    assert store.check_access(Principal(10, 1, 99), project["id"])["channel_id"] == 99
    with pytest.raises(ProjectDenied):
        store.check_access(MEMBER, project["id"])  # old channel audience is gone
    with pytest.raises(ProjectDenied):
        store.relocate(OTHER, project["id"], channel_id=99)


def test_delete_purges_blobs_and_manifest(store, project):
    store.save(MEMBER, project["id"], task_id="task-a1", files=rust_project(),
               provenance="doomed", verified=True)
    result = store.delete_project(MEMBER, project["id"])
    assert result["deleted"] is True
    assert not [p for sub in store.blob_dir.iterdir() for p in sub.iterdir()]
    with pytest.raises(ProjectDenied):
        store.list_files(MEMBER, project["id"])
    assert store.list_projects(MEMBER) == []


def test_shared_blob_survives_other_owners_deletion(store, project):
    """Content-addressed blobs are referenced by every manifest using them; one
    owner's purge must not corrupt another project's bytes."""
    shared = b"fn common() -> u32 { 7 }"
    store.save(MEMBER, project["id"], task_id="task-a1",
               files={"lib.rs": shared}, provenance="victim", verified=True)
    attacker = store.create_project(OTHER, name="attacker", task_id="task-evil")
    store.save(OTHER, attacker["id"], task_id="task-evil",
               files={"copy.rs": shared}, provenance="independent copy", verified=True)
    store.delete_project(MEMBER, project["id"])
    assert store.read_file(OTHER, attacker["id"], "copy.rs") == shared

# ------------------------------------------------------- forged task references

def test_task_id_is_bound_to_one_project_across_reopen(store, tmp_path, project):
    store.save(MEMBER, project["id"], task_id="task-a1", files=rust_project(),
               provenance="origin", verified=True)
    other = store.create_project(MEMBER, name="other", task_id="task-z9")
    store.save(MEMBER, other["id"], task_id="task-z9", files={"a.rs": b"a"},
               provenance="origin", verified=True)
    store.close()
    reopened = make(tmp_path)
    try:
        # task-a1 legitimately restores its own project.
        assert reopened.restore(MEMBER, project["id"], task_id="task-a1")["version"] == 1
        # The same task id against a different project is a forgery.
        with pytest.raises(ProjectDenied):
            reopened.restore(MEMBER, other["id"], task_id="task-a1")
        with pytest.raises(ProjectDenied):
            reopened.save(MEMBER, other["id"], task_id="task-a1", files={"x.rs": b"x"},
                          provenance="forged", verified=True)
        # A fresh continuation task binds to the first project it restores.
        reopened.restore(MEMBER, project["id"], task_id="task-new")
        with pytest.raises(ProjectDenied):
            reopened.restore(MEMBER, other["id"], task_id="task-new")
    finally:
        reopened.close()


def test_forged_manifest_reference_cannot_reach_another_projects_blobs(store, project):
    """Replaying a victim's task id or project id must not read foreign bytes."""
    store.save(MEMBER, project["id"], task_id="task-a1", files=rust_project(),
               provenance="victim work", verified=True)
    attacker = store.create_project(OTHER, name="attacker", task_id="task-evil")
    # No API addresses blobs by digest alone; reads go through a visible
    # manifest row only.
    assert not hasattr(store, "read_blob") and not hasattr(store, "get_files")
    with pytest.raises(ProjectDenied):
        store.restore(OTHER, project["id"], task_id="task-evil")
    with pytest.raises(ProjectDenied):
        store.save(OTHER, project["id"], task_id="task-evil",
                   files={"x.rs": b"x"}, provenance="forged", verified=True)
    # The victim's bound task id cannot be replayed against another project.
    with pytest.raises(ProjectDenied):
        store.restore(OTHER, attacker["id"], task_id="task-a1")
    with pytest.raises(ProjectDenied):
        store.save(OTHER, attacker["id"], task_id="task-a1",
                   files={"x.rs": b"x"}, provenance="forged", verified=True)
    # The attacker re-saving a victim-looking path only ever gets their own bytes.
    store.save(OTHER, attacker["id"], task_id="task-evil",
               files={"src/main.rs": b"mine"}, provenance="mine", verified=True)
    assert dict(store.restore(OTHER, attacker["id"], task_id="task-evil")["files"]) \
        == {"src/main.rs": b"mine"}


def test_principal_override_arguments_do_not_exist(store, project):
    with pytest.raises(TypeError):
        store.list_projects(MEMBER, user_id=1)
    with pytest.raises(TypeError):
        store.create_project(MEMBER, name="x", task_id="t", owner_user_id=1)
    with pytest.raises(AttributeError):
        Principal(10, 1, 20).with_guild(11)


# ----------------------------------------------------------------- file safety

@pytest.mark.parametrize("name", [
    "../escape.rs", "src/../../escape.rs", "/etc/passwd", "C:/windows/system32",
    "src\\main.rs", "src//main.rs", "..", ".", "a/../b.rs", "src/./main.rs",
    "CON.rs", "nul.txt", "com1.log", "trailing.rs.", "trailing.rs ",
    "x" * 241, "a" * 300 + ".rs", "bad\x00name.rs", "tab\tname.rs",
    "zwj\u200bname.rs", "caf\u0065\u0301.rs", "surrogate\ud800.rs",
])
def test_unsafe_filenames_are_rejected(store, project, name):
    with pytest.raises(ProjectViolation):
        store.save(MEMBER, project["id"], task_id="task-a1", files={name: b"x"},
                   provenance="x", verified=True)


def test_prefix_conflict_between_file_and_directory(store, project):
    with pytest.raises(ProjectViolation):
        store.save(MEMBER, project["id"], task_id="task-a1",
                   files={"a": b"file", "a/b": b"nested"}, provenance="x", verified=True)


def test_duplicate_and_case_colliding_names_are_rejected(store, project):
    with pytest.raises(ProjectViolation):
        store.save(MEMBER, project["id"], task_id="task-a1",
                   files=[("main.rs", b"one"), ("main.rs", b"two")],
                   provenance="x", verified=True)
    with pytest.raises(ProjectViolation):
        store.save(MEMBER, project["id"], task_id="task-a1",
                   files=[("Main.rs", b"one"), ("main.rs", b"two")],
                   provenance="x", verified=True)


@pytest.mark.parametrize("name,payload", [
    ("drop.zip", b"PK\x03\x04rest"),
    ("shell.sh.gz", b"\x1f\x8b\x08\x00rest"),
    ("crate.tar", b"x" * 257 + b"ustar"),
    ("crate.tar", b"\x00" * 257 + b"ustar"),
    ("pack.7z", b"7z\xbc\xaf\x27\x1cdata"),
    ("pack.rar", b"Rar!\x1a\x07\x01\x00"),
    ("fold.zst", b"\x28\xb5\x2f\xfddata"),
    ("pkg.deb", b"!<arch>\n"),
    ("pkg.rpm", b"\xed\xab\xee\xdbdata"),
    ("plain.txt", b"PK\x03\x04zip-that-pretends-to-be-text"),
])
def test_archive_and_compressed_inputs_are_rejected(store, project, name, payload):
    with pytest.raises(ProjectViolation):
        store.save(MEMBER, project["id"], task_id="task-a1", files={name: payload},
                   provenance="x", verified=True)


def test_zip_bomb_style_payload_is_rejected_not_deflated(store, tmp_path):
    """A real high-compression bomb: 1 MiB of zeros zips to ~1 KiB. The store
    refuses the container outright; nothing is ever decompressed."""
    bomb = bytearray(b"\x00" * (1024 * 1024))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("payload", bytes(bomb))
    tiny = buf.getvalue()
    assert len(tiny) * 100 < len(bomb)
    store = make(tmp_path, max_file_bytes=len(tiny) + 10)
    try:
        project = store.create_project(MEMBER, name="bomb", task_id="task-bomb")
        with pytest.raises(ProjectViolation):
            store.save(MEMBER, project["id"], task_id="task-bomb",
                       files={"payload.zip": tiny}, provenance="x", verified=True)
        # Renamed without an archive extension, the magic prefix still trips it.
        with pytest.raises(ProjectViolation):
            store.save(MEMBER, project["id"], task_id="task-bomb",
                       files={"payload.rs": tiny}, provenance="x", verified=True)
        assert store.list_projects(MEMBER)[0]["total_bytes"] == 0
    finally:
        store.close()


def test_tar_member_payload_rejected_by_magic(store, tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as archive:
        info = tarfile.TarInfo("../../evil.rs")
        data = b"fn evil() {}"
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    store = make(tmp_path)
    project = store.create_project(MEMBER, name="tar", task_id="task-tar")
    with pytest.raises(ProjectViolation):
        store.save(MEMBER, project["id"], task_id="task-tar",
                   files={"drop.rs": buf.getvalue()}, provenance="x", verified=True)
    store.close()


def test_oversized_file_and_version_are_rejected(tmp_path):
    store = make(tmp_path, max_file_bytes=128, max_version_bytes=200)
    project = store.create_project(MEMBER, name="sizes", task_id="task-s")
    with pytest.raises(ProjectQuota):
        store.save(MEMBER, project["id"], task_id="task-s",
                   files={"big.rs": b"x" * 129}, provenance="x", verified=True)
    with pytest.raises(ProjectQuota):
        store.save(MEMBER, project["id"], task_id="task-s",
                   files={"a.rs": b"x" * 128, "b.rs": b"y" * 100},
                   provenance="x", verified=True)
    store.close()


def test_non_bytes_content_rejected(store, project):
    with pytest.raises(ProjectViolation):
        store.save(MEMBER, project["id"], task_id="task-a1",
                   files={"main.rs": "text not bytes"}, provenance="x", verified=True)
    with pytest.raises(ProjectViolation):
        store.save(MEMBER, project["id"], task_id="task-a1", files="main.rs",
                   provenance="x", verified=True)


# -------------------------------------------------------------- integrity

def test_tampered_blob_fails_integrity_before_any_bytes_escap(store, project):
    store.save(MEMBER, project["id"], task_id="task-a1", files=rust_project(),
               provenance="integrity", verified=True)
    target = store._blob_path(digest(MAIN_V1))
    target.write_bytes(b"fn main(){ evil(); }")  # same-ish size differs; force size too
    with pytest.raises(ProjectIntegrityError):
        store.verify(MEMBER, project["id"])
    with pytest.raises(ProjectIntegrityError):
        store.read_file(MEMBER, project["id"], "src/main.rs")
    with pytest.raises(ProjectIntegrityError):
        store.restore(MEMBER, project["id"], task_id="task-a1")


def test_truncated_blob_fails_size_check(store, project):
    store.save(MEMBER, project["id"], task_id="task-a1", files=rust_project(),
               provenance="integrity", verified=True)
    store._blob_path(digest(MAIN_V1)).write_bytes(b"cut")
    with pytest.raises(ProjectIntegrityError):
        store.read_file(MEMBER, project["id"], "src/main.rs")


def test_symlinked_blob_is_refused(store, project):
    store.save(MEMBER, project["id"], task_id="task-a1", files=rust_project(),
               provenance="integrity", verified=True)
    real = store._blob_path(digest(MAIN_V1))
    decoy = store.root / "decoy.bin"
    decoy.write_bytes(b"fn main(){ stolen(); }")
    shutil.move(real, store.root / "hidden.bin")
    os.symlink(decoy, real)
    with pytest.raises(ProjectIntegrityError):
        store.read_file(MEMBER, project["id"], "src/main.rs")
    with pytest.raises(ProjectIntegrityError):
        store.verify(MEMBER, project["id"])


def test_manifest_provenance_version_and_hash_are_accurate(store, project):
    files = rust_project()
    store.save(MEMBER, project["id"], task_id="task-a1", files=files,
               provenance="task-a1 via hermes worker", verified=True,
               dependency_instructions="no crates; std only")
    listing = store.list_files(MEMBER, project["id"])
    assert listing["state"] == "verified"
    assert listing["provenance"] == "task-a1 via hermes worker"
    assert listing["dependency_instructions"] == "no crates; std only"
    for entry in listing["files"]:
        raw = store.read_file(MEMBER, project["id"], entry["name"])
        assert entry["sha256"] == digest(raw)
        assert entry["size"] == len(raw)
    summary = store.describe(MEMBER, project["id"])
    assert summary["latest_version"] == 1
    assert summary["total_bytes"] == sum(len(b) for b in files.values())
    assert summary["origin_task_id"] == "task-a1"
    with pytest.raises(ProjectViolation):
        store.list_files(MEMBER, project["id"], version=7)


def test_provenance_and_name_validation(store, project):
    with pytest.raises(ProjectViolation):
        store.save(MEMBER, project["id"], task_id="task-a1", files={"a.rs": b"a"},
                   provenance="with\nnewline", verified=True)
    with pytest.raises(ProjectViolation):
        store.save(MEMBER, project["id"], task_id="task-a1", files={"a.rs": b"a"},
                   provenance="", verified=True)
    with pytest.raises(ProjectViolation):
        store.save(MEMBER, project["id"], task_id="task-a1", files={"a.rs": b"a"},
                   provenance="x", verified="yes")
    with pytest.raises(ProjectViolation):
        store.save(MEMBER, project["id"], task_id=" ", files={"a.rs": b"a"},
                   provenance="x", verified=True)
    with pytest.raises(ProjectViolation):
        store.create_project(MEMBER, name="zero\x00width\u200b", task_id="t-ok")


# --------------------------------------------------- partial + quota + retention

def test_best_effort_partial_checkpoint_keeps_valid_files(store, project):
    result = store.save(
        MEMBER, project["id"], task_id="task-a1",
        files={"good.rs": b"fn good() {}", "../evil.rs": b"x",
               "payload.zip": b"PK\x03\x04zip", "big.bin": b"z" * 64},
        provenance="salvaged after timeout", verified=False, best_effort=True)
    assert result["state"] == "partial"
    assert result["file_count"] == 2
    reasons = {r["name"] for r in result["rejected"]}
    assert {"../evil.rs", "payload.zip"} <= reasons
    restored = store.restore(MEMBER, project["id"], task_id="task-a1")
    assert restored["state"] == "partial"
    assert dict(restored["files"]) == {"big.bin": b"z" * 64, "good.rs": b"fn good() {}"}
    # Strict mode still fails closed on the same input.
    with pytest.raises(ProjectViolation):
        store.save(MEMBER, project["id"], task_id="task-a1",
                   files={"ok.rs": b"a", "../evil.rs": b"x"},
                   provenance="strict", verified=True)
    # A save of only-invalid entries claims nothing.
    with pytest.raises(ProjectViolation):
        store.save(MEMBER, project["id"], task_id="task-a1",
                   files={"../x": b"y"}, provenance="junk", verified=False, best_effort=True)


def test_project_byte_quota_bounds_growth(tmp_path):
    store = make(tmp_path, max_project_bytes=300)
    project = store.create_project(MEMBER, name="quota", task_id="task-q")
    store.save(MEMBER, project["id"], task_id="task-q",
               files={"a.rs": b"a" * 100}, provenance="one", verified=True)
    store.save(MEMBER, project["id"], task_id="task-q",
               files={"a.rs": b"b" * 100}, provenance="two", verified=True)
    with pytest.raises(ProjectQuota):
        store.save(MEMBER, project["id"], task_id="task-q",
                   files={"a.rs": b"c" * 120}, provenance="three", verified=True)
    # Rejected save changed nothing observable.
    assert store.describe(MEMBER, project["id"])["latest_version"] == 2
    store.close()


def test_user_byte_quota_across_projects(tmp_path):
    store = make(tmp_path, max_user_bytes=250)
    first = store.create_project(MEMBER, name="p1", task_id="t1")
    second = store.create_project(MEMBER, name="p2", task_id="t2")
    store.save(MEMBER, first["id"], task_id="t1", files={"a": b"x" * 200},
               provenance="big", verified=True)
    with pytest.raises(ProjectQuota):
        store.save(MEMBER, second["id"], task_id="t2", files={"b": b"y" * 100},
                   provenance="more", verified=True)
    store.close()


def test_version_count_evicts_oldest_and_gcs_its_blobs(tmp_path):
    store = make(tmp_path, max_versions_per_project=2)
    project = store.create_project(MEMBER, name="evict", task_id="task-v")
    for n in range(3):
        store.save(MEMBER, project["id"], task_id="task-v",
                   files={"main.rs": f"v{n}".encode()}, provenance=f"v{n}", verified=True)
    summary = store.describe(MEMBER, project["id"])
    assert summary["latest_version"] == 3 and summary["state"] == "verified"
    with pytest.raises(ProjectViolation):
        store.list_files(MEMBER, project["id"], version=1)
    # v1's unique blob is gone from disk; v3's is present.
    assert not store._blob_path(digest(b"v0")).exists()
    assert store._blob_path(digest(b"v2")).exists()
    assert store.verify(MEMBER, project["id"])["ok"] is True
    store.close()


def test_per_user_project_count_cap(tmp_path):
    store = make(tmp_path, max_projects_per_user=2)
    store.create_project(MEMBER, name="p1", task_id="t1")
    store.create_project(MEMBER, name="p2", task_id="t2")
    with pytest.raises(ProjectQuota):
        store.create_project(MEMBER, name="p3", task_id="t3")
    # The cap is per user, not global.
    assert store.create_project(OTHER, name=" theirs ", task_id="t4")["name"] == "theirs"
    store.close()


def test_retention_drops_old_versions_keeps_latest_and_aged_project(tmp_path):
    now = [1_000_000]
    make.clock = lambda: now[0]
    try:
        store = make(tmp_path, retention_days=90)
        kept = store.create_project(MEMBER, name="kept", task_id="t-keep")
        aged = store.create_project(MEMBER, name="aged", task_id="t-aged")
        store.save(MEMBER, kept["id"], task_id="t-keep", files={"a": b"one"},
                   provenance="old v1", verified=True)
        store.save(MEMBER, aged["id"], task_id="t-aged", files={"old": b"three"},
                   provenance="whole old project", verified=True)
        now[0] += 91 * 86_400
        # kept gets a fresh latest version; aged is fully abandoned.
        store.save(MEMBER, kept["id"], task_id="t-keep", files={"a": b"two"},
                   provenance="newest v2", verified=True)
        report = store.retention_sweep()
        assert report["versions_removed"] == 2  # kept/v1 and aged/v1
        assert report["projects_removed"] == 1  # aged: nothing survived
        assert store.describe(MEMBER, kept["id"])["latest_version"] == 2
        assert store.read_file(MEMBER, kept["id"], "a") == b"two"
        assert all(p["id"] != aged["id"] for p in store.list_projects(MEMBER))
        assert not store._blob_path(digest(b"one")).exists()
        assert not store._blob_path(digest(b"three")).exists()
        assert store._blob_path(digest(b"two")).exists()
        store.close()
    finally:
        del make.clock


def test_future_project_schema_fails_closed(tmp_path):
    root = tmp_path / "projects"
    store = ProjectStore(root)
    store._conn.execute("PRAGMA user_version=200")
    store.close()
    with pytest.raises(ProjectIntegrityError, match="newer"):
        ProjectStore(root)


def _age_blobs(store, ts):
    """Backdate every blob so the sweep's one-hour grace window has passed."""
    for sub in store.blob_dir.iterdir():
        if sub.is_dir():
            for entry in sub.iterdir():
                os.utime(entry, (ts - 7200, ts - 7200))


def test_orphan_blob_from_crashed_save_is_reclaimed(tmp_path):
    now = [2_000_000]
    make.clock = lambda: now[0]
    try:
        store = make(tmp_path, retention_days=90)
        project = store.create_project(MEMBER, name="crash", task_id="t-crash")
        store.save(MEMBER, project["id"], task_id="t-crash", files={"a.rs": b"kept"},
                   provenance="ok", verified=True)
        # Crash between blob write and manifest commit: bytes exist, no row.
        orphan = digest(b"never committed")
        store._write_blob(orphan, b"never committed")
        assert store._blob_path(orphan).exists()
        # Sweep with a young clock must NOT touch it (in-flight save race).
        report = store.retention_sweep()
        assert report["orphan_blobs_removed"] == 0
        assert store._blob_path(orphan).exists()
        # Past the grace window it is reclaimed; the referenced blob survives.
        _age_blobs(store, now[0])
        report = store.retention_sweep()
        assert report["orphan_blobs_removed"] == 1
        assert not store._blob_path(orphan).exists()
        assert store.read_file(MEMBER, project["id"], "a.rs") == b"kept"
        store.close()
    finally:
        del make.clock


def test_sweep_never_deletes_foreign_or_symlinked_blob_entries(tmp_path):
    now = [3_000_000]
    make.clock = lambda: now[0]
    try:
        store = make(tmp_path, retention_days=90)
        project = store.create_project(MEMBER, name="weird", task_id="t-weird")
        store.save(MEMBER, project["id"], task_id="t-weird", files={"a.rs": b"x"},
                   provenance="ok", verified=True)
        sub = next(p for p in store.blob_dir.iterdir() if p.is_dir())
        stranger = sub / "not-a-blob.bin"
        stranger.write_bytes(b"operator file")
        fake = sub / ("e" * 64)
        link_target = store.root / "outside.bin"
        link_target.write_bytes(b"important elsewhere")
        os.symlink(link_target, fake)
        _age_blobs(store, now[0])
        os.utime(link_target, (now[0] - 7200, now[0] - 7200))
        report = store.retention_sweep()
        assert report["orphan_blobs_removed"] == 0
        assert stranger.exists() and link_target.exists()
        assert fake.is_symlink()  # symlinks are never followed or unlinked
        store.close()
    finally:
        del make.clock


def test_shared_blob_survives_sweep_and_only_full_deletion_frees_it(tmp_path):
    now = [4_000_000]
    make.clock = lambda: now[0]
    try:
        store = make(tmp_path, retention_days=90)
        shared = b"fn common() -> u32 { 7 }"
        victim = store.create_project(MEMBER, name="victim", task_id="t-victim")
        store.save(MEMBER, victim["id"], task_id="t-victim", files={"lib.rs": shared},
                   provenance="v", verified=True)
        # Holder stays live past the sweep: its manifest must keep the shared
        # blob even though the victim's manifest (and delete-time GC) ran.
        now[0] += 91 * 86_400
        holder = store.create_project(OTHER, name="holder", task_id="t-holder")
        store.save(OTHER, holder["id"], task_id="t-holder", files={"copy.rs": shared},
                   provenance="h", verified=True)
        store.delete_project(MEMBER, victim["id"])
        _age_blobs(store, now[0])
        report = store.retention_sweep()
        assert report["blobs_removed"] == 0 and report["orphan_blobs_removed"] == 0
        assert store.read_file(OTHER, holder["id"], "copy.rs") == shared
        # Only after the last manifest drops it is the blob reclaimed.
        store.delete_project(OTHER, holder["id"])
        assert not store._blob_path(digest(shared)).exists()
        store.close()
    finally:
        del make.clock
