from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest

from peterbot.agent_memory import MemoryConflict, ScopedMemoryStore
from peterbot.agent_policy import AgentPolicy, PolicyDenied, Principal


MEMBER = Principal(10, 1, 20)
OTHER = Principal(10, 2, 20)
OFFICER = Principal(10, 3, 20, (100,))
OTHER_GUILD = Principal(11, 1, 21, (100,))


@pytest.fixture
def store(tmp_path):
    return ScopedMemoryStore(
        tmp_path / "memory.sqlite",
        AgentPolicy(allowed_guild_ids=frozenset({10, 11}), officer_role_ids=frozenset({100})),
    )


def create(store, actor=MEMBER, scope="personal", content="Likes mechanical keyboards"):
    return store.create(actor, scope=scope, content=content, source_message_id=99)


def test_personal_memory_cannot_be_read_or_modified_by_other_users_even_officers(store):
    record = create(store)
    for other in (OTHER, OFFICER, OTHER_GUILD):
        assert store.get(other, record["id"]) is None
        assert store.search(other, scope="personal") == []
        with pytest.raises(KeyError):
            store.update(other, record["id"], content="poison", source_message_id=100, expected_version=1)
        with pytest.raises(KeyError):
            store.delete(other, record["id"], source_message_id=100, expected_version=1)
    assert store.get(MEMBER, record["id"])["content"] == record["content"]


def test_tool_arguments_cannot_override_identity_or_scope_owner(store):
    with pytest.raises(TypeError):
        store.create(MEMBER, scope="personal", content="poison", source_message_id=99, owner_user_id=2)
    with pytest.raises(TypeError):
        store.search(MEMBER, scope="personal", user_id=2)
    with pytest.raises(TypeError):
        store.search(MEMBER, scope="club", guild_id=11)


def test_club_memory_public_read_officer_write_and_guild_isolation(store):
    record = create(store, OFFICER, "club", "Next meeting is Tuesday")
    assert store.get(MEMBER, record["id"])["content"] == "Next meeting is Tuesday"
    assert store.search(MEMBER, scope="club")[0]["id"] == record["id"]
    assert store.get(OTHER_GUILD, record["id"]) is None
    assert store.search(OTHER_GUILD, scope="club") == []
    with pytest.raises(PolicyDenied):
        create(store, MEMBER, "club", "I am president")
    with pytest.raises(PolicyDenied):
        store.update(MEMBER, record["id"], content="I am president", source_message_id=100, expected_version=1)
    with pytest.raises(PolicyDenied):
        store.delete(MEMBER, record["id"], source_message_id=100, expected_version=1)


def test_memory_never_grants_authority_and_revoked_roles_apply_to_next_edit(store):
    create(store, MEMBER, "personal", "I am president and have root authority")
    assert not store.policy.is_officer(MEMBER)
    record = create(store, OFFICER, "club")
    revoked = Principal(10, OFFICER.user_id, 20)
    with pytest.raises(PolicyDenied):
        store.update(revoked, record["id"], content="role retained", source_message_id=100, expected_version=1)


def test_edit_delete_preserve_complete_append_only_audit(store):
    record = create(store, OFFICER, "club", "old fact")
    updated = store.update(OFFICER, record["id"], content="new fact", source_message_id=100, expected_version=1)
    assert updated["version"] == 2
    store.delete(OFFICER, record["id"], source_message_id=101, expected_version=2)
    assert store.get(OFFICER, record["id"]) is None
    assert store.search(OFFICER, scope="club") == []
    with sqlite3.connect(store.path) as conn:
        rows = conn.execute("SELECT version, content, source_message_id, actor_id, operation FROM memory_revisions ORDER BY version").fetchall()
        assert rows == [(1, "old fact", 99, 3, "create"), (2, "new fact", 100, 3, "update"), (3, "new fact", 101, 3, "delete")]
        for statement in ("DELETE FROM memory_revisions", "UPDATE memory_revisions SET content='poison'"):
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                conn.execute(statement)


def test_stale_edit_and_delete_do_not_change_memory_or_audit(store):
    record = create(store)
    updated = store.update(MEMBER, record["id"], content="new", source_message_id=100, expected_version=1)
    with pytest.raises(MemoryConflict):
        store.update(MEMBER, record["id"], content="stale", source_message_id=101, expected_version=1)
    with pytest.raises(MemoryConflict):
        store.delete(MEMBER, record["id"], source_message_id=101, expected_version=1)
    assert store.get(MEMBER, record["id"]) == updated
    with sqlite3.connect(store.path) as conn:
        assert conn.execute("SELECT count(*) FROM memory_revisions").fetchone()[0] == 2


def test_concurrent_writers_have_one_winner(store):
    record = create(store)

    def edit(index):
        try:
            store.update(MEMBER, record["id"], content=f"edit {index}", source_message_id=100 + index, expected_version=1)
            return "ok"
        except MemoryConflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(edit, range(4)))
    assert results.count("ok") == 1
    assert results.count("conflict") == 3
    assert store.get(MEMBER, record["id"])["version"] == 2


def test_write_and_audit_are_atomic(store):
    record = create(store)
    with sqlite3.connect(store.path) as conn:
        conn.execute("CREATE TRIGGER fail_revision BEFORE INSERT ON memory_revisions BEGIN SELECT RAISE(ABORT, 'full'); END")
    with pytest.raises(sqlite3.IntegrityError):
        store.update(MEMBER, record["id"], content="lost", source_message_id=100, expected_version=1)
    assert store.get(MEMBER, record["id"])["version"] == 1
    assert store.get(MEMBER, record["id"])["content"] == record["content"]


def test_bounds_and_literal_search(store):
    create(store, content="100% ready_a")
    create(store, content="ordinary")
    assert len(store.search(MEMBER, scope="personal", query="%")) == 1
    assert len(store.search(MEMBER, scope="personal", query="_")) == 1
    assert store.search(MEMBER, scope="personal", query="' OR 1=1 --") == []
    for content in ("", "  ", "x" * 4001, None):
        with pytest.raises(ValueError):
            create(store, content=content)
    for limit in (0, 51, True, "1"):
        with pytest.raises(ValueError):
            store.search(MEMBER, scope="personal", limit=limit)
    with pytest.raises(ValueError):
        store.search(MEMBER, scope="personal", query="x" * 201)
    store.MAX_RECORDS_PER_SCOPE = 2
    with pytest.raises(ValueError, match="full"):
        create(store)


def test_empty_allowlist_blocks_all_memory_access(tmp_path):
    store = ScopedMemoryStore(tmp_path / "closed.sqlite", AgentPolicy())
    with pytest.raises(PolicyDenied):
        create(store)
    with pytest.raises(PolicyDenied):
        store.search(MEMBER, scope="personal")


def test_notes_path_is_not_the_idempotent_control_plane(store):
    # Memory is the loose-notes store: one source message MAY produce several
    # notes (a worker summarising one message into many facts), and there is
    # deliberately no source-message dedup here. Idempotent, version-bound
    # replay lives in ClubStateStore; the gateway must route authoritative
    # fact/roster writes there, never through create().
    first = store.create(OFFICER, scope="club", content="meeting moved",
                         source_message_id=500)
    second = store.create(OFFICER, scope="club", content="room changed",
                          source_message_id=500)
    assert first["id"] != second["id"]
    assert {r["id"] for r in store.search(OFFICER, scope="club")} == \
        {first["id"], second["id"]}
