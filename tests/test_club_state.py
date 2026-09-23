"""PETER-09/PETER-10 store-level proofs for the club fact/roster plane.

All members are synthetic. These tests prove the store/policy contract only;
they are not gateway integration evidence (see handoff report for the exact
gateway calls still required).
"""
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pytest

from peterbot.agent_policy import AgentPolicy, ControlIntent, PolicyDenied, Principal
from peterbot.club_state import (
    AmbiguousIdentityError,
    ClubStateConflict,
    ClubStateStore,
    OfficeAssignment,
    OfficerRequest,
    UnresolvedIdentityError,
    resolve_officers,
)
from peterbot.knowledge import parse_markdown_knowledge

GUILD = 10
CONTROL = 20        # configured private control channel
GENERAL = 21        # ordinary public channel
OFFICER_ROLE = 100

DIRECTORY = {"Alex": 1001, "Sam": 1002, "Robin": 1003, "Pat": 1004}


def policy(**overrides):
    kwargs = dict(allowed_guild_ids=frozenset({GUILD}),
                  officer_role_ids=frozenset({OFFICER_ROLE}),
                  control_channel_ids=frozenset({CONTROL}))
    kwargs.update(overrides)
    return AgentPolicy(**kwargs)


@pytest.fixture
def store(tmp_path):
    return ClubStateStore(tmp_path / "club.sqlite3", policy())


def officer(user_id=1, channel_id=CONTROL, roles=(OFFICER_ROLE,)):
    return Principal(GUILD, user_id, channel_id, roles)


def intent(user_id=1, channel_id=CONTROL, message_id=500, action="roster"):
    return ControlIntent(GUILD, user_id, channel_id, message_id, action)


def assignments(*pairs):
    return [OfficeAssignment(office, DIRECTORY[name], name) for office, name in pairs]


def test_denials_leave_no_trace(store):
    denied_both = [
        # same officer speaking in a general channel
        (officer(channel_id=GENERAL), intent(channel_id=GENERAL), True),
        # non-officer speaking in the control channel
        (officer(roles=()), intent(), True),
        # renamed impostor: verified member ID has no officer role; only the
        # display name claims "President Alex"
        (officer(user_id=77, roles=()), intent(user_id=77), True),
        # spoofed/quoted source: intent bound to another user's message
        (officer(user_id=2), intent(user_id=1, message_id=555), True),
        # intent names the control channel but the verified channel is public
        (officer(), intent(message_id=501), False),
    ]
    for principal, request, private in denied_both:
        with pytest.raises(PolicyDenied):
            store.set_officers(principal, request, assignments(("president", "Alex")),
                               channel_is_private=private, term="Fall 2026",
                               expected_version=0)
        with pytest.raises(PolicyDenied):
            store.set_fact(principal, ControlIntent(GUILD, request.user_id,
                                                    request.channel_id,
                                                    request.source_message_id + 90,
                                                    "club_fact"),
                           channel_is_private=private, key="club_mascot",
                           value="A raccoon", visibility="public", expected_version=0)
    # Action-class mismatch in both directions:
    with pytest.raises(PolicyDenied):
        store.set_officers(officer(), intent(action="club_fact"),
                           assignments(("president", "Alex")), channel_is_private=True,
                           term="Fall 2026", expected_version=0)
    with pytest.raises(PolicyDenied):
        store.set_fact(officer(), intent(action="roster"), channel_is_private=True,
                       key="club_mascot", value="A raccoon", visibility="public",
                       expected_version=0)
    assert store.current(GUILD) == {"version": 0, "offices": [], "facts": []}


def test_published_title_never_grants_discord_authority(store):
    # The roster factually records that user 999 is president.
    store.set_officers(officer(), intent(),
                       [OfficeAssignment("president", 999, "Alex (President)")],
                       channel_is_private=True, term="Fall 2026", expected_version=0)
    claimed = Principal(GUILD, 999, GENERAL, ())
    # Discord roles, not the published roster, decide authority:
    assert not store.policy.is_officer(claimed)
    with pytest.raises(PolicyDenied):
        store.policy.require_admission(claimed)
    with pytest.raises(PolicyDenied):
        store.set_fact(claimed, ControlIntent(GUILD, 999, CONTROL, 700, "club_fact"),
                       channel_is_private=True, key="anything", value="x",
                       visibility="public", expected_version=1)
    # And a role revoked from the recorded holder takes effect immediately:
    store.role_advice(GUILD, {"president": OFFICER_ROLE}, {999: ()})  # advisory only
    advice = store.role_advice(GUILD, {"president": OFFICER_ROLE}, {999: ()})
    assert advice and "granted no access" in advice[0]
    # Advising must not mutate policy or state:
    assert store.current(GUILD)["version"] == 1


def test_guild_isolation(store):
    store.set_officers(officer(), intent(), assignments(("president", "Alex")),
                       channel_is_private=True, term="Fall 2026", expected_version=0)
    other = Principal(11, 1, CONTROL, (OFFICER_ROLE,))
    with pytest.raises(PolicyDenied):
        store.set_officers(other, ControlIntent(11, 1, CONTROL, 500, "roster"),
                           assignments(("president", "Alex")),
                           channel_is_private=True, term="Fall 2026", expected_version=0)
    with pytest.raises(PolicyDenied):
        store.snapshot(11)


# ------------------------------------------------------------ atomic reshuffle


def test_reshuffle_supersedes_instead_of_accumulating(store):
    store.set_officers(officer(), intent(), assignments(("president", "Alex")),
                       channel_is_private=True, term="Fall 2026", expected_version=0)
    # Fall shuffle: Sam takes over. One president, not two contradictory rows.
    store.set_officers(officer(), intent(message_id=501),
                       assignments(("president", "Sam")),
                       channel_is_private=True, term="Spring 2027",
                       expected_version=1)
    offices = store.public_officers(GUILD)
    assert [o["holder_label"] for o in offices] == ["Sam"]
    text, version = store.chat_context(GUILD, "who is president")
    assert "Sam" in text and "Alex" not in text and version == 2


def test_partial_amend_keeps_other_offices(store):
    store.set_officers(officer(), intent(),
                       assignments(("president", "Alex"), ("treasurer", "Robin")),
                       channel_is_private=True, term="Fall 2026", expected_version=0)
    store.set_officers(officer(), intent(message_id=502), assignments(("secretary", "Pat")),
                       channel_is_private=True, term="Fall 2026",
                       replace_all=False, expected_version=1)
    assert {o["office"] for o in store.public_officers(GUILD)} == \
        {"president", "treasurer", "secretary"}


def test_public_roster_respects_effective_and_expiry_dates(store):
    with pytest.raises(ValueError, match="ISO date"):
        store.set_officers(officer(), intent(message_id=499),
                           assignments(("president", "Alex")), channel_is_private=True,
                           term="Fall 2026", effective_from="20260901", expected_version=0)
    store.set_officers(officer(), intent(), assignments(("president", "Alex")),
                       channel_is_private=True, term="Fall 2026",
                       effective_from="2026-09-01", expires_at="2026-12-31",
                       expected_version=0)
    assert store.public_officers(GUILD, today=date(2026, 8, 31)) == ()
    assert "Alex" not in store.snapshot(GUILD, today=date(2026, 8, 31)).text
    assert store.public_officers(GUILD, today=date(2026, 9, 1))[0]["holder_label"] == "Alex"
    assert store.public_officers(GUILD, today=date(2026, 12, 31))[0]["holder_label"] == "Alex"
    assert store.public_officers(GUILD, today=date(2027, 1, 1)) == ()
    assert "Alex" not in store.snapshot(GUILD, today=date(2027, 1, 1)).text
    assert "Old Bob" not in store.snapshot(
        GUILD, today=date(2027, 1, 1),
        static_chunks=parse_markdown_knowledge("## Officers\nOld Bob is president.\n"),
        query="who is president").text
    # The historical record stays available to the operator for audit/undo.
    assert store.current(GUILD)["offices"][0]["holder_label"] == "Alex"


# ------------------------------------------------------------ versions/replay


def test_version_conflict_rejects_stale_writer(store):
    store.set_officers(officer(), intent(), assignments(("president", "Alex")),
                       channel_is_private=True, term="Fall 2026", expected_version=0)
    with pytest.raises(ClubStateConflict):
        store.set_officers(officer(), intent(message_id=501),
                           assignments(("president", "Sam")),
                           channel_is_private=True, term="Fall 2026",
                           expected_version=0)
    assert store.current(GUILD)["version"] == 1
    assert store.public_officers(GUILD)[0]["holder_label"] == "Alex"


def test_concurrent_writers_have_one_winner(store):
    def edit(index):
        try:
            store.set_fact(officer(), intent(message_id=600 + index, action="club_fact"),
                           channel_is_private=True, key=f"fact_{index}",
                           value=f"value {index}", visibility="public",
                           expected_version=0)
            return "ok"
        except ClubStateConflict:
            return "conflict"
    results = list(ThreadPoolExecutor(max_workers=4).map(edit, range(4)))
    assert results.count("ok") == 1 and results.count("conflict") == 3
    assert store.current(GUILD)["version"] == 1


def test_replayed_source_message_never_applies_twice(store):
    first = store.set_fact(officer(), intent(action="club_fact"), channel_is_private=True,
                           key="meeting_day", value="Tuesdays 7pm",
                           visibility="public", expected_version=0)
    replay = store.set_fact(officer(), intent(action="club_fact"), channel_is_private=True,
                            key="meeting_day", value="totally different",
                            visibility="public", expected_version=99)
    assert replay["replayed"] and replay["version"] == first["version"] == 1
    facts = store.public_facts(GUILD)
    assert list(facts) == [{"key": "meeting_day", "value": "Tuesdays 7pm",
                            "updated_at": facts[0]["updated_at"]}]


def test_replay_under_other_actor_or_action_is_denied(store):
    store.set_officers(officer(), intent(), assignments(("president", "Alex")),
                       channel_is_private=True, term="Fall 2026", expected_version=0)
    # Same source message number, but the second officer's own verified edit:
    with pytest.raises(PolicyDenied):
        store.set_fact(officer(user_id=2, roles=(OFFICER_ROLE,)),
                       ControlIntent(GUILD, 2, CONTROL, 500, "club_fact"),
                       channel_is_private=True, key="evil", value="x",
                       visibility="public", expected_version=1)
    # Same message, mismatched action class for the first officer:
    with pytest.raises(PolicyDenied):
        store.set_fact(officer(), ControlIntent(GUILD, 1, CONTROL, 500, "club_fact"),
                       channel_is_private=True, key="evil", value="x",
                       visibility="public", expected_version=1)
    assert store.current(GUILD)["version"] == 1


# --------------------------------------------------------------------- undo


def test_undo_restores_previous_state_atomically(store):
    store.set_officers(officer(), intent(), assignments(("president", "Alex")),
                       channel_is_private=True, term="Fall 2026", expected_version=0)
    store.set_officers(officer(), intent(message_id=501), assignments(("president", "Sam")),
                       channel_is_private=True, term="Fall 2026", expected_version=1)
    reverted = store.undo(officer(), intent(message_id=502),
                          channel_is_private=True, expected_version=2)
    assert reverted["version"] == 3
    assert store.public_officers(GUILD)[0]["holder_label"] == "Alex"
    audit = sqlite3.connect(store.path).execute(
        "SELECT version, action, operation, actor_id FROM club_revisions ORDER BY version"
    ).fetchall()
    assert audit == [(1, "roster", "officers_set", 1), (2, "roster", "officers_set", 1),
                     (3, "roster", "undo", 1)]


def test_undo_requires_matching_action_and_latest_version(store):
    store.set_officers(officer(), intent(), assignments(("president", "Alex")),
                       channel_is_private=True, term="Fall 2026", expected_version=0)
    store.set_fact(officer(), intent(message_id=501, action="club_fact"),
                   channel_is_private=True, key="meeting_day", value="Tuesdays 7pm",
                   visibility="public", expected_version=1)
    # Roster intent cannot undo a fact revision:
    with pytest.raises(PolicyDenied):
        store.undo(officer(), intent(message_id=502), channel_is_private=True,
                   expected_version=2)
    # Fact intent cannot undo a roster revision: action class is checked first.
    fact_undo = ControlIntent(GUILD, 1, CONTROL, 503, "club_fact")
    with pytest.raises(PolicyDenied):
        store.undo(officer(), fact_undo, channel_is_private=True, expected_version=1)
    # Latest fact revision undoes cleanly with the fact action class:
    undone = store.undo(officer(), ControlIntent(GUILD, 1, CONTROL, 504, "club_fact"),
                        channel_is_private=True, expected_version=2)
    assert undone["version"] == 3 and store.public_facts(GUILD) == ()
    # Replaying the undo message is idempotent:
    again = store.undo(officer(), ControlIntent(GUILD, 1, CONTROL, 504, "club_fact"),
                       channel_is_private=True, expected_version=99)
    assert again["replayed"] and again["version"] == 3


# ------------------------------------------------------------ persistence


def test_state_survives_restart_and_more_than_ten_newer_facts(store, tmp_path):
    store.set_officers(officer(), intent(), assignments(("president", "Alex")),
                       channel_is_private=True, term="Fall 2026", expected_version=0)
    store.set_fact(officer(), intent(message_id=501, action="club_fact"),
                   channel_is_private=True, key="meeting_day", value="Tuesdays 7pm",
                   visibility="public", expected_version=1)
    for index in range(12):  # more than the old "10 most recent" window
        store.set_fact(officer(),
                       intent(message_id=600 + index, action="club_fact"),
                       channel_is_private=True, key=f"widget_note_{index}",
                       value=f"unrelated note {index}", visibility="public",
                       expected_version=2 + index)
    reopened = ClubStateStore(tmp_path / "club.sqlite3", policy())
    text, version = reopened.chat_context(GUILD, "what day is the meeting")
    assert version == 14
    assert "Tuesdays 7pm" in text and "president: Alex" in text
    # Offices never fall out of the snapshot:
    assert "Sam" not in text  # sanity: nobody named Sam exists
    snapshot = reopened.snapshot(GUILD, query="meeting day")
    assert snapshot.offices[0]["holder_label"] == "Alex"
    # Relevance outranks recency: the older committed fact beats the 12 newer
    # unrelated notes for a matching query, and survives the restart.
    assert snapshot.facts[0]["key"] == "meeting_day"


def test_private_facts_never_enter_public_snapshot(store):
    store.set_fact(officer(), intent(action="club_fact"), channel_is_private=True,
                   key="meeting_day", value="Tuesdays 7pm",
                   visibility="public", expected_version=0)
    store.set_fact(officer(), intent(message_id=501, action="club_fact"),
                   channel_is_private=True, key="treasury_balance",
                   value="$4,200 reserve", visibility="private", expected_version=1)
    store.set_fact(officer(), intent(message_id=502, action="club_fact"),
                   channel_is_private=True, key="officer_meeting_notes",
                   value="private discussion summary", visibility="private",
                   expected_version=2)
    snapshot = store.snapshot(GUILD, query="treasury balance officer meeting notes")
    assert "4,200" not in snapshot.text
    assert "private discussion" not in snapshot.text
    assert "Tuesdays 7pm" in snapshot.text
    assert [f["key"] for f in store.public_facts(GUILD)] == ["meeting_day"]
    assert {f["key"] for f in store.current(GUILD)["facts"]} == \
        {"meeting_day", "treasury_balance", "officer_meeting_notes"}


# ---------------------------------------------------------------- identity


def test_identity_resolution_requires_stable_trusted_ids():
    result = resolve_officers(
        [OfficerRequest("president", "alex"), OfficerRequest("vice_president", "Sam")],
        DIRECTORY)
    assert [(a.office, a.holder_user_id) for a in result] == \
        [("president", 1001), ("vice_president", 1002)]
    with pytest.raises(UnresolvedIdentityError):
        resolve_officers([OfficerRequest("president", "Morgan")], DIRECTORY)
    with pytest.raises(AmbiguousIdentityError):
        resolve_officers([OfficerRequest("president", "Alex")],
                         {"Alex": 1001, "alex": 2002})


def test_roster_commit_uses_resolved_ids_and_ignores_display_claims(store):
    resolved = resolve_officers([OfficerRequest("president", "Alex")], DIRECTORY)
    store.set_officers(officer(), intent(), resolved, channel_is_private=True,
                       term="Fall 2026", expected_version=0)
    rows = store.current(GUILD)["offices"]
    # A renamed impostor cannot take the seat by changing display text: the
    # stored holder is the resolved member ID, labels are decorative.
    assert rows[0]["holder_user_id"] == 1001


def test_forge_offices_fail_validation(store):
    # The typed roster rejects forged rows at construction; none can reach the store.
    for bad in (lambda: OfficeAssignment("President", 1001, "Alex"),   # not a slug
                lambda: OfficeAssignment("president", 0, "Alex"),      # invalid ID
                lambda: OfficeAssignment("president", 1001, ""),       # blank label
                lambda: OfficeAssignment("president", 1001, "A\nB")):  # multiline
        with pytest.raises(ValueError):
            store.set_officers(officer(), intent(), [bad()], channel_is_private=True,
                               term="Fall 2026", expected_version=0)
    with pytest.raises(ValueError):
        store.set_officers(officer(), intent(), assignments(("president", "Alex"),
                                                            ("president", "Sam")),
                           channel_is_private=True, term="Fall 2026", expected_version=0)
    assert store.current(GUILD)["version"] == 0


# ------------------------------------------------------- static supersession

STATIC_KNOWLEDGE = """## Officers
Old Bob is president and Case is vice president.

## Room
Room 214 in the science building.
expires: 2026-05-01

## Funding
Club gets $500 a semester.
"""


def test_committed_roster_supersedes_stale_static_knowledge(store):
    chunks = parse_markdown_knowledge(STATIC_KNOWLEDGE)
    # Before any committed roster, static text remains as background.
    before = store.snapshot(GUILD, query="who is president", static_chunks=chunks,
                            today=date(2026, 9, 22))
    assert "Old Bob" in before.text and "Room 214" not in before.text  # room expired
    store.set_officers(officer(), intent(), assignments(("president", "Alex")),
                       channel_is_private=True, term="Fall 2026", expected_version=0)
    after = store.snapshot(GUILD, query="who is president room funding",
                           static_chunks=chunks, today=date(2026, 9, 22))
    assert "Alex" in after.text
    assert "Old Bob" not in after.text          # newer committed record wins
    assert "Room 214" not in after.text          # expired static chunk dropped
    assert "500 a semester" in after.text       # live background survives
    assert len(after.text) <= 2200


def test_snapshot_is_bounded_and_fresh_per_call(store):
    store.set_officers(officer(), intent(), assignments(("president", "Alex")),
                       channel_is_private=True, term="Fall 2026", expected_version=0)
    bounded = store.snapshot(GUILD, query="president", max_chars=128)
    assert len(bounded.text) <= 128
    with pytest.raises(ValueError):
        store.snapshot(GUILD, query="x" * 301)
    with pytest.raises(ValueError):
        store.snapshot(GUILD, max_items=0)
