import pytest

from peterbot.agent_policy import AgentPolicy, ControlIntent, PolicyDenied, Principal
from peterbot.announcement_outbox import AnnouncementOutbox, OutboxConflict


@pytest.fixture
def outbox(tmp_path):
    policy = AgentPolicy(allowed_guild_ids=frozenset({10}), officer_role_ids=frozenset({100}),
                         control_channel_ids=frozenset({20}))
    value = AnnouncementOutbox(tmp_path / "outbox.sqlite3", policy, {10: frozenset({30})})
    yield value
    value.close()


def actor(*, user=1, channel=20, roles=(100,)):
    return Principal(10, user, channel, roles)


def request(*, user=1, channel=20, source=40, action="announcement"):
    return ControlIntent(10, user, channel, source, action)


def propose(outbox, *, principal=None, intent=None, destination=30, content="Welcome to the club!", private=True):
    return outbox.propose(principal or actor(), intent or request(), target_channel_id=destination,
                          content=content, channel_is_private=private)


def test_one_authorized_intent_has_a_real_receipt_only_after_send(outbox):
    record = propose(outbox)
    assert record["status"] == "pending"
    assert outbox.receipt_url(record["id"]) is None
    assert outbox.begin_send(record["id"], actor(), request(), channel_is_private=True)
    assert not outbox.begin_send(record["id"], actor(), request(), channel_is_private=True)
    assert outbox.mark_sent(record["id"], 50)
    assert outbox.receipt_url(record["id"]) == "https://discord.com/channels/10/30/50"
    assert outbox.pending() == []


def test_replayed_request_is_idempotent_and_changed_payload_needs_new_source(outbox):
    first = propose(outbox)
    assert propose(outbox)["id"] == first["id"]
    with pytest.raises(OutboxConflict):
        propose(outbox, content="Different public message")
    assert len(outbox.pending()) == 1


def test_authority_destination_and_mass_mentions_fail_closed(outbox):
    for arguments in (
        {"principal": actor(roles=())},
        {"principal": actor(channel=21), "intent": request(channel=21)},
        {"principal": actor(), "private": False},
        {"principal": actor(), "intent": request(user=2)},
        {"destination": 31},
        {"intent": request(action="style")},
    ):
        with pytest.raises(PolicyDenied):
            propose(outbox, **arguments)
    for content in ("@everyone hello", "@here hello", "hi <@123>", "hello <@&456>"):
        with pytest.raises(ValueError, match="mentions"):
            propose(outbox, content=content)
    assert outbox.pending() == []


def test_uncertain_send_is_not_replayed_after_restart(outbox, tmp_path):
    record = propose(outbox)
    assert outbox.begin_send(record["id"], actor(), request(), channel_is_private=True)
    reopened = AnnouncementOutbox(tmp_path / "outbox.sqlite3", outbox.policy,
                                  {10: frozenset({30})})
    try:
        assert reopened.get(record["id"])["status"] == "unknown"
        assert reopened.pending() == []
        assert reopened.receipt_url(record["id"]) is None
        with pytest.raises(ValueError):
            reopened.reconcile_unknown(record["id"])
        assert reopened.reconcile_unknown(record["id"], confirmed_message_id=51)
        assert reopened.receipt_url(record["id"]).endswith("/51")
    finally:
        reopened.close()


def test_known_rejection_retries_with_bound_and_denial_stops_send(outbox):
    record = propose(outbox)
    for attempt in range(8):
        assert outbox.begin_send(record["id"], actor(), request(), channel_is_private=True)
        assert outbox.rejected_retry(record["id"]) == ("failed" if attempt == 7 else "pending")
    assert outbox.pending() == []
    other = propose(outbox, intent=request(source=41))
    assert outbox.mark_denied(other["id"])
    assert not outbox.begin_send(other["id"], actor(), request(source=41), channel_is_private=True)


def test_unknown_send_retries_only_after_explicit_checked_decision(outbox):
    record = propose(outbox)
    assert outbox.begin_send(record["id"], actor(), request(), channel_is_private=True)
    assert outbox.mark_unknown(record["id"])
    assert outbox.pending() == []
    assert outbox.reconcile_unknown(record["id"], retry_after_check=True)
    assert outbox.pending()[0]["id"] == record["id"]
    assert outbox.pending()[0]["nonce"] == record["nonce"]


def test_revoked_officer_cannot_dispatch_a_previously_authorized_intent(outbox):
    record = propose(outbox)
    with pytest.raises(PolicyDenied):
        outbox.begin_send(record["id"], actor(roles=()), request(), channel_is_private=True)
    assert outbox.get(record["id"])["status"] == "pending"


def test_in_flight_send_cannot_be_reported_as_denied(outbox):
    record = propose(outbox)
    assert outbox.begin_send(record["id"], actor(), request(), channel_is_private=True)
    assert not outbox.mark_denied(record["id"])
    assert outbox.mark_unknown(record["id"])


def test_future_outbox_schema_fails_closed(outbox, tmp_path):
    outbox.db.execute("PRAGMA user_version=200")
    with pytest.raises(ValueError, match="newer"):
        AnnouncementOutbox(tmp_path / "outbox.sqlite3", outbox.policy,
                           {10: frozenset({30})})


def test_receipt_conflict_can_freeze_pending_but_never_reopens_sent(outbox):
    pending = propose(outbox)
    assert outbox.mark_unknown(pending['id'])
    assert outbox.pending() == []
    sent = propose(outbox, intent=request(source=42))
    assert outbox.begin_send(sent['id'], actor(), request(source=42), channel_is_private=True)
    assert outbox.mark_sent(sent['id'], 55)
    assert not outbox.mark_unknown(sent['id'])
    assert outbox.get(sent['id'])['status'] == 'sent'
