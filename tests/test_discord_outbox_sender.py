import asyncio
from types import SimpleNamespace

import pytest

from peterbot.discord_outbox_sender import InvalidAnnouncementReceipt, send_announcement


RECORD = {"status": "sending", "target_channel_id": 30, "guild_id": 10,
          "nonce": "0123456789abcdef", "content": "Meeting Friday at six."}
CHANNEL = SimpleNamespace(id=30, guild=SimpleNamespace(id=10))


class HTTP:
    def __init__(self, receipt=None):
        self.calls = []
        self.receipt = {"id": "44", "channel_id": "30"} if receipt is None else receipt

    async def request(self, route, **kwargs):
        self.calls.append((route, kwargs))
        return self.receipt


def test_enforced_nonce_uses_rate_limited_discord_client():
    http = HTTP()
    bot = SimpleNamespace(http=http)
    message_id = asyncio.run(send_announcement(bot, RECORD, CHANNEL))
    assert message_id == 44
    route, kwargs = http.calls[0]
    assert route.method == "POST" and route.url.endswith("/channels/30/messages")
    assert kwargs["json"] == {
        "content": RECORD["content"], "nonce": RECORD["nonce"], "enforce_nonce": True,
        "allowed_mentions": {"parse": [], "users": [], "roles": [], "replied_user": False},
        "flags": 4,
    }


def test_unclaimed_or_wrong_destination_does_not_send():
    http = HTTP()
    bot = SimpleNamespace(http=http)
    with pytest.raises(ValueError):
        asyncio.run(send_announcement(bot, {**RECORD, "status": "pending"}, CHANNEL))
    with pytest.raises(ValueError):
        asyncio.run(send_announcement(bot, RECORD, SimpleNamespace(id=31, guild=CHANNEL.guild)))
    with pytest.raises(ValueError):
        asyncio.run(send_announcement(bot, RECORD, SimpleNamespace(id=30,
                           guild=SimpleNamespace(id=11))))
    assert http.calls == []


def test_unbound_receipt_is_unknown_to_caller():
    for receipt in ({"id": "44", "channel_id": "31"},
                    {"id": "bad", "channel_id": "30"},
                    {"id": "44"}, []):
        with pytest.raises(InvalidAnnouncementReceipt):
            asyncio.run(send_announcement(SimpleNamespace(http=HTTP(receipt)),
                                          RECORD, CHANNEL))
