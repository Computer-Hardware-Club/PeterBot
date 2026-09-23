import asyncio
import signal
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from peterbot.app import register_signal_handlers


def test_sigterm_closes_bot_and_saves_reminders_once(monkeypatch):
    handlers = {}
    monkeypatch.setattr(signal, "signal", lambda kind, handler: handlers.setdefault(kind, handler))
    close = AsyncMock()
    reminders = SimpleNamespace(save_shutdown_time=Mock(), save_reminders=Mock())
    register_signal_handlers(SimpleNamespace(bot=SimpleNamespace(close=close), reminder_manager=reminders))

    async def scenario():
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        handlers[signal.SIGINT](signal.SIGINT, None)
        await asyncio.sleep(0)

    asyncio.run(scenario())
    close.assert_awaited_once()
    reminders.save_shutdown_time.assert_called_once()
    reminders.save_reminders.assert_called_once()
