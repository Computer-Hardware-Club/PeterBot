import asyncio
from types import SimpleNamespace

from peterbot.commands import IMAGE_ATTACHMENT_READ_FAILURE_MESSAGE, resolve_mention_images


def test_resolve_mention_images_reports_when_all_images_are_unusable() -> None:
    class OversizedAttachment:
        filename = "case-photo.png"
        content_type = "image/png"
        size = 20

        async def read(self) -> bytes:
            return b"x" * 20

    message = SimpleNamespace(
        attachments=[OversizedAttachment()],
        id=1,
        author=SimpleNamespace(id=2),
        channel=SimpleNamespace(id=3),
        guild=None,
    )

    images, error = asyncio.run(
        resolve_mention_images(
            message,
            image_limit=2,
            max_image_bytes=10,
        )
    )

    assert images == []
    assert error == IMAGE_ATTACHMENT_READ_FAILURE_MESSAGE
