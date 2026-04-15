"""Screen capture + OCR via Apple Vision framework.

Uses `/usr/sbin/screencapture -x -t png -` CLI for capture (Apple-signed, TCC stable).
OCR via pyobjc Vision VNRecognizeTextRequest at .fast level in an executor thread.
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Protocol, runtime_checkable

log = logging.getLogger("ace_buddy.vision")

SCREENCAPTURE_BIN = "/usr/sbin/screencapture"
CAPTURE_TIMEOUT_SEC = 2.0
OCR_TIMEOUT_SEC = 15.0  # first call on a cold process can load ML models

# Pre-import pyobjc frameworks at module load time — avoids cold-import delay
# inside executor threads and ensures thread-safety of Foundation/Quartz.
_VISION_AVAILABLE = False
try:
    import Vision  # type: ignore
    import Quartz  # type: ignore
    from Foundation import NSData  # type: ignore
    _VISION_AVAILABLE = True
except ImportError as _e:
    log.warning("Vision/Quartz/Foundation import failed: %s", _e)


@runtime_checkable
class ScreenSource(Protocol):
    paused: bool
    async def capture_and_ocr(self) -> str: ...
    async def aclose(self) -> None: ...


def _run_vision_ocr(png_bytes: bytes) -> str:
    """Synchronous OCR call — run in executor.

    Uses pyobjc Vision framework. Returns empty string on any failure.
    """
    if not _VISION_AVAILABLE:
        return ""
    if not png_bytes:
        return ""

    try:
        data = NSData.dataWithBytes_length_(png_bytes, len(png_bytes))
        src = Quartz.CGImageSourceCreateWithData(data, None)
        if src is None or Quartz.CGImageSourceGetCount(src) == 0:
            log.warning("CGImageSource could not parse PNG bytes")
            return ""
        cg_image = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
        if cg_image is None:
            return ""

        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(1)  # fast
        request.setUsesLanguageCorrection_(False)

        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, {})
        ok, err = handler.performRequests_error_([request], None)
        if not ok:
            log.warning("Vision performRequests failed: %s", err)
            return ""

        lines: list[str] = []
        results = request.results() or []
        for obs in results:
            candidates = obs.topCandidates_(1)
            if candidates and len(candidates) > 0:
                lines.append(str(candidates[0].string()))
        return "\n".join(lines).strip()
    except Exception as e:
        log.exception("OCR crashed: %s", e)
        return ""


async def ocr_png_bytes(png_bytes: bytes) -> str:
    """Async wrapper: OCR runs in thread pool."""
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(None, _run_vision_ocr, png_bytes),
            timeout=OCR_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        log.warning("OCR timed out after %ss", OCR_TIMEOUT_SEC)
        return ""


class ScreencaptureScreenSource:
    """Shells out to `/usr/sbin/screencapture` per capture. Apple-signed binary."""

    def __init__(self, screencapture_path: str = SCREENCAPTURE_BIN):
        self.path = screencapture_path
        self.paused = False

    async def capture_and_ocr(self) -> str:
        if self.paused:
            return ""
        if not Path(self.path).exists():
            log.error("%s does not exist", self.path)
            return ""
        try:
            proc = await asyncio.create_subprocess_exec(
                self.path,
                "-x",  # silent (no sound)
                "-t", "png",
                "-",  # stdout
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=CAPTURE_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            log.warning("screencapture timed out")
            return ""
        except FileNotFoundError:
            log.error("screencapture not found")
            return ""

        if proc.returncode != 0:
            log.warning("screencapture exit code %s: %s", proc.returncode, stderr[:200])
            return ""

        if not stdout:
            return ""

        return await ocr_png_bytes(stdout)

    async def aclose(self) -> None:
        pass


class FixtureScreenSource:
    """Test implementation: reads a canned PNG file and OCRs it. No subprocess."""

    def __init__(self, png_path: str | Path):
        self.png_path = Path(png_path)
        self.paused = False

    async def capture_and_ocr(self) -> str:
        if self.paused:
            return ""
        if not self.png_path.exists():
            log.error("fixture png not found: %s", self.png_path)
            return ""
        png_bytes = self.png_path.read_bytes()
        return await ocr_png_bytes(png_bytes)

    async def aclose(self) -> None:
        pass


def _smoke_main() -> int:
    """Manual smoke test: capture current desktop and print OCR output."""
    logging.basicConfig(level=logging.INFO)

    async def run():
        src = ScreencaptureScreenSource()
        text = await src.capture_and_ocr()
        print(f"--- OCR output ({len(text)} chars) ---")
        print(text or "(empty — nothing recognized)")
        return 0

    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(_smoke_main())
