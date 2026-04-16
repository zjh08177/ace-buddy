"""ace-buddy main entrypoint: lifecycle, hotkey, preflight, menubar."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import secrets
import signal
import sys
from pathlib import Path
from typing import Optional

import uvicorn

from ace_buddy.audio import AudioSource, FixtureAudioSource, SoundDeviceAudioSensor
from ace_buddy.cheatsheet import compute as compute_cheatsheet
from ace_buddy.llm import LLMClient, MockLLMClient, OpenAILLMClient
from ace_buddy.pipeline import Pipeline
from ace_buddy.prompt import ContextBundle, PromptBuilder
from ace_buddy.server import ServerState, auth_url, build_qr_png, create_app, get_local_ip
from ace_buddy.vision import FixtureScreenSource, ScreencaptureScreenSource, ScreenSource

log = logging.getLogger("ace_buddy")


def load_env(config_dir: Path) -> None:
    from dotenv import load_dotenv
    env_path = config_dir / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    # also allow repo-level .env for dev
    load_dotenv(".env", override=False)


def expand_config_dir() -> Path:
    raw = os.environ.get("ACE_BUDDY_CONFIG_DIR", "~/.ace-buddy")
    return Path(raw).expanduser()


def print_qr_to_terminal(url: str) -> None:
    try:
        import qrcode
        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.make()
        qr.print_ascii(invert=True)
    except Exception as e:
        log.warning("qrcode terminal print failed: %s", e)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="ace-buddy")
    p.add_argument("--lan", action="store_true", help="bind 0.0.0.0 (phone on same Wi-Fi)")
    p.add_argument("--port", type=int, default=int(os.environ.get("ACE_BUDDY_PORT", "8765")))
    p.add_argument("--no-auth", action="store_true", help="skip token auth (tests only)")
    p.add_argument("--headless", action="store_true", help="no menu bar (tests)")
    p.add_argument("--debug", action="store_true", help="enable /debug/* routes")
    p.add_argument("--fixture-audio", type=str, default=None)
    p.add_argument("--fixture-screen", type=str, default=None)
    p.add_argument("--mock-llm", type=str, default=None, help="JSON list of tokens to emit")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def build_state(args: argparse.Namespace) -> ServerState:
    config_dir = expand_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    load_env(config_dir)

    host = "0.0.0.0" if args.lan else "127.0.0.1"
    display_host = get_local_ip() if args.lan else "127.0.0.1"
    token = "dev" if args.no_auth else secrets.token_urlsafe(24)

    state = ServerState(
        token=token,
        host=display_host,
        port=args.port,
        config_dir=config_dir,
        auth_required=not args.no_auth,
        debug=args.debug or os.environ.get("ACE_BUDDY_DEBUG") == "1",
    )
    return state


async def run_server(app, state: ServerState, bind_host: str) -> None:
    cfg = uvicorn.Config(
        app,
        host=bind_host,
        port=state.port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(cfg)
    await server.serve()


def build_audio(args) -> AudioSource:
    if args.fixture_audio:
        log.info("using fixture audio: %s", args.fixture_audio)
        return FixtureAudioSource(args.fixture_audio)
    if args.headless:
        # Headless mode in tests: empty audio source unless fixture given
        return FixtureAudioSource("/nonexistent.wav")
    return SoundDeviceAudioSensor()


def build_screen(args) -> ScreenSource:
    if args.fixture_screen:
        log.info("using fixture screen: %s", args.fixture_screen)
        return FixtureScreenSource(args.fixture_screen)
    return ScreencaptureScreenSource()


def build_llm(args) -> LLMClient:
    if args.mock_llm:
        try:
            tokens = json.loads(args.mock_llm)
            log.info("using mock LLM with %d tokens", len(tokens))
            return MockLLMClient(tokens=tokens, delay_s=0.0)
        except json.JSONDecodeError as e:
            log.error("invalid --mock-llm JSON: %s", e)
            sys.exit(2)
    if os.environ.get("MOCK_ANSWER"):
        return MockLLMClient(delay_s=0.0)
    return OpenAILLMClient(model=os.environ.get("ACE_BUDDY_MODEL", "gpt-4o"))


def register_hotkey(fire_fn) -> Optional[object]:
    """Best-effort global hotkey via pynput. Returns listener or None."""
    try:
        from pynput import keyboard
    except Exception as e:
        log.warning("pynput unavailable: %s", e)
        return None

    hotkey_str = os.environ.get("ACE_BUDDY_HOTKEY", "<cmd>+<shift>+<space>")
    try:
        combo = keyboard.HotKey.parse(hotkey_str)
    except Exception as e:
        log.warning("invalid hotkey %r: %s", hotkey_str, e)
        return None

    hotkey = keyboard.HotKey(combo, fire_fn)

    def on_press(key):
        try:
            hotkey.press(listener.canonical(key))
        except Exception:
            pass

    def on_release(key):
        try:
            hotkey.release(listener.canonical(key))
        except Exception:
            pass

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    try:
        listener.start()
        log.info("hotkey registered: %s", hotkey_str)
    except Exception as e:
        log.warning("hotkey listener start failed: %s", e)
        return None
    return listener


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("ACE_BUDDY_LOG", "INFO"),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = parse_args(argv)
    logging.getLogger().setLevel(args.log_level)

    state = build_state(args)
    bind_host = "0.0.0.0" if args.lan else "127.0.0.1"

    # Load context + build pipeline components
    ctx = ContextBundle.from_dir(state.config_dir)
    state.job_title = ctx.job_title
    prompt_builder = PromptBuilder(ctx)

    audio = build_audio(args)
    screen = build_screen(args)
    llm = build_llm(args)
    pipeline = Pipeline(
        state=state,
        audio=audio,
        screen=screen,
        llm=llm,
        prompt_builder=prompt_builder,
    )

    # Start audio sensor in background thread
    try:
        audio.start()
    except Exception as e:
        log.warning("audio.start failed (non-fatal in headless/fixture mode): %s", e)

    # Wire pipeline fire + sensor refs into server state
    state.fire_callback = pipeline.fire_from_any_thread
    state.audio_source = audio
    state.screen_source = screen

    # Build the FastAPI app
    app = create_app(state)

    url = auth_url(state)
    print(f"\nace-buddy ready")
    print(f"  URL: {url}")
    if args.lan:
        print(f"  Scan this QR on your phone (must be on same Wi-Fi):")
        print_qr_to_terminal(url)
    else:
        print(f"  (bound to 127.0.0.1 — use --lan to expose on Wi-Fi)")
    print()

    # Signal-based shutdown
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    pipeline.attach_loop(loop)
    stop_event = asyncio.Event()

    def _stop(*_):
        log.info("shutdown signal received")
        loop.call_soon_threadsafe(stop_event.set)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    # Register global hotkey (non-fatal if it fails — /fire still works)
    hotkey_listener = register_hotkey(pipeline.fire_from_any_thread) if not args.headless else None

    async def _run():
        # Compute cheatsheet in background (non-blocking)
        async def _cheat():
            if args.mock_llm or args.headless:
                return
            try:
                qs = await compute_cheatsheet(ctx)
                state.cheatsheet = qs
                log.info("cheatsheet computed: %d questions", len(qs))
            except Exception as e:
                log.warning("cheatsheet compute failed: %s", e)

        asyncio.create_task(_cheat())
        server_task = asyncio.create_task(run_server(app, state, bind_host))
        stop_task = asyncio.create_task(stop_event.wait())
        done, pending = await asyncio.wait(
            {server_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()

    try:
        loop.run_until_complete(_run())
    except KeyboardInterrupt:
        pass
    finally:
        try:
            audio.stop()
        except Exception:
            pass
        if hotkey_listener is not None:
            try:
                hotkey_listener.stop()
            except Exception:
                pass
        loop.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
