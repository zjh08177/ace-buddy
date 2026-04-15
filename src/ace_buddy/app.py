"""ace-buddy main entrypoint: lifecycle, hotkey, preflight, menubar."""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import secrets
import signal
import sys
from pathlib import Path

import uvicorn

from ace_buddy.server import ServerState, auth_url, build_qr_png, create_app, get_local_ip

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
        debug=os.environ.get("ACE_BUDDY_DEBUG") == "1",
    )
    # Try to read job title for display banner
    job_md = config_dir / "job.md"
    if job_md.exists():
        first = job_md.read_text(errors="ignore").strip().splitlines()
        if first:
            state.job_title = first[0].lstrip("# ").strip() or state.job_title
    return state


async def run_server(state: ServerState, bind_host: str) -> None:
    app = create_app(state)
    cfg = uvicorn.Config(
        app,
        host=bind_host,
        port=state.port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(cfg)
    await server.serve()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("ACE_BUDDY_LOG", "INFO"),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = parse_args(argv)
    logging.getLogger().setLevel(args.log_level)

    state = build_state(args)
    bind_host = "0.0.0.0" if args.lan else "127.0.0.1"

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
    stop_event = asyncio.Event()

    def _stop(*_):
        log.info("shutdown signal received")
        loop.call_soon_threadsafe(stop_event.set)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    async def _run():
        server_task = asyncio.create_task(run_server(state, bind_host))
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
        loop.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
