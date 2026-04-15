"""FastAPI server: phone UI, WS answer stream, /fire redundant trigger, /qr, /auth."""
from __future__ import annotations

import asyncio
import io
import json
import logging
import secrets
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import qrcode
from fastapi import Cookie, FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, JSONResponse

log = logging.getLogger("ace_buddy.server")

UI_PATH = Path(__file__).parent / "ui" / "index.html"


@dataclass
class AnswerState:
    """Server-side buffered answer for WS reconnect replay. Bounded to 1 answer."""
    id: int = 0
    tokens: list[str] = field(default_factory=list)
    complete: bool = False

    def reset(self, new_id: int) -> None:
        self.id = new_id
        self.tokens = []
        self.complete = False

    def append(self, token: str) -> None:
        self.tokens.append(token)

    def finish(self) -> None:
        self.complete = True


@dataclass
class ServerState:
    token: str
    host: str
    port: int
    config_dir: Path
    auth_required: bool = True
    debug: bool = False
    job_title: str = "(no job.md loaded)"

    # Single-client policy (Eric has one phone)
    current_ws: Optional[WebSocket] = None
    current_answer: AnswerState = field(default_factory=AnswerState)

    # Injected by app.py once pipeline is ready
    fire_callback: Optional[Callable[[], None]] = None

    # Injected later for cheatsheet
    cheatsheet: list[dict] = field(default_factory=list)


def get_local_ip() -> str:
    """Best-effort local LAN IP."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def auth_url(state: ServerState) -> str:
    return f"http://{state.host}:{state.port}/auth?k={state.token}"


def build_qr_png(url: str) -> bytes:
    img = qrcode.make(url)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def check_auth(state: ServerState, cookie_token: Optional[str]) -> None:
    if not state.auth_required:
        return
    if cookie_token != state.token:
        raise HTTPException(status_code=401, detail="auth required — visit /auth?k=TOKEN")


def create_app(state: ServerState) -> FastAPI:
    app = FastAPI(title="ace-buddy", docs_url=None, redoc_url=None)

    # --- auth / index ---

    @app.get("/auth")
    async def auth(k: str = ""):
        if state.auth_required and not secrets.compare_digest(k, state.token):
            return JSONResponse({"error": "invalid token"}, status_code=401)
        resp = RedirectResponse(url="/", status_code=302)
        resp.set_cookie("ab_token", state.token, httponly=False, samesite="lax", max_age=86400)
        return resp

    @app.get("/")
    async def index(ab_token: Optional[str] = Cookie(default=None)):
        try:
            check_auth(state, ab_token)
        except HTTPException:
            return JSONResponse(
                {"error": "visit /auth?k=TOKEN first", "hint": "scan the QR from the menu bar"},
                status_code=401,
            )
        html = UI_PATH.read_text()
        html = html.replace("__JOB_TITLE__", state.job_title)
        html = html.replace("__TOKEN__", state.token if not state.auth_required else "")
        return HTMLResponse(html)

    @app.get("/qr")
    async def qr(ab_token: Optional[str] = Cookie(default=None)):
        # QR contains the auth URL so phone can scan to get cookie
        url = auth_url(state)
        return StreamingResponse(io.BytesIO(build_qr_png(url)), media_type="image/png")

    @app.get("/status")
    async def status():
        return {
            "ok": True,
            "debug": state.debug,
            "job_title": state.job_title,
            "cheatsheet_count": len(state.cheatsheet),
            "answer_id": state.current_answer.id,
            "answer_complete": state.current_answer.complete,
        }

    # --- cheatsheet ---

    @app.get("/cheatsheet")
    async def cheatsheet(ab_token: Optional[str] = Cookie(default=None)):
        check_auth(state, ab_token)
        return {"questions": state.cheatsheet}

    # --- redundant trigger (phone tap) ---

    @app.post("/fire")
    async def fire(ab_token: Optional[str] = Cookie(default=None)):
        check_auth(state, ab_token)
        if state.fire_callback is None:
            return JSONResponse({"error": "pipeline not wired"}, status_code=503)
        state.fire_callback()
        return {"ok": True, "trigger": "phone"}

    # --- debug (ACE_BUDDY_DEBUG=1) ---

    if state.debug:
        @app.post("/debug/fire")
        async def debug_fire():
            if state.fire_callback:
                state.fire_callback()
            return {"ok": True, "trigger": "debug"}

        @app.get("/debug/state")
        async def debug_state():
            return {
                "answer_id": state.current_answer.id,
                "tokens_buffered": len(state.current_answer.tokens),
                "complete": state.current_answer.complete,
                "has_client": state.current_ws is not None,
            }

    # --- WebSocket ---

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        if state.auth_required:
            cookie_token = ws.cookies.get("ab_token")
            if cookie_token != state.token:
                await ws.close(code=4401)
                return
        await ws.accept()

        # Drop previous client if any (single-client policy)
        if state.current_ws is not None:
            try:
                await state.current_ws.close(code=4000)
            except Exception:
                pass
        state.current_ws = ws

        # Send hello + replay current answer if any
        await ws.send_text(json.dumps({"type": "hello", "job_title": state.job_title}))
        ans = state.current_answer
        if ans.id > 0 and ans.tokens:
            await ws.send_text(json.dumps({
                "type": "answer_start", "id": ans.id,
            }))
            for tok in ans.tokens:
                await ws.send_text(json.dumps({
                    "type": "answer_token", "id": ans.id, "text": tok,
                }))
            if ans.complete:
                await ws.send_text(json.dumps({
                    "type": "answer_complete", "id": ans.id,
                }))

        try:
            while True:
                # Phone can send: {type: "fire"} as redundant trigger
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if msg.get("type") == "fire" and state.fire_callback:
                    state.fire_callback()
        except WebSocketDisconnect:
            pass
        finally:
            if state.current_ws is ws:
                state.current_ws = None

    return app


async def broadcast(state: ServerState, message: dict) -> None:
    """Send message to current WS client, if any, with timeout."""
    ws = state.current_ws
    if ws is None:
        return
    try:
        await asyncio.wait_for(ws.send_text(json.dumps(message)), timeout=0.5)
    except (asyncio.TimeoutError, Exception) as e:
        log.warning("WS send failed: %s", e)
        try:
            await ws.close(code=4000)
        except Exception:
            pass
        if state.current_ws is ws:
            state.current_ws = None
