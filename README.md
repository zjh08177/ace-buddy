# ace-buddy

> Personal-use macOS interview copilot. Listens to your mic + OCRs your screen, then streams a GPT-4o answer to your phone over local Wi-Fi. **Nothing is rendered on your Mac display**, so screen-sharing is safe by design.

```
┌──────────── MacBook ─────────────┐         ┌─── iPhone/iPad ───┐
│                                   │         │                    │
│  Mic ──► Whisper ──┐              │         │  Safari            │
│                    ├─► GPT-4o ────┼──WS────►│  http://mac:8765   │
│  Screen ──► Vision ┘              │         │                    │
│      OCR                          │         │  ┌──────────────┐  │
│                                   │         │  │ **TL;DR**: …  │  │
│  ⌘⇧Space (hotkey)                 │         │  │ • anchor 1    │  │
│  or POST /fire                    │         │  │ • anchor 2    │  │
│                                   │         │  │ • anchor 3    │  │
│  ✅ Architecturally invisible —    │         │  │ **Data**: …   │  │
│     the answer never touches      │         │  └──────────────┘  │
│     your Mac's framebuffer        │         │                    │
└───────────────────────────────────┘         └────────────────────┘
```

## Why this design

The "invisible overlay" trick that Cluely / Final Round AI rely on (`NSWindowSharingNone` / `setContentProtection`) is **broken on macOS 15 Sequoia** — Apple's ScreenCaptureKit captures the final composited framebuffer, and Zoom/Meet/Teams all migrated to it. Any window-level stealth flag is now ignored.

ace-buddy sidesteps the entire problem by **never putting the answer on the Mac display**. The phone is a separate device, not part of the Mac display graph, so there is nothing to capture. Architecturally immune to future macOS updates.

See `Projects/ace-buddy/research-cluely-findround.md` and `finding-001-oss-clone-survey.md` in the vault for the full investigation.

---

## Quick start

### Prerequisites

- macOS 13+ (developed on 15.6.1 Sequoia)
- Python 3.11+
- An iPhone or iPad on the same Wi-Fi as your Mac
- An OpenAI API key with access to `gpt-4o` and `whisper-1`

### 1. Install

```bash
git clone https://github.com/zjh08177/ace-buddy.git
cd ace-buddy
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configure

Create `~/.ace-buddy/` with three markdown files and a `.env`:

```bash
mkdir -p ~/.ace-buddy
cp .env.example ~/.ace-buddy/.env
$EDITOR ~/.ace-buddy/.env
```

Edit `~/.ace-buddy/.env`:
```bash
OPENAI_API_KEY=sk-...                     # required
ACE_BUDDY_MODEL=gpt-4o                    # or gpt-4o-mini for cheaper
ACE_BUDDY_HOTKEY=<cmd>+<shift>+<space>    # optional override
COST_BOUND_USD=1.0                        # safety cap per process
```

Then create three context files (the app will create stubs on first launch if you skip this — but you'll get better answers if you write real content):

**`~/.ace-buddy/resume.md`** — your resume in markdown. Bullet points work great.
```markdown
# Eric

## Senior ML Engineer at Foo (2021-)
- Led 5-person team building recommendation pipeline
- 40% engagement lift; reduced p99 latency 2.5s → 600ms
- Tech stack: PyTorch, Triton, Ray, GCP

## ML Engineer at Bar (2018-2021)
- ...
```

**`~/.ace-buddy/job.md`** — the role you're interviewing for. First line becomes the title shown on your phone.
```markdown
# Staff ML Engineer at Anthropic

## Role
Building Claude's safety infrastructure...

## Key requirements from JD
- Production ML systems
- Distributed systems
- ...

## Interviewer (if known)
- Jane Doe, Eng Manager
```

**`~/.ace-buddy/system.md`** — optional tone/style override. The default is good; only customize if you want a different voice. The injection-protection rules and output format are always appended automatically.

### 3. Grant macOS permissions

On first launch, macOS will prompt for these. Grant them all and **relaunch** (Screen Recording requires a fresh process):

| Permission | Why |
|---|---|
| **Microphone** | Capture audio for Whisper |
| **Screen Recording** | OCR the interviewer's shared content |
| **Accessibility** | Global hotkey `Cmd+Shift+Space` |

System Settings → Privacy & Security → grant to your terminal app or to Python.

### 4. Run

```bash
make run            # binds 127.0.0.1 (only from this Mac)
make run-lan        # binds 0.0.0.0 + prints QR code (phone on same Wi-Fi)
```

You'll see something like:
```
ace-buddy ready
  URL: http://192.168.1.42:8765/auth?k=AbCdEf1234...
  Scan this QR on your phone (must be on same Wi-Fi):

  ███████ ██  █  ██ ███████
  ██   ██ █████ ██  ██   ██
  ...
```

### 5. Open on your phone

Scan the QR with your phone's camera. Safari opens, the auth cookie is set, and the dark-glass UI loads. You should see:

- A green status dot ("ready")
- The job title from your `job.md`
- An "Ask" button
- A drawer with anticipated questions (computed at boot)

### 6. Use it

When the interviewer asks a question:

1. **Press `Cmd+Shift+Space`** on your Mac, OR **tap "Ask"** on your phone.
2. Within ~1-2 seconds, the answer streams onto your phone:
   - **TL;DR**: one spoken-style sentence (your opener)
   - 3 bullet anchors
   - One specific data point from your resume

3. Glance at the phone like you're checking notes, then speak the answer in your own words.

### 7. While you share your screen

Click the menu bar icon → **"⏸ I'm sharing — pause screen OCR"** before clicking Share in Zoom. This prevents the macOS "screen recording" indicator from appearing in your shared display. The mic + LLM still work; only screen capture pauses.

After unsharing, click **"▶️ Resume screen OCR"**.

---

## Architecture (TL;DR)

```
src/ace_buddy/
├── app.py          ← main, hotkey registration, lifecycle, signal handling
├── server.py       ← FastAPI: /, /auth, /ws, /fire, /qr, /cheatsheet, /debug/*
├── ui/index.html   ← dark glass phone UI (vanilla JS + WebSocket)
├── audio.py        ← sounddevice 60s ring buffer + Whisper non-streaming
├── vision.py       ← /usr/sbin/screencapture CLI + pyobjc Vision OCR
├── pipeline.py     ← parallel sensors → GPT-4o stream → WS broadcast
├── prompt.py       ← byte-stable system prompt with sha256 cache check
├── llm.py          ← OpenAI client + MockLLMClient for tests
├── cheatsheet.py   ← anticipated questions precompute at boot
└── preflight.py    ← 5-check startup health verification
```

**Key design choices** (full reasoning in `Projects/ace-buddy/tech-solution-mvp.md`):
- Single Python process, asyncio-based
- Apple's `screencapture` CLI for capture (Apple-signed, TCC-stable)
- Whisper non-streaming on a 60-second rolling audio window
- Hotkey-triggered (auto-detect deferred to v1.1)
- Token-cookie auth on a per-launch random secret; QR carries the auth URL
- Phone tap (`POST /fire`) as redundant trigger if the hotkey collides

---

## Testing

```bash
make verify          # L0 unit + L4 e2e (no API calls) — ~15s
make verify-l0       # L0 only (fast, no network)
make verify-l4       # E2E only (spawns app, drives via WebSocket)
make verify-live     # L1 with real OpenAI (needs OPENAI_API_KEY)
```

**Test layers**:
- **L0**: 43 unit tests — buffers, prompt builder, preflight, server endpoints
- **L4**: 3 end-to-end tests — spawn app headless with mock LLM + fixture audio + fixture screen, drive via WebSocket, assert streaming tokens arrive
- **L1**: real-API tests, gated on `RUN_LIVE_TESTS=1` + `OPENAI_API_KEY`

Before a real interview, run live verification:
```bash
RUN_LIVE_TESTS=1 OPENAI_API_KEY=sk-... make verify-live
```

---

## Troubleshooting

**"Hotkey doesn't fire"** — Accessibility permission is the most common culprit. System Settings → Privacy & Security → Accessibility → check your terminal/Python. If still broken, use the phone "Ask" button (the redundant trigger).

**"OCR returns empty"** — Screen Recording permission. Grant it, then **fully quit and relaunch** (TCC requires a fresh process for Screen Recording).

**"Phone can't connect"** — Make sure both devices are on the same Wi-Fi (not cellular). Use `make run-lan` (not `make run`). The QR contains the local IP — if your DHCP changed it, regenerate by relaunching.

**"Cheat sheet says (loading…)"** — That's normal for ~3-5s after launch. It's a one-shot GPT-4o call from `resume.md` + `job.md`.

**"Latency too slow"** — Whisper non-streaming on a 20s window is ~800ms p50. If yours is worse, check your wifi. The app also caches the GPT-4o system prompt (sha256 verified), so subsequent calls should be faster.

---

## Development

```bash
. .venv/bin/activate
make verify         # before any commit
make run-headless   # for testing without menu bar / hotkey
```

Edit and re-run. Hot reload is not configured — restart the process.

---

## What's NOT in v0.1 (deferred to v1.1)

- **Auto end-of-question detection** — currently you press a hotkey
- **AirPod TTS** alternative output — currently visual on phone only
- **System audio loopback** (BlackHole) — currently mic-only
- **Multi-display OCR selection** — captures primary display only
- **`py2app` bundle** — currently runs as `python -m ace_buddy.app`
- **Auto share-detection** — manual menu bar pause toggle in v0.1

See `Projects/ace-buddy/tech-solution-mvp.md` §12 for the full deferred list.

---

## License

Personal use only. Not licensed for distribution.

---

## Credits

Built with reviewer feedback from 8 parallel agents across 2 review rounds. Pivoted from Electron/overlay (broken on Sequoia) to companion-device architecture mid-design after deep OSS clone survey. See `Projects/ace-buddy/finding-001-oss-clone-survey.md` for the architecture-decision history.
