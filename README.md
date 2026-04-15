# ace-buddy

Personal macOS interview copilot. Listens to your mic + OCRs your screen, streams a GPT-4o answer to your phone over local Wi-Fi.

**Design docs**: see `docs/` or the project vault (`Projects/ace-buddy/` in the second-brain).

## Quickstart

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Config
mkdir -p ~/.ace-buddy
cp .env.example ~/.ace-buddy/.env
# Edit ~/.ace-buddy/.env with your OPENAI_API_KEY
# Add ~/.ace-buddy/resume.md and ~/.ace-buddy/job.md

# 3. Grant permissions on first run
#   System Settings → Privacy & Security → Microphone → allow Terminal/Python
#   System Settings → Privacy & Security → Screen Recording → allow Terminal/Python
#   System Settings → Privacy & Security → Accessibility → allow Terminal/Python

# 4. Run
make run           # binds 127.0.0.1 (phone must be on same Mac)
make run-lan       # binds 0.0.0.0 (phone on same Wi-Fi)

# 5. Scan the QR code printed in terminal on your phone
# 6. Press Cmd+Shift+Space during your interview — answer appears on phone
```

## Testing

```bash
make verify        # L0 unit + L2 Playwright
make verify-l0     # L0 only (fast, no network)
make verify-l1     # L1 with real OpenAI (costs ~$0.05/run)
```
