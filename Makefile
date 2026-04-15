.PHONY: run test verify verify-l0 verify-l1 verify-l2 verify-l4 clean install

PY ?= python3
PIP ?= $(PY) -m pip

install:
	$(PIP) install -e ".[dev]"

run:
	$(PY) -m ace_buddy.app

run-lan:
	$(PY) -m ace_buddy.app --lan

run-headless:
	$(PY) -m ace_buddy.app --headless --no-auth

test: verify-l0

verify: verify-l0 verify-l2
	@echo "\n✅ All layers green"

verify-l0:
	$(PY) -m pytest tests/ -x -q --tb=short -m "not live and not e2e"

verify-l1:
	RUN_LIVE_TESTS=1 $(PY) -m pytest tests/ -x -q -m "live"

verify-l2:
	$(PY) -m pytest tests/ -x -q -m "e2e"

verify-l4:
	RUN_LIVE_TESTS=1 $(PY) -m pytest tests/test_full_loop.py -x -q

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
