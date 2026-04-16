.PHONY: run test verify verify-l0 verify-l1 verify-l2 verify-l4 clean install

PY ?= python3
PIP ?= $(PY) -m pip

install:
	$(PIP) install -e ".[dev]"

run:
	$(PY) -m ace_buddy.app

run-lan:
	$(PY) -m ace_buddy.app --lan --debug

run-headless:
	$(PY) -m ace_buddy.app --headless --no-auth

test: verify-l0

verify: verify-l0 verify-l4
	@echo "\n✅ All layers green"

verify-l0:
	$(PY) -m pytest tests/ -x -q --tb=short --ignore=tests/test_e2e_full_loop.py --ignore=tests/test_live_api.py

verify-l4:
	$(PY) -m pytest tests/test_e2e_full_loop.py -x -q --tb=short

verify-live:
	RUN_LIVE_TESTS=1 $(PY) -m pytest tests/test_live_api.py -x -q --tb=short

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
