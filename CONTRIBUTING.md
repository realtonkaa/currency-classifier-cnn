# Contributing

Thanks for your interest in contributing! This is a personal project but PRs and issues are welcome.

## Getting started

1. Fork the repo and clone your fork
2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

3. Create a branch for your change:

```bash
git checkout -b my-feature
```

## Running tests

```bash
pytest tests/
```

For the E2E tests (requires Playwright and a running Streamlit instance):

```bash
pytest tests/test_app_e2e.py -v
```

## Code style

- Follow PEP 8
- Use type hints for all function signatures
- Add docstrings to new functions

## Submitting changes

- Keep PRs focused — one thing per PR
- Make sure tests pass before opening a PR
- Describe what the change does and why in the PR description

## Reporting bugs

Open an issue and include:
- What you did
- What you expected
- What actually happened
- Your Python version and OS
