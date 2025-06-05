# Runpod traning serverless worker

## Build

This project uses Github Actions. To build, push to the `main` branch
```bash
git checkout main
git rebase origin/master
git push
```

### Skipping Builds

You can skip builds in several ways:

#### 1. Skip Entire Workflow (Tests + Build)
Add `[skip ci]` or `[ci skip]` to your commit message:
```bash
git commit -m "Update documentation [skip ci]"
git commit -m "Fix typo [ci skip]"
```

#### 2. Skip Only Build (Tests Still Run)
Add `[skip build]` to your commit message:
```bash
git commit -m "Add logging [skip build]"
```

#### 3. Manual Workflow Control
- Go to GitHub Actions → Your workflow → "Run workflow"
- Choose "Skip the build step: true"
- Tests will run, but build will be skipped

#### 4. Automatic Skip for Documentation
Builds are automatically skipped when only these files change:
- `*.md` files (README, docs, etc.)
- Files in `docs/` directory
- `.gitignore`
- `LICENSE`

| Method | Tests Run? | Build Runs? | Use Case |
|--------|------------|-------------|----------|
| `[skip ci]` | ❌ No | ❌ No | Skip everything |
| `[ci skip]` | ❌ No | ❌ No | Skip everything |
| `[skip build]` | ✅ Yes | ❌ No | Test code but don't build |
| Manual dispatch | ✅ Yes | ❌ No | Interactive control |
| Doc-only changes | ❌ No | ❌ No | Automatic optimization |

## Test

You might need to install these first:

```bash
sudo apt-get update
sudo apt-get install -y libsqlite3-dev libffi-dev libbz2-dev libncurses-dev \
                       libreadline-dev libssl-dev zlib1g-dev libgdbm-dev \
                       liblzma-dev tk-dev

pyenv uninstall 3.10.12
pyenv install 3.10.12
```

Run tests:

```bash
make test
```

The Makefile includes several helpful commands:

```bash
# Full setup from scratch (system deps + venv + python deps)
make setup

# Run tests (includes automatic dependency setup)
make test

# Quick test run (assumes setup is already done)
make test-quick

# Clean up __pycache__ and virtual environment
make clean

# See all available commands
make help
```