# Runpod traning serverless worker

## Build

This project uses Github Actions. To build, push to the `main` branch
```bash
git checkout main
git rebase origin/master
git push
```

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

Prepare venv:

```bash
python -m venv venv
source venv/bin/activate
pip3 install -r requirements-dev.txt
```

Run tests:

```bash
make test
```