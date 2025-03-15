# Runpod traning serverless worker

## Build

This project uses Github Actions. To build, push to the `build` branch
```bash
git checkout build
git rebase origin/master
git push
```

## Test

```bash
docker compose -f docker-compose.dev.yaml run test
```