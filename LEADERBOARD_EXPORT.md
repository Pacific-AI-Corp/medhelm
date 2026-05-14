# MedHELM leaderboard export

Export the MedHELM React leaderboard as static files for nginx, S3, or any static host.

## Prerequisites

- Python 3.10+ and [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) (`gcloud`) for downloading public results
- This repository checked out and installed in a virtual environment
- Optional: Node.js and npm if you change `helm-frontend/`

## Setup

```bash
cd /path/to/medhelm
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
helm-server -h | grep export-path
```

Use the `helm-server` from this venv. A global or Homebrew install may not support `--export-path`.

## 1. Populate leaderboard data

Download the public MedHELM release tree from GCS:

```bash
export OUTPUT_PATH="./benchmark_output"
export GCS_BENCHMARK_OUTPUT_PATH="gs://crfm-helm-public/medhelm/benchmark_output"

mkdir -p "$OUTPUT_PATH"
gcloud storage rsync -r "$GCS_BENCHMARK_OUTPUT_PATH" "$OUTPUT_PATH"
```

Confirm release folders exist, for example:

- `benchmark_output/releases/v2.0.0/summary.json`
- `benchmark_output/releases/v1.0.0/summary.json`

To generate your own runs instead of downloading, use `medhelm-run` and `helm-summarize` against the same `OUTPUT_PATH`. See [docs/medhelm.md](docs/medhelm.md).

### Copy runs with only metrics and no instance and response data
```bash
rsync -av \
  --exclude='scenario_state.json' \
  --exclude='instances.json' \
  --exclude='display_requests.json' \
  --exclude='display_predictions.json' \
  --exclude='per_instance_stats.json' \
  . $OUTPUT_PATH
```

## 2. Optional: rebuild the UI

Only needed after editing `helm-frontend/`:


```bash
cd /path/to/medhelm/helm-frontend
npm install --global yarn   # if you don’t already have Yarn
yarn install
yarn build --outDir '../src/helm/benchmark/static_build' --emptyOutDir
cd /path/to/medhelm
```
OR
```bash
cd helm-frontend
npm install
npm run build
cd ..

rm -rf src/helm/benchmark/static_build/assets
cp -r helm-frontend/dist/* src/helm/benchmark/static_build/
```

## 3. Export static sites

Pick a site root and portal base URL (your public origin and path prefix). Export **one directory per release**. Each `--export-path` must not exist yet.

```bash
export SITE_ROOT="/tmp/medhelm_site"
export OUTPUT_PATH="./benchmark_output"
export PORTAL_BASE="https://your.domain/medhelm"

rm -rf "$SITE_ROOT"
mkdir -p "$SITE_ROOT/medhelm"

helm-server \
  --release v2.0.0 \
  -o "$OUTPUT_PATH" \
  --project medhelm \
  --export-path "$SITE_ROOT/medhelm/v2.0.0" \
  --helm-portal-base-url "$PORTAL_BASE" \
  --helm-project-metadata-url "$PORTAL_BASE/project_metadata.json"

helm-server \
  --release v1.0.0 \
  -o "$OUTPUT_PATH" \
  --project medhelm \
  --export-path "$SITE_ROOT/medhelm/v1.0.0" \
  --helm-portal-base-url "$PORTAL_BASE" \
  --helm-project-metadata-url "$PORTAL_BASE/project_metadata.json"
```

Repeat for other tags under `benchmark_output/releases/`.

## 4. Release menu metadata

The navbar release dropdown needs `project_metadata.json` at the URL you set above.

```bash
 cat > "$SITE_ROOT/project_metadata.json" <<'EOF'
[
  {
    "title": "MedHELM",
    "description": "MedHELM leaderboard",
    "id": "medhelm",
    "releases": ["v4.0.0", "v3.0.0", "v2.0.0", "v1.0.0"]
  }
]
EOF
```

`HELM_PROJECT_METADATA_URL` in each export’s `config.js` must match where this file is served. If the file lives under `/medhelm/`, do not point metadata at `/helm/`.

## 5. Preview locally

```bash
python -m http.server -d "$SITE_ROOT"
```

Open:

- `http://127.0.0.1:8000/medhelm/v2.0.0/`
- `http://127.0.0.1:8000/medhelm/project_metadata.json`

Hard-refresh if `config.js` is cached.

## 6. Deploy

Upload `$SITE_ROOT` to your host or map nginx locations to each version folder and to `medhelm/project_metadata.json`.

