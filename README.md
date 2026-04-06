# One-shot WAV Slicer (Local-first)

Local-first web app that accepts one WAV file containing multiple one-shot sounds, detects slices using silence/transient-style envelope analysis, exports each slice as an individual WAV, and returns a ZIP for download.

## Publish to GitHub

The project already has a local Git history (`main`). To upload it:

1. On GitHub, create a **new empty** repository (no README, no `.gitignore`, no license) named e.g. `oneshot-slicer`.
2. From this folder on your machine:

```bash
cd /Users/rutz/Documents/PROG/PROJECTS/oneshot-slicer
git remote add origin git@github.com:YOUR_GITHUB_USER/oneshot-slicer.git
# or HTTPS:
# git remote add origin https://github.com/YOUR_GITHUB_USER/oneshot-slicer.git
git push -u origin main
```

Replace `YOUR_GITHUB_USER` with your GitHub username or organization.

If you use [GitHub CLI](https://cli.github.com/): `gh repo create oneshot-slicer --public --source=. --remote=origin --push`

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

## Architecture

- **Backend**: FastAPI app (`backend/app/main.py`)
- **Segmentation engine**: isolated module (`backend/app/services/segmentation.py`)
- **Frontend**: lightweight static HTML/CSS/JS served by FastAPI (`backend/app/static`)
- **Testing**: basic segmentation tests with synthetic waveforms (`backend/tests`)
- **Packaging**: Dockerfile + docker-compose for local development and future server deployment

Flow:
1. Browser uploads WAV + settings to `/api/process`
2. Backend validates input + reads audio
3. Segmentation detects active regions separated by configurable silence
4. Each region is exported as `shot_001.wav`, `shot_002.wav`, etc.
5. Files are zipped and returned as a downloadable response

## Features

- WAV upload and processing in browser
- Configurable detection:
  - `silence_threshold_db`
  - `min_silence_ms`
  - `min_clip_ms`
  - `padding_ms`
  - `min_peak_db`
  - `normalize`
- Handles:
  - leading/trailing silence
  - low-level noise via threshold + min silence
  - very short clips via minimum clip duration
  - stereo WAV (preserves channel count)
  - original sample rate preservation
- Frontend:
  - clean UI with advanced settings
  - waveform preview
  - progress/status text
  - detected slice count + filenames
  - ZIP download button
- Robustness:
  - file type validation
  - upload size limit (80MB)
  - temporary workspace cleanup
  - logging + exception handling
- Optional **near-duplicate exclusion**: compares each slice to earlier kept slices using (1) peak-normalized waveform with small time alignment (best Pearson |r|), and (2) **spectral cosine** on log magnitude bins so similar samples still match when the time-domain correlation is a bit low. Disabled by default so intentional repeats are kept unless you enable it in Advanced settings.

**Dedupe tuning — remove more duplicates:** lower **waveform |r|** (e.g. `0.82`–`0.86`), lower **spectral match** (e.g. `0.85`–`0.88`), lower **min waveform for spectral rule** slightly (e.g. `0.68`–`0.72`), and/or lower **envelope prefilter** (e.g. `0.50`) so more pairs are fully compared. **If different hits get merged:** raise waveform and spectral thresholds and/or raise **envelope prefilter** to reduce comparisons.

**If near-dupes differ a lot in length:** raise **max length ratio**. **If alignment is off:** raise **max time align (ms)** or **compare points**.

## Run locally (without Docker)

From project root:

```bash
cd oneshot-slicer/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open: [http://localhost:8000](http://localhost:8000)

## Run with Docker Compose

From project root:

```bash
cd oneshot-slicer
docker compose up --build
```

Open: [http://localhost:8000](http://localhost:8000)

## Run tests

```bash
cd oneshot-slicer/backend
python -m pytest -q
```

## API

### `POST /api/process`

`multipart/form-data`:
- `file`: WAV file
- `silence_threshold_db` (float, default `-40`)
- `min_silence_ms` (int, default `80`)
- `min_clip_ms` (int, default `25`)
- `padding_ms` (int, default `5`)
- `normalize` (bool, default `false`)
- `min_peak_db` (float, default `-60`)
- `dedupe` (bool, default `false`)
- `dedupe_correlation_threshold` (float, default `0.87`, range `0.5`–`1`) — waveform Pearson |r| (lower = more aggressive)
- `dedupe_max_length_ratio` (float, default `1.5`, min `1`) — only compare clips if `max(len)/min(len)` is within this ratio
- `dedupe_compare_points` (int, default `2048`, range `128`–`32768`) — resample both clips to `min(this, shorter clip length)` samples for comparison
- `dedupe_max_lag_ms` (float, default `10`, range `0`–`200`) — try aligning copies by shifting up to this many milliseconds on the resampled grid
- `dedupe_prefilter_threshold` (float, default `0.58`, range `0.35`–`0.95`) — envelope fingerprint gate before full compare (lower = more aggressive)
- `dedupe_spectral_threshold` (float, default `0.90`, range `0.5`–`1`) — spectral cosine threshold for the secondary duplicate rule
- `dedupe_spectral_min_waveform` (float, default `0.74`, range `0.5`–`0.99`) — minimum waveform |r| when using the spectral rule

Response:
- `application/zip` download (`one_shots.zip`)
- headers include:
  - `X-Detected-Slices`
  - `X-Exported-Filenames`
  - `X-Discarded-Duplicates` (count dropped as duplicates when `dedupe` is enabled)

## Extensibility notes

The code is intentionally split so later features can be added with minimal churn:
- drag-and-drop uploads in `static/app.js`
- preview/play individual slices by adding a metadata endpoint
- batch upload support by looping files server-side
- richer transient logic by extending `detect_slices()`
