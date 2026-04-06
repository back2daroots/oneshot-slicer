# One-shot WAV Slicer (Local-first)

Local-first web app that accepts one WAV file containing multiple one-shot sounds, detects slices using silence/transient-style envelope analysis, exports each slice as an individual WAV, and returns a ZIP for download.

## Repository

```bash
git clone https://github.com/YOUR_USERNAME/oneshot-slicer.git
cd oneshot-slicer
```

Replace `YOUR_USERNAME` with the GitHub owner after you create the remote (or use SSH: `git@github.com:YOUR_USERNAME/oneshot-slicer.git`).

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

Response:
- `application/zip` download (`one_shots.zip`)
- headers include:
  - `X-Detected-Slices`
  - `X-Exported-Filenames`

## Extensibility notes

The code is intentionally split so later features can be added with minimal churn:
- drag-and-drop uploads in `static/app.js`
- preview/play individual slices by adding a metadata endpoint
- batch upload support by looping files server-side
- richer transient logic by extending `detect_slices()`
