# Harvest Planner

A Flask application for estimating South African regional harvest windows from
`crop_calendar_gaul_asap1.csv`. It optionally compares that estimate with the
provided Random Forest model and can request general agricultural guidance from
Gemini.

## Run locally

Use Python 3.10 or later. In PowerShell, from this project directory:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>. Keep the CSV and model file beside `app.py`.

The calendar prediction works without an API key. To enable Gemini guidance,
create a `.env` file beside `app.py` with:

```text
GEMINI_API_KEY=your-key
```

Restart the Flask server after saving `.env`. The app also accepts
`GEMINI_API_KEY` from the process environment, which takes precedence over the
`.env` value. Never share your key in chat, add it to source code, or commit the
`.env` file. The `.env.example` file is a safe template.

The supplied model was serialized with scikit-learn 1.7.2, so the requirements
pin that version for compatibility. A missing or unreadable model does not
prevent calendar estimates.

Set `HARVEST_DATA_PATH` or `HARVEST_MODEL_PATH` to use alternate data/model
files. `FLASK_HOST`, `FLASK_PORT`, and `FLASK_DEBUG=true` are optional runtime
settings; debug mode should only be used for local development.

## API

- `GET /api/health` — application and model status.
- `GET /api/options` — crop names.
- `GET /api/options?crop=Maize` — regions with data for that crop.
- `POST /api/predict` — JSON body: `{"crop":"Maize","region":"Gauteng","planting_date":"2026-10-12"}`.
- `POST /api/analysis` — a prediction response as JSON; requires `GEMINI_API_KEY`.

Planting dates outside the regional planting period return HTTP 422 with the
calendar planting window. All dates in the API use ISO format (`YYYY-MM-DD`).
Regional and model output are seasonal estimates, not field-level observations.
