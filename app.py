import logging
import os
import warnings
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
DATA_FILE = Path(os.environ.get("HARVEST_DATA_PATH", BASE_DIR / "crop_calendar_gaul_asap1.csv"))
MODEL_FILE = Path(os.environ.get("HARVEST_MODEL_PATH", BASE_DIR / "south_africa_harvest_window_model.pkl"))
GEMINI_MODEL = "gemini-2.5-flash"
LOCATION_COLUMNS = [
    "uniqid_asap_0_1",
    "asap0_id",
    "name0_shr",
    "asap1_id",
    "name1_shr",
]
DECODED_COLUMNS = [
    "planting_start",
    "planting_end",
    "harvest_start",
    "harvest_end",
]

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def decode_code(code):
    try:
        parts = [int(part) for part in str(code).split("_")]
    except (TypeError, ValueError):
        return None
    if len(parts) != 4 or any(value < 1 or value > 36 for value in parts):
        return None
    return parts


def load_calendar(path):
    if not path.is_file():
        raise FileNotFoundError(f"Crop calendar CSV not found: {path}")

    source = pd.read_csv(path)
    missing = [column for column in LOCATION_COLUMNS if column not in source.columns]
    if missing:
        raise ValueError(f"Crop calendar is missing required columns: {', '.join(missing)}")

    calendar = source.melt(
        id_vars=LOCATION_COLUMNS,
        value_vars=[column for column in source.columns if column not in LOCATION_COLUMNS],
        var_name="crop",
        value_name="calendar_code",
    )
    calendar = calendar.dropna(subset=["calendar_code"]).copy()
    calendar = calendar[
        calendar["name0_shr"].astype(str).str.strip().str.casefold() == "south africa"
    ].copy()
    calendar["calendar_code"] = calendar["calendar_code"].astype(str).str.strip()
    calendar["decoded"] = calendar["calendar_code"].map(decode_code)
    calendar = calendar.dropna(subset=["decoded"]).copy()

    if calendar.empty:
        raise ValueError("No valid South African crop-calendar records were found.")

    calendar[DECODED_COLUMNS] = pd.DataFrame(
        calendar["decoded"].tolist(), index=calendar.index
    )
    for column in DECODED_COLUMNS:
        calendar[column] = calendar[column].astype(int)
    return calendar


def load_model(path):
    if not path.is_file():
        logger.warning("No trained model found at %s; calendar predictions remain available.", path)
        return None
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = joblib.load(path)
        for warning in caught:
            logger.warning("Model loading warning: %s", warning.message)
        return model
    except Exception:
        logger.exception("Could not load Random Forest model from %s", path)
        return None


def date_to_dekad(value):
    period = 1 if value.day <= 10 else 2 if value.day <= 20 else 3
    return (value.month - 1) * 3 + period


def dekad_to_date(dekad, year):
    dekad = int(dekad)
    if not 1 <= dekad <= 36:
        raise ValueError("Dekad must be between 1 and 36.")
    month = (dekad - 1) // 3 + 1
    period = (dekad - 1) % 3
    day = 1 if period == 0 else 11 if period == 1 else 21
    return date(year, month, day)


def is_dekad_in_range(value, start, end):
    if start <= end:
        return start <= value <= end
    return value >= start or value <= end


def dekad_window(start, end, year):
    end_year = year + (1 if end < start else 0)
    return dekad_to_date(start, year), dekad_to_date(end, end_year)


def get_harvest_dates(planting_date, planting_dekad, harvest_start, harvest_end):
    start_year = planting_date.year + (1 if harvest_start < planting_dekad else 0)
    return dekad_window(harvest_start, harvest_end, start_year)


def format_date(value):
    return value.strftime("%d %B %Y")


def create_app(data_path=None, model_path=None):
    app = Flask(__name__)
    app.json.ensure_ascii = False

    calendar = load_calendar(Path(data_path) if data_path else DATA_FILE)
    model = load_model(Path(model_path) if model_path else MODEL_FILE)
    app.config["CALENDAR"] = calendar
    app.config["MODEL"] = model

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "model_available": app.config["MODEL"] is not None})

    @app.get("/api/options")
    def options():
        frame = app.config["CALENDAR"]
        crops = sorted(frame["crop"].unique().tolist())
        crop = request.args.get("crop")
        regions = []
        if crop in crops:
            regions = sorted(
                frame.loc[frame["crop"] == crop, "name1_shr"]
                .dropna()
                .unique()
                .tolist()
            )
        return jsonify({"crops": crops, "regions": regions})

    @app.post("/api/predict")
    def predict():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Send a JSON object with crop, region, and planting_date."}), 400

        crop = payload.get("crop")
        region = payload.get("region")
        raw_date = payload.get("planting_date")
        if not all(isinstance(value, str) and value.strip() for value in (crop, region, raw_date)):
            return jsonify({"error": "Crop, region, and planting date are required."}), 400
        try:
            planting_date = datetime.strptime(raw_date, "%Y-%m-%d").date()
            if planting_date.isoformat() != raw_date:
                raise ValueError("Date must use zero-padded ISO format.")
        except ValueError:
            return jsonify({"error": "Planting date must use YYYY-MM-DD format."}), 400

        frame = app.config["CALENDAR"]
        match = frame[(frame["crop"] == crop) & (frame["name1_shr"] == region)]
        if match.empty:
            return jsonify({"error": "No crop-calendar record was found for that crop and region."}), 404

        row = match.iloc[0]
        planting_dekad = date_to_dekad(planting_date)
        planting_start = int(row["planting_start"])
        planting_end = int(row["planting_end"])
        if not is_dekad_in_range(planting_dekad, planting_start, planting_end):
            window_start, window_end = dekad_window(
                planting_start, planting_end, planting_date.year
            )
            return jsonify({
                "error": "Planting date is outside the regional planting period.",
                "planting_window": {
                    "start": window_start.isoformat(),
                    "end": window_end.isoformat(),
                    "start_label": format_date(window_start),
                    "end_label": format_date(window_end),
                },
            }), 422

        harvest_start, harvest_end = get_harvest_dates(
            planting_date,
            planting_dekad,
            int(row["harvest_start"]),
            int(row["harvest_end"]),
        )

        ml_result = None
        if app.config["MODEL"] is not None:
            features = pd.DataFrame([{
                "crop": crop,
                "name1_shr": region,
                "planting_start": planting_start,
                "planting_end": planting_end,
            }])
            try:
                prediction = app.config["MODEL"].predict(features)[0]
                ml_start = int(np.clip(round(prediction[0]), 1, 36))
                ml_end = int(np.clip(round(prediction[1]), 1, 36))
                ml_year = planting_date.year + (1 if ml_start < planting_dekad else 0)
                ml_start_date, ml_end_date = dekad_window(ml_start, ml_end, ml_year)
                ml_result = {
                    "start": ml_start_date.isoformat(),
                    "end": ml_end_date.isoformat(),
                    "start_label": format_date(ml_start_date),
                    "end_label": format_date(ml_end_date),
                }
            except Exception:
                logger.exception("Random Forest prediction failed for %s / %s", crop, region)

        return jsonify({
            "crop": crop,
            "region": region,
            "country": "South Africa",
            "planting_date": planting_date.isoformat(),
            "planting_date_label": format_date(planting_date),
            "harvest_window": {
                "start": harvest_start.isoformat(),
                "end": harvest_end.isoformat(),
                "start_label": format_date(harvest_start),
                "end_label": format_date(harvest_end),
            },
            "calendar_code": row["calendar_code"],
            "model_available": app.config["MODEL"] is not None,
            "model_window": ml_result,
            "note": "This is a regional seasonal estimate, not a verified field-level maturity assessment.",
        })

    @app.post("/api/analysis")
    def analysis():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Send a prediction result as a JSON object."}), 400
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            return jsonify({
                "error": "AI analysis is not configured. Set GEMINI_API_KEY on the server to enable it."
            }), 503

        required = ("crop", "region", "planting_date_label", "calendar_code", "harvest_window")
        if any(key not in payload for key in required):
            return jsonify({"error": "Run a valid harvest prediction before requesting AI analysis."}), 400
        window = payload["harvest_window"]
        if not isinstance(window, dict) or not all(key in window for key in ("start_label", "end_label")):
            return jsonify({"error": "Prediction is missing its harvest window."}), 400

        try:
            from google import genai

            prompt = f"""You are an agricultural analysis assistant helping farmers understand regional harvest estimates in South Africa.

CROP: {payload['crop']}
REGION: {payload['region']}, South Africa
PLANTING DATE: {payload['planting_date_label']}
REGIONAL CALENDAR CODE: {payload['calendar_code']}
EXPECTED HARVEST START: {window['start_label']}
EXPECTED HARVEST END: {window['end_label']}
MACHINE LEARNING RESULT: {payload.get('model_window') or 'No trained ML estimate is available.'}

Use these headings:
1. HARVEST WINDOW ANALYSIS
2. CROP MANAGEMENT
3. WEATHER CONSIDERATIONS
4. HARVEST PREPARATION
5. IMPORTANT LIMITATIONS

Give clear, practical advice for South African farmers. Do not claim the crop is ready,
invent live weather, crop observations, soil moisture, or yields. Distinguish regional
estimates from actual field maturity, and describe the Random Forest estimate as experimental."""
            with genai.Client(api_key=api_key) as client:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                )
            if not response.text:
                return jsonify({"error": "The AI service returned an empty response."}), 502
            return jsonify({"analysis": response.text})
        except Exception:
            logger.exception("Gemini analysis request failed")
            return jsonify({"error": "AI analysis failed. Check the server logs and try again."}), 502

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.environ.get("FLASK_HOST", "127.0.0.1"),
        port=int(os.environ.get("FLASK_PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG", "").lower() == "true",
    )
