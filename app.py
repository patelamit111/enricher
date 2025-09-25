import csv
import io
import os
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

try:
    import replicate
except ImportError:  # pragma: no cover - handled at runtime
    replicate = None  # type: ignore

load_dotenv()

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", "15")) * 1024 * 1024  # 15 MB default

AVAILABLE_MODELS: List[Dict[str, str]] = [
    {"id": "openai/gpt-5.0-mini", "label": "GPT‑5 Mini"},
    {"id": "openai/gpt-5.0-large", "label": "GPT‑5 Large"},
]
DEFAULT_MODEL = os.getenv("REPLICATE_DEFAULT_MODEL", AVAILABLE_MODELS[0]["id"])
_client: "replicate.Client | None" = None


def get_replicate_client() -> "replicate.Client":
    if replicate is None:
        raise RuntimeError("replicate package is not installed. Install dependencies from requirements.txt")

    global _client
    if _client is not None:
        return _client

    token = os.getenv("REPLICATE_API_TOKEN")
    if not token:
        raise RuntimeError("Missing REPLICATE_API_TOKEN environment variable")

    _client = replicate.Client(api_token=token)
    return _client


def normalize_model_id(model_id: str) -> str:
    available_ids = {m["id"] for m in AVAILABLE_MODELS}
    if model_id in available_ids:
        return model_id
    return DEFAULT_MODEL


def extract_csv_preview(csv_text: str, preview_rows: int = 5) -> Tuple[List[str], List[List[str]], int]:
    reader = csv.reader(io.StringIO(csv_text))
    headers: List[str] = next(reader, [])
    preview: List[List[str]] = []
    total_rows = 0
    for row in reader:
        if total_rows < preview_rows:
            preview.append(row)
        total_rows += 1
    return headers, preview, total_rows


def truncate_text(text: str, limit: int = 8000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def build_analysis_prompt(headers: List[str], preview: List[List[str]], total_rows: int) -> str:
    preview_lines = [", ".join(row) for row in preview]
    preview_text = "\n".join(preview_lines) if preview_lines else "(no data rows found)"
    column_text = ", ".join(headers) if headers else "(no headers)"

    return (
        "You are a senior data enrichment strategist. Given a CSV dataset, identify smart ways to "
        "augment the data with external signals or derived insights. Suggest value-driving enrichments, "
        "potential data sources, and outline steps to automate them."
        "\n\n"
        f"CSV column headers: {column_text}"
        "\nSample rows (may be truncated):\n"
        f"{preview_text}"
        "\n\nTotal data rows: "
        f"{total_rows}"
        "\n\nProvide 3-5 concise enrichment ideas. Each idea should include:"
        "\n- A short title"
        "\n- What the enrichment adds or derives"
        "\n- Suggested external data source or technique"
        "\n- Implementation notes"
        "\nFormat as bullet points."
    )


def build_enrichment_prompt(original_csv: str, instructions: str) -> str:
    return (
        "You are an expert data engineer. You will receive a CSV dataset and instructions for enrichment."
        "\nAnalyze the instructions and perform the requested enrichment to the dataset."
        "\nReturn a CSV with the original headers plus any new enriched columns."
        "\nEnsure the CSV stays aligned row by row. If information is unavailable, leave the field blank."
        "\nIf the instructions cannot be satisfied, explain why instead of fabricating data."
        "\n\nInstructions:\n"
        f"{instructions.strip()}"
        "\n\nOriginal CSV (UTF-8):\n"
        f"{truncate_text(original_csv, 12000)}"
        "\n\nReturn only the enriched CSV data."
    )


def run_model(prompt: str, model: str, temperature: float = 0.2, max_output_tokens: int = 1200) -> str:
    client = get_replicate_client()
    payload: Dict[str, Any] = {
        "messages": [
            {"role": "system", "content": "You are a precise and reliable data analyst."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }

    output = client.run(model, input=payload)

    if isinstance(output, list):
        text_output = "".join(segment for segment in output if isinstance(segment, str))
    elif isinstance(output, dict) and "output" in output:
        text_output = str(output["output"])
    else:
        text_output = str(output)

    return text_output.strip()


@app.route("/")
def index() -> str:
    return render_template(
        "index.html",
        models=AVAILABLE_MODELS,
        default_model=DEFAULT_MODEL,
    )


@app.route("/api/analyze", methods=["POST"])
def analyze_csv() -> Any:
    upload = request.files.get("file")
    model = normalize_model_id(request.form.get("model", DEFAULT_MODEL))

    if upload is None or upload.filename == "":
        return jsonify({"error": "Please upload a CSV file."}), 400

    try:
        csv_bytes = upload.read()
        csv_text = csv_bytes.decode("utf-8-sig", errors="ignore")
    finally:
        upload.close()

    headers, preview, total_rows = extract_csv_preview(csv_text)
    prompt = build_analysis_prompt(headers, preview, total_rows)

    try:
        analysis = run_model(prompt, model=model, temperature=0.1)
    except Exception as err:  # pragma: no cover - runtime error path
        return jsonify({"error": f"Failed to contact model: {err}"}), 500

    return jsonify(
        {
            "analysis": analysis,
            "metadata": {
                "columns": headers,
                "rows_sampled": len(preview),
                "rows_total_estimate": total_rows,
            },
        }
    )


@app.route("/api/enrich", methods=["POST"])
def enrich_csv() -> Any:
    upload = request.files.get("file")
    instructions = request.form.get("instructions", "").strip()
    model = normalize_model_id(request.form.get("model", DEFAULT_MODEL))

    if upload is None or upload.filename == "":
        return jsonify({"error": "Please upload the CSV to enrich."}), 400

    if not instructions:
        return jsonify({"error": "Provide enrichment instructions first."}), 400

    try:
        csv_bytes = upload.read()
        csv_text = csv_bytes.decode("utf-8-sig", errors="ignore")
    finally:
        upload.close()

    prompt = build_enrichment_prompt(csv_text, instructions)

    try:
        enriched_csv = run_model(prompt, model=model, temperature=0.15, max_output_tokens=2000)
    except Exception as err:  # pragma: no cover - runtime error path
        return jsonify({"error": f"Failed to run enrichment: {err}"}), 500

    return jsonify({"enriched_csv": enriched_csv})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5200, debug=True)
