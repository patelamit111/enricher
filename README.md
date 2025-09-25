# Enricher

Local prototype for a CSV enrichment assistant powered by GPT‑5 models served via Replicate.

## Features
- Upload a CSV from the browser
- Ask GPT‑5 for tailored enrichment ideas based on headers and sample rows
- Refine instructions, run enrichment, and download an updated CSV
- Runs locally on port `5200`

## Prerequisites
- Python 3.9+
- A Replicate account and API token with access to the desired OpenAI GPT‑5 models

## Getting started
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and paste your REPLICATE_API_TOKEN
python app.py
```

The app will be available at http://localhost:5200.

## Environment variables
| Name | Description |
| ---- | ----------- |
| `REPLICATE_API_TOKEN` | **Required.** Your Replicate API token. |
| `REPLICATE_DEFAULT_MODEL` | Optional. Default GPT‑5 model id (must be from the dropdown options). |
| `MAX_UPLOAD_MB` | Optional. Maximum upload size in MB (default `15`). |

## Notes
- The sample implementation limits payloads sent to GPT‑5 to keep prompts manageable. Large CSVs may need additional chunking before enrichment.
- Update `AVAILABLE_MODELS` in `app.py` if you have access to different GPT‑5 variants on Replicate.
