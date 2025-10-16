# Text2Speech (Multi-language UI + Tiny NN Backend)

A minimal Flask backend with a simple HTML/CSS/JS frontend for Text-to-Speech (TTS). The frontend uses the browser's Web Speech API to synthesize speech locally. The backend provides two optional neural helpers built with PyTorch:

- Predict speech parameters (`pitch`, `rate`) from input text.
- Detect language among a few demo labels: `en`, `hi`, `es`, `hinglish`.

The UI lets you pick a voice, adjust pitch and rate, and speak your text. A helper button can call the backend to suggest pitch/rate and show the detected language.

---

## Project Structure

```
Text2speech/
├─ backend/
│  ├─ app.py                 # Flask app, routes and CORS
│  ├─ model.py               # Tiny PyTorch models for params + language
│  ├─ requirements.txt       # Python dependencies
│  └─ templates/
│     ├─ index.html          # Frontend (Web Speech API)
│     └─ style.css           # Frontend styles
└─ README.md                 # This file
```

---

## Tech Stack

- Backend: Flask + flask-cors
- ML: PyTorch (CPU)
- Frontend: Plain HTML/CSS/JS using `window.speechSynthesis`

---

## Features

- Select a voice from installed/browser-provided voices.
- Control pitch and rate via sliders.
- Speak text locally with Web Speech API.
- Optional: Click "Suggest from NN" to:
  - Get suggested `pitch` and `rate` from `/nn_params`.
  - Detect language from `/nn_language`.

---

## Requirements

- Python 3.9+ recommended
- pip
- Windows/macOS/Linux (frontend works where the Web Speech API is supported; best in Chrome/Edge)

Python packages (from `backend/requirements.txt`):
- Flask>=2.3.0
- flask-cors>=4.0.0
- torch>=2.1.0

Note: Installing PyTorch can take time and may require extra system libraries; see https://pytorch.org for platform-specific wheels if `pip install` is slow or fails.

---

## Setup

1) Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
# Windows PowerShell
. .venv\Scripts\Activate.ps1
# macOS/Linux (bash)
source .venv/bin/activate
```

2) Install dependencies:

```bash
pip install -r backend/requirements.txt
```

3) Run the backend:

```bash
python backend/app.py
```

- The server starts at: http://127.0.0.1:5000/
- The root route `/` serves `backend/templates/index.html`.
- Styles are served from `/style.css` mapped to `backend/templates/style.css`.

---

## Usage

- Open http://127.0.0.1:5000/ in a modern browser (Chrome/Edge recommended).
- Allow the page a moment for voices to populate.
- Enter text, optionally click "Suggest from NN" to prefill pitch/rate and see language.
- Click "Speak".

Tip: If no voices appear initially, wait a second; the page listens for `onvoiceschanged` to repopulate.

---

## API

Base URL: `http://127.0.0.1:5000`

- POST `/nn_params`
  - Body: `{ "text": "your text" }`
  - Response: `{ "pitch": <float>, "rate": <float> }` in browser-friendly ranges `[0.5, 2.0]`.

- POST `/nn_language`
  - Body: `{ "text": "your text" }`
  - Response: `{ "language": "en|hi|es|hinglish", "scores": {"en": p, ...} }`

### cURL examples

```bash
curl -s -X POST http://127.0.0.1:5000/nn_params \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world!"}'

curl -s -X POST http://127.0.0.1:5000/nn_language \
  -H "Content-Type: application/json" \
  -d '{"text":"Hola señor!"}'
```

---

## How it works

- Frontend (`templates/index.html`):
  - Uses `window.speechSynthesis` and `SpeechSynthesisUtterance` to speak locally.
  - Populates voices via `speechSynthesis.getVoices()`.
  - Sends optional requests to Flask for suggested params and language detection.

- Backend (`app.py`):
  - `/` serves the UI.
  - `/style.css` serves the stylesheet from `templates/` without changing the HTML.
  - `/nn_params` and `/nn_language` accept JSON `{text}` and call helpers in `model.py`.

- Models (`model.py`):
  - `predict_params(text)` uses a tiny untrained NN to map simple text features to `[pitch, rate]`.
  - `detect_language(text)` uses a shallow linear classifier with handcrafted weights to produce softmax scores for `en|hi|es|hinglish`.

These are demo-quality models meant for UX prototyping; they are deterministic and lightweight but not production-grade.

---

## Troubleshooting

- No sound or no voices
  - Use Chrome/Edge desktop. Some browsers restrict the Web Speech API.
  - Ensure system TTS voices are installed (OS-dependent).
  - If speech starts but is silent after repeated clicks, the code calls `synth.cancel()` before speaking to avoid queue buildup; try again.

- PyTorch install issues
  - Check https://pytorch.org/get-started/locally/ for a wheel compatible with your Python version and GPU/CPU.

- CORS / Fetch errors
  - `flask-cors` is enabled via `CORS(app)`. Ensure you are calling the same origin `http://127.0.0.1:5000` from the page.

---

## Extending

- Replace the helper models with your own trained models and load weights.
- Add server-side TTS (e.g., edge-tts, Coqui TTS) if you need consistent voices across browsers.
- Persist user presets in localStorage and add more controls (volume, pause/resume UI).
- Containerize (Docker) or deploy behind a production server (gunicorn + reverse proxy).

---

