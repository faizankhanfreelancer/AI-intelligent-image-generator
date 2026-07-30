# Multimodal Image Generation Studio

A production-ready, provider-agnostic text-to-image platform built with Streamlit.
Describe an image in natural language, tune craft settings (style, aspect ratio,
resolution, quality), and generate through **OpenAI** or **Stability AI** behind a
single, swappable pipeline — no UI changes required to add new providers.

Visually, the studio follows a **"Darkroom Atelier"** design language: an ink-black
canvas, a warm amber "safelight" accent, a Fraunces display serif paired with Inter
for UI text and JetBrains Mono for technical readouts, and a rotating camera-aperture
mark as the signature motif — tying the interface back to the craft of image-making.

---

## Features

- **Structured prompt engineering** — raw prompts are never sent as-is; every
  request is rebuilt into a professional, provider-ready prompt (subject, style,
  aspect ratio, quality, resolution, negative-prompt guidance, cinematic lighting cues).
- **Provider abstraction** — `BaseImageProvider` + `provider_factory.py` make adding
  Replicate, Hugging Face, Fal AI, Together AI, or Runware a one-class, zero-UI-change
  addition.
- **Full craft controls** — style (15 options), resolution (6 presets), aspect ratio
  (6 presets), quality tier, image count (1/2/4/8), seed locking, safety level, and
  provider-specific advanced parameters (CFG scale, inference steps, scheduler) that
  only appear when the selected provider actually supports them.
- **Prompt tools** — ready-made prompt library, a random creative-prompt generator,
  and a local prompt enhancer.
- **Results gallery** — responsive image cards with dimensions, file size, generation
  time, the prompt used, and one-click download.
- **Timestamped local archive** — every image is saved to `outputs/` as
  `YYYY_MM_DD_HHMMSS_NN.png`.
- **Session history & favorites** — reopen any past generation, star favorites,
  export the whole session's history as JSON.
- **Session dashboard** — generation counts, average generation time, provider
  usage, and a style-frequency chart.
- **Dark / light theme toggle**, friendly error handling for invalid keys, rate
  limits, timeouts, and empty prompts, and structured logging (console + rotating
  file log under `history/logs/app.log`).

---

## Architecture

```
UI (app.py)
   │  builds a GenerationRequest from sidebar + prompt inputs
   ▼
services/image_generator.py   (orchestrator)
   │  validates → builds structured prompt → calls provider → saves files
   ▼
services/provider_factory.py  →  services/openai_provider.py
                               →  services/stability_provider.py
                               (implements services/base_provider.py)
```

- **models/** — Pydantic models (`GenerationRequest`, `GeneratedImage`,
  `GenerationResult`) give the whole pipeline validation and type safety.
- **prompts/** — structured prompt template, prompt library, random/enhance helpers.
- **services/** — provider abstraction (factory pattern + dependency injection),
  the two concrete providers, and the orchestrating `ImageGenerationService`.
- **utils/** — logging, input validation/sanitization, Pillow helpers, and the
  file-download/save logic.
- **config.py** — the single source of truth for settings, loaded from `.env`.

Adding a new provider never touches `app.py`:

```python
# services/replicate_provider.py
class ReplicateProvider(BaseImageProvider):
    name = "Replicate"
    ...

# anywhere at startup
from services.provider_factory import register_provider
register_provider("Replicate", ReplicateProvider)
```

---

## Folder structure

```
image_generation_studio/
├── app.py                     # Streamlit entrypoint (UI only)
├── config.py                  # Settings loaded from .env
├── requirements.txt
├── README.md
├── .env                       # Your API keys (never commit this)
├── .env.example                # Template for .env
├── .streamlit/config.toml     # Base Streamlit theme
│
├── services/
│   ├── base_provider.py       # Abstract provider interface
│   ├── openai_provider.py     # OpenAI SDK implementation
│   ├── stability_provider.py  # Stability AI REST implementation
│   ├── provider_factory.py    # Factory / registry / DI
│   └── image_generator.py     # Orchestration service
│
├── prompts/
│   └── prompt_builder.py      # Structured prompt template, library, random/enhance
│
├── models/
│   └── request_model.py       # Pydantic request/result models + enums
│
├── utils/
│   ├── downloader.py          # Save-to-disk + download helpers
│   ├── logger.py              # Rich console + rotating file logger
│   ├── image_utils.py         # Pillow helpers
│   └── validators.py          # Input validation & sanitization
│
├── assets/                    # Static assets
├── outputs/                   # Saved generated images (timestamped)
└── history/
    └── logs/app.log           # Rotating application log
```

---

## Installation

```bash
# 1. Clone / unzip the project, then enter it
cd image_generation_studio

# 2. Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API keys
cp .env.example .env
# then edit .env and paste in your keys

# 5. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Environment variables

| Variable                   | Required | Description                                  |
|-----------------------------|----------|-----------------------------------------------|
| `OPENAI_API_KEY`            | For OpenAI generation | Your OpenAI API key |
| `STABILITY_API_KEY`         | For Stability generation | Your Stability AI API key |
| `REQUEST_TIMEOUT_SECONDS`   | No | HTTP timeout per request (default `120`) |
| `MAX_RETRIES`               | No | Reserved for retry logic (default `2`) |
| `DEFAULT_PROVIDER`          | No | Provider selected by default (default `OpenAI`) |
| `LOG_LEVEL`                 | No | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

You only need a key for the provider(s) you intend to use — the sidebar shows a
live "● configured / ○ no key" indicator per provider, and the app will show a
friendly warning rather than crash if a key is missing.

---

## Screenshots

_Add screenshots of the running app here, e.g.:_

```
assets/screenshot-generate.png
assets/screenshot-history.png
assets/screenshot-dashboard.png
```

---

## Technologies

- Python 3.12+
- Streamlit — UI framework
- Official `openai` Python SDK
- Stability AI REST API (`requests`)
- Pydantic — request/response validation
- Pillow — image decoding, sizing, encoding
- python-dotenv — environment configuration
- Rich — structured, readable logging

---

## Future improvements

- Additional providers: Replicate, Hugging Face Inference, Fal AI, Together AI, Runware
- Image-to-image variations and upscaling
- Persistent (disk/DB-backed) history instead of session-only state
- Prompt-strength / creativity sliders wired into provider-specific guidance params
- Generation queue for large batch jobs
- User accounts and per-user galleries

---

## License

Built as a portfolio-grade reference implementation. Adapt freely for your own projects.
