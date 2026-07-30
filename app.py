"""
app.py

Multimodal Image Generation Studio — main Streamlit entrypoint.

A professional, provider-agnostic text-to-image studio. This module is
UI-only: all business logic (prompt engineering, provider calls, file
persistence, validation, logging) lives in services/, prompts/, utils/,
and models/. app.py just wires user input to ImageGenerationService and
renders the results.
"""

from __future__ import annotations

import json
from datetime import datetime

import streamlit as st

from config import settings
from models.request_model import (
    AspectRatio,
    GenerationRequest,
    QualityOption,
    Resolution,
    SafetyLevel,
    StyleOption,
)
from prompts.prompt_builder import PROMPT_LIBRARY, enhance_prompt, generate_random_prompt
from services.image_generator import ImageGenerationService
from services.openai_provider import ProviderError
from services.provider_factory import get_provider, list_providers
from utils.downloader import image_bytes_for_download
from utils.image_utils import human_readable_size
from utils.logger import logger
from utils.validators import ValidationError

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Multimodal Image Generation Studio",
    page_icon="◐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
_DEFAULTS = {
    "history": [],
    "current_result": None,
    "prompt_text": "",
    "negative_prompt_text": "blurry, low quality, distorted, extra fingers, text, watermark",
    "dark_mode": True,
    "favorites": set(),
    "show_advanced": False,
}
for key, value in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------------------------------
# Theme / CSS — "Darkroom Atelier" design system
# --------------------------------------------------------------------------
def inject_theme(dark: bool) -> None:
    """Inject the full design-token CSS system. Signature element: the
    aperture iris — a rotating camera-aperture motif used as the brand
    mark and the generation loading state, tying the visual identity to
    photographic image-making."""

    if dark:
        bg_void, bg_surface, bg_elevated = "#0B0B0E", "#15141A", "#1D1C24"
        border = "#2B2934"
        text_primary, text_muted = "#F4F1EA", "#9C98A8"
        card_shadow = "0 12px 32px rgba(0,0,0,0.45)"
    else:
        bg_void, bg_surface, bg_elevated = "#F6F4EF", "#FFFFFF", "#FFFFFF"
        border = "#E4E0D6"
        text_primary, text_muted = "#1B1A1F", "#6B6875"
        card_shadow = "0 12px 32px rgba(20,15,10,0.08)"

    accent_amber = "#E3A857"
    accent_amber_soft = "#E3A85722"
    accent_violet = "#7C6FF0"

    st.markdown(
        f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {{
        --bg-void: {bg_void};
        --bg-surface: {bg_surface};
        --bg-elevated: {bg_elevated};
        --border: {border};
        --text-primary: {text_primary};
        --text-muted: {text_muted};
        --accent-amber: {accent_amber};
        --accent-amber-soft: {accent_amber_soft};
        --accent-violet: {accent_violet};
        --shadow: {card_shadow};
    }}

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background: var(--bg-void);
        color: var(--text-primary);
    }}

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {{
        background: var(--bg-surface);
        border-right: 1px solid var(--border);
    }}
    section[data-testid="stSidebar"] .stMarkdown h3 {{
        font-family: 'Fraunces', serif;
        font-weight: 600;
        color: var(--text-primary);
        letter-spacing: 0.02em;
        margin-top: 1.4rem;
        margin-bottom: 0.4rem;
        font-size: 0.95rem;
        text-transform: uppercase;
        opacity: 0.85;
    }}

    /* ---------- Hero ---------- */
    .hero-wrap {{
        display: flex;
        align-items: center;
        gap: 22px;
        padding: 8px 0 6px 0;
        margin-bottom: 6px;
        border-bottom: 1px solid var(--border);
    }}
    .hero-title {{
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-size: 2.6rem;
        line-height: 1.05;
        letter-spacing: -0.01em;
        color: var(--text-primary);
        margin: 0;
    }}
    .hero-title .accent {{ color: var(--accent-amber); font-style: italic; }}
    .hero-sub {{
        font-family: 'Inter', sans-serif;
        color: var(--text-muted);
        font-size: 0.98rem;
        margin-top: 6px;
        max-width: 640px;
    }}
    .eyebrow {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--accent-amber);
        margin-bottom: 6px;
    }}

    /* ---------- Aperture signature mark ---------- */
    .aperture {{
        width: 64px; height: 64px; flex-shrink: 0;
        animation: spin 18s linear infinite;
    }}
    @keyframes spin {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
    .aperture-spin-fast {{ animation: spin 1.6s linear infinite; }}

    /* ---------- Buttons ---------- */
    .stButton > button {{
        border-radius: 8px;
        border: 1px solid var(--border);
        background: var(--bg-elevated);
        color: var(--text-primary);
        font-weight: 500;
        transition: all 0.15s ease;
    }}
    .stButton > button:hover {{
        border-color: var(--accent-amber);
        color: var(--accent-amber);
        transform: translateY(-1px);
    }}
    div[data-testid="stFormSubmitButton"] > button,
    .primary-generate button {{
        background: linear-gradient(135deg, var(--accent-amber), #C98A3A);
        color: #1B140A;
        font-weight: 600;
        border: none;
        box-shadow: 0 6px 18px var(--accent-amber-soft);
    }}
    .primary-generate button:hover {{
        filter: brightness(1.08);
        color: #1B140A;
    }}

    /* ---------- Text inputs ---------- */
    .stTextArea textarea, .stTextInput input {{
        background: var(--bg-elevated);
        border: 1px solid var(--border);
        color: var(--text-primary);
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
    }}
    .stTextArea textarea:focus, .stTextInput input:focus {{
        border-color: var(--accent-amber);
        box-shadow: 0 0 0 1px var(--accent-amber);
    }}

    /* ---------- Cards ---------- */
    .studio-card {{
        background: var(--bg-surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 16px;
        box-shadow: var(--shadow);
        margin-bottom: 14px;
    }}
    .meta-row {{
        display: flex;
        justify-content: space-between;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        color: var(--text-muted);
        border-top: 1px dashed var(--border);
        margin-top: 10px;
        padding-top: 8px;
    }}
    .meta-pill {{
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: var(--accent-amber);
        background: var(--accent-amber-soft);
        border-radius: 20px;
        padding: 2px 10px;
        margin-right: 6px;
        margin-bottom: 4px;
    }}
    .prompt-used {{
        font-size: 0.78rem;
        color: var(--text-muted);
        font-style: italic;
        margin-top: 8px;
        line-height: 1.4;
    }}

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        border-bottom: 1px solid var(--border);
    }}
    .stTabs [data-baseweb="tab"] {{
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        color: var(--text-muted);
    }}
    .stTabs [aria-selected="true"] {{
        color: var(--accent-amber) !important;
    }}

    /* ---------- Metrics ---------- */
    div[data-testid="stMetric"] {{
        background: var(--bg-surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 10px 14px;
    }}
    div[data-testid="stMetricValue"] {{
        font-family: 'JetBrains Mono', monospace;
        color: var(--accent-amber);
    }}

    /* ---------- Misc ---------- */
    hr {{ border-color: var(--border); }}
    .stSpinner > div {{ border-top-color: var(--accent-amber) !important; }}
    ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
    ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 8px; }}

    .empty-state {{
        text-align: center;
        padding: 60px 20px;
        color: var(--text-muted);
        font-family: 'Inter', sans-serif;
    }}
    .empty-state .eyebrow {{ display: block; margin-bottom: 8px; }}
    </style>
    """,
        unsafe_allow_html=True,
    )


APERTURE_SVG = """
<svg class="aperture" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="50" cy="50" r="46" fill="none" stroke="var(--accent-amber)" stroke-width="1.5" opacity="0.35"/>
  <g fill="var(--accent-amber)">
    <path d="M50 50 L50 8 A42 42 0 0 1 86 29 Z" opacity="0.9"/>
    <path d="M50 50 L86 29 A42 42 0 0 1 86 71 Z" opacity="0.75"/>
    <path d="M50 50 L86 71 A42 42 0 0 1 50 92 Z" opacity="0.6"/>
    <path d="M50 50 L50 92 A42 42 0 0 1 14 71 Z" opacity="0.45"/>
    <path d="M50 50 L14 71 A42 42 0 0 1 14 29 Z" opacity="0.3"/>
    <path d="M50 50 L14 29 A42 42 0 0 1 50 8 Z" opacity="0.15"/>
  </g>
  <circle cx="50" cy="50" r="14" fill="var(--bg-void)" stroke="var(--accent-amber)" stroke-width="1.5"/>
</svg>
"""


def render_hero() -> None:
    st.markdown(
        f"""
        <div class="hero-wrap">
            {APERTURE_SVG}
            <div>
                <div class="eyebrow">Provider-agnostic · Text-to-Image</div>
                <h1 class="hero-title">Multimodal <span class="accent">Image</span> Studio</h1>
                <div class="hero-sub">
                    Describe an image, tune the craft settings, and generate with OpenAI or
                    Stability AI through one unified, swappable pipeline.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Sidebar — generation controls
# --------------------------------------------------------------------------
def render_sidebar() -> GenerationRequest:
    with st.sidebar:
        st.markdown(f"### Theme")
        st.session_state.dark_mode = st.toggle("Dark mode", value=st.session_state.dark_mode)

        st.markdown("### AI Provider")
        provider_status = settings.provider_status()
        provider_labels = [
            f"{p} {'●' if provider_status.get(p) else '○ (no key)'}" for p in list_providers()
        ]
        provider_choice_idx = st.selectbox(
            "Provider",
            options=range(len(list_providers())),
            format_func=lambda i: provider_labels[i],
            label_visibility="collapsed",
        )
        provider_name = list_providers()[provider_choice_idx]
        provider = get_provider(provider_name)

        st.markdown("### Image Model")
        model = st.selectbox("Model", options=provider.available_models, label_visibility="collapsed")

        st.markdown("### Resolution")
        resolution = st.selectbox(
            "Resolution",
            options=[r.value for r in Resolution],
            index=2,
            label_visibility="collapsed",
        )

        st.markdown("### Aspect Ratio")
        aspect_ratio = st.select_slider(
            "Aspect Ratio",
            options=[a.value for a in AspectRatio],
            value="1:1",
            label_visibility="collapsed",
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### Images")
            num_images = st.selectbox("Number", options=[1, 2, 4, 8], label_visibility="collapsed")
        with col_b:
            st.markdown("### Quality")
            quality = st.selectbox(
                "Quality", options=[q.value for q in QualityOption], label_visibility="collapsed"
            )

        st.markdown("### Style")
        style = st.selectbox(
            "Style", options=[s.value for s in StyleOption], label_visibility="collapsed"
        )

        st.markdown("### Safety Level")
        safety_level = st.select_slider(
            "Safety Level",
            options=[s.value for s in SafetyLevel],
            value=SafetyLevel.STANDARD.value,
            label_visibility="collapsed",
        )

        seed_enabled = st.checkbox("Fix random seed", value=False)
        seed = None
        if seed_enabled:
            seed = st.number_input("Seed", min_value=0, max_value=2_147_483_647, value=42, step=1)

        # Only show advanced params the selected provider actually supports.
        supported = provider.supported_advanced_params
        cfg_scale, inference_steps, scheduler = 7.0, 30, "DDIM"
        if supported & {"cfg_scale", "inference_steps"}:
            with st.expander("Advanced Parameters", expanded=False):
                if "cfg_scale" in supported:
                    cfg_scale = st.slider("CFG Scale (prompt strength)", 0.0, 35.0, 7.0, 0.5)
                if "inference_steps" in supported:
                    inference_steps = st.slider("Inference Steps", 10, 150, 30, 5)
                    scheduler = st.selectbox("Scheduler", ["DDIM", "DPM++ 2M", "Euler a", "K-LMS"])

        st.markdown("---")
        st.markdown(
            f"<span class='meta-pill'>{provider_name}</span>"
            f"<span class='meta-pill'>{model}</span>"
            f"<span class='meta-pill'>{resolution}</span>",
            unsafe_allow_html=True,
        )

        if not provider.is_configured():
            st.warning(
                f"No API key detected for **{provider_name}**. "
                f"Add it to your `.env` file to enable generation.",
                icon="⚠️",
            )

    return GenerationRequest(
        prompt="placeholder",  # overwritten by caller before use
        provider=provider_name,
        model=model,
        style=StyleOption(style),
        resolution=Resolution(resolution),
        aspect_ratio=AspectRatio(aspect_ratio),
        quality=QualityOption(quality),
        num_images=num_images,
        seed=seed,
        safety_level=SafetyLevel(safety_level),
        cfg_scale=cfg_scale,
        inference_steps=inference_steps,
        scheduler=scheduler,
    )


# --------------------------------------------------------------------------
# Image card rendering
# --------------------------------------------------------------------------
def render_image_card(image_meta, request: GenerationRequest, generation_seconds: float, final_prompt: str) -> None:
    with st.container(border=True):
        st.image(image_meta.file_path, use_container_width=True)
        st.markdown(
            f"<div class='meta-row'>"
            f"<span>{image_meta.width}×{image_meta.height}px</span>"
            f"<span>{human_readable_size(image_meta.file_size_bytes)}</span>"
            f"<span>{generation_seconds:.2f}s</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='prompt-used'>“{request.prompt[:140]}"
            f"{'…' if len(request.prompt) > 140 else ''}”</div>",
            unsafe_allow_html=True,
        )
        with open(image_meta.file_path, "rb") as f:
            st.download_button(
                "⬇ Download",
                data=f.read(),
                file_name=image_meta.file_path.split("/")[-1],
                mime="image/png",
                use_container_width=True,
                key=f"dl_{image_meta.file_path}",
            )


# --------------------------------------------------------------------------
# Main generate tab
# --------------------------------------------------------------------------
def render_generate_tab(base_request: GenerationRequest, service: ImageGenerationService) -> None:
    st.markdown("<div class='eyebrow'>Prompt</div>", unsafe_allow_html=True)
    st.session_state.prompt_text = st.text_area(
        "Describe your image",
        value=st.session_state.prompt_text,
        height=120,
        placeholder="A futuristic cyberpunk city during sunset with flying cars and neon reflections.",
        label_visibility="collapsed",
    )

    with st.expander("Negative prompt (what to avoid)", expanded=False):
        st.session_state.negative_prompt_text = st.text_area(
            "Negative prompt",
            value=st.session_state.negative_prompt_text,
            height=68,
            label_visibility="collapsed",
        )

    btn_cols = st.columns([1.3, 1, 1, 1])
    with btn_cols[0]:
        st.markdown("<div class='primary-generate'>", unsafe_allow_html=True)
        generate_clicked = st.button("✦ Generate Images", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with btn_cols[1]:
        random_clicked = st.button("🎲 Random Prompt", use_container_width=True)
    with btn_cols[2]:
        enhance_clicked = st.button("✨ Enhance Prompt", use_container_width=True)
    with btn_cols[3]:
        clear_clicked = st.button("✕ Clear", use_container_width=True)

    if random_clicked:
        st.session_state.prompt_text = generate_random_prompt()
        st.rerun()
    if enhance_clicked and st.session_state.prompt_text.strip():
        st.session_state.prompt_text = enhance_prompt(st.session_state.prompt_text)
        st.rerun()
    if clear_clicked:
        st.session_state.prompt_text = ""
        st.session_state.current_result = None
        st.rerun()

    if generate_clicked:
        base_request.prompt = st.session_state.prompt_text
        base_request.negative_prompt = st.session_state.negative_prompt_text
        try:
            with st.spinner("Opening the aperture — rendering your image…"):
                result = service.generate(base_request)
            st.session_state.current_result = result
            st.session_state.history.insert(0, result)
            st.toast("Generation complete.", icon="✅")
        except (ValidationError, ProviderError) as exc:
            st.error(str(exc), icon="🚫")
            logger.error("Generation failed: %s", exc)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Unexpected error: {exc}", icon="🚫")
            logger.exception("Unexpected generation failure")

    st.markdown("---")

    result = st.session_state.current_result
    if result is None:
        st.markdown(
            "<div class='empty-state'>"
            "<span class='eyebrow'>Nothing rendered yet</span>"
            "Write a prompt above and press <b>Generate Images</b> to begin."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"<div class='eyebrow'>Result · {result.provider_used} · {result.model_used} · "
        f"{result.generation_seconds}s total</div>",
        unsafe_allow_html=True,
    )
    with st.expander("View final engineered prompt sent to the provider"):
        st.code(result.final_prompt, language="text")

    cols = st.columns(min(len(result.images), 4) or 1)
    for i, image_meta in enumerate(result.images):
        with cols[i % len(cols)]:
            render_image_card(image_meta, result.request, result.generation_seconds, result.final_prompt)


# --------------------------------------------------------------------------
# History / Gallery tab
# --------------------------------------------------------------------------
def render_history_tab() -> None:
    history = st.session_state.history
    if not history:
        st.markdown(
            "<div class='empty-state'><span class='eyebrow'>No history yet</span>"
            "Every generation you run this session will appear here as a gallery.</div>",
            unsafe_allow_html=True,
        )
        return

    top_cols = st.columns([3, 1])
    with top_cols[0]:
        st.markdown(f"<div class='eyebrow'>{len(history)} generation(s) this session</div>", unsafe_allow_html=True)
    with top_cols[1]:
        export_payload = json.dumps(
            [
                {
                    "prompt": r.request.prompt,
                    "negative_prompt": r.request.negative_prompt,
                    "provider": r.provider_used,
                    "model": r.model_used,
                    "resolution": r.request.resolution.value,
                    "aspect_ratio": r.request.aspect_ratio.value,
                    "timestamp": r.created_at.isoformat(),
                }
                for r in history
            ],
            indent=2,
        )
        st.download_button(
            "Export history (JSON)",
            data=export_payload,
            file_name="generation_history.json",
            mime="application/json",
            use_container_width=True,
        )

    for idx, result in enumerate(history):
        fav_key = f"fav_{idx}_{result.created_at.timestamp()}"
        is_fav = fav_key in st.session_state.favorites
        with st.container(border=True):
            header_cols = st.columns([5, 1])
            with header_cols[0]:
                st.markdown(
                    f"**{result.request.prompt[:90]}{'…' if len(result.request.prompt) > 90 else ''}**  \n"
                    f"<span class='meta-pill'>{result.provider_used}</span>"
                    f"<span class='meta-pill'>{result.request.style.value}</span>"
                    f"<span class='meta-pill'>{result.request.aspect_ratio.value}</span>"
                    f"<span class='meta-pill'>{result.created_at.strftime('%Y-%m-%d %H:%M:%S')}</span>",
                    unsafe_allow_html=True,
                )
            with header_cols[1]:
                if st.button("★" if is_fav else "☆", key=f"btn_{fav_key}"):
                    if is_fav:
                        st.session_state.favorites.discard(fav_key)
                    else:
                        st.session_state.favorites.add(fav_key)
                    st.rerun()

            thumb_cols = st.columns(min(len(result.images), 4) or 1)
            for i, image_meta in enumerate(result.images):
                with thumb_cols[i % len(thumb_cols)]:
                    st.image(image_meta.file_path, use_container_width=True)
                    with open(image_meta.file_path, "rb") as f:
                        st.download_button(
                            "⬇",
                            data=f.read(),
                            file_name=image_meta.file_path.split("/")[-1],
                            mime="image/png",
                            key=f"hist_dl_{idx}_{i}_{image_meta.file_path}",
                            use_container_width=True,
                        )

            reopen_col, _ = st.columns([1, 3])
            with reopen_col:
                if st.button("Reopen in Generate tab", key=f"reopen_{idx}"):
                    st.session_state.current_result = result
                    st.session_state.prompt_text = result.request.prompt
                    st.session_state.negative_prompt_text = result.request.negative_prompt
                    st.toast("Loaded into the Generate tab.", icon="↩️")


# --------------------------------------------------------------------------
# Prompt Library tab
# --------------------------------------------------------------------------
def render_library_tab() -> None:
    st.markdown(
        "<div class='eyebrow'>Ready-made prompts</div>"
        "<div class='hero-sub'>Pick a starting point, then refine it in the Generate tab.</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    items = list(PROMPT_LIBRARY.items())
    cols = st.columns(3)
    for i, (title, prompt) in enumerate(items):
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(prompt)
                if st.button("Use this prompt", key=f"lib_{title}", use_container_width=True):
                    st.session_state.prompt_text = prompt
                    st.toast(f"Loaded '{title}' into the prompt box.", icon="📚")


# --------------------------------------------------------------------------
# Dashboard tab
# --------------------------------------------------------------------------
def render_dashboard_tab() -> None:
    history = st.session_state.history
    total_images = sum(len(r.images) for r in history)
    total_time = sum(r.generation_seconds for r in history)
    avg_time = (total_time / len(history)) if history else 0.0
    providers_used = {r.provider_used for r in history}

    st.markdown("<div class='eyebrow'>Session dashboard</div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Generations", len(history))
    m2.metric("Images created", total_images)
    m3.metric("Avg. generation time", f"{avg_time:.2f}s")
    m4.metric("Providers used", len(providers_used) if history else 0)

    st.write("")
    key_status = settings.provider_status()
    st.markdown("<div class='eyebrow'>Provider configuration</div>", unsafe_allow_html=True)
    status_cols = st.columns(len(key_status))
    for col, (name, ok) in zip(status_cols, key_status.items()):
        with col:
            with st.container(border=True):
                st.markdown(f"**{name}**")
                st.markdown("🟢 Key configured" if ok else "🔴 Missing key")

    if history:
        st.write("")
        st.markdown("<div class='eyebrow'>Recent styles</div>", unsafe_allow_html=True)
        style_counts: dict[str, int] = {}
        for r in history:
            style_counts[r.request.style.value] = style_counts.get(r.request.style.value, 0) + 1
        st.bar_chart(style_counts)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    inject_theme(st.session_state.dark_mode)
    render_hero()

    base_request = render_sidebar()
    service = ImageGenerationService()

    tab_generate, tab_history, tab_library, tab_dashboard = st.tabs(
        ["🖼 Generate", "🗂 History & Gallery", "📚 Prompt Library", "📊 Dashboard"]
    )

    with tab_generate:
        render_generate_tab(base_request, service)
    with tab_history:
        render_history_tab()
    with tab_library:
        render_library_tab()
    with tab_dashboard:
        render_dashboard_tab()

    st.markdown(
        "<div style='text-align:center; color:var(--text-muted); font-size:0.75rem; "
        "font-family:JetBrains Mono, monospace; margin-top:2rem;'>"
        f"{settings.app_name} · v{settings.app_version}"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    logger.info("Application startup | %s v%s", settings.app_name, settings.app_version)
    main()
