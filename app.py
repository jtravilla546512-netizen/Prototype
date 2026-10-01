from __future__ import annotations

import re
import time
from difflib import SequenceMatcher

import streamlit as st


st.set_page_config(page_title="DA-RoBERTa | Media Bias Analysis", page_icon="📰", layout="wide")

st.markdown(
    """
    <style>
      .block-container {max-width: 1120px; padding-top: 2rem; padding-bottom: 2rem;}
      .topline {color:#54708a; font-size:.86rem; letter-spacing:.08em; text-transform:uppercase; font-weight:700;}
      .hero-title {font-size:2.1rem; font-weight:750; color:#122b45; margin:.15rem 0 .25rem;}
      .subtitle {font-size:1rem; color:#5c6f82; margin:0 0 1.2rem;}
      .panel-title {font-weight:700; color:#183b56; font-size:1.15rem; margin-bottom:.3rem;}
      .analysis-box {background:#f7fafc; border:1px solid #dce6ee; border-radius:14px;
        padding:1rem 1.15rem; min-height:170px;}
      .step-done {color:#256b53; padding:.28rem 0;}
      .step-active {color:#1e5aa8; font-weight:700; padding:.28rem 0;}
      .result-chip {display:inline-block; padding:.32rem .72rem; border-radius:999px;
        background:#e8f4ee; color:#236344; font-weight:700; font-size:.9rem;}
      .muted {color:#687b8d; font-size:.88rem;}
      div[data-testid="stTextArea"] textarea {border-radius:10px;}
      div.stButton > button {border-radius:10px; min-height:2.8rem; font-weight:700;}
    </style>
    <div class="topline">Research demonstration</div>
    <div class="hero-title">DA-RoBERTa Media Bias Analysis</div>
    <div class="subtitle">Analyze a target sentence with the surrounding article context.</div>
    """,
    unsafe_allow_html=True,
)


def split_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return []
    return [part.strip() for part in re.split(r'(?<=[.!?])\s+(?=["“‘(]*[A-Z0-9])', cleaned) if part.strip()]


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", (text or "").casefold()).split())


def locate_target(target: str, sentences: list[str]) -> int | None:
    wanted = normalize(target)
    if not wanted:
        return None
    normalized = [normalize(sentence) for sentence in sentences]
    for index, candidate in enumerate(normalized):
        if candidate == wanted or wanted in candidate or candidate in wanted:
            return index
    if not normalized:
        return None
    similarities = [SequenceMatcher(None, wanted, candidate).ratio() for candidate in normalized]
    best_index = max(range(len(similarities)), key=similarities.__getitem__)
    return best_index if similarities[best_index] >= 0.82 else None


def simulated_prediction(target: str, context: str) -> tuple[str, int, int]:
    """Small illustrative scoring rule; this does not load or run DA-RoBERTa."""
    cues = {
        "reckless", "shocking", "disastrous", "corrupt", "heroic", "radical", "outrageous",
        "devastated", "catastrophic", "failure", "brave", "extreme", "so-called", "biased",
    }
    tokens = set(re.findall(r"[a-z][a-z'-]+", f"{target} {context}".casefold()))
    hits = len(tokens & cues)
    target_hits = len(set(re.findall(r"[a-z][a-z'-]+", target.casefold())) & cues)
    biased_score = min(91, 54 + 11 * target_hits + 4 * max(0, hits - target_hits))
    if hits == 0:
        biased_score = 23 + (len(target.split()) % 4) * 3
    label = "Biased" if biased_score >= 50 else "Non-biased"
    return label, biased_score, 100 - biased_score


EXAMPLES = {
    "Biased example": {
        "target": "The mayor’s reckless decision devastated the community, critics said.",
        "article": (
            "Residents gathered outside city hall on Tuesday to discuss the new budget. "
            "The mayor’s reckless decision devastated the community, critics said. "
            "The mayor’s office said the changes were intended to reduce costs. "
            "City council members are expected to review the plan next month."
        ),
    },
    "Non-biased example": {
        "target": "The city council approved the revised transportation budget on Tuesday.",
        "article": (
            "City council members met on Tuesday to review the proposed budget. "
            "The city council approved the revised transportation budget on Tuesday. "
            "The plan allocates funds for road maintenance and public transit. "
            "Officials said implementation will begin next month."
        ),
    },
}


def load_example() -> None:
    example_name = st.session_state.get("example_choice", "Custom text")
    if example_name in EXAMPLES:
        st.session_state["target_sentence"] = EXAMPLES[example_name]["target"]
        st.session_state["article_text"] = EXAMPLES[example_name]["article"]


st.selectbox(
    "Load a seeded example (optional)",
    ["Custom text", *EXAMPLES.keys()],
    key="example_choice",
    on_change=load_example,
)

with st.container(border=True):
    st.markdown('<div class="panel-title">News text</div>', unsafe_allow_html=True)
    left, right = st.columns([0.9, 1.1], gap="large")
    with left:
        target_sentence = st.text_area(
            "Target sentence",
            height=145,
            placeholder="Paste the specific news sentence you want to analyze…",
            key="target_sentence",
        )
        st.caption("The sentence the enhanced model will assess.")
    with right:
        article_text = st.text_area(
            "Full news article",
            height=145,
            placeholder="Paste the full article so nearby sentences can provide context…",
            key="article_text",
        )
        st.caption("Used to find the target and retrieve neighboring sentences.")

    analyze = st.button("Analyze", type="primary", use_container_width=True,
                        disabled=not (target_sentence.strip() and article_text.strip()))

st.write("")
st.markdown('<div class="panel-title">Live analysis</div>', unsafe_allow_html=True)
analysis_panel = st.container(border=True)

if analyze:
    article_sentences = split_sentences(article_text)
    target_index = locate_target(target_sentence, article_sentences)
    previous = article_sentences[max(0, target_index - 4):target_index] if target_index is not None else []
    following = article_sentences[target_index + 1:target_index + 5] if target_index is not None else []
    context_text = " ".join(previous + following)

    steps = [
        f"Reading the article and splitting it into {len(article_sentences)} ordered sentences",
        "Matching the target sentence to its location in the article",
        "Selecting up to four preceding and four following sentences",
        "Applying the target–context gated fusion pathway",
        "Passing the fused representation to the two-class classifier",
    ]
    with analysis_panel:
        progress = st.progress(0, text="Starting analysis…")
        live_steps = st.empty()
        for step_index, step in enumerate(steps):
            live_steps.markdown(
                "<div class='analysis-box'>" + "".join(
                    f"<div class='{'step-done' if i < step_index else 'step-active' if i == step_index else 'muted'}'>"
                    f"{'✓' if i < step_index else '◉' if i == step_index else '○'} &nbsp; {item}</div>"
                    for i, item in enumerate(steps)
                ) + "</div>",
                unsafe_allow_html=True,
            )
            progress.progress((step_index + 1) / len(steps), text=step)
            time.sleep(0.62)

        if target_index is None:
            st.warning("I couldn't confidently match that target sentence in the article. The demo will still show a result using the sentence and the article text provided.")
            context_text = article_text
        else:
            st.success(f"Target located in article sentence {target_index + 1} of {len(article_sentences)}.")

        label, biased_score, nonbiased_score = simulated_prediction(target_sentence, context_text)
        st.divider()
        result_col, score_col = st.columns([1, 1.25])
        with result_col:
            st.markdown("**Enhanced model result**")
            st.markdown(f"<span class='result-chip'>{label}</span>", unsafe_allow_html=True)
            st.caption("Illustrative prototype output")
        with score_col:
            st.markdown("**Example class scores**")
            st.write(f"Biased · {biased_score}%")
            st.progress(biased_score / 100)
            st.write(f"Non-biased · {nonbiased_score}%")
            st.progress(nonbiased_score / 100)

        with st.expander("See the context selected for this analysis", expanded=False):
            st.markdown("**Target sentence**")
            st.write(target_sentence)
            st.markdown("**Nearby sentences provided to context fusion**")
            if previous:
                st.markdown("*Before the target*\n\n" + "\n\n".join(previous))
            if following:
                st.markdown("*After the target*\n\n" + "\n\n".join(following))
            if target_index is None:
                st.write("Target not matched; article text is used as illustrative context.")

elif not target_sentence.strip() or not article_text.strip():
    with analysis_panel:
        st.info("Enter the target sentence and article, then select **Analyze** to view the processing steps and example result.")
else:
    with analysis_panel:
        st.info("Ready to analyze. Select **Analyze** to begin.")
