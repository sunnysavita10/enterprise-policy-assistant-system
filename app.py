from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import settings
from src.evaluation import run_evaluation, run_red_team
from src.rag_pipeline import EnterprisePolicyAssistant
from src.vector_store import rebuild_vector_store


st.set_page_config(
    page_title="Sentinel | Enterprise Policy AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _load_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
        :root { --ink:#10233f; --muted:#65758b; --line:#dce5ef; --canvas:#f4f7fb; }
        html, body, [class*="css"] { font-family:"DM Sans",sans-serif; }
        .stApp {
            background:radial-gradient(circle at 82% -8%,rgba(37,99,235,.12),transparent 26rem),
                       radial-gradient(circle at 4% 40%,rgba(6,182,212,.07),transparent 24rem),var(--canvas);
            color:var(--ink);
        }
        h1,h2,h3,h4 { font-family:"Manrope",sans-serif!important; color:var(--ink); letter-spacing:-.025em; }
        [data-testid="stSidebar"] {
            background:linear-gradient(180deg,#081a32 0%,#0c2646 58%,#0d3154 100%);
            border-right:1px solid rgba(255,255,255,.08);
        }
        [data-testid="stSidebar"] * { color:#e8f0fa; }
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color:#cbd8e8; }
        [data-testid="stSidebar"] hr { border-color:rgba(255,255,255,.12); }
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
            background:rgba(255,255,255,.06); border:1px dashed rgba(147,197,253,.45); border-radius:14px;
        }
        .block-container { max-width:1480px; padding-top:1.7rem; padding-bottom:3rem; }
        .brand { display:flex; align-items:center; gap:.75rem; margin:.1rem 0 1.5rem; }
        .brand-mark {
            display:grid; place-items:center; width:42px; height:42px; border-radius:12px;
            background:linear-gradient(135deg,#3b82f6,#06b6d4); box-shadow:0 10px 24px rgba(6,182,212,.22); font-size:1.25rem;
        }
        .brand-name { font:800 1.1rem/1.1 "Manrope",sans-serif; color:white; }
        .brand-tag { color:#9fb2ca; font-size:.72rem; margin-top:.22rem; letter-spacing:.08em; text-transform:uppercase; }
        .hero {
            position:relative; overflow:hidden; min-height:205px; padding:2.1rem 2.35rem; border-radius:24px;
            background:linear-gradient(120deg,#0a1e38 0%,#123b65 62%,#075b73 100%);
            box-shadow:0 22px 55px rgba(15,35,60,.17); color:white; margin-bottom:1.3rem;
        }
        .hero:after {
            content:""; position:absolute; width:340px; height:340px; border-radius:50%; right:-80px; top:-180px;
            border:54px solid rgba(255,255,255,.06);
        }
        .eyebrow {
            display:inline-flex; align-items:center; gap:.45rem; padding:.38rem .7rem;
            border:1px solid rgba(125,211,252,.32); border-radius:999px; background:rgba(14,165,233,.12);
            color:#bdeeff; font-size:.72rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase;
        }
        .pulse-dot { width:7px; height:7px; border-radius:50%; background:#34d399; box-shadow:0 0 0 5px rgba(52,211,153,.12); }
        .hero h1 { color:white!important; margin:.9rem 0 .45rem; font-size:clamp(2rem,4vw,3.25rem); line-height:1.05; }
        .hero p { color:#c8d9ea; max-width:760px; font-size:1rem; margin:0; }
        .feature-row { display:flex; flex-wrap:wrap; gap:.55rem; margin-top:1.3rem; }
        .feature-pill {
            padding:.36rem .68rem; border-radius:8px; background:rgba(255,255,255,.08); color:#e5f3ff;
            font-size:.76rem; border:1px solid rgba(255,255,255,.1);
        }
        div[data-testid="stMetric"] {
            background:rgba(255,255,255,.86); border:1px solid var(--line); padding:1rem 1.1rem;
            border-radius:16px; box-shadow:0 6px 22px rgba(23,43,77,.055);
        }
        div[data-testid="stMetric"] label { color:var(--muted); }
        div[data-testid="stMetricValue"] { color:var(--ink); font-family:"Manrope",sans-serif; }
        .section-heading { margin:.4rem 0 .15rem; font:800 1.3rem/1.3 "Manrope",sans-serif; color:var(--ink); }
        .section-copy { color:var(--muted); margin-bottom:1rem; font-size:.92rem; }
        .welcome-card {
            padding:1.45rem 1.55rem; border:1px solid var(--line); border-radius:18px;
            background:rgba(255,255,255,.8); margin:.45rem 0 1rem;
        }
        .welcome-card h3 { margin:0 0 .35rem; font-size:1.1rem; }
        .welcome-card p { color:var(--muted); margin:0; font-size:.9rem; }
        [data-testid="stChatMessage"] {
            border:1px solid #e0e7f0; background:rgba(255,255,255,.9); border-radius:17px;
            padding:.35rem .55rem; box-shadow:0 5px 18px rgba(23,43,77,.04);
        }
        [data-testid="stChatMessageContent"],
        [data-testid="stChatMessageContent"] p,
        [data-testid="stChatMessageContent"] li,
        [data-testid="stChatMessageContent"] span,
        [data-testid="stChatMessageContent"] strong {
            color:#10233f!important;
        }
        .user-message-row {
            display:flex;
            justify-content:flex-end;
            align-items:flex-end;
            gap:.65rem;
            margin:.85rem 0;
        }
        .user-message-bubble {
            max-width:min(76%,780px);
            padding:.8rem 1rem;
            border-radius:17px 17px 4px 17px;
            background:linear-gradient(135deg,#2563eb,#1d4ed8);
            color:#ffffff!important;
            box-shadow:0 8px 20px rgba(37,99,235,.18);
            line-height:1.55;
            overflow-wrap:anywhere;
        }
        .user-message-avatar {
            display:grid;
            place-items:center;
            flex:0 0 34px;
            width:34px;
            height:34px;
            border-radius:50%;
            background:#dbeafe;
            border:1px solid #bfdbfe;
            font-size:1rem;
        }
        [data-testid="stChatInput"] {
            border-radius:15px;
            border:1px solid #b8c7da;
            background:#ffffff;
            box-shadow:0 8px 26px rgba(23,43,77,.08);
        }
        [data-testid="stChatInput"] textarea,
        [data-testid="stChatInput"] textarea:focus {
            color:#10233f!important;
            background:#ffffff!important;
            caret-color:#2563eb!important;
            -webkit-text-fill-color:#10233f!important;
        }
        [data-testid="stChatInput"] div[data-baseweb="textarea"],
        [data-testid="stChatInput"] div[data-baseweb="base-input"] {
            background-color:#ffffff!important;
        }
        [data-testid="stChatInput"] textarea::placeholder {
            color:#718096!important;
            opacity:1!important;
            -webkit-text-fill-color:#718096!important;
        }
        [data-testid="stChatInput"] button {
            color:#ffffff!important;
            background:#2563eb!important;
        }
        [data-testid="stChatInput"]:focus-within {
            border-color:#3b82f6;
            box-shadow:0 0 0 3px rgba(59,130,246,.16),0 8px 26px rgba(23,43,77,.08);
        }
        /* Keep all standard text fields readable when the browser uses dark mode. */
        [data-baseweb="input"] input,
        [data-baseweb="textarea"] textarea {
            color:#10233f!important;
            background-color:#ffffff!important;
            caret-color:#2563eb!important;
            -webkit-text-fill-color:#10233f!important;
        }
        .stButton>button,.stDownloadButton>button { border-radius:11px; font-weight:700; transition:transform .15s ease,box-shadow .15s ease; }
        .stButton>button:hover,.stDownloadButton>button:hover { transform:translateY(-1px); box-shadow:0 7px 18px rgba(37,99,235,.13); }
        button[data-baseweb="tab"] { font-weight:700; padding:.75rem 1rem; }
        div[data-baseweb="tab-list"] {
            gap:.35rem; background:rgba(255,255,255,.72); padding:.35rem; border:1px solid var(--line); border-radius:14px;
        }
        .source-card {
            padding:.9rem 1rem; margin:.55rem 0; border:1px solid #dbe6f1; border-left:4px solid #3b82f6;
            border-radius:12px; background:#f8fbff;
        }
        .source-title { font-weight:700; color:var(--ink); margin-bottom:.3rem; }
        .source-excerpt { color:var(--muted); font-size:.84rem; line-height:1.55; }
        .flow-grid { display:grid; grid-template-columns:repeat(5,1fr); gap:.75rem; margin:1rem 0; }
        .flow-step {
            min-height:120px; padding:1rem; border-radius:15px; border:1px solid var(--line);
            background:white; box-shadow:0 6px 20px rgba(23,43,77,.045);
        }
        .flow-number { color:#2563eb; font:800 .72rem "Manrope",sans-serif; letter-spacing:.08em; }
        .flow-title { color:var(--ink); font-weight:800; margin:.55rem 0 .25rem; }
        .flow-copy { color:var(--muted); font-size:.78rem; line-height:1.45; }
        .sidebar-status {
            display:flex; align-items:center; gap:.55rem; padding:.7rem .8rem; margin-bottom:1rem; border-radius:11px;
            background:rgba(52,211,153,.1); border:1px solid rgba(52,211,153,.18); color:#baf7db!important;
            font-size:.82rem; font-weight:600;
        }
        @media(max-width:900px) { .flow-grid{grid-template-columns:repeat(2,1fr)} .hero{padding:1.6rem} }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _document_count() -> int:
    extensions = {".pdf", ".docx", ".txt", ".md"}
    directories = [settings.policy_dir, settings.upload_dir]
    return sum(
        1
        for directory in directories
        if directory.exists()
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def _render_sources(sources: list) -> None:
    if not sources:
        return
    with st.expander(f"Evidence used · {len(sources)} sources", expanded=True):
        for source in sources:
            details = []
            if source.page:
                details.append(f"Page {source.page}")
            if source.section:
                details.append(escape(source.section))
            suffix = f" · {' · '.join(details)}" if details else ""
            st.markdown(
                f'<div class="source-card"><div class="source-title">[{escape(source.id)}] '
                f'{escape(source.source)}{suffix}</div><div class="source-excerpt">'
                f'{escape(source.excerpt)}</div></div>',
                unsafe_allow_html=True,
            )


def _safe_mean(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else 0.0


def _render_user_message(content: str) -> None:
    safe_content = escape(content).replace("\n", "<br>")
    st.markdown(
        f'<div class="user-message-row"><div class="user-message-bubble">{safe_content}</div>'
        '<div class="user-message-avatar">🧑‍💼</div></div>',
        unsafe_allow_html=True,
    )


_load_styles()
if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.markdown(
        """
        <div class="brand"><div class="brand-mark">🛡️</div><div><div class="brand-name">Sentinel</div>
        <div class="brand-tag">Policy Intelligence</div></div></div>
        <div class="sidebar-status"><span class="pulse-dot"></span> Guardrails active</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Knowledge base")
    st.caption("Upload approved company policies, then rebuild the searchable index.")
    uploads = st.file_uploader(
        "Policy documents", type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True, label_visibility="collapsed",
    )
    if uploads and st.button("Save uploaded files", use_container_width=True):
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        for uploaded in uploads:
            destination = settings.upload_dir / Path(uploaded.name).name
            destination.write_bytes(uploaded.getbuffer())
        st.success(f"Saved {len(uploads)} file(s). Rebuild the index to activate them.")

    if st.button("↻  Rebuild policy index", type="primary", use_container_width=True):
        try:
            with st.spinner("Parsing and indexing policies…"):
                _, count = rebuild_vector_store()
            st.success(f"Index ready with {count:,} chunks")
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")

    st.divider()
    st.markdown("### Runtime")
    st.caption(f"MODEL  ·  {settings.openai_model}")
    st.caption(f"EMBEDDINGS  ·  {settings.embedding_model}")
    st.caption(f"RETRIEVAL  ·  TOP {settings.top_k}")
    guardrail_col, ground_col = st.columns(2)
    guardrail_col.metric("Injection", "ON" if settings.enable_llm_injection_check else "OFF")
    ground_col.metric("Grounding", "ON" if settings.enable_groundedness_check else "OFF")
    st.divider()
    if st.button("Clear conversation", use_container_width=True, disabled=not st.session_state.messages):
        st.session_state.messages = []
        st.rerun()


st.markdown(
    """
    <section class="hero">
        <div class="eyebrow"><span class="pulse-dot"></span> Secure enterprise AI workspace</div>
        <h1>Policy answers you can trust.</h1>
        <p>Ask workplace questions and get grounded answers backed by your approved policy library—with built-in injection defense, PII protection, and source-level traceability.</p>
        <div class="feature-row"><span class="feature-pill">✓ Evidence grounded</span>
        <span class="feature-pill">✓ Prompt-injection defense</span><span class="feature-pill">✓ PII redaction</span>
        <span class="feature-pill">✓ Continuous evaluation</span></div>
    </section>
    """,
    unsafe_allow_html=True,
)

overview_cols = st.columns(4)
overview_cols[0].metric("Policy documents", f"{_document_count():,}", help="Supported policy files currently available")
overview_cols[1].metric("Retrieval depth", f"Top {settings.top_k}", help="Maximum evidence chunks retrieved per question")
threshold = getattr(settings, "groundedness_threshold", 0.7)
overview_cols[2].metric("Grounding threshold", f"{threshold:.0%}")
overview_cols[3].metric("Security posture", "Protected" if settings.enable_llm_injection_check else "Review")

st.write("")
chat_tab, eval_tab, red_tab, architecture_tab = st.tabs(
    ["💬  Policy Assistant", "📊  Evaluation Lab", "🎯  Red Team", "◈  Architecture"]
)


with chat_tab:
    st.markdown('<div class="section-heading">Policy copilot</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">Get concise, cited guidance from your organization’s approved documents.</div>', unsafe_allow_html=True)
    suggested_question = None
    if not st.session_state.messages:
        st.markdown(
            '<div class="welcome-card"><h3>What can I help you find?</h3>'
            '<p>Choose a common question below or write your own. Every material claim is checked against retrieved policy evidence.</p></div>',
            unsafe_allow_html=True,
        )
        prompt_cols = st.columns(3)
        suggestions = [
            "What is the travel reimbursement policy?",
            "How should employees handle sensitive data?",
            "What are the rules for using AI tools at work?",
        ]
        for index, suggestion in enumerate(suggestions):
            if prompt_cols[index].button(suggestion, key=f"suggestion_{index}", use_container_width=True):
                suggested_question = suggestion

    for item in st.session_state.messages:
        if item["role"] == "user":
            _render_user_message(item["content"])
        else:
            with st.chat_message("assistant", avatar="🛡️"):
                st.markdown(item["content"])

    typed_question = st.chat_input("Ask about travel, HR, security, compliance, or AI-use policy…")
    question = typed_question or suggested_question
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        _render_user_message(question)
        with st.chat_message("assistant", avatar="🛡️"):
            try:
                with st.status("Running secure policy analysis…", expanded=True) as status:
                    st.write("Scanning the request for prompt injection and sensitive data")
                    st.write("Retrieving the most relevant policy evidence")
                    result = EnterprisePolicyAssistant().ask(question)
                    st.write("Validating answer groundedness and citations")
                    status.update(label="Analysis complete", state="complete", expanded=False)
                st.markdown(result.answer)
                signal_cols = st.columns(4)
                signal_cols[0].metric("Request", "Blocked" if result.blocked else "Allowed")
                signal_cols[1].metric("Injection risk", result.injection_risk.title())
                signal_cols[2].metric("PII handling", "Redacted" if result.pii_redacted else "Clear")
                score = "N/A" if result.groundedness_score is None else f"{result.groundedness_score:.0%}"
                signal_cols[3].metric("Groundedness", score)
                _render_sources(result.sources)
                if result.block_reason:
                    with st.expander("Why this request was blocked"):
                        st.warning(result.block_reason)
                assistant_text = result.answer
            except Exception as exc:
                assistant_text = f"Error: {exc}\n\nIf this is your first run, add `OPENAI_API_KEY` to `.env` and rebuild the policy index."
                st.error(assistant_text)
        st.session_state.messages.append({"role": "assistant", "content": assistant_text})


with eval_tab:
    st.markdown('<div class="section-heading">Quality evaluation lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">Benchmark retrieval and answer quality against the curated golden dataset.</div>', unsafe_allow_html=True)
    control_col, info_col = st.columns([1, 2])
    with control_col:
        limit = st.slider("Evaluation cases", 1, 10, 5)
        run_eval = st.button("Run quality evaluation", type="primary", use_container_width=True)
    with info_col:
        st.info("Measures source retrieval, groundedness, answer relevance, and reference-answer correctness.")
    if run_eval:
        try:
            with st.status("Evaluating policy assistant…", expanded=True) as status:
                st.write(f"Running {limit} golden-dataset cases")
                df = run_evaluation(limit=limit)
                status.update(label="Evaluation complete", state="complete", expanded=False)
            summary_cols = st.columns(4)
            summary_cols[0].metric("Retrieval hit", f"{_safe_mean(df, 'retrieval_hit'):.0%}")
            summary_cols[1].metric("Groundedness", f"{_safe_mean(df, 'groundedness'):.0%}")
            summary_cols[2].metric("Relevance", f"{_safe_mean(df, 'answer_relevance'):.0%}")
            summary_cols[3].metric("Correctness", f"{_safe_mean(df, 'correctness'):.0%}")
            chart_data = pd.DataFrame(
                {"Score": [_safe_mean(df, "retrieval_hit"), _safe_mean(df, "groundedness"),
                           _safe_mean(df, "answer_relevance"), _safe_mean(df, "correctness")]},
                index=["Retrieval", "Grounding", "Relevance", "Correctness"],
            )
            chart_col, table_col = st.columns([1, 2])
            with chart_col:
                st.markdown("#### Score profile")
                st.bar_chart(chart_data, horizontal=True, height=300)
            with table_col:
                st.markdown("#### Case-level results")
                st.dataframe(
                    df, use_container_width=True, hide_index=True,
                    column_config={
                        "retrieval_hit": st.column_config.CheckboxColumn("Retrieved"),
                        "groundedness": st.column_config.ProgressColumn("Grounding", min_value=0, max_value=1),
                        "answer_relevance": st.column_config.ProgressColumn("Relevance", min_value=0, max_value=1),
                        "correctness": st.column_config.ProgressColumn("Correctness", min_value=0, max_value=1),
                    },
                )
            st.download_button("Download evaluation CSV", data=df.to_csv(index=False).encode("utf-8"),
                               file_name="policy-assistant-evaluation.csv", mime="text/csv")
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")


with red_tab:
    st.markdown('<div class="section-heading">Adversarial resilience</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">Probe injection defense, PII handling, and unsupported-policy behavior.</div>', unsafe_allow_html=True)
    intro_col, action_col = st.columns([2, 1])
    with intro_col:
        st.warning("Red-team prompts intentionally attempt to bypass safeguards. Run after changes to prompts, models, or retrieval.")
    with action_col:
        run_red = st.button("Launch red-team suite", type="primary", use_container_width=True)
    if run_red:
        try:
            with st.status("Running adversarial probes…", expanded=True) as status:
                st.write("Testing injection, privacy, and unsupported claims")
                df = run_red_team()
                status.update(label="Red-team run complete", state="complete", expanded=False)
            passed = int(df["passed"].astype(bool).sum())
            metric_cols = st.columns(3)
            metric_cols[0].metric("Pass rate", f"{_safe_mean(df, 'passed'):.0%}")
            metric_cols[1].metric("Defenses passed", passed)
            metric_cols[2].metric("Needs review", len(df) - passed)
            category_summary = (df.assign(passed=df["passed"].astype(float))
                                .groupby("category", as_index=False)["passed"].mean()
                                .rename(columns={"passed": "pass_rate"}))
            chart_col, table_col = st.columns([1, 2])
            with chart_col:
                st.markdown("#### Defense coverage")
                st.bar_chart(category_summary.set_index("category"), height=300)
            with table_col:
                st.markdown("#### Probe results")
                st.dataframe(
                    df, use_container_width=True, hide_index=True,
                    column_config={
                        "passed": st.column_config.CheckboxColumn("Passed"),
                        "blocked": st.column_config.CheckboxColumn("Blocked"),
                        "pii_redacted": st.column_config.CheckboxColumn("PII redacted"),
                    },
                )
        except Exception as exc:
            st.error(f"Red-team suite failed: {exc}")


with architecture_tab:
    st.markdown('<div class="section-heading">Trust-by-design architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">Every request moves through a layered retrieval and safety pipeline before an answer reaches the user.</div>', unsafe_allow_html=True)
    steps = [
        ("01", "Policy library", "Approved PDF, DOCX, TXT, and Markdown documents."),
        ("02", "Secure intake", "Parsing, chunking, metadata capture, and embeddings."),
        ("03", "Input defense", "Prompt-injection detection and PII redaction."),
        ("04", "Grounded RAG", "Semantic retrieval and evidence-constrained generation."),
        ("05", "Output assurance", "Groundedness judging, redaction, and citations."),
    ]
    cards = "".join(
        f'<div class="flow-step"><div class="flow-number">STEP {number}</div>'
        f'<div class="flow-title">{title}</div><div class="flow-copy">{copy}</div></div>'
        for number, title, copy in steps
    )
    st.markdown(f'<div class="flow-grid">{cards}</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown("#### Controls included")
        st.markdown("- Untrusted-context isolation\n- Prompt-injection screening\n- Input and output PII redaction\n- Source-level answer citations\n- LLM groundedness assessment")
    with right:
        st.markdown("#### Production hardening")
        st.markdown("- Enterprise identity and authorization\n- Document-level ACL filtering\n- Centralized secrets management\n- Immutable audit logging\n- Rate limits and provider controls")
    st.info("Sentinel is a teaching/reference implementation. Add the production-hardening controls before using it with sensitive enterprise data.")
