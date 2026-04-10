# Causal Information Flow System -- Streamlit Community Cloud version
# Reads data from Google Drive via service account (read-only)
# No emojis, no non-ASCII characters anywhere in this file

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import json
import pickle
import io
import os
from pathlib import Path

# Google Drive API
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ============================================================
# Configuration
# ============================================================

CLASS_COLOR = {
    "equity":       "#2563EB",
    "sector":       "#7C3AED",
    "commodity":    "#D97706",
    "fx":           "#059669",
    "fixed_income": "#DC2626",
    "volatility":   "#9333EA",
    "unknown":      "#6B7280",
}
NAME_COLOR = {
    "bull":    "#059669",
    "bear":    "#D97706",
    "crisis":  "#DC2626",
    "neutral": "#6B7280",
}
STRENGTH_COLOR = {
    "Strong":   "#059669",
    "Moderate": "#D97706",
    "Weak":     "#9CA3AF",
}

st.set_page_config(
    page_title="Causal Flow System",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Google Drive reader via service account
# ============================================================

@st.cache_resource
def get_drive_service():
    """
    Build a Google Drive API service using the service account credentials
    stored in Streamlit secrets. The service account only has read access
    to the causal_flow data folder -- no access to the rest of Drive.
    """
    creds_dict = dict(st.secrets["google_service_account"])
    creds = service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    return build("drive", "v3", credentials=creds)


@st.cache_data(ttl=3600)
def list_folder(folder_id):
    """List all files in a Drive folder by ID."""
    svc = get_drive_service()
    results = (
        svc.files()
        .list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="files(id, name, mimeType)",
            pageSize=200,
        )
        .execute()
    )
    return {f["name"]: f["id"] for f in results.get("files", [])}


@st.cache_data(ttl=3600)
def download_file(file_id):
    """Download a file from Drive by ID, return as bytes."""
    svc = get_drive_service()
    request = svc.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    buf.seek(0)
    return buf


def load_parquet(file_id):
    return pd.read_parquet(download_file(file_id))


def load_json(file_id):
    return json.loads(download_file(file_id).read().decode("utf-8"))


def load_pickle(file_id):
    return pickle.loads(download_file(file_id).read())


@st.cache_data(ttl=3600)
def load_all_data():
    """
    Load all required data files from Google Drive.
    Folder IDs come from Streamlit secrets.
    """
    folder_ids = {
        "processed": st.secrets["drive_folders"]["processed"],
        "phase3":    st.secrets["drive_folders"]["phase3"],
        "phase4":    st.secrets["drive_folders"]["phase4"],
        "phase5":    st.secrets["drive_folders"]["phase5"],
        "phase6":    st.secrets["drive_folders"]["phase6"],
    }

    files = {k: list_folder(v) for k, v in folder_ids.items()}

    def get(folder, name):
        fid = files[folder].get(name)
        if fid is None:
            raise FileNotFoundError(f"{name} not found in {folder} folder")
        return fid

    # Load all files
    meta  = load_json(   get("processed", "phase1_meta.json"))
    summ  = load_json(   get("phase6",    "update_summary.json"))
    crit  = load_json(   get("phase6",    "regime_criteria.json"))
    Gf    = load_pickle( get("phase4",    "final_graph.pkl"))
    rg    = load_pickle( get("phase3",    "regime_graphs.pkl"))
    cent  = load_parquet(get("phase4",    "final_centrality.parquet"))
    rcent = load_parquet(get("phase6",    "regime_centrality.parquet"))
    ret   = load_parquet(get("phase6",    "latest_returns_daily.parquet"))
    reg   = load_parquet(get("phase6",    "latest_regime.parquet"))["regime"]
    rpb   = load_parquet(get("phase6",    "latest_regime_probs.parquet"))
    sig   = load_parquet(get("phase6",    "current_signal.parquet"))["weight_pct"]
    etf   = load_parquet(get("phase6",    "edge_table_final.parquet"))
    etc   = load_parquet(get("phase6",    "edge_table_current.parquet"))

    perf  = (load_parquet(get("phase5", "performance_table.parquet"))
             if "performance_table.parquet" in files["phase5"]
             else pd.DataFrame())
    strat = (load_parquet(get("phase5", "strategy_returns.parquet"))
             if "strategy_returns.parquet" in files["phase5"]
             else pd.DataFrame())

    return (meta, summ, crit, Gf, rg, cent, rcent,
            ret, reg, rpb, sig, etf, etc, perf, strat)


# ============================================================
# Load data
# ============================================================

try:
    (meta, summ, crit, G_final, rg, cent, rcent,
     ret, reg, rpb, sig, etf, etc, perf, strat) = load_all_data()
    AC     = meta.get("asset_classes", {})
    ASSETS = meta.get("assets", list(ret.columns))
    CUREG  = summ["current_regime"]
    GCUR   = rg.get(CUREG, G_final)
except Exception as e:
    st.error("Could not load data from Google Drive.")
    st.code(str(e))
    st.info(
        "Check that:\n"
        "1. Your service account credentials are in .streamlit/secrets.toml\n"
        "2. The service account has been granted access to the causal_flow folder\n"
        "3. The Phase 6 weekly refresh has been run at least once"
    )
    st.stop()


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("### Causal Flow System")
    st.caption("Updated: " + summ.get("updated_at", "")[:10])
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Home",
            "Causal Graph",
            "Regime Analysis",
            "Portfolio Signal",
            "Performance",
            "Asset Explorer",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Asset class colours**")
    for cls, col in CLASS_COLOR.items():
        st.markdown(
            f'<span style="display:inline-block;width:11px;height:11px;'
            f'border-radius:2px;background:{col};margin-right:6px;'
            f'vertical-align:middle"></span>'
            f'<span style="font-size:13px">{cls}</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("**Edge strength tiers**")
    st.markdown(
        '<span style="color:#059669;font-weight:600">Strong</span>'
        ' -- top 25% of TE values<br>'
        '<span style="color:#D97706;font-weight:600">Moderate</span>'
        ' -- middle 50%<br>'
        '<span style="color:#9CA3AF;font-weight:600">Weak</span>'
        ' -- bottom 25%',
        unsafe_allow_html=True,
    )


# ============================================================
# HOME
# ============================================================

if page == "Home":
    st.title("Causal Information Flow System")
    st.caption(
        "Tracking "
        + str(len(ASSETS))
        + " global assets across equities, commodities, FX and fixed income"
        + "  |  Data as of "
        + str(summ.get("as_of_date", ""))
        + "  |  Current regime: "
        + CUREG.upper()
    )
    st.markdown("---")

    with st.expander(
        "What is this system and how does it work? (click to read)",
        expanded=True,
    ):
        st.markdown(
            """
### The problem with correlation

Most finance tools measure **correlation** between assets. Correlation tells you two
assets move together, but it is symmetric -- it cannot tell you who moves first.
SPY->GLD looks the same as GLD->SPY. It also cannot distinguish a genuine causal
link from two assets both reacting to the same piece of news.

This system asks a different question:

> "Does knowing what Oil did last week reduce my uncertainty about what Energy
> stocks will do this week -- beyond what Energy stocks' own history already tells me?"

If yes, and if the relationship survives rigorous statistical testing, then Oil is
a **causal information leader** of Energy stocks.

---

### What Transfer Entropy measures

Transfer Entropy (TE) from A to B is measured in **bits** -- the unit of information
in information theory. It quantifies how much predictive information A's past contains
about B's future, above what B already knows about itself.

| TE value | What it means |
|---|---|
| 0 bits | A tells you nothing new about B. No causal link. |
| 0.001 to 0.01 bits | Weak but real signal. A slightly reduces uncertainty about B. |
| 0.01 to 0.05 bits | Moderate signal. A is a meaningful leader of B. |
| 0.05+ bits | Strong signal. A strongly leads B. |

Small numbers are normal in efficient markets. What matters is the **ranking**
(who scores highest relative to others) and the **direction** (who leads, who follows).

---

### The four validation tests every edge must pass

1. **Surrogate significance (p < 0.05)** -- the signal is not random noise.
   The source series is scrambled 200 times; real TE must exceed scrambled TE.

2. **Conditional TE** -- the relationship survives after controlling for macro factors
   (VIX, yield curve slope, USD index, credit spreads). This removes spurious links
   where both assets were simply reacting to the same macro event.

3. **False discovery rate correction** -- with 400+ pairs tested, some false positives
   are expected by chance. Benjamini-Hochberg correction controls this.

4. **Temporal stability** -- the relationship must hold in at least 4 of 5 historical
   sub-periods. A genuine structural link should be persistent across time.

---

### Why market regimes matter

The causal structure of markets changes dramatically across conditions.
A Hidden Markov Model (HMM) detects which regime is currently active.
The causal graph shown is specific to the current regime.

| Regime | Characteristics | What it means for leaders/followers |
|---|---|---|
| Bull | Positive returns, low volatility, low correlation | Sector fundamentals drive assets; leadership is dispersed |
| Bear | Negative returns, elevated volatility, rising correlation | Macro fear starts to synchronise assets |
| Crisis | Sharp losses, very high volatility, very high correlation | Risk sentiment overwhelms all fundamentals simultaneously |

---

### How to navigate

- **Causal Graph** -- who leads whom, with strength and lag quantified
- **Regime Analysis** -- what the current regime is, what defines it empirically
- **Portfolio Signal** -- which assets to overweight based on the causal graph
- **Performance** -- out-of-sample backtest results
- **Asset Explorer** -- drill into any single asset's causal neighbourhood
"""
        )

    st.markdown("### Current system status")
    c1, c2, c3, c4, c5 = st.columns(5)
    conf = summ.get("regime_confidence", {})
    mx   = max(conf.values()) if conf else 0
    c1.metric("Regime", CUREG.upper())
    c2.metric(
        "Regime confidence",
        f"{mx * 100:.0f}%",
        help="HMM posterior probability for the current regime label",
    )
    c3.metric("Assets tracked", len(ASSETS))
    c4.metric(
        "Validated edges (full)",
        summ.get("n_edges_final", 0),
        help="Causal edges surviving all four validation filters",
    )
    c5.metric(
        "Active edges (regime)",
        summ.get("n_edges_regime", 0),
        help="Edges active within the current market regime specifically",
    )

    st.markdown("---")
    st.markdown(
        "### Information leaders and followers -- " + CUREG.upper() + " regime"
    )
    st.caption(
        "Net flow = out-strength minus in-strength, in Transfer Entropy bits. "
        "Positive = net information sender (leader). "
        "Negative = net information receiver (follower)."
    )

    col_l, col_f = st.columns(2)
    with col_l:
        st.markdown("**Leaders** -- their past predicts others' futures")
        for a in summ.get("top_leaders", []):
            nf  = float(rcent["net_flow_regime"].get(a, 0))
            cls = AC.get(a, "unknown")
            fol = list(GCUR.successors(a)) if a in GCUR else []
            st.markdown(
                f'<div style="margin:5px 0;padding:10px 14px;border-radius:7px;'
                f'background:#F0FDF4;border-left:3px solid #059669">'
                f'<b style="font-size:14px;color:#065F46">{a}</b>'
                f'<span style="font-size:11px;color:#6B7280;margin-left:6px">[{cls}]</span><br>'
                f'<span style="font-size:12px;color:#047857">'
                f'Net outflow: <b>+{nf:.5f} bits</b></span><br>'
                f'<span style="font-size:11px;color:#6B7280">'
                f'Leads: {", ".join(fol[:4]) if fol else "none in current regime"}'
                f'</span></div>',
                unsafe_allow_html=True,
            )

    with col_f:
        st.markdown("**Followers** -- they react to what leaders do")
        for a in summ.get("top_followers", []):
            nf  = float(rcent["net_flow_regime"].get(a, 0))
            cls = AC.get(a, "unknown")
            led = list(GCUR.predecessors(a)) if a in GCUR else []
            st.markdown(
                f'<div style="margin:5px 0;padding:10px 14px;border-radius:7px;'
                f'background:#FFF7ED;border-left:3px solid #D97706">'
                f'<b style="font-size:14px;color:#92400E">{a}</b>'
                f'<span style="font-size:11px;color:#6B7280;margin-left:6px">[{cls}]</span><br>'
                f'<span style="font-size:12px;color:#B45309">'
                f'Net inflow: <b>{nf:.5f} bits</b></span><br>'
                f'<span style="font-size:11px;color:#6B7280">'
                f'Led by: {", ".join(led[:4]) if led else "none in current regime"}'
                f'</span></div>',
                unsafe_allow_html=True,
            )

    st.info(
        "On the bit values: a net flow of 0.01 bits means this asset's past movements "
        "explain roughly 1% of the remaining uncertainty in its followers' next-week "
        "returns, above what those followers already know from their own history. "
        "Focus on ranking and direction rather than the absolute size."
    )


# ============================================================
# CAUSAL GRAPH
# ============================================================

elif page == "Causal Graph":
    st.title("Directed Causal Information Graph")

    with st.expander("How to read this graph", expanded=False):
        st.markdown(
            """
**Nodes:** Each circle is an asset. Larger size means higher out-strength -- the
asset sends more information to others. Node colour indicates asset class.

**Arrows:** An arrow from A to B means A's past price movements contain validated
predictive information about B's future price movements.

**Arrow thickness:** Proportional to net Transfer Entropy = TE(A->B) minus TE(B->A).
Thicker means A dominates B more strongly in the causal direction.

**Arrow colour (strength tier):**
- Green: Strong -- top 25% of TE values in this graph
- Orange: Moderate -- middle 50%
- Grey: Weak -- bottom 25%

**What is not shown:** Edges that failed any of the four validation tests.
Absent edges do not mean no correlation -- they mean no validated directional
causality survived the testing pipeline.

**Lag:** How many weeks back the leader's signal is predictive.
Lag 1 means last week's A predicts this week's B.
"""
        )

    g_choice = st.radio(
        "Graph to display:",
        [
            "Current regime only (" + CUREG.upper() + ")",
            "Full sample (all regimes combined)",
        ],
        horizontal=True,
    )
    G_show = GCUR if "Current" in g_choice else G_final
    et_show = etc  if "Current" in g_choice else etf

    col_graph, col_table = st.columns([3, 2])

    with col_graph:
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.set_facecolor("#F9FAFB")

        if G_show.number_of_edges() == 0:
            ax.text(
                0.5, 0.5,
                "No validated edges found.\nRun Phases 2 to 4 first.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="#6B7280",
            )
        else:
            pos    = nx.spring_layout(G_show, k=2.8, seed=42, weight="weight")
            ow     = dict(G_show.out_degree(weight="weight"))
            max_ow = max(ow.values(), default=1)
            te_vals = [d.get("weight", 0) for _, _, d in G_show.edges(data=True)]
            p25 = np.percentile(te_vals, 25)
            p75 = np.percentile(te_vals, 75)
            sizes = [120 + 1600 * (ow.get(n, 0) / (max_ow + 1e-9)) for n in G_show.nodes()]
            ncolors = [CLASS_COLOR.get(AC.get(n, "unknown"), "#888") for n in G_show.nodes()]

            nx.draw_networkx_nodes(G_show, pos, ax=ax, node_size=sizes,
                                   node_color=ncolors, alpha=0.92)
            nx.draw_networkx_labels(G_show, pos, ax=ax, font_size=7, font_weight="bold")

            for src, tgt, d in G_show.edges(data=True):
                te  = d.get("weight", 0)
                nte = d.get("net_te", te)
                col = ("#059669" if te >= p75 else "#D97706" if te >= p25 else "#9CA3AF")
                nx.draw_networkx_edges(
                    G_show, pos, edgelist=[(src, tgt)], ax=ax,
                    width=max(nte * 18, 0.5), edge_color=col, alpha=0.75,
                    arrows=True, arrowsize=13,
                    connectionstyle="arc3,rad=0.12",
                )

            patches = [
                mpatches.Patch(color=CLASS_COLOR.get(c, "#888"), label=c)
                for c in set(AC.get(a, "unknown") for a in G_show.nodes())
            ]
            patches += [
                mpatches.Patch(color="#059669", label="Strong edge (top 25%)"),
                mpatches.Patch(color="#D97706", label="Moderate edge"),
                mpatches.Patch(color="#9CA3AF", label="Weak edge (bottom 25%)"),
            ]
            ax.legend(handles=patches, loc="lower left", fontsize=7, ncol=2, framealpha=0.9)

        lbl = CUREG.upper() + " regime" if "Current" in g_choice else "Full sample"
        ax.set_title(
            f"Causal graph -- {lbl}  |  {G_show.number_of_edges()} edges\n"
            "Node size = out-strength  |  Thickness = net TE  |  Colour = strength tier",
            fontsize=9, color="#374151",
        )
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_table:
        st.markdown("**All validated edges ranked by Transfer Entropy**")
        st.caption(
            "TE (bits): Transfer Entropy from source to target.  "
            "Net TE: TE(A->B) minus TE(B->A).  "
            "Lag: weeks back.  Stability: significant in how many of 5 windows."
        )
        if not et_show.empty:
            show_df = et_show[
                ["source", "target", "te_bits", "net_te_bits",
                 "lag_weeks", "stability", "strength"]
            ].head(30).rename(columns={
                "te_bits":     "TE (bits)",
                "net_te_bits": "Net TE",
                "lag_weeks":   "Lag (wks)",
                "stability":   "Stability /5",
                "strength":    "Strength",
            })

            def color_strength(val):
                return {
                    "Strong":   "background-color:#D1FAE5",
                    "Moderate": "background-color:#FEF3C7",
                    "Weak":     "background-color:#F3F4F6",
                }.get(val, "")

            st.dataframe(
                show_df.style.applymap(color_strength, subset=["Strength"]),
                use_container_width=True, height=500,
            )
        else:
            st.info("No edge data found. Run Phases 2 to 4 first.")


# ============================================================
# REGIME ANALYSIS
# ============================================================

elif page == "Regime Analysis":
    st.title("Market Regime Analysis")

    with st.expander("How regimes are detected and what defines them", expanded=False):
        st.markdown(
            """
A **Gaussian Hidden Markov Model (HMM)** is fitted on five observable market signals:
average cross-asset return, cross-sectional dispersion, realised volatility,
average pairwise correlation, and equity-bond return spread.

The HMM learns from historical data what each regime typically looks like across
these signals. It then assigns a posterior probability to each regime for every day.

**Regime labels are assigned by volatility, not by return.** The regime with the
highest annualised volatility is labelled "crisis", the lowest is "bull", and the
middle is "bear". This is more robust than sorting by return, because short-lived
recovery rallies inside a high-volatility period can produce a misleadingly high
average return for the crisis regime if it is sorted by mean.

The criteria table below shows the empirically observed average values of each
signal during all days assigned to each regime. These are not hand-coded rules.

**On the average return in crisis:** If the crisis regime shows a high positive
return, this almost certainly indicates a regime labelling issue. The crisis
regime should have the most negative or near-zero average return combined with
very high volatility. If yours does not, rerun Phase 3 with the corrected
labelling code provided in the regime_label_fix file.
"""
        )

    rc = NAME_COLOR.get(CUREG, "#6B7280")
    st.markdown(
        f'<div style="padding:14px 18px;border-radius:9px;border-left:5px solid {rc};'
        f'background:{rc}18;margin-bottom:12px">'
        f'<span style="font-size:18px;font-weight:700;color:{rc}">'
        f'Current regime: {CUREG.upper()}</span><br>'
        f'<span style="font-size:12px;color:#374151">'
        f'As of {summ.get("as_of_date", "")}'
        f'</span></div>',
        unsafe_allow_html=True,
    )

    conf = summ.get("regime_confidence", {})
    if conf:
        st.markdown("**Model confidence -- posterior probabilities**")
        st.caption(
            "The HMM gives a probability for each regime simultaneously. "
            "They sum to 100%. The highest probability determines the assigned label."
        )
        cols_conf = st.columns(len(conf))
        for i, (k, p) in enumerate(conf.items()):
            rname_clean = k.replace("p_", "")
            rc2 = NAME_COLOR.get(rname_clean, "#6B7280")
            is_max = p == max(conf.values())
            cols_conf[i].markdown(
                f'<div style="text-align:center;padding:12px;border-radius:8px;'
                f'border:{"2px" if is_max else "1px"} solid {rc2};'
                f'background:{rc2}{"25" if is_max else "10"}">'
                f'<div style="font-size:22px;font-weight:700;color:{rc2}">'
                f'{p * 100:.1f}%</div>'
                f'<div style="font-size:13px;color:#374151">{rname_clean.upper()}</div>'
                f'{"<div style=margin-top:4px><b style=font-size:11px;color:" + rc2 + ">active</b></div>" if is_max else ""}'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown(
        "### Regime criteria -- empirically derived from historical data"
    )
    st.caption(
        "Average values of observable signals across all days assigned to each regime. "
        "Learned from data, not hand-coded. "
        "Crisis should have the highest volatility and most negative return. "
        "If crisis shows a high positive return, the Phase 3 labelling needs the fix. "
        "Highlighted row = currently active regime."
    )

    if crit:
        crit_df = pd.DataFrame(crit).T.rename(
            columns={
                "avg_ann_return_pct":  "Avg ann. return (%)",
                "avg_ann_vol_pct":     "Avg ann. vol (%)",
                "avg_pairwise_corr":   "Avg corr.",
                "avg_eq_bond_spread":  "Eq-bond spread (%/wk)",
                "n_days":              "Days in history",
                "pct_of_sample":       "% of sample",
                "avg_duration_days":   "Avg duration (days)",
            }
        )

        def highlight_current(df):
            styles = pd.DataFrame("", index=df.index, columns=df.columns)
            for rn in df.index:
                if rn == CUREG:
                    styles.loc[rn] = (
                        "background-color:"
                        + NAME_COLOR.get(rn, "#6B7280")
                        + "22;font-weight:600"
                    )
            return styles

        st.dataframe(
            crit_df.style.apply(highlight_current, axis=None),
            use_container_width=True,
        )

        st.markdown(
            """
**Interpreting the table:**

- **Bull:** High positive return, low volatility, low cross-asset correlation.
  Assets move on their own fundamentals rather than in lockstep.
- **Bear:** Negative or flat return, elevated volatility, rising correlation.
  Macro concerns start to dominate individual fundamentals.
- **Crisis:** Sharply negative return, very high volatility, very high correlation.
  Risk sentiment overwhelms all asset classes simultaneously. Bonds rally as
  equities and commodities fall, which is why the equity-bond spread is deeply
  negative in this regime.

**If crisis shows a high positive return in your table:** this is a bug, not
a real finding. Short V-shaped recoveries inside a high-volatility period can
inflate the average return if the regime window captures the rebound but not
the decline. Apply the regime_label_fix patch to Phase 3 and rerun.
"""
        )

    st.markdown("---")
    st.markdown("### Regime history")

    fig2, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    ax = axes[0]
    for rn, rc4 in NAME_COLOR.items():
        mask = reg == rn
        ax.fill_between(reg.index, 0, 1, where=mask, color=rc4, alpha=0.7, label=rn)
    ax.set_yticks([])
    ax.set_ylabel("Regime", fontsize=10)
    ax.set_title("Regime classification over time", fontsize=11)
    ax.legend(loc="upper right", fontsize=8, ncol=4)

    ax2 = axes[1]
    spy_col = next((c for c in ret.columns if "SP500" in c), ret.columns[0])
    cum = (1 + ret[spy_col]).cumprod()
    ax2.plot(cum.index, cum, color="black", linewidth=0.9)
    for rn, rc5 in NAME_COLOR.items():
        mask = (reg == rn).reindex(cum.index, method="ffill").fillna(False)
        ax2.fill_between(cum.index, cum.min() * 0.95, cum,
                         where=mask, color=rc5, alpha=0.15)
    ax2.set_ylabel("Cumulative return", fontsize=10)
    ax2.set_title(spy_col + " cumulative return shaded by regime", fontsize=11)
    ax2.set_yscale("log")
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()

    st.caption(
        "Crisis periods (red) should visually align with major drawdowns such as "
        "March 2020 and Q4 2022. If they do not, apply the regime_label_fix patch."
    )


# ============================================================
# PORTFOLIO SIGNAL
# ============================================================

elif page == "Portfolio Signal":
    st.title("Current Portfolio Signal")

    with st.expander("How this signal is constructed", expanded=False):
        st.markdown(
            """
Each week we identify the current regime from the HMM posterior probabilities.
We load the regime-specific causal graph and set portfolio weights proportional
to each asset's **out-strength** in that regime's graph.

Assets that send more causal information to others receive higher portfolio weight.
The idea is to align the portfolio with the current information generators rather
than the reactive assets that trail behind them.

Constraints: maximum 20% per asset. All weights sum to 100%.

This is a structural tilt signal, not a market timing signal. It works best as
an overlay on an existing base allocation, not as a standalone strategy.
"""
        )

    st.markdown(
        "Active regime: **" + CUREG.upper() + "**  |  "
        + str(GCUR.number_of_edges())
        + " validated edges in regime graph"
    )

    col_s1, col_s2 = st.columns([1, 1])

    with col_s1:
        st.markdown("**Suggested weights by asset**")
        st.caption(
            "Weight % is out-strength in the current regime graph, normalised to 100%."
        )
        if not sig.empty:
            sdf = sig.sort_values(ascending=False).to_frame()
            sdf["Asset class"] = sdf.index.map(lambda a: AC.get(a, "unknown"))
            sdf["Out-strength (bits)"] = sdf.index.map(
                lambda a: round(
                    float(GCUR.out_degree(a, weight="weight")) if a in GCUR else 0.0, 5
                )
            )
            nf_map = rcent["net_flow_regime"].to_dict()
            sdf["Role"] = sdf.index.map(
                lambda a: (
                    "Leader"   if nf_map.get(a, 0) > 0.001 else
                    "Follower" if nf_map.get(a, 0) < -0.001 else
                    "Neutral"
                )
            )
            sdf = sdf.rename(columns={"weight_pct": "Weight (%)"})

            def color_role(val):
                return {
                    "Leader":   "background-color:#D1FAE5",
                    "Follower": "background-color:#FEF3C7",
                    "Neutral":  "background-color:#F3F4F6",
                }.get(val, "")

            st.dataframe(
                sdf[["Weight (%)", "Out-strength (bits)", "Asset class", "Role"]]
                .style.applymap(color_role, subset=["Role"]),
                use_container_width=True, height=460,
            )
        else:
            st.info("Run the Phase 6 weekly refresh to generate signal data.")

    with col_s2:
        st.markdown("**Weight by asset class**")
        if not sig.empty:
            class_wts = {}
            for a, w in sig.items():
                cls = AC.get(a, "unknown")
                class_wts[cls] = class_wts.get(cls, 0) + float(w)
            fig3, ax3 = plt.subplots(figsize=(6, 5))
            ax3.pie(
                list(class_wts.values()),
                labels=list(class_wts.keys()),
                colors=[CLASS_COLOR.get(c, "#888") for c in class_wts],
                autopct="%1.1f%%", startangle=90,
                textprops={"fontsize": 10},
            )
            ax3.set_title("Weight by class -- " + CUREG + " regime", fontsize=11)
            st.pyplot(fig3, use_container_width=True)
            plt.close()

        st.markdown("---")
        st.markdown("**Net information flow by asset**")
        st.caption("Positive = leader (overweighted). Negative = follower.")
        if "net_flow_regime" in rcent.columns:
            nf_s = rcent["net_flow_regime"].sort_values(ascending=False)
            fig4, ax4 = plt.subplots(figsize=(6, 5))
            colors_bar = ["#059669" if v >= 0 else "#DC2626" for v in nf_s.values]
            ax4.barh(nf_s.index, nf_s.values, color=colors_bar, alpha=0.8)
            ax4.axvline(0, color="black", linewidth=0.8)
            ax4.set_xlabel("Net flow (bits)")
            ax4.set_title("Information role -- current regime", fontsize=11)
            ax4.tick_params(labelsize=8)
            plt.tight_layout()
            st.pyplot(fig4, use_container_width=True)
            plt.close()


# ============================================================
# PERFORMANCE
# ============================================================

elif page == "Performance":
    st.title("Backtest Performance")
    st.caption(
        "All results are out-of-sample. "
        "The causal graph was estimated on training data only."
    )

    with st.expander("How to interpret these results", expanded=False):
        st.markdown(
            """
**Walk-forward design:** The causal graph and HMM are estimated on data up to the
training cutoff. All numbers shown are from the test period -- data the model
never saw. This is the only honest evaluation.

**Signal 1 -- Lead-lag momentum:** When a top leader asset moves more than 1.5%
in a week, go long its documented followers for 2 weeks.

**Signal 2 -- Regime rotation:** Each week, overweight assets with high out-strength
in the current regime's causal graph. Rebalanced weekly.

**Sharpe ratio above 0.5 after costs** is considered meaningful for a systematic
strategy. Above 1.0 is strong.
"""
        )

    if not perf.empty:
        st.markdown("### Summary -- out-of-sample test period")

        def highlight_perf(df):
            styles = pd.DataFrame("", index=df.index, columns=df.columns)
            for col in ["sharpe", "ann_return_%", "sortino"]:
                if col in df.columns:
                    mx = df[col].max()
                    styles.loc[df[col] == mx, col] = (
                        "background-color:#D1FAE5;font-weight:600"
                    )
            for col in ["max_drawdown_%", "ann_vol_%"]:
                if col in df.columns:
                    mn = df[col].min()
                    styles.loc[df[col] == mn, col] = (
                        "background-color:#D1FAE5;font-weight:600"
                    )
            return styles

        st.dataframe(
            perf.style.apply(highlight_perf, axis=None),
            use_container_width=True,
        )
        st.caption("Green highlights = best value in each column.")

    if not strat.empty:
        st.markdown("### Cumulative returns -- all strategies")
        STYLE_MAP = {
            "Signal1_LeadLag":      ("#2563EB", 2.2, "-"),
            "Signal2_RegimeRotate": ("#7C3AED", 2.2, "-"),
            "Equal_Weight":         ("#6B7280", 1.2, "--"),
        }
        fig5, ax5 = plt.subplots(figsize=(14, 5))
        for col in strat.columns:
            cum   = (1 + strat[col]).cumprod()
            color, lw, ls = STYLE_MAP.get(col, ("#D97706", 1.2, "--"))
            ax5.plot(cum.index, cum, label=col, linewidth=lw, linestyle=ls, color=color)
        ax5.set_ylabel("Growth of $1 invested")
        ax5.set_title("Out-of-sample cumulative returns vs benchmarks")
        ax5.legend(fontsize=9)
        ax5.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
        st.pyplot(fig5, use_container_width=True)
        plt.close()

        st.markdown("### Drawdowns")
        fig6, ax6 = plt.subplots(figsize=(14, 4))
        for col in strat.columns:
            cum  = (1 + strat[col]).cumprod()
            peak = cum.cummax()
            dd   = (cum - peak) / peak
            color = STYLE_MAP.get(col, ("#D97706", 1.2, "--"))[0]
            ax6.fill_between(dd.index, dd, 0, alpha=0.2, color=color)
            ax6.plot(dd.index, dd, linewidth=0.8, color=color, label=col)
        ax6.set_ylabel("Drawdown")
        ax6.set_title("Drawdown comparison -- less negative is better")
        ax6.legend(fontsize=8)
        st.pyplot(fig6, use_container_width=True)
        plt.close()
    else:
        st.info("Run Phase 5 to generate backtest results.")


# ============================================================
# ASSET EXPLORER
# ============================================================

elif page == "Asset Explorer":
    st.title("Asset Deep-Dive")

    sel = st.selectbox("Select an asset", sorted(ret.columns.tolist()))
    if not sel:
        st.stop()

    cls_sel = AC.get(sel, "unknown")
    st.markdown("### " + sel + "  --  " + cls_sel)

    nf_val  = float(rcent["net_flow_regime"].get(sel, 0))
    out_val = float(rcent["out_strength_regime"].get(sel, 0))
    in_val  = float(rcent["in_strength_regime"].get(sel, 0))

    if nf_val > 0.001:
        role_html = (
            '<span style="color:#059669;font-weight:700">Information Leader</span>'
            " -- sends more information than it receives in the current regime"
        )
    elif nf_val < -0.001:
        role_html = (
            '<span style="color:#D97706;font-weight:700">Information Follower</span>'
            " -- receives more information than it sends in the current regime"
        )
    else:
        role_html = (
            '<span style="color:#6B7280;font-weight:700">Neutral / Isolated</span>'
            " -- minimal directional causal flow detected in current regime"
        )

    st.markdown("Role: " + role_html, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Out-strength (bits sent)", f"{out_val:.5f}",
              help="Total TE bits sent to all followers in current regime")
    m2.metric("In-strength (bits received)", f"{in_val:.5f}",
              help="Total TE bits received from all leaders in current regime")
    m3.metric("Net flow (bits)", f"{nf_val:+.5f}",
              help="Out minus In. Positive = net leader.")

    st.markdown("---")

    all_te_cur = [d.get("weight", 0) for _, _, d in GCUR.edges(data=True)]
    p25_cur = np.percentile(all_te_cur, 25) if all_te_cur else 0
    p75_cur = np.percentile(all_te_cur, 75) if all_te_cur else 1

    col_out, col_in = st.columns(2)

    with col_out:
        st.markdown("**Assets that " + sel + " leads (->)**")
        st.caption("Their future returns are partially predicted by " + sel + "'s past.")
        out_n = list(GCUR.successors(sel)) if sel in GCUR else []
        if out_n:
            for follower in out_n:
                d    = GCUR.edges.get((sel, follower), {})
                te   = d.get("weight", 0)
                nte  = d.get("net_te", te)
                lag  = d.get("lag", "?")
                stab = d.get("stability", "?")
                slab = ("Strong" if te >= p75_cur else "Moderate" if te >= p25_cur else "Weak")
                sc   = STRENGTH_COLOR.get(slab, "#888")
                st.markdown(
                    f'<div style="margin:4px 0;padding:9px 12px;border-radius:6px;'
                    f'background:#F0FDF4;border-left:3px solid {sc}">'
                    f'<b>{sel} -> {follower}</b><br>'
                    f'<span style="font-size:12px;color:#374151">'
                    f'TE: <b>{te:.5f} bits</b>  |  '
                    f'Net TE: <b>{nte:.5f} bits</b>  |  '
                    f'Lag: <b>{lag} wks</b>  |  '
                    f'Stability: <b>{stab}/5</b>  |  '
                    f'<span style="color:{sc}">{slab}</span>'
                    f'</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info(sel + " has no validated followers in the " + CUREG + " regime.")

    with col_in:
        st.markdown("**Assets that lead " + sel + " (<-)**")
        st.caption("Their past returns have validated predictive power over " + sel + "'s future.")
        in_n = list(GCUR.predecessors(sel)) if sel in GCUR else []
        if in_n:
            for leader in in_n:
                d    = GCUR.edges.get((leader, sel), {})
                te   = d.get("weight", 0)
                nte  = d.get("net_te", te)
                lag  = d.get("lag", "?")
                stab = d.get("stability", "?")
                slab = ("Strong" if te >= p75_cur else "Moderate" if te >= p25_cur else "Weak")
                sc   = STRENGTH_COLOR.get(slab, "#888")
                st.markdown(
                    f'<div style="margin:4px 0;padding:9px 12px;border-radius:6px;'
                    f'background:#FFF7ED;border-left:3px solid {sc}">'
                    f'<b>{leader} -> {sel}</b><br>'
                    f'<span style="font-size:12px;color:#374151">'
                    f'TE: <b>{te:.5f} bits</b>  |  '
                    f'Net TE: <b>{nte:.5f} bits</b>  |  '
                    f'Lag: <b>{lag} wks</b>  |  '
                    f'Stability: <b>{stab}/5</b>  |  '
                    f'<span style="color:{sc}">{slab}</span>'
                    f'</span></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No validated leaders found for " + sel + " in the " + CUREG + " regime.")

    st.markdown("---")
    st.markdown("**" + sel + " -- recent price performance**")
    if sel in ret.columns:
        fig7, ax7 = plt.subplots(figsize=(12, 3))
        cum7   = (1 + ret[sel]).cumprod()
        color7 = CLASS_COLOR.get(AC.get(sel, "unknown"), "#2563EB")
        ax7.plot(cum7.index, cum7, color=color7, linewidth=1.2)
        ax7.fill_between(cum7.index, 1, cum7, where=cum7 >= 1, alpha=0.15, color=color7)
        ax7.fill_between(cum7.index, 1, cum7, where=cum7 < 1, alpha=0.15, color="#DC2626")
        ax7.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
        ax7.set_ylabel("Cumulative return")
        ax7.set_title(sel + " -- recent 2 years")
        plt.tight_layout()
        st.pyplot(fig7, use_container_width=True)
        plt.close()
