# Causal Information Flow System -- Streamlit Community Cloud version
# Fixed: better error messages, recursive folder search, subfolder handling

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import json
import pickle
import io
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

CLASS_COLOR = {
    "equity": "#2563EB", "sector": "#7C3AED", "commodity": "#D97706",
    "fx": "#059669", "fixed_income": "#DC2626", "volatility": "#9333EA",
    "unknown": "#6B7280",
}
NAME_COLOR = {
    "bull": "#059669", "bear": "#D97706",
    "crisis": "#DC2626", "neutral": "#6B7280",
}
STRENGTH_COLOR = {
    "Strong": "#059669", "Moderate": "#D97706", "Weak": "#9CA3AF",
}

st.set_page_config(
    page_title="Causal Flow System",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------
# Google Drive helpers
# -----------------------------------------------------------------------

@st.cache_resource
def get_drive_service():
    creds = service_account.Credentials.from_service_account_info(
        dict(st.secrets["google_service_account"]),
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


@st.cache_data(ttl=3600)
def list_folder(folder_id):
    """Return dict of {filename: file_id} for all files in a folder."""
    svc = get_drive_service()
    all_files = {}
    page_token = None
    while True:
        resp = svc.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id, name, mimeType)",
            pageSize=200,
            pageToken=page_token,
        ).execute()
        for f in resp.get("files", []):
            all_files[f["name"]] = f["id"]
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return all_files


@st.cache_data(ttl=3600)
def download_bytes(file_id):
    svc = get_drive_service()
    request = svc.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = dl.next_chunk()
    buf.seek(0)
    return buf.read()


def load_parquet(file_id):
    return pd.read_parquet(io.BytesIO(download_bytes(file_id)))


def load_json_file(file_id):
    return json.loads(download_bytes(file_id).decode("utf-8"))


def load_pickle_file(file_id):
    return pickle.loads(download_bytes(file_id))


def get_file_id(folder_files, filename, folder_name):
    """Get file ID or raise a clear error."""
    fid = folder_files.get(filename)
    if fid is None:
        available = sorted(folder_files.keys())
        raise FileNotFoundError(
            f"'{filename}' not found in the '{folder_name}' folder.\n\n"
            f"Files actually found in that folder ({len(available)} total):\n"
            + "\n".join(f"  - {f}" for f in available[:20])
            + ("\n  ..." if len(available) > 20 else "")
            + "\n\nCheck that:\n"
            "1. The folder ID in your secrets is correct\n"
            "2. The Phase 1-6 notebooks have been run\n"
            "3. The service account has Viewer access to THIS specific subfolder"
        )
    return fid


@st.cache_data(ttl=3600)
def load_all_data():
    folder_ids = {
        "processed": st.secrets["drive_folders"]["processed"],
        "phase3":    st.secrets["drive_folders"]["phase3"],
        "phase4":    st.secrets["drive_folders"]["phase4"],
        "phase5":    st.secrets["drive_folders"]["phase5"],
        "phase6":    st.secrets["drive_folders"]["phase6"],
    }

    # List contents of all folders upfront
    folder_contents = {}
    for name, fid in folder_ids.items():
        try:
            folder_contents[name] = list_folder(fid)
        except Exception as e:
            raise RuntimeError(
                f"Cannot list folder '{name}' (ID: {fid}).\n"
                f"Error: {e}\n\n"
                "Check:\n"
                "1. The folder ID is correct (open in Drive, check the URL)\n"
                "2. The service account has been given Viewer access to this folder\n"
                "3. The Google Drive API is enabled in your GCP project"
            )

    def get(folder, name):
        return get_file_id(folder_contents[folder], name, folder)

    meta  = load_json_file(  get("processed", "phase1_meta.json"))
    summ  = load_json_file(  get("phase6",    "update_summary.json"))
    crit  = load_json_file(  get("phase6",    "regime_criteria.json"))
    Gf    = load_pickle_file(get("phase4",    "final_graph.pkl"))
    rg    = load_pickle_file(get("phase3",    "regime_graphs.pkl"))
    cent  = load_parquet(    get("phase4",    "final_centrality.parquet"))
    rcent = load_parquet(    get("phase6",    "regime_centrality.parquet"))
    ret   = load_parquet(    get("phase6",    "latest_returns_daily.parquet"))
    reg   = load_parquet(    get("phase6",    "latest_regime.parquet"))["regime"]
    rpb   = load_parquet(    get("phase6",    "latest_regime_probs.parquet"))
    sig   = load_parquet(    get("phase6",    "current_signal.parquet"))["weight_pct"]
    etf   = load_parquet(    get("phase6",    "edge_table_final.parquet"))
    etc   = load_parquet(    get("phase6",    "edge_table_current.parquet"))

    perf  = (load_parquet(get("phase5", "performance_table.parquet"))
             if "performance_table.parquet" in folder_contents["phase5"]
             else pd.DataFrame())
    strat = (load_parquet(get("phase5", "strategy_returns.parquet"))
             if "strategy_returns.parquet" in folder_contents["phase5"]
             else pd.DataFrame())

    return (meta, summ, crit, Gf, rg, cent, rcent,
            ret, reg, rpb, sig, etf, etc, perf, strat)


# -----------------------------------------------------------------------
# Load data with helpful error display
# -----------------------------------------------------------------------

try:
    (meta, summ, crit, G_final, rg, cent, rcent,
     ret, reg, rpb, sig, etf, etc, perf, strat) = load_all_data()
    AC     = meta.get("asset_classes", {})
    ASSETS = meta.get("assets", list(ret.columns))
    CUREG  = summ["current_regime"]
    GCUR   = rg.get(CUREG, G_final)

except FileNotFoundError as e:
    st.error("A required data file was not found on Google Drive.")
    st.code(str(e), language="text")
    with st.expander("How to fix this"):
        st.markdown("""
**Most common causes and fixes:**

1. **Wrong folder ID in secrets.toml**
   - Open each folder in Google Drive in your browser
   - The URL looks like: `drive.google.com/drive/folders/FOLDER_ID_HERE`
   - Copy the ID and paste it into the corresponding field in secrets.toml

2. **File not yet created**
   - The file is only created after you run the corresponding Phase notebook
   - `phase1_meta.json` requires Phase 1 to have completed
   - Files in `phase6/` require the Phase 6 weekly refresh to have run

3. **Service account does not have access to the subfolder**
   - Sharing the parent `causal_flow` folder is NOT enough
   - You must share EACH of the five subfolders individually with the service account
   - Right-click the subfolder in Drive > Share > add the service account email > Viewer

4. **How to confirm what files are in each folder**
   - Paste and run the `diagnostic.py` cell in a new Colab cell
   - It will print exactly what files exist in each folder path on your Drive
        """)
    st.stop()

except RuntimeError as e:
    st.error("Could not connect to Google Drive.")
    st.code(str(e), language="text")
    with st.expander("How to fix this"):
        st.markdown("""
**Check these in order:**

1. **Google Drive API enabled?**
   - Go to console.cloud.google.com > APIs & Services > Enabled APIs
   - Search for "Google Drive API" -- it must be enabled

2. **Correct folder ID?**
   - Open the folder in Drive, check the URL contains the ID you put in secrets

3. **Service account has access?**
   - Go to IAM console > Service Accounts > find causal-flow-reader
   - Make sure a key exists and is not deleted
        """)
    st.stop()

except Exception as e:
    st.error("Unexpected error loading data.")
    st.code(str(e), language="text")
    st.info("Check the Streamlit Cloud logs for the full traceback.")
    st.stop()


# -----------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Causal Flow System")
    st.caption("Updated: " + summ.get("updated_at", "")[:10])
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Home", "Causal Graph", "Regime Analysis",
         "Portfolio Signal", "Performance", "Asset Explorer"],
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


# -----------------------------------------------------------------------
# HOME
# -----------------------------------------------------------------------

if page == "Home":
    st.title("Causal Information Flow System")
    st.caption(
        "Tracking " + str(len(ASSETS))
        + " global assets  |  Data as of " + str(summ.get("as_of_date", ""))
        + "  |  Current regime: " + CUREG.upper()
        
    )
    st.markdown("---")

    with st.expander("Brief overview about the model", expanded=True):
        st.markdown("""
### Rethinking how assets influence each other

Most finance tools measure **correlation** between assets. Correlation tells you two
assets move together, but it is symmetric, it cannot tell you who moves first.
It also cannot distinguish a genuine causal link from two assets both reacting to
the same piece of news.

This system asks a different question:

> "Does knowing what Oil did last week reduce my uncertainty about what Energy
> stocks will do this week; beyond what Energy stocks' own history already tells me?"

If yes, and if the relationship survives rigorous testing, then Oil is a
**causal information leader** of Energy stocks.

---
### what the model accomplishes
                    
**Measures directional influence (not just co-movement)**
Identifies which assets lead and which ones follow, capturing the flow of information across markets.
**Quantifies predictive information gain**
Uses information-theoretic methods to test whether one asset meaningfully improves forecasts of another.
**Builds a dynamic causal network**
Represents markets as a graph where nodes are assets and edges capture statistically validated influence.
**Filters for robustness**
Only retains relationships that survive significance testing, reducing noise and spurious signals.
**Tracks evolving market structure**
Updates relationships over time to capture regime shifts and changing leadership dynamics.
**Generates actionable signals**
Highlights potential leading indicators that can be used for portfolio construction and risk management.
                    
---

### What Transfer Entropy measures

Transfer Entropy (TE) is measured in **bits**. It quantifies how much predictive
information A's past contains about B's future, above what B already knows about itself.

| TE value | Meaning |
|---|---|
| 0 bits | A tells you nothing new about B. No causal link. |
| 0.001 to 0.01 bits | Weak but real. A slightly reduces uncertainty about B. |
| 0.01 to 0.05 bits | Moderate. A is a meaningful leader of B. |
| 0.05+ bits | Strong. A strongly leads B. |

Small numbers are normal in efficient markets. We focus on ranking and direction.

---

### Four validation tests every edge must pass

1. **Surrogate significance (p < 0.05)** -- the signal is not random noise
2. **Conditional TE** -- survives after controlling for VIX, yield curve, USD index
3. **FDR correction** -- controls false discoveries across 400+ pairs tested
4. **Temporal stability** -- holds in at least 4 of 5 historical sub-periods

---

### Why market regimes matter

A Hidden Markov Model detects which regime is currently active.
The causal graph is estimated separately for each regime.

| Regime | Typical characteristics |
|---|---|
| Bull | Positive returns, low volatility, low correlation |
| Bear | Negative returns, elevated volatility, rising correlation |
| Crisis | Sharp losses, very high volatility, very high correlation, risk-off across all assets |


        """)

    st.markdown("### Current system status")
    c1, c2, c3, c4, c5 = st.columns(5)
    conf = summ.get("regime_confidence", {})
    mx   = max(conf.values()) if conf else 0
    c1.metric("Regime", CUREG.upper())
    c2.metric("Regime confidence", f"{mx * 100:.0f}%",
              help="HMM posterior probability for the current regime label")
    c3.metric("Assets tracked", len(ASSETS))
    c4.metric("Validated edges (full)", summ.get("n_edges_final", 0),
              help="Edges surviving all four validation filters")
    c5.metric("Active edges (regime)", summ.get("n_edges_regime", 0),
              help="Edges active in the current market regime specifically")

    st.markdown("---")
    st.markdown("### Information leaders and followers -- " + CUREG.upper() + " regime")
    st.caption(
        "Net flow = out-strength minus in-strength in Transfer Entropy bits. "
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
                f'<span style="font-size:12px;color:#047857">Net outflow: <b>+{nf:.5f} bits</b></span><br>'
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
                f'<span style="font-size:12px;color:#B45309">Net inflow: <b>{nf:.5f} bits</b></span><br>'
                f'<span style="font-size:11px;color:#6B7280">'
                f'Led by: {", ".join(led[:4]) if led else "none in current regime"}'
                f'</span></div>',
                unsafe_allow_html=True,
            )

    st.info(
        "On the bit values: a net flow of 0.01 bits means this asset's past movements "
        "explain roughly 1% of the remaining uncertainty in its followers' next-week returns, "
        "above what those followers already know from their own history. "
        "We focus on ranking and direction rather than the absolute size."
    )


# -----------------------------------------------------------------------
# CAUSAL GRAPH
# -----------------------------------------------------------------------

elif page == "Causal Graph":
    st.title("Directed Causal Information Graph")

    with st.expander("How to read this graph", expanded=False):
        st.markdown("""
**Nodes:** Each circle is an asset. Larger size = higher out-strength (sends more info to others).
Node colour = asset class.

**Arrows:** A -> B means A's past price movements have validated predictive information
about B's future price movements.

**Arrow thickness:** Proportional to net TE = TE(A->B) minus TE(B->A).
Thicker = A dominates B more strongly in the causal direction.

**Arrow colour:**
- Green: Strong -- top 25% of TE values in this graph
- Orange: Moderate -- middle 50%
- Grey: Weak -- bottom 25%

Absent edges = failed one or more of the four validation tests.
Absent does not mean no correlation -- it means no validated directional causality.
        """)

    g_choice = st.radio(
        "Graph to display:",
        ["Current regime only (" + CUREG.upper() + ")", "Full sample (all regimes combined)"],
        horizontal=True,
    )
    G_show = GCUR if "Current" in g_choice else G_final
    et_show = etc  if "Current" in g_choice else etf

    col_graph, col_table = st.columns([3, 2])

    with col_graph:
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.set_facecolor("#F9FAFB")
        if G_show.number_of_edges() == 0:
            ax.text(0.5, 0.5, "No validated edges found.\nRun Phases 2 to 4 first.",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=12, color="#6B7280")
        else:
            pos    = nx.spring_layout(G_show, k=2.8, seed=42, weight="weight")
            ow     = dict(G_show.out_degree(weight="weight"))
            max_ow = max(ow.values(), default=1)
            te_v   = [d.get("weight", 0) for _, _, d in G_show.edges(data=True)]
            p25, p75 = np.percentile(te_v, 25), np.percentile(te_v, 75)
            sizes  = [120 + 1600 * (ow.get(n, 0) / (max_ow + 1e-9)) for n in G_show.nodes()]
            ncolors= [CLASS_COLOR.get(AC.get(n, "unknown"), "#888") for n in G_show.nodes()]
            nx.draw_networkx_nodes(G_show, pos, ax=ax, node_size=sizes,
                                   node_color=ncolors, alpha=0.92)
            nx.draw_networkx_labels(G_show, pos, ax=ax, font_size=7, font_weight="bold")
            for src, tgt, d in G_show.edges(data=True):
                te  = d.get("weight", 0)
                nte = d.get("net_te", te)
                col = "#059669" if te >= p75 else "#D97706" if te >= p25 else "#9CA3AF"
                nx.draw_networkx_edges(G_show, pos, edgelist=[(src, tgt)], ax=ax,
                    width=max(nte * 18, 0.5), edge_color=col, alpha=0.75,
                    arrows=True, arrowsize=13, connectionstyle="arc3,rad=0.12")
            patches = ([mpatches.Patch(color=CLASS_COLOR.get(c, "#888"), label=c)
                        for c in set(AC.get(a, "unknown") for a in G_show.nodes())]
                       + [mpatches.Patch(color="#059669", label="Strong (top 25%)"),
                          mpatches.Patch(color="#D97706", label="Moderate"),
                          mpatches.Patch(color="#9CA3AF", label="Weak (bottom 25%)")])
            ax.legend(handles=patches, loc="lower left", fontsize=7, ncol=2, framealpha=0.9)
        lbl = CUREG.upper() + " regime" if "Current" in g_choice else "Full sample"
        ax.set_title(
            f"Causal graph -- {lbl}  |  {G_show.number_of_edges()} edges\n"
            "Node size = out-strength  |  Thickness = net TE  |  Colour = strength tier",
            fontsize=9, color="#374151")
        ax.axis("off")
        st.pyplot(fig, width="stretch")
        plt.close()

    with col_table:
        st.markdown("**All validated edges ranked by Transfer Entropy**")
        st.caption(
            "TE (bits): Transfer Entropy source -> target.  "
            "Net TE: TE(A->B) minus TE(B->A).  "
            "Lag: weeks. Stability: of 5 windows."
        )
        if not et_show.empty:
            show_df = et_show[["source", "target", "te_bits", "net_te_bits",
                                "lag_weeks", "stability", "strength"]].head(30).rename(columns={
                "te_bits": "TE (bits)", "net_te_bits": "Net TE",
                "lag_weeks": "Lag (wks)", "stability": "Stability /5", "strength": "Strength"})
            def cs(v):
                return {"Strong": "background-color:#D1FAE5",
                        "Moderate": "background-color:#FEF3C7",
                        "Weak": "background-color:#F3F4F6"}.get(v, "")
            st.dataframe(show_df.style.map(cs, subset=["Strength"]),
                         width="stretch", height=500)
        else:
            st.info("No edge data found. Run Phases 2 to 4.")


# -----------------------------------------------------------------------
# REGIME ANALYSIS
# -----------------------------------------------------------------------

elif page == "Regime Analysis":
    st.title("Market Regime Analysis")

    with st.expander("How regimes are detected and what defines them", expanded=False):
        st.markdown("""
A **Gaussian Hidden Markov Model (HMM)** is fitted on five observable signals:
average cross-asset return, cross-sectional dispersion, realised volatility,
average pairwise correlation, and equity-bond return spread.

**Regime labels are assigned by volatility, not return.**
Highest volatility = crisis, lowest = bull, middle = bear.
This is more robust than sorting by return, because short V-shaped recovery rallies
within a high-volatility period can produce misleadingly high average returns
if the crisis regime is defined by return rather than risk level.

The criteria table below shows empirically observed average values of each signal
during all days assigned to each regime. These are learned from data, not hard-coded.


        """)

    rc = NAME_COLOR.get(CUREG, "#6B7280")
    st.markdown(
        f'<div style="padding:14px 18px;border-radius:9px;border-left:5px solid {rc};'
        f'background:{rc}18;margin-bottom:12px">'
        f'<span style="font-size:18px;font-weight:700;color:{rc}">'
        f'Current regime: {CUREG.upper()}</span><br>'
        f'<span style="font-size:12px;color:#374151">As of {summ.get("as_of_date", "")}</span>'
        f'</div>', unsafe_allow_html=True)

    conf = summ.get("regime_confidence", {})
    if conf:
        st.markdown("**Model confidence -- posterior probabilities**")
        st.caption("The HMM assigns a probability to each regime simultaneously. They sum to 100%.")
        cols_c = st.columns(len(conf))
        for i, (k, p) in enumerate(conf.items()):
            nm  = k.replace("p_", "")
            rc2 = NAME_COLOR.get(nm, "#6B7280")
            is_max = p == max(conf.values())
            cols_c[i].markdown(
                f'<div style="text-align:center;padding:12px;border-radius:8px;'
                f'border:{"2px" if is_max else "1px"} solid {rc2};'
                f'background:{rc2}{"25" if is_max else "10"}">'
                f'<div style="font-size:22px;font-weight:700;color:{rc2}">{p*100:.1f}%</div>'
                f'<div style="font-size:13px;color:#374151">{nm.upper()}</div>'
                f'{"<div style=margin-top:4px><b style=font-size:11px;color:" + rc2 + ">active</b></div>" if is_max else ""}'
                f'</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Regime criteria -- empirically derived from historical data")
    st.caption(
        "Average values of observable signals during all days assigned to each regime. "
        "Crisis should have the most negative return and highest volatility. "
        "If crisis shows a high positive return, apply the regime_label_fix patch. "
        "Highlighted row = currently active regime."
    )

    if crit:
        crit_df = pd.DataFrame(crit).T.rename(columns={
            "avg_ann_return_pct":  "Avg ann. return (%)",
            "avg_ann_vol_pct":     "Avg ann. vol (%)",
            "avg_pairwise_corr":   "Avg corr.",
            "avg_eq_bond_spread":  "Eq-bond spread (%/wk)",
            "n_days":              "Days in history",
            "pct_of_sample":       "% of sample",
            "avg_duration_days":   "Avg duration (days)",
        })
        def hl(df):
            s = pd.DataFrame("", index=df.index, columns=df.columns)
            for rn in df.index:
                if rn == CUREG:
                    s.loc[rn] = "background-color:" + NAME_COLOR.get(rn, "#6B7280") + "22;font-weight:600"
            return s
        st.dataframe(crit_df.style.apply(hl, axis=None), width="stretch")
        st.markdown("""
**Interpreting the table:**
- **Bull:** High positive return, low volatility, low cross-asset correlation
- **Bear:** Negative or flat return, elevated volatility, rising correlation
- **Crisis:** Sharply negative return, very high volatility, very high correlation,
  deeply negative equity-bond spread (bonds rally as equities sell off)


        """)

    st.markdown("---")
    st.markdown("### Regime history")
    fig2, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    ax = axes[0]
    for rn, rc4 in NAME_COLOR.items():
        mask = reg == rn
        ax.fill_between(reg.index, 0, 1, where=mask, color=rc4, alpha=0.7, label=rn)
    ax.set_yticks([]); ax.set_ylabel("Regime", fontsize=10)
    ax.set_title("Regime classification over time", fontsize=11)
    ax.legend(loc="upper right", fontsize=8, ncol=4)
    ax2 = axes[1]
    spy_col = next((c for c in ret.columns if "SP500" in c), ret.columns[0])
    cum = (1 + ret[spy_col]).cumprod()
    ax2.plot(cum.index, cum, color="black", linewidth=0.9)
    for rn, rc5 in NAME_COLOR.items():
        mask = (reg == rn).reindex(cum.index, method="ffill").fillna(False)
        ax2.fill_between(cum.index, cum.min() * 0.95, cum, where=mask, color=rc5, alpha=0.15)
    ax2.set_ylabel("Cumulative return", fontsize=10)
    ax2.set_title(spy_col + " cumulative return shaded by regime", fontsize=11)
    ax2.set_yscale("log")
    plt.tight_layout()
    st.pyplot(fig2, width="stretch")
    plt.close()
    st.caption("")


# -----------------------------------------------------------------------
# PORTFOLIO SIGNAL
# -----------------------------------------------------------------------

elif page == "Portfolio Signal":
    st.title("Current Portfolio Signal")

    with st.expander("How this signal is constructed", expanded=False):
        st.markdown("""
Each week we identify the current regime from the HMM posterior probabilities.
We load the regime-specific causal graph and set portfolio weights proportional
to each asset's out-strength -- assets that send more information to others
receive higher portfolio weight.

Constraints: maximum 20% per asset. All weights sum to 100%.

This is a structural tilt signal, not a market timing signal.
It works best as an overlay on an existing allocation.
        """)

    st.markdown(
        "Active regime: **" + CUREG.upper() + "**  |  "
        + str(GCUR.number_of_edges()) + " validated edges in regime graph"
    )
    col_s1, col_s2 = st.columns([1, 1])

    with col_s1:
        st.markdown("**Suggested weights by asset**")
        st.caption("Weight % is out-strength in the current regime graph, normalised to 100%.")
        if not sig.empty:
            sdf = sig.sort_values(ascending=False).to_frame()
            sdf["Asset class"] = sdf.index.map(lambda a: AC.get(a, "unknown"))
            sdf["Out-strength (bits)"] = sdf.index.map(
                lambda a: round(float(GCUR.out_degree(a, weight="weight"))
                                if a in GCUR else 0.0, 5))
            nf_map = rcent["net_flow_regime"].to_dict()
            sdf["Role"] = sdf.index.map(
                lambda a: ("Leader" if nf_map.get(a, 0) > 0.001
                           else "Follower" if nf_map.get(a, 0) < -0.001
                           else "Neutral"))
            sdf = sdf.rename(columns={"weight_pct": "Weight (%)"})
            def cr(v):
                return {"Leader": "background-color:#D1FAE5",
                        "Follower": "background-color:#FEF3C7",
                        "Neutral": "background-color:#F3F4F6"}.get(v, "")
            st.dataframe(
                sdf[["Weight (%)", "Out-strength (bits)", "Asset class", "Role"]]
                .style.map(cr, subset=["Role"]),
                width="stretch", height=460)
        else:
            st.info("Run the Phase 6 weekly refresh to generate signal data.")

    with col_s2:
        st.markdown("**Weight by asset class**")
        if not sig.empty:
            cw = {}
            for a, w in sig.items():
                cls = AC.get(a, "unknown")
                cw[cls] = cw.get(cls, 0) + float(w)
            fig3, ax3 = plt.subplots(figsize=(6, 5))
            ax3.pie(list(cw.values()), labels=list(cw.keys()),
                    colors=[CLASS_COLOR.get(c, "#888") for c in cw],
                    autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10})
            ax3.set_title("Weight by class -- " + CUREG + " regime", fontsize=11)
            st.pyplot(fig3, width="stretch")
            plt.close()

        st.markdown("---")
        st.markdown("**Net information flow by asset**")
        st.caption("Positive = leader (overweighted). Negative = follower.")
        if "net_flow_regime" in rcent.columns:
            nf_s = rcent["net_flow_regime"].sort_values(ascending=False)
            fig4, ax4 = plt.subplots(figsize=(6, 5))
            ax4.barh(nf_s.index, nf_s.values,
                     color=["#059669" if v >= 0 else "#DC2626" for v in nf_s.values],
                     alpha=0.8)
            ax4.axvline(0, color="black", linewidth=0.8)
            ax4.set_xlabel("Net flow (bits)")
            ax4.set_title("Information role -- current regime", fontsize=11)
            ax4.tick_params(labelsize=8)
            plt.tight_layout()
            st.pyplot(fig4, width="stretch")
            plt.close()


# -----------------------------------------------------------------------
# PERFORMANCE
# -----------------------------------------------------------------------

elif page == "Performance":
    st.title("Backtest Performance")
    st.caption("All results are out-of-sample. Causal graph estimated on training data only.")

    with st.expander("How to interpret", expanded=False):
        st.markdown("""
**Walk-forward design:** All numbers are from the test period -- data the model never saw.

**Signal 1 -- Lead-lag momentum:** When a top leader moves >1.5%, go long its followers for 2 weeks.

**Signal 2 -- Regime rotation:** Overweight assets with high out-strength in the
current regime's causal graph. Rebalanced weekly.

**Sharpe above 0.5** after costs = meaningful. Above 1.0 = strong.
        """)

    if not perf.empty:
        st.markdown("### Summary -- out-of-sample test period")
        def hp(df):
            s = pd.DataFrame("", index=df.index, columns=df.columns)
            for col in ["sharpe", "ann_return_%", "sortino"]:
                if col in df.columns:
                    s.loc[df[col] == df[col].max(), col] = "background-color:#D1FAE5;font-weight:600"
            for col in ["max_drawdown_%", "ann_vol_%"]:
                if col in df.columns:
                    s.loc[df[col] == df[col].min(), col] = "background-color:#D1FAE5;font-weight:600"
            return s
        st.dataframe(perf.style.apply(hp, axis=None), width="stretch")
        st.caption("Green = best value in each column.")

    if not strat.empty:
        SM = {"Signal1_LeadLag": ("#2563EB", 2.2, "-"),
              "Signal2_RegimeRotate": ("#7C3AED", 2.2, "-"),
              "Equal_Weight": ("#6B7280", 1.2, "--")}
        fig5, ax5 = plt.subplots(figsize=(14, 5))
        for col in strat.columns:
            cum = (1 + strat[col]).cumprod()
            color, lw, ls = SM.get(col, ("#D97706", 1.2, "--"))
            ax5.plot(cum.index, cum, label=col, linewidth=lw, linestyle=ls, color=color)
        ax5.set_ylabel("Growth of $1"); ax5.legend(fontsize=9)
        ax5.set_title("Out-of-sample cumulative returns")
        ax5.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
        st.pyplot(fig5,width="stretch" ); plt.close()

        fig6, ax6 = plt.subplots(figsize=(14, 4))
        for col in strat.columns:
            cum = (1 + strat[col]).cumprod(); peak = cum.cummax(); dd = (cum - peak) / peak
            color = SM.get(col, ("#D97706", 1.2, "--"))[0]
            ax6.fill_between(dd.index, dd, 0, alpha=0.2, color=color)
            ax6.plot(dd.index, dd, linewidth=0.8, color=color, label=col)
        ax6.set_ylabel("Drawdown"); ax6.legend(fontsize=8)
        ax6.set_title("Drawdown comparison -- less negative is better")
        st.pyplot(fig6, width="stretch"); plt.close()
    else:
        st.info("Run Phase 5 to generate backtest results.")


# -----------------------------------------------------------------------
# ASSET EXPLORER
# -----------------------------------------------------------------------

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
        role_html = ('<span style="color:#059669;font-weight:700">Information Leader</span>'
                     " -- sends more information than it receives in the current regime")
    elif nf_val < -0.001:
        role_html = ('<span style="color:#D97706;font-weight:700">Information Follower</span>'
                     " -- receives more information than it sends in the current regime")
    else:
        role_html = ('<span style="color:#6B7280;font-weight:700">Neutral / Isolated</span>'
                     " -- minimal directional causal flow in current regime")

    st.markdown("Role: " + role_html, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Out-strength (bits sent)", f"{out_val:.5f}",
              help="Total TE bits sent to all followers in current regime")
    m2.metric("In-strength (bits received)", f"{in_val:.5f}",
              help="Total TE bits received from all leaders in current regime")
    m3.metric("Net flow (bits)", f"{nf_val:+.5f}",
              help="Out minus In. Positive = net leader.")

    st.markdown("---")
    all_te = [d.get("weight", 0) for _, _, d in GCUR.edges(data=True)]
    p25c = np.percentile(all_te, 25) if all_te else 0
    p75c = np.percentile(all_te, 75) if all_te else 1

    col_o, col_i = st.columns(2)

    with col_o:
        st.markdown("**Assets that " + sel + " leads (->)**")
        st.caption("Their future returns are partially predicted by " + sel + "'s past.")
        out_n = list(GCUR.successors(sel)) if sel in GCUR else []
        if out_n:
            for f in out_n:
                d = GCUR.edges.get((sel, f), {})
                te = d.get("weight", 0); nte = d.get("net_te", te)
                lag = d.get("lag", "?"); stab = d.get("stability", "?")
                slab = "Strong" if te >= p75c else "Moderate" if te >= p25c else "Weak"
                sc = STRENGTH_COLOR.get(slab, "#888")
                st.markdown(
                    f'<div style="margin:4px 0;padding:9px 12px;border-radius:6px;'
                    f'background:#F0FDF4;border-left:3px solid {sc}">'
                    f'<b>{sel} -> {f}</b><br>'
                    f'<span style="font-size:12px;color:#374151">'
                    f'TE: <b>{te:.5f} bits</b>  |  Net TE: <b>{nte:.5f} bits</b>  |  '
                    f'Lag: <b>{lag} wks</b>  |  Stability: <b>{stab}/5</b>  |  '
                    f'<span style="color:{sc}">{slab}</span></span></div>',
                    unsafe_allow_html=True)
        else:
            st.info(sel + " has no validated followers in the " + CUREG + " regime.")

    with col_i:
        st.markdown("**Assets that lead " + sel + " (<-)**")
        st.caption("Their past returns have validated predictive power over " + sel + "'s future.")
        in_n = list(GCUR.predecessors(sel)) if sel in GCUR else []
        if in_n:
            for l in in_n:
                d = GCUR.edges.get((l, sel), {})
                te = d.get("weight", 0); nte = d.get("net_te", te)
                lag = d.get("lag", "?"); stab = d.get("stability", "?")
                slab = "Strong" if te >= p75c else "Moderate" if te >= p25c else "Weak"
                sc = STRENGTH_COLOR.get(slab, "#888")
                st.markdown(
                    f'<div style="margin:4px 0;padding:9px 12px;border-radius:6px;'
                    f'background:#FFF7ED;border-left:3px solid {sc}">'
                    f'<b>{l} -> {sel}</b><br>'
                    f'<span style="font-size:12px;color:#374151">'
                    f'TE: <b>{te:.5f} bits</b>  |  Net TE: <b>{nte:.5f} bits</b>  |  '
                    f'Lag: <b>{lag} wks</b>  |  Stability: <b>{stab}/5</b>  |  '
                    f'<span style="color:{sc}">{slab}</span></span></div>',
                    unsafe_allow_html=True)
        else:
            st.info("No validated leaders for " + sel + " in the " + CUREG + " regime.")

    st.markdown("---")
    st.markdown("**" + sel + " -- recent price performance**")
    if sel in ret.columns:
        fig7, ax7 = plt.subplots(figsize=(12, 3))
        cum7 = (1 + ret[sel]).cumprod()
        color7 = CLASS_COLOR.get(AC.get(sel, "unknown"), "#2563EB")
        ax7.plot(cum7.index, cum7, color=color7, linewidth=1.2)
        ax7.fill_between(cum7.index, 1, cum7, where=cum7 >= 1, alpha=0.15, color=color7)
        ax7.fill_between(cum7.index, 1, cum7, where=cum7 < 1, alpha=0.15, color="#DC2626")
        ax7.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
        ax7.set_ylabel("Cumulative return"); ax7.set_title(sel + " -- recent 2 years")
        plt.tight_layout()
        st.pyplot(fig7, width="stretch")
        plt.close()