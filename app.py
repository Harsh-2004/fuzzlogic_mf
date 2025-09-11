# app.py
import io
import subprocess
import sys
from rapidfuzz import process, fuzz

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Scheme Matcher", layout="wide")

st.title("Scheme Master ↔ Tester — fuzzy matcher (pick top match and download)")

# --- Helper functions ---
# --- add these imports at top ---
import numpy as np

# --- helper: sanitize a dataframe for Streamlit / pyarrow ---
def decode_bytes(x):
    # turn bytes/bytearray into str, leave others unchanged
    try:
        if isinstance(x, (bytes, bytearray)):
            return x.decode('utf-8', errors='ignore')
    except Exception:
        pass
    return x

def sanitize_df(df, convert_objects_to_str=True, na_to_empty=True):
    """
    - Decodes bytes in object columns
    - Optionally converts object dtype columns to str (ensures homogeneous types)
    - Optionally replaces NaN with empty string
    Returns a copy (doesn't mutate original)
    """
    df = df.copy()
    # decode bytes in object columns first
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda x: decode_bytes(x))
    if convert_objects_to_str:
        for c in df.columns:
            if df[c].dtype == object:
                # Convert to string; keep numerics alone (so numbers remain numeric if dtype isn't object)
                df[c] = df[c].astype(str)
    if na_to_empty:
        # replace common NA variants with empty string
        df = df.replace({np.nan: ""})
        # also replace literal 'nan' created by astype(str)
        df = df.replace({"nan": ""})
    return df

@st.cache_data
def build_scheme_name_list(df, scheme_name_col):
    return df[scheme_name_col].astype(str).fillna("").tolist()

@st.cache_data
def get_top_matches(query, choices, top_k=3, scorer=fuzz.token_sort_ratio, score_cutoff=0):
    results = process.extract(query, choices, scorer=scorer, limit=top_k)
    if score_cutoff:
        results = [r for r in results if r[1] >= score_cutoff]
    return results

# --- File uploads ---
st.markdown("**Step 1 — Upload files**")
col1, col2 = st.columns(2)
with col1:
    tester_u = st.file_uploader("Upload `tester` Excel (contains `security_name`)", type=["xlsx", "xls"])
with col2:
    master_u = st.file_uploader("Upload `Scheme_master` Excel", type=["xlsx", "xls"])

if tester_u is None or master_u is None:
    st.info("Please upload both files to proceed.")
    st.stop()

tester_df = pd.read_excel(tester_u)
master_df = pd.read_excel(master_u)
st.success("Files loaded.")

# --- Column selectors ---
st.markdown("**Step 2 — choose columns**")
c1, c2 = st.columns(2)
with c1:
    tester_name_col = st.selectbox("Column in tester with security names", options=list(tester_df.columns), index=0)
with c2:
    default_master_idx = list(master_df.columns).index("Scheme Name") if "Scheme Name" in master_df.columns else 0
    master_name_col = st.selectbox("Column in Scheme_master with scheme names", options=list(master_df.columns), index=default_master_idx)

# --- Output columns (dynamic from Scheme_master) ---
available_master_cols = list(master_df.columns)
cols_to_show = st.multiselect(
    "Columns from Scheme_master to include in result (order preserved)",
    options=available_master_cols,
    default=available_master_cols,
)

# --- Matching parameters ---
st.markdown("**Step 3 — matching parameters**")
param_col1, param_col2, param_col3 = st.columns([2, 1, 1])
with param_col1:
    top_k = st.number_input("Top matches per security", min_value=1, max_value=10, value=3, step=1)
with param_col2:
    score_cutoff = st.slider("Minimum match score to show", min_value=0, max_value=100, value=50)
with param_col3:
    max_display = st.number_input("Max securities to display (for large files)", min_value=1, max_value=500, value=200)

# --- Do the matching ---
st.markdown("**Step 4 — review top matches and select intended match**")
choices = build_scheme_name_list(master_df, master_name_col)
security_series = tester_df[tester_name_col].astype(str).fillna("").reset_index(drop=True)
n_to_process = min(len(security_series), max_display)

selected_matches = []

for i in range(n_to_process):
    sec_name = security_series.iloc[i]
    expander_label = f"{i+1}. Tester: {sec_name}"
    with st.expander(expander_label, expanded=(i < 10)):
        matches = get_top_matches(sec_name, choices, top_k=top_k, scorer=fuzz.token_sort_ratio)
        matches = [m for m in matches if m[1] >= score_cutoff]

        if not matches:
            st.write("No matches above score cutoff.")
            continue

        rows = []
        for match_name, score, idx in matches:
            row = master_df.iloc[idx].to_dict()
            row["_match_score"] = score
            row["_scheme_name_matched_to"] = match_name
            row["_tester_security_name"] = sec_name
            rows.append(row)

        matches_df = pd.DataFrame(rows)
        display_cols = ["_tester_security_name", "_scheme_name_matched_to", "_match_score"] + cols_to_show
        display_cols = [c for c in display_cols if c in matches_df.columns]
        display_df = matches_df[display_cols].rename(
            columns={
                "_tester_security_name": "Tester security_name",
                "_scheme_name_matched_to": "Matched scheme name",
                "_match_score": "Match score",
            }
        )
        st.dataframe(sanitize_df(display_df), use_container_width=True)


        # Single-choice selection
        radio_options = ["No match"] + [
            f"{r['_match_score']} — {r.get('Scheme Name', r['_scheme_name_matched_to'])}"
            for _, r in matches_df.reset_index(drop=True).iterrows()
        ]
        choice = st.radio("Choose intended match (one only)", options=radio_options, key=f"radio_{i}", index=0)

        if choice != "No match":
            chosen_idx = radio_options.index(choice) - 1
            chosen_row = matches_df.reset_index(drop=True).iloc[chosen_idx].to_dict()
            chosen_row_prefilled = {f"tester_{c}": tester_df[c].iloc[i] for c in tester_df.columns}
            merged = {**chosen_row_prefilled, **chosen_row}
            selected_matches.append(merged)

# --- Results + download ---
st.markdown("**Step 5 — preview selected matches**")
if selected_matches:
    out_df = pd.DataFrame(selected_matches)
    clean_out_df = sanitize_df(out_df)
    st.dataframe(clean_out_df, use_container_width=True)

    # Excel download (no .save()!)
    to_xlsx = io.BytesIO()
    with pd.ExcelWriter(to_xlsx, engine="openpyxl") as writer:
        clean_out_df.to_excel(writer, index=False, sheet_name="Selected_Matches")
    to_xlsx.seek(0)

    st.download_button(
        label="Download selected matches as Excel",
        data=to_xlsx.getvalue(),
        file_name="selected_scheme_matches.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # CSV download
    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download selected as CSV", csv_bytes, file_name="selected_scheme_matches.csv", mime="text/csv")
else:
    st.info("No matches were selected yet. Use the expanders above to pick intended matches.")
