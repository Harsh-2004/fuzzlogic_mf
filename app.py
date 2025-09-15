import io
import re
import time
import pandas as pd
import streamlit as st
import numpy as np
from rapidfuzz import process, fuzz

# --- Page Configuration ---
st.set_page_config(
    page_title="Scheme Matcher",
    page_icon="🔗",
    layout="wide"
)

# --- Helper Functions (No changes here) ---

def decode_bytes(x):
    """turn bytes/bytearray into str, leave others unchanged"""
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
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda x: decode_bytes(x))
    if convert_objects_to_str:
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype(str)
    if na_to_empty:
        df = df.replace({np.nan: ""})
        df = df.replace({"nan": ""})
    return df

@st.cache_data
def build_scheme_name_list(df, scheme_name_col):
    return df[scheme_name_col].astype(str).fillna("").tolist()

def custom_scorer(s1, s2, **kwargs):
    s1, s2 = s1.lower(), s2.lower()
    s1 = re.sub(r'\breg\b', 'regular', s1)
    s1 = re.sub(r'\bdir\b', 'direct', s1)
    s2 = re.sub(r'\breg\b', 'regular', s2)
    s2 = re.sub(r'\bdir\b', 'direct', s2)
    base_score = fuzz.token_sort_ratio(s1, s2)
    s1_first, s2_first = s1.split(' ', 1)[0], s2.split(' ', 1)[0]
    if s1_first == s2_first:
        base_score += 10
    if "regular" in s2:
        base_score += 5
    return min(base_score, 100)

def get_top_matches(query, choices, top_k=3, scorer=custom_scorer, score_cutoff=0):
    results = process.extract(query, choices, scorer=scorer, limit=top_k)
    return [r for r in results if r[1] >= score_cutoff]

# --- Main App UI ---

st.title("🔗 Scheme Matcher Pro")
st.markdown("Fuzzy match `tester` securities against a `Scheme_master` list and download your hand-picked selections.")

# --- Sidebar for Controls ---
with st.sidebar:
    st.image("https://www.google.com/s2/favicons?domain=streamlit.io&sz=128", width=50)
    st.header("⚙️ Matching Controls")

    top_k = st.number_input("Top matches per security", min_value=1, max_value=10, value=3, step=1,
                            help="How many potential matches to show for each security name.")
    score_cutoff = st.slider("Minimum Match Score", min_value=0, max_value=100, value=60,
                             help="Don't show any matches below this score.")
    max_display = st.number_input("Max Securities to Process", min_value=1, max_value=5000, value=200,
                                  help="Limit the number of rows from the 'tester' file to process in this run.")

# --- Step 1: File Uploads ---
with st.container(border=True):
    st.header("Step 1: Upload Your Files", divider="rainbow")
    col1, col2 = st.columns(2)
    with col1:
        tester_u = st.file_uploader("📂 Upload `tester` Excel", type=["xlsx", "xls"])
    with col2:
        master_u = st.file_uploader("📂 Upload `Scheme_master` Excel", type=["xlsx", "xls"])

if tester_u is None or master_u is None:
    st.info("👋 Welcome! Please upload both Excel files to begin the matching process.")
    st.stop()

# --- Load Data (with caching) ---
@st.cache_data
def load_data(tester_file, master_file):
    tester_df = pd.read_excel(tester_file)
    master_df = pd.read_excel(master_file)
    return tester_df, master_df

tester_df, master_df = load_data(tester_u, master_u)
st.success(f"✅ Files loaded! Tester: `{tester_u.name}` ({len(tester_df)} rows), Master: `{master_u.name}` ({len(master_df)} rows)")


# --- Step 2: Column Selection ---
with st.container(border=True):
    st.header("Step 2: Choose Your Columns", divider="rainbow")
    c1, c2 = st.columns(2)
    with c1:
        tester_name_col = st.selectbox("Column in `tester` with security names", options=list(tester_df.columns), index=0)
    with c2:
        default_master_idx = list(master_df.columns).index("Scheme Name") if "Scheme Name" in master_df.columns else 0
        master_name_col = st.selectbox("Column in `Scheme_master` with scheme names", options=list(master_df.columns), index=default_master_idx)

    available_master_cols = list(master_df.columns)
    cols_to_show = st.multiselect(
        "Columns from `Scheme_master` to include in the final result",
        options=available_master_cols,
        default=available_master_cols,
    )

# --- Initialize Matching Process ---
master_df[master_name_col] = master_df[master_name_col].astype(str)
security_series = tester_df[tester_name_col].astype(str).fillna("").reset_index(drop=True)
n_to_process = min(len(security_series), max_display)
selected_matches = []

# --- Step 3 & 4: Tabs for Review and Download ---
tab1, tab2 = st.tabs(["🔍 Review & Select Matches", "⬇️ Final Results & Download"])

with tab1:
    st.header(f"Step 3: Review Top Matches ({n_to_process} securities)", divider="rainbow")
    st.write("Go through each security, review the potential matches, and select the correct one using the radio buttons.")
    
    progress_bar = st.progress(0, text="Starting match process...")
    
    for i in range(n_to_process):
        sec_name = security_series.iloc[i]
        
        # Update progress bar
        progress_text = f"Processing {i+1}/{n_to_process}: {sec_name[:40]}..."
        progress_bar.progress((i + 1) / n_to_process, text=progress_text)
        
        expander_label = f"**{i+1}.** **Tester:** `{sec_name}`"
        
        with st.expander(expander_label, expanded=(i < 3)):
            cleaned_sec_name = sec_name.split('(', 1)[0].strip()
            first_word = cleaned_sec_name.split(' ', 1)[0]
            
            master_df_filtered = master_df[master_df[master_name_col].str.lower().str.startswith(first_word.lower(), na=False)]
            current_master_df = master_df if master_df_filtered.empty else master_df_filtered
            choices = build_scheme_name_list(current_master_df, master_name_col)
            
            if not choices:
                st.warning("No potential matches found in Scheme Master based on the first word.")
                continue

            matches = get_top_matches(cleaned_sec_name, choices, top_k, custom_scorer, score_cutoff)

            if not matches:
                st.info("No matches found above the score cutoff.")
                continue

            rows = []
            for match_name, score, idx in matches:
                row = current_master_df.iloc[idx].to_dict()
                row.update({
                    "_match_score": round(score),
                    "_scheme_name_matched_to": match_name,
                    "_tester_security_name": sec_name
                })
                rows.append(row)

            matches_df = pd.DataFrame(rows)
            display_cols = ["_match_score", "_scheme_name_matched_to"] + [c for c in cols_to_show if c in matches_df.columns]
            
            st.dataframe(sanitize_df(matches_df[display_cols]), use_container_width=True, hide_index=True)
            
            radio_options = ["**(No Match)**"] + [
                f"**Score {r['_match_score']}**: {r['_scheme_name_matched_to']}"
                for _, r in matches_df.iterrows()
            ]
            choice = st.radio("Select the correct match:", options=radio_options, key=f"radio_{i}", index=1 if matches else 0, horizontal=True)

            if choice != radio_options[0]:
                chosen_idx = radio_options.index(choice) - 1
                chosen_row = matches_df.iloc[chosen_idx].to_dict()
                tester_info = {f"tester_{c}": tester_df[c].iloc[i] for c in tester_df.columns}
                selected_matches.append({**tester_info, **chosen_row})

    progress_bar.empty() # Clear progress bar when done
    st.success("Review complete! Go to the 'Final Results & Download' tab to get your file.")


with tab2:
    st.header("Step 4: Download Your Selections", divider="rainbow")
    if selected_matches:
        out_df = pd.DataFrame(selected_matches)
        clean_out_df = sanitize_df(out_df)
        st.dataframe(clean_out_df, use_container_width=True)

        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            # Excel download
            to_xlsx = io.BytesIO()
            with pd.ExcelWriter(to_xlsx, engine="openpyxl") as writer:
                clean_out_df.to_excel(writer, index=False, sheet_name="Selected_Matches")
            
            st.download_button(
                label="📥 Download as Excel",
                data=to_xlsx.getvalue(),
                file_name="selected_scheme_matches.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        with col2:
            # CSV download
            csv_bytes = clean_out_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download as CSV",
                data=csv_bytes,
                file_name="selected_scheme_matches.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.warning("No matches have been selected yet. Please select your intended matches in the 'Review' tab. 👆")