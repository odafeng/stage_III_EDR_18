import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import os

# ==========================================
# 1. Page Configuration (Added CSS Fix)
# ==========================================
st.set_page_config(
    page_title="OncoPredict: Stage III Colon Cancer",
    page_icon="⚕️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS Fix: Added responsive width and font adjustment to prevent HTML layout issues
st.markdown("""
    <style>
    .stApp {
        background-color: #ffffff;
    }
    .report-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-top: 20px;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .report-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 15px;
        margin-bottom: 20px;
    }
    .risk-badge-high {
        background-color: #ffebee;
        color: #c62828;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: bold;
        border: 1px solid #ffcdd2;
        font-size: 0.9em;
        white-space: nowrap; /* Prevent line break */
    }
    .risk-badge-low {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: bold;
        border: 1px solid #c8e6c9;
        font-size: 0.9em;
        white-space: nowrap; /* Prevent line break */
    }
    .prob-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #eee;
        margin-bottom: 20px;
    }
    .prob-val {
        font-size: 2.5em;
        font-weight: 800;
        color: #2c3e50;
        line-height: 1.2;
    }
    .rec-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #2196f3;
        font-size: 0.95em;
        color: #0d47a1;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. Load Model
# ==========================================
@st.cache_resource
def load_model():
    try:
        # Please confirm this is the correct path on your computer
        # Use absolute path relative to this script for better deployment stability
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'model', 'final_model_calibrated.pkl')
        return joblib.load(model_path)
    except FileNotFoundError:
        return None

model = load_model()

# ==========================================
# 3. Language Settings
# ==========================================
with st.sidebar:
    st.header("Settings")
    lang_choice = st.selectbox("Language", ["English", "Traditional Chinese (繁體中文)"])
    lang = "en" if lang_choice == "English" else "zh"

t = {
    "en": {
        "title": "🏥 Stage III Colon Cancer Risk Assessment",
        "subtitle": "AI-Driven EDR Prediction Tool",
        "sec_pt": "Patient Demographics",
        "sec_clini": "Clinicopathological Features",
        "sex": "Sex",
        "sex_opts": ["Male", "Female"],
        "age": "Age",
        "ajcc": "AJCC 8th Substage",
        "pni": "Perineural Invasion (PNI)",
        "pni_opts": ["Negative", "Positive"],
        "diff": "Histological Grade",
        "diff_opts": ["Well Differentiated", "Moderately Differentiated", "Poorly Differentiated"],
        "ln_header": "Lymph Node Ratio (LNR)",
        "ln_pos": "Positive Nodes (pN)",
        "ln_tot": "Total Harvested",
        "btn": "Calculate Risk",
        "res_title": "Risk Analysis Report",
        "high_risk": "HIGH RISK",
        "low_risk": "LOW RISK",
        "rec_high": "Recommendation: Consider intensified adjuvant therapy (e.g., 6 months FOLFOX).",
        "rec_low": "Recommendation: Standard adjuvant therapy (e.g., 3 months CAPOX) likely sufficient.",
        "prob": "18-Month EDR Probability",
        "thresh": "Threshold"
    },
    "zh": {
        "title": "🏥 第 3 期大腸癌風險評估系統",
        "subtitle": "AI 驅動之早期遠端復發 (EDR) 預測工具",
        "sec_pt": "病患基本資料",
        "sec_clini": "臨床病理特徵",
        "sex": "性別",
        "sex_opts": ["男性", "女性"],
        "age": "年齡",
        "ajcc": "AJCC 第八版子分期",
        "pni": "神經侵犯 (PNI)",
        "pni_opts": ["陰性 (-)", "陽性 (+)"],
        "diff": "組織分化度 (Grade)",
        "diff_opts": ["良好分化 (Well)", "中度分化 (Moderately)", "分化不良 (Poorly)"],
        "ln_header": "淋巴結比率 (LNR)",
        "ln_pos": "陽性淋巴結數",
        "ln_tot": "總摘除淋巴結數",
        "btn": "開始分析",
        "res_title": "風險分析報告",
        "high_risk": "高風險群 (High Risk)",
        "low_risk": "低風險群 (Low Risk)",
        "rec_high": "臨床建議：此病患具有較高生物學惡性度。建議考慮加強輔助化療強度 (如 6 個月 FOLFOX)。",
        "rec_low": "臨床建議：此病患預後相對良好。建議依循標準治療指引 (如 3 個月 CAPOX) 即可。",
        "prob": "預測 18 個月內復發機率",
        "thresh": "切點"
    }
}

if model is None:
    st.error("⚠️ Model not found. Please check the file path.")
    st.stop()

st.title(t[lang]["title"])
st.markdown(f"**{t[lang]['subtitle']}**")
st.divider()

# ==========================================
# 4. Input Interface
# ==========================================
with st.form("main_form"):
    
    st.subheader(f"👤 {t[lang]['sec_pt']}")
    c1, c2 = st.columns(2)
    with c1: sex = st.selectbox(t[lang]["sex"], t[lang]["sex_opts"])
    with c2: age = st.number_input(t[lang]["age"], 20, 100, 65)
    
    st.write("")

    st.subheader(f"🧬 {t[lang]['sec_clini']}")
    m1, m2 = st.columns(2)
    with m1: ajcc_val = st.selectbox(t[lang]["ajcc"], ["3A", "3B", "3C"], index=1)
    with m2:
        diff_str = st.selectbox(t[lang]["diff"], t[lang]["diff_opts"])
        if diff_str == t[lang]["diff_opts"][0]: diff_val = 1
        elif diff_str == t[lang]["diff_opts"][1]: diff_val = 2
        else: diff_val = 3
    
    st.write(f"**{t[lang]['pni']}**")
    pni_str = st.radio(t[lang]["pni"], t[lang]["pni_opts"], horizontal=True, label_visibility="collapsed")
    pni_val = 1 if "+" in pni_str or "Positive" in pni_str else 0

    st.write("---")
    st.write(f"**{t[lang]['ln_header']}**")
    l1, l2, l3 = st.columns([1, 1, 1])
    with l1: ln_pos = st.number_input(t[lang]["ln_pos"], 0, 100, 2)
    with l2: ln_tot = st.number_input(t[lang]["ln_tot"], 1, 100, 15)
    with l3:
        lnr_val = ln_pos / ln_tot if ln_tot >= ln_pos and ln_tot > 0 else 0.0
        if ln_tot < ln_pos: st.error("Error")
        else: st.metric("LNR", f"{lnr_val:.3f}")

    st.write("")
    submit = st.form_submit_button(t[lang]["btn"], use_container_width=True, type="primary")

# ==========================================
# 5. Calculation and Report Output (HTML Structure Optimization)
# ==========================================
if submit:
    with st.spinner("Calculating..."):
        time.sleep(0.5)

    # 1. Prepare Data (Ignore order for now, just get all columns ready)
    # Note: The column names here must exactly match the names from pd.get_dummies during training
    # Based on our final training code, the name should be composed of 'AJCC_Substage' + '_' + '3A'
    input_data = pd.DataFrame({
        'PNI': [pni_val],
        'LNR': [lnr_val],
        'Differentiation': [diff_val],
        'AJCC_Substage_3A': [1 if ajcc_val == "3A" else 0],
        'AJCC_Substage_3B': [1 if ajcc_val == "3B" else 0],
        'AJCC_Substage_3C': [1 if ajcc_val == "3C" else 0]
    })
    
    # 2. [Critical Fix] Auto-align column order
    # Attempt to read the column order "remembered" by the model during training
    try:
        if hasattr(model, 'feature_names_in_'):
            # If the model has a record, reorder according to it
            correct_order = model.feature_names_in_
            input_data = input_data[correct_order]
        else:
            # In case the model has no record (older version), we manually specify (this is the likely order from the last training)
            # According to the default behavior of pd.get_dummies, it usually puts dummies at the end or replaces in place
            # Here is a backup of the most likely order
            fallback_order = ['PNI', 'LNR', 'Differentiation', 'AJCC_Substage_3A', 'AJCC_Substage_3B', 'AJCC_Substage_3C']
            # Check if columns match, if not, try to run anyway
            if set(fallback_order).issubset(input_data.columns):
                input_data = input_data[fallback_order]
    except Exception as e:
        st.warning(f"Auto-alignment failed, using default order. ({e})")

    try:
        # 3. Prediction
        prob = model.predict_proba(input_data)[:, 1][0]
        
        # Cutoff (Your golden cutoff)
        CUTOFF = 0.120 
        
        st.divider()
        st.subheader(f"📋 {t[lang]['res_title']}")
        
        # Display result title
        if prob >= CUTOFF:
            st.error(f"#### {t[lang]['high_risk']}")
            rec_box = st.warning
            rec_text = t[lang]["rec_high"]
        else:
            st.success(f"#### {t[lang]['low_risk']}")
            rec_box = st.info
            rec_text = t[lang]["rec_low"]
            
        # Display large numbers and progress bar
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric(label=t[lang]["prob"], value=f"{prob:.1%}", delta=f"Threshold: {CUTOFF:.1%}", delta_color="off")
        with c2:
            st.write("") # Spacer
            st.progress(float(prob))
            st.caption(f"Patient Profile: {sex} | {age} y/o")
        
        # Display recommendations
        rec_box(f"**💡 Recommendation:**\n\n{rec_text}")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        # If it still errors, show debug info to help you
        st.write("--- Debug Info ---")
        st.write("Input Shape:", input_data.shape)
        st.write("Input Columns:", input_data.columns.tolist())
        if hasattr(model, 'feature_names_in_'):
             st.write("Expected Columns:", model.feature_names_in_.tolist())

# ==========================================
# 6. Footer / Citation
# ==========================================
st.divider()
st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.85em;">
        <i>'Ruling Out Early Distant Recurrence in Stage III Colon Cancer: A Parsimonious Machine Learning Model with External Validation'</i><br>
        <b>Shih-Feng Fredric Huang, et al.</b>
    </div>
    """, unsafe_allow_html=True)