import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px

# ────────────────────────────────────────────────────────────
# PAGE CONFIG — must be FIRST streamlit command
# ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "X Education — Lead Scorer",
    page_icon  = "🎯",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ────────────────────────────────────────────────────────────
# CUSTOM CSS
# ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(90deg, #1A5276, #2980B9);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem; color: #7F8C8D; margin-bottom: 2rem;
    }
    .score-box {
        border-radius: 16px; padding: 28px 20px;
        text-align: center; margin: 10px 0;
    }
    .hot-box   { background: linear-gradient(135deg,#1E8449,#27AE60); color:white; }
    .warm-box  { background: linear-gradient(135deg,#BA4A00,#E67E22); color:white; }
    .cold-box  { background: linear-gradient(135deg,#922B21,#E74C3C); color:white; }
    .score-number { font-size: 4rem; font-weight: 900; line-height: 1; }
    .score-label  { font-size: 1.4rem; font-weight: 700; margin-top: 6px; }
    .score-action { font-size: 0.85rem; margin-top: 8px; opacity: 0.9; }
    .metric-card {
        background: #F8F9FA; border-radius: 12px;
        padding: 16px; text-align: center;
        border: 1px solid #E8E8E8;
    }
    .metric-val { font-size: 1.8rem; font-weight: 700; color: #1A5276; }
    .metric-lbl { font-size: 0.8rem; color: #7F8C8D; margin-top: 4px; }
    .stButton>button {
        background: linear-gradient(90deg, #1A5276, #2980B9);
        color: white; font-weight: 700; font-size: 1.1rem;
        padding: 0.6rem 2rem; border-radius: 8px; border: none;
        width: 100%; margin-top: 10px;
    }
    .stButton>button:hover { opacity: 0.9; }
    .section-title {
        font-size: 1.2rem; font-weight: 700;
        color: #1A5276; border-bottom: 2px solid #2980B9;
        padding-bottom: 6px; margin: 20px 0 14px 0;
    }
    div[data-testid="stSidebar"] { background: #F0F4F8; }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────
# LOAD MODEL AND ENCODERS
# ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load("best_model.pkl")
    encoders = joblib.load("label_encoders.pkl")
    return model, encoders

try:
    model, label_encoders = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"⚠️ Could not load model files: {e}")
    st.info("Make sure best_model.pkl and label_encoders.pkl are in the same folder as app.py")
    st.stop()


# ────────────────────────────────────────────────────────────
# FEATURE DEFINITIONS
# ────────────────────────────────────────────────────────────
# These must match EXACTLY what was used during training

BINARY_COLS = [
    "Do Not Email", "Search", "Magazine", "Newspaper Article",
    "X Education Forums", "Newspaper", "Digital Advertisement",
    "Through Recommendations", "Receive More Updates About Our Courses",
    "Update me on Supply Chain Content", "Get updates on DM Content",
    "A free copy of Mastering The Interview",
    "I agree to pay the amount through cheque"
]

CAT_COLS = [
    "Lead Origin", "Lead Source", "Last Activity",
    "Country", "Specialization", "Last Notable Activity",
    "What is your current occupation",
    "What matters most to you in choosing a course",
    "City", "Tags"
]

NUM_COLS = ["Total Time Spent on Website"]

# Common dropdown values based on your dataset
LEAD_SOURCES = [
    "Google", "Direct Traffic", "Organic Search", "Reference",
    "Welingak Website", "Olark Chat", "Social Media", "Facebook",
    "bing", "Click2call", "Live Chat", "Unknown"
]
LEAD_ORIGINS = [
    "Landing Page Submission", "API", "Lead Add Form",
    "Lead Import", "Quick Add Form"
]
LAST_ACTIVITIES = [
    "Email Opened", "SMS Sent", "Olark Chat Conversation",
    "Page Visited on Website", "Email Link Clicked",
    "Had a Phone Conversation", "Email Bounced",
    "Form Submitted on Website", "Unreachable", "Unsubscribed",
    "View in browser link Clicked", "Email Received",
    "Approached upfront", "Resubscribed to emails", "Unknown"
]
LAST_NOTABLE = [
    "Modified", "Email Opened", "SMS Sent",
    "Had a Phone Conversation", "Email Link Clicked",
    "Email Bounced", "Olark Chat Conversation",
    "View in browser link Clicked", "Page Visited on Website",
    "Unsubscribed", "Unknown"
]
SPECIALIZATIONS = [
    "Select", "Finance Management", "Human Resource Management",
    "Marketing Management", "Operations Management",
    "Business Administration", "IT Projects Management",
    "Supply Chain Management", "Banking, Investment And Insurance",
    "Travel and Tourism", "Media and Advertising",
    "International Business", "Healthcare Management",
    "Hospitality Management", "Retail Management",
    "Rural and Agribusiness", "E-COMMERCE", "E-Business",
    "Services Excellence", "Unknown"
]
OCCUPATIONS = [
    "Unemployed", "Working Professional", "Student",
    "Businessman", "Housewife", "Other", "Unknown"
]
WHAT_MATTERS = [
    "Better Career Prospects", "Flexibility & Convenience",
    "Unknown", "Other"
]
COUNTRIES = [
    "India", "United Arab Emirates", "United Kingdom",
    "United States", "Singapore", "Saudi Arabia", "Unknown"
]
CITIES = [
    "Mumbai", "Thane & Outskirts", "Other Cities",
    "Other Cities of Maharashtra", "Pune", "Other Metro Cities",
    "Tier II Cities", "Unknown"
]
TAGS = [
    "Will revert after reading the email", "Ringing",
    "Interested in other courses", "switched off",
    "Already a student", "Not doing further education",
    "In confusion whether part time or DLP",
    "Interested in full time MBA",
    "Diploma holder (Not Eligible)", "invalid number",
    "Lost to EINS", "Busy", "Unknown"
]


# ────────────────────────────────────────────────────────────
# SCORING FUNCTION
# ────────────────────────────────────────────────────────────
def encode_and_score(input_data: dict) -> tuple[float, str, str]:
    """
    Take raw input dict → encode → predict → return score, tier, action.
    Returns: (score_0_to_100, tier_string, action_string)
    """
    row = {}

    # Numeric
    row["Total Time Spent on Website"] = float(
        input_data.get("Total Time Spent on Website", 0))

    # Binary Yes/No → 1/0
    for col in BINARY_COLS:
        row[col] = 1 if input_data.get(col, "No") == "Yes" else 0

    # Categorical → LabelEncoder
    for col in CAT_COLS:
        val = str(input_data.get(col, "Unknown"))
        if col in label_encoders:
            le = label_encoders[col]
            # Handle unseen labels gracefully
            if val in le.classes_:
                row[col] = int(le.transform([val])[0])
            else:
                # Use most common class (index 0) for unknown values
                row[col] = 0
        else:
            row[col] = 0

    # Build DataFrame in correct column order
    all_cols = NUM_COLS + BINARY_COLS + CAT_COLS
    # Only keep columns that exist in our model
    model_features = [c for c in all_cols if c in row]
    df_input = pd.DataFrame([row])[model_features]

    # Get probability
    prob  = model.predict_proba(df_input)[0][1]
    score = round(prob * 100, 1)

    # Tier and action
    if score >= 70:
        tier   = "🔥 HOT"
        action = "Call within 24 hours — Priority queue"
    elif score >= 40:
        tier   = "⚡ WARM"
        action = "Send email + follow-up call this week"
    else:
        tier   = "❄️ COLD"
        action = "Automated email only — low priority"

    return score, tier, action


# ────────────────────────────────────────────────────────────
# GAUGE CHART
# ────────────────────────────────────────────────────────────
def make_gauge(score: float) -> go.Figure:
    color = ("#27AE60" if score >= 70 else
             "#E67E22" if score >= 40 else
             "#E74C3C")
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = score,
        number= {"suffix": "/100", "font": {"size": 36, "color": color}},
        gauge = {
            "axis"      : {"range": [0, 100], "tickwidth": 1},
            "bar"       : {"color": color, "thickness": 0.25},
            "steps"     : [
                {"range": [0,  40], "color": "#FADBD8"},
                {"range": [40, 70], "color": "#FDEBD0"},
                {"range": [70,100], "color": "#D5F5E3"},
            ],
            "threshold" : {
                "line" : {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": score
            }
        },
        domain={"x": [0,1], "y": [0,1]}
    ))
    fig.update_layout(
        margin     = dict(t=20, b=10, l=30, r=30),
        height     = 250,
        paper_bgcolor = "rgba(0,0,0,0)",
        font       = {"family": "Arial"}
    )
    return fig


# ────────────────────────────────────────────────────────────
# MAIN APP LAYOUT
# ────────────────────────────────────────────────────────────
# ── Header ───────────────────────────────────────────────────
st.markdown('<div class="main-header">🎯 X Education — Lead Conversion Scorer</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by XGBoost · ROC-AUC: 0.9786 · Enter lead details to get an instant conversion probability score</div>',
            unsafe_allow_html=True)

# ── Model metrics strip ───────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
for col, label, value in zip(
    [c1, c2, c3, c4, c5],
    ["Model",    "ROC-AUC", "Precision", "Recall",  "F1 Score"],
    ["XGBoost",  "0.9786",  "0.9125",   "0.9228",  "0.9176"]
):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-val">{value}</div>
        <div class="metric-lbl">{label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Two-column layout: inputs left, result right ─────────────
left, right = st.columns([1.6, 1], gap="large")

with left:
    st.markdown('<div class="section-title">📋 Lead Details</div>',
                unsafe_allow_html=True)

    # Row 1
    r1a, r1b = st.columns(2)
    with r1a:
        lead_source = st.selectbox("Lead Source", LEAD_SOURCES,
            help="Where did this lead come from?")
    with r1b:
        lead_origin = st.selectbox("Lead Origin", LEAD_ORIGINS,
            help="How did the lead enter the system?")

    # Row 2
    r2a, r2b = st.columns(2)
    with r2a:
        last_activity = st.selectbox("Last Activity", LAST_ACTIVITIES,
            help="Most recent interaction with the lead")
    with r2b:
        last_notable = st.selectbox("Last Notable Activity", LAST_NOTABLE,
            help="Most noteworthy recent event")

    # Row 3
    r3a, r3b = st.columns(2)
    with r3a:
        specialization = st.selectbox("Specialization", SPECIALIZATIONS,
            help="Course specialization the lead is interested in")
    with r3b:
        occupation = st.selectbox("Current Occupation", OCCUPATIONS,
            help="Lead's current employment status")

    # Row 4
    r4a, r4b = st.columns(2)
    with r4a:
        what_matters = st.selectbox("What Matters Most", WHAT_MATTERS,
            help="What the lead prioritises when choosing a course")
    with r4b:
        tags = st.selectbox("Tags", TAGS,
            help="CRM tag assigned to this lead")

    # Row 5
    r5a, r5b = st.columns(2)
    with r5a:
        country = st.selectbox("Country", COUNTRIES)
    with r5b:
        city = st.selectbox("City", CITIES)

    # Row 6 — numeric
    time_on_site = st.slider(
        "Total Time Spent on Website (minutes)",
        min_value = 0,
        max_value = 2500,
        value     = 200,
        step      = 10,
        help      = "Converted leads median: 832 min | Not converted: 179 min"
    )
    st.caption(f"📊 Converted leads average 832 min · Not converted: 179 min · "
               f"Your input: **{time_on_site} min** "
               f"({'above' if time_on_site >= 832 else 'below'} converted median)")

    # Row 7 — flags
    st.markdown('<div class="section-title">📧 Communication Flags</div>',
                unsafe_allow_html=True)
    fl1, fl2, fl3 = st.columns(3)
    with fl1:
        do_not_email = st.selectbox("Do Not Email", ["No", "Yes"],
            help="Yes = lead opted out of emails. Converts at only 16.1%")
    with fl2:
        through_rec = st.selectbox("Through Recommendations", ["No", "Yes"])
    with fl3:
        digital_ad = st.selectbox("Digital Advertisement", ["No", "Yes"])

    # Score button
    st.markdown("<br>", unsafe_allow_html=True)
    score_btn = st.button("🎯 Calculate Lead Score", use_container_width=True)


# ── RIGHT COLUMN — Result ─────────────────────────────────────
with right:
    st.markdown('<div class="section-title">📊 Lead Score Result</div>',
                unsafe_allow_html=True)

    if score_btn:
        # Build input dict
        input_data = {
            "Total Time Spent on Website"               : time_on_site,
            "Lead Source"                               : lead_source,
            "Lead Origin"                               : lead_origin,
            "Last Activity"                             : last_activity,
            "Last Notable Activity"                     : last_notable,
            "Specialization"                            : specialization,
            "What is your current occupation"           : occupation,
            "What matters most to you in choosing a course": what_matters,
            "Tags"                                      : tags,
            "Country"                                   : country,
            "City"                                      : city,
            "Do Not Email"                              : do_not_email,
            "Through Recommendations"                   : through_rec,
            "Digital Advertisement"                     : digital_ad,
            # Defaults for other binary cols
            "Search"                                    : "No",
            "Magazine"                                  : "No",
            "Newspaper Article"                         : "No",
            "X Education Forums"                        : "No",
            "Newspaper"                                 : "No",
            "Receive More Updates About Our Courses"    : "No",
            "Update me on Supply Chain Content"         : "No",
            "Get updates on DM Content"                 : "No",
            "A free copy of Mastering The Interview"    : "No",
            "I agree to pay the amount through cheque"  : "No",
        }

        score, tier, action = encode_and_score(input_data)

        # Gauge chart
        st.plotly_chart(make_gauge(score),
                        use_container_width=True, config={"displayModeBar": False})

        # Score box
        tier_class = ("hot-box"  if score >= 70 else
                      "warm-box" if score >= 40 else
                      "cold-box")
        st.markdown(f"""
        <div class="score-box {tier_class}">
            <div class="score-number">{score}</div>
            <div class="score-label">{tier}</div>
            <div class="score-action">📌 {action}</div>
        </div>
        """, unsafe_allow_html=True)

        # Key signals
        st.markdown('<div class="section-title">🔍 Key Signals Detected</div>',
                    unsafe_allow_html=True)

        signals = []
        if time_on_site >= 832:
            signals.append(("✅", f"High time on site ({time_on_site} min) — above converted median"))
        elif time_on_site >= 300:
            signals.append(("⚡", f"Moderate time on site ({time_on_site} min)"))
        else:
            signals.append(("❌", f"Low time on site ({time_on_site} min) — below average"))

        if last_activity == "SMS Sent":
            signals.append(("✅", "SMS Sent = FOCUS activity (62.9% conv rate)"))
        elif last_activity == "Had a Phone Conversation":
            signals.append(("✅", "Phone conversation = strong buying signal (73.3%)"))
        elif last_activity == "Email Bounced":
            signals.append(("❌", "Email bounced = weak signal (8% conv rate)"))

        if tags == "Will revert after reading the email":
            signals.append(("✅", "'Will revert' tag = near-perfect signal (96.9%)"))
        elif tags in ["Ringing", "switched off", "invalid number"]:
            signals.append(("❌", f"Tag '{tags}' = low conversion signal"))

        if lead_source in ["Reference", "Welingak Website", "Live Chat"]:
            signals.append(("✅", f"{lead_source} = GROW tier (90%+ conversion)"))
        elif lead_source == "Google":
            signals.append(("✅", "Google = FOCUS source (40% conv, 31% of leads)"))
        elif lead_source in ["Facebook", "bing", "blog"]:
            signals.append(("❌", f"{lead_source} = DROP tier (0% conversion historically)"))

        if do_not_email == "Yes":
            signals.append(("❌", "Email opt-out: converts at only 16.1% vs 40.5%"))

        if occupation == "Working Professional":
            signals.append(("✅", "Working Professional = 91.6% conversion rate"))
        elif occupation == "Student":
            signals.append(("⚡", "Student = 31.1% conversion (below average)"))

        for icon, msg in signals:
            st.markdown(f"{icon} {msg}")

    else:
        # Default empty state
        st.info("👈 Fill in the lead details on the left and click **Calculate Lead Score**")
        st.markdown("""
        **How the score works:**
        - 🔥 **70–100** → HOT lead. Call within 24 hours.
        - ⚡ **40–69** → WARM lead. Email + follow-up this week.
        - ❄️ **0–39** → COLD lead. Automated email only.

        **Model performance:**
        - XGBoost trained on 9,240 leads
        - ROC-AUC = 0.9786 (excellent)
        - Catches 92.3% of all actual conversions
        """)


# ────────────────────────────────────────────────────────────
# BOTTOM SECTION — BATCH SCORING
# ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">📂 Batch Scoring — Upload CSV</div>',
            unsafe_allow_html=True)

st.markdown("Upload `leads_cleaned.csv` or any new leads file to score all leads at once.")

uploaded = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded:
    batch_df = pd.read_csv(uploaded)
    st.success(f"✅ Loaded {len(batch_df):,} leads")

    if st.button("🚀 Score All Leads"):
        scores, tiers, actions = [], [], []

        progress = st.progress(0)
        for i, (_, row) in enumerate(batch_df.iterrows()):
            inp = row.to_dict()
            try:
                s, t, a = encode_and_score(inp)
            except Exception:
                s, t, a = 0.0, "❄️ COLD", "Error scoring this lead"
            scores.append(s)
            tiers.append(t)
            actions.append(a)
            if i % 100 == 0:
                progress.progress(min(i / len(batch_df), 1.0))

        progress.progress(1.0)
        batch_df["Lead_Score"]  = scores
        batch_df["Score_Tier"]  = tiers
        batch_df["Action"]      = actions
        batch_df = batch_df.sort_values("Lead_Score", ascending=False)

        st.markdown("**Score Distribution:**")
        bc1, bc2, bc3 = st.columns(3)
        hot  = (batch_df["Lead_Score"] >= 70).sum()
        warm = ((batch_df["Lead_Score"] >= 40) & (batch_df["Lead_Score"] < 70)).sum()
        cold = (batch_df["Lead_Score"] < 40).sum()
        bc1.metric("🔥 Hot Leads",  f"{hot:,}",  f"{hot/len(batch_df)*100:.1f}%")
        bc2.metric("⚡ Warm Leads", f"{warm:,}", f"{warm/len(batch_df)*100:.1f}%")
        bc3.metric("❄️ Cold Leads", f"{cold:,}", f"{cold/len(batch_df)*100:.1f}%")

        st.dataframe(
            batch_df[["Lead_Score", "Score_Tier", "Action"] +
                      [c for c in ["Lead Source","Last Activity","Tags",
                                   "What is your current occupation",
                                   "Total Time Spent on Website"]
                       if c in batch_df.columns]].head(50),
            use_container_width=True
        )

        csv_out = batch_df.to_csv(index=False)
        st.download_button(
            label     = "⬇️ Download Scored CSV",
            data      = csv_out,
            file_name = "leads_with_scores.csv",
            mime      = "text/csv"
        )

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>X Education Lead Conversion Scorer · "
    "XGBoost Model · AUC 0.9786 · "
    "Built with Streamlit</small></center>",
    unsafe_allow_html=True
)
