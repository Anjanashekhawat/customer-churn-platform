import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import io

# Page Config
st.set_page_config(
    page_title="RetainPulse | AI Churn & Retention Suite",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="collapsed"
)

# --- MODERN CUSTOM CSS (UI / UX STYLING) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hero Header Container */
    .hero-container {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        padding: 35px 30px;
        border-radius: 16px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px -5px rgba(67, 56, 202, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 8px;
        background: linear-gradient(to right, #ffffff, #c7d2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #e0e7ff;
        font-weight: 400;
        max-width: 800px;
        line-height: 1.5;
    }
    
    .hero-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(8px);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 12px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Glassmorphism Card Containers */
    .glass-card {
        background: #ffffff;
        padding: 24px;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .glass-card:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }

    /* Modern Pill Badges */
    .badge-high {
        background-color: #fee2e2;
        color: #dc2626;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #fecaca;
    }
    
    .badge-medium {
        background-color: #fef3c7;
        color: #d97706;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #fde68a;
    }
    
    .badge-low {
        background-color: #dcfce7;
        color: #16a34a;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #bbf7d0;
    }

    /* Action Card */
    .action-card {
        background: #f8fafc;
        border-left: 4px solid #4f46e5;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 12px;
        font-size: 0.95rem;
    }

    /* Styled Action Button */
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 12px 28px;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(79, 70, 229, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# --- HERO BANNER ---
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">⚡ AI-Powered Customer Intelligence</div>
    <div class="hero-title">RetainPulse™ Retention & Churn Analytics</div>
    <div class="hero-subtitle">
        Evaluate customer churn probabilities in real time, extract game-theory root cause drivers (SHAP), and automatically generate tailored retention interventions.
    </div>
</div>
""", unsafe_allow_html=True)

# Load AI model
@st.cache_resource
def load_model():
    with open("churn_model.pkl", "rb") as f:
        return pickle.load(f)

artifacts = load_model()
model = artifacts["model"]
preprocessor = artifacts["preprocessor"]
feature_names = artifacts["feature_names"]
explainer = artifacts["explainer"]

def get_retention_actions(row, prob):
    if prob < 0.35:
        return "Customer loyal. Offer loyalty perks or feature upsells."
    actions = []
    if row["ComplaintsCount"] >= 3:
        actions.append("Escalate to Senior Support (24h callback)")
    if row["Contract"] == "Month-to-month":
        actions.append("Offer 15% discount for 1-year contract transition")
    if row["MonthlyCharges"] > 80:
        actions.append("Suggest plan optimization bundle")
    if row["LatePayments"] >= 2:
        actions.append("Provide flexible payment schedule")
    if not actions:
        actions.append("Issue $10 courtesy loyalty credit")
    return " | ".join(actions)

tab1, tab2 = st.tabs(["👤 Individual Customer Diagnosis", "📁 Enterprise Batch Assessment"])

# ==========================================
# TAB 1: SINGLE CUSTOMER ANALYSIS
# ==========================================
with tab1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📋 Customer Attributes & Telemetry")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**👤 Demographics**")
        name = st.text_input("Customer Name", value="Alex Morgan")
        age = st.slider("Age", 18, 80, 42)
        tenure = st.slider("Tenure (Months with Company)", 1, 72, 8)

    with col2:
        st.markdown("**📄 Subscription & Telemetry**")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        data_usage = st.number_input("Monthly Data Usage (GB)", value=210.0, step=10.0)

    with col3:
        st.markdown("**💳 Financials & Support Tickets**")
        monthly_charges = st.number_input("Monthly Bill ($)", value=98.0, step=5.0)
        late_payments = st.slider("Late Payments Count", 0, 5, 1)
        complaints = st.slider("Support Interactions / Complaints", 0, 10, 4)

    st.markdown('</div>', unsafe_allow_html=True)

    input_df = pd.DataFrame([{
        "Age": age, "TenureMonths": tenure, "Contract": contract,
        "InternetService": internet, "MonthlyDataGB": data_usage,
        "MonthlyCharges": monthly_charges, "LatePayments": late_payments,
        "ComplaintsCount": complaints
    }])

    if st.button("🚀 Run Risk & Retention Diagnosis", type="primary"):
        # Prediction
        processed = preprocessor.transform(input_df)
        prob = float(model.predict_proba(processed)[0][1])

        # SHAP Reasons
        shap_vals = explainer.shap_values(processed)[0]
        reasons = []
        for feat, val in zip(feature_names, shap_vals):
            if val > 0:
                clean_f = feat.replace("cat__", "").replace("num__", "")
                reasons.append((clean_f, val))
        reasons = sorted(reasons, key=lambda x: x[1], reverse=True)[:3]

        # Results Display
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        r_col1, r_col2 = st.columns([1, 1.2])

        with r_col1:
            st.markdown("### 📊 Churn Probability")
            if prob >= 0.70:
                st.markdown('<span class="badge-high">🚨 CRITICAL RISK (High Churn)</span>', unsafe_allow_html=True)
            elif prob >= 0.40:
                st.markdown('<span class="badge-medium">⚡ ELEVATED RISK (Monitor)</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-low">✅ HEALTHY (Low Churn)</span>', unsafe_allow_html=True)

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={'suffix': "%", 'font': {'color': "#1e1b4b", 'size': 42}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#cbd5e1"},
                    'bar': {'color': "#ef4444" if prob > 0.5 else "#10b981"},
                    'bgcolor': "white",
                    'steps': [
                        {'range': [0, 40], 'color': "#f0fdf4"},
                        {'range': [40, 70], 'color': "#fffbeb"},
                        {'range': [70, 100], 'color': "#fef2f2"}
                    ]
                }
            ))
            fig.update_layout(height=230, margin=dict(l=15, r=15, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with r_col2:
            st.markdown("### 🔍 Root-Cause Attribution (SHAP)")
            st.write("Specific attributes pushing this customer toward churn:")
            if reasons:
                for r, v in reasons:
                    st.markdown(f"""
                    <div style="background: #f1f5f9; padding: 10px 14px; border-radius: 8px; margin-bottom: 8px; border-left: 3px solid #6366f1;">
                        <strong>{r}</strong> &nbsp;—&nbsp; <span style="color: #dc2626; font-weight:600;">+{v:.2f} Risk Impact</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No adverse churn drivers detected. All metrics within healthy thresholds.")

        st.markdown('</div>', unsafe_allow_html=True)

        # Retention Action & Outreach Plan
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 💡 Prescribed Retention Strategy & Draft Email")
        retention_advice = get_retention_actions(input_df.iloc[0], prob)
        
        st.markdown(f"""
        <div class="action-card">
            <strong>🎯 Recommended Tactical Intervention:</strong><br/>
            {retention_advice}
        </div>
        """, unsafe_allow_html=True)

        email_body = f"""Subject: Dedicated Check-in & Special Consideration from Customer Success

Dear {name},

We noticed you've recently had to reach out to our support department {complaints} time(s). We want to personally apologize if any service aspect didn't meet your standard.

Because we deeply value your partnership with us, we'd like to extend:
👉 A complimentary review of your account to optimize your ${monthly_charges:.2f}/month plan, along with a 15% loyalty credit applied to your next cycle.

Please reply directly to this message or schedule time with our senior retention specialist.

Warm regards,
Customer Success Operations"""

        st.text_area("Ready-to-Send Retention Outreach Draft:", value=email_body, height=210)
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 2: BATCH CSV ANALYSIS
# ==========================================
with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📁 Bulk Account Risk Evaluation")
    st.write("Upload an account roster in CSV format to rank high-risk customers and download an operational intervention list.")

    sample_data = pd.DataFrame({
        "CustomerID": ["C-101", "C-102", "C-103", "C-104", "C-105"],
        "CustomerName": ["Sarah Connor", "John Doe", "Emma Watson", "Bruce Wayne", "Clark Kent"],
        "Age": [34, 52, 28, 45, 38],
        "TenureMonths": [4, 36, 12, 2, 48],
        "Contract": ["Month-to-month", "Two year", "Month-to-month", "Month-to-month", "One year"],
        "InternetService": ["Fiber optic", "DSL", "Fiber optic", "Fiber optic", "DSL"],
        "MonthlyDataGB": [320.0, 150.0, 280.0, 450.0, 190.0],
        "MonthlyCharges": [105.0, 60.0, 92.0, 115.0, 65.0],
        "LatePayments": [2, 0, 1, 3, 0],
        "ComplaintsCount": [5, 0, 3, 4, 1]
    })
    
    csv_buffer = io.StringIO()
    sample_data.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Sample Batch Template (CSV)",
        data=csv_buffer.getvalue(),
        file_name="sample_customers.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        predict_cols = ["Age", "TenureMonths", "Contract", "InternetService", "MonthlyDataGB", "MonthlyCharges", "LatePayments", "ComplaintsCount"]
        
        try:
            X_batch = preprocessor.transform(batch_df[predict_cols])
            probs = model.predict_proba(X_batch)[:, 1]

            batch_df["ChurnProbability"] = np.round(probs * 100, 1)
            batch_df["RiskLevel"] = pd.cut(
                batch_df["ChurnProbability"],
                bins=[-1, 40, 70, 100],
                labels=["Low Risk", "Medium Risk", "High Risk"]
            )
            batch_df["SuggestedAction"] = [
                get_retention_actions(row, p/100) for row, p in zip(batch_df.to_dict('records'), batch_df["ChurnProbability"])
            ]

            m1, m2, m3 = st.columns(3)
            high_risk_count = (batch_df["RiskLevel"] == "High Risk").sum()
            med_risk_count = (batch_df["RiskLevel"] == "Medium Risk").sum()
            low_risk_count = (batch_df["RiskLevel"] == "Low Risk").sum()

            m1.metric("🚨 High Risk", high_risk_count)
            m2.metric("⚡ Medium Risk", med_risk_count)
            m3.metric("✅ Low Risk", low_risk_count)

            sorted_df = batch_df.sort_values(by="ChurnProbability", ascending=False)
            st.dataframe(sorted_df[["CustomerID", "CustomerName", "ChurnProbability", "RiskLevel", "SuggestedAction", "MonthlyCharges", "ComplaintsCount"]], use_container_width=True)

            export_csv = io.StringIO()
            sorted_df.to_csv(export_csv, index=False)
            st.download_button(
                label="💾 Download Prioritized Retention Action Plan (CSV)",
                data=export_csv.getvalue(),
                file_name="retention_action_plan.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error parsing file: {e}")

    st.markdown('</div>', unsafe_allow_html=True)
