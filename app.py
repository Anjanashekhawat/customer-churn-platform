import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import io

st.set_page_config(page_title="Customer Retention Platform", layout="wide", page_icon="🏢")

# Title & Header
st.title("🏢 Enterprise Churn & Retention Analytics Platform")
st.markdown("Predict customer churn, identify root causes, and prescribe retention strategies.")

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

# Helper function for retention strategies
def get_retention_actions(row, prob):
    if prob < 0.35:
        return "Customer loyal. Offer loyalty perks or feature upsells."
    actions = []
    if row["ComplaintsCount"] >= 3:
        actions.append("Escalate to Senior Support (24h callback)")
    if row["Contract"] == "Month-to-month":
        actions.append("Offer 15% discount for 1-year contract")
    if row["MonthlyCharges"] > 80:
        actions.append("Suggest plan optimization bundle")
    if row["LatePayments"] >= 2:
        actions.append("Provide flexible payment schedule")
    if not actions:
        actions.append("Issue $10 courtesy loyalty credit")
    return " | ".join(actions)

# Create 2 Tabs
tab1, tab2 = st.tabs(["👤 Single Customer Analysis", "📁 Batch CSV Analysis (Bulk)"])

# ==========================================
# TAB 1: SINGLE CUSTOMER ANALYSIS
# ==========================================
with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Profile**")
        name = st.text_input("Customer Name", value="Alex Morgan")
        age = st.slider("Age", 18, 80, 42)
        tenure = st.slider("Tenure (Months with Company)", 1, 72, 8)

    with col2:
        st.markdown("**Subscription & Usage**")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        data_usage = st.number_input("Monthly Data Usage (GB)", value=210.0, step=10.0)

    with col3:
        st.markdown("**Billing & Complaints**")
        monthly_charges = st.number_input("Monthly Bill ($)", value=98.0, step=5.0)
        late_payments = st.slider("Late Payments Count", 0, 5, 1)
        complaints = st.slider("Support Complaints", 0, 10, 4)

    input_df = pd.DataFrame([{
        "Age": age, "TenureMonths": tenure, "Contract": contract,
        "InternetService": internet, "MonthlyDataGB": data_usage,
        "MonthlyCharges": monthly_charges, "LatePayments": late_payments,
        "ComplaintsCount": complaints
    }])

    if st.button("🚀 Analyze Single Customer", type="primary"):
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

        # Display results
        r_col1, r_col2 = st.columns([1, 1.2])
        with r_col1:
            st.subheader("Churn Risk Gauge")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={'suffix': "%"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#EF4444" if prob > 0.5 else "#10B981"},
                    'steps': [
                        {'range': [0, 40], 'color': "#D1FAE5"},
                        {'range': [40, 70], 'color': "#FEF3C7"},
                        {'range': [70, 100], 'color': "#FEE2E2"}
                    ]
                }
            ))
            fig.update_layout(height=240, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with r_col2:
            st.subheader("🔍 Primary Drivers For Risk")
            if reasons:
                for r, v in reasons:
                    st.write(f"• **{r}** is pushing the churn risk higher.")
            else:
                st.write("No severe risk factors.")

        st.divider()

        # Retention Action & Ready Email Draft
        st.subheader("✉️ Auto-Drafted Retention Email")
        retention_advice = get_retention_actions(input_df.iloc[0], prob)
        
        st.info(f"**Recommended Action:** {retention_advice}")

        # Email draft
        email_body = f"""Subject: Special Offer & Personal Check-in from your Customer Success Team

Dear {name},

We noticed you've recently had to reach out to our support team {complaints} time(s). We sincerely apologize if any experience fell short of your expectations.

Because we value your partnership with us, we would love to offer you:
👉 A complimentary review of your account to optimize your ${monthly_charges:.2f}/month plan, plus a 15% loyalty credit.

Please reply directly to this email or pick a time with our dedicated specialist. We are here to help!

Warm regards,
Customer Success Team"""

        st.text_area("Copy-paste ready email for your support team:", value=email_body, height=220)

# ==========================================
# TAB 2: BATCH CSV ANALYSIS
# ==========================================
with tab2:
    st.subheader("📁 Upload Customer List (CSV)")
    st.write("Upload your customer records to identify all high-risk accounts and export an action plan.")

    # Button to download a sample template
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
        label="📥 Download Sample CSV Template",
        data=csv_buffer.getvalue(),
        file_name="sample_customers.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        
        # Keep copy for predictions
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

            # Summary Metrics
            m1, m2, m3 = st.columns(3)
            high_risk_count = (batch_df["RiskLevel"] == "High Risk").sum()
            med_risk_count = (batch_df["RiskLevel"] == "Medium Risk").sum()
            low_risk_count = (batch_df["RiskLevel"] == "Low Risk").sum()

            m1.metric("🚨 High Risk Customers", high_risk_count)
            m2.metric("⚡ Medium Risk Customers", med_risk_count)
            m3.metric("✅ Low Risk Customers", low_risk_count)

            # Sort so most endangered customers appear at the top
            sorted_df = batch_df.sort_values(by="ChurnProbability", ascending=False)
            st.dataframe(sorted_df[["CustomerID", "CustomerName", "ChurnProbability", "RiskLevel", "SuggestedAction", "MonthlyCharges", "ComplaintsCount"]])

            # Export button
            export_csv = io.StringIO()
            sorted_df.to_csv(export_csv, index=False)
            st.download_button(
                label="💾 Download Prioritized Retention Action Plan (CSV)",
                data=export_csv.getvalue(),
                file_name="retention_action_plan.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error processing file. Make sure columns match the template! Error: {e}")