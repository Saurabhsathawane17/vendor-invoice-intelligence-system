import streamlit as st
import os
from src.predict import predict_invoice_risk

st.set_page_config(page_title="Vendor Invoice Intelligence", page_icon="🛡️", layout="centered")

st.title("🛡️ Vendor Invoice Intelligence System")
st.markdown("Predict expected freight costs and detect anomalous or risky vendor invoices instantly.")

st.sidebar.header("📦 Invoice Details")

# Sidebar Inputs
quantity = st.sidebar.number_input("Quantity", min_value=1, value=150, step=10)
unit_price = st.sidebar.number_input("Unit Price ($)", min_value=0.01, value=2.50, step=0.5)
provided_freight = st.sidebar.number_input("Provided Freight Cost ($) [Optional]", min_value=0.0, value=0.0, step=1.0, help="Leave at 0 if you only want to estimate the cost.")

st.sidebar.markdown("---")
st.sidebar.header("🏢 Vendor History (Optional)")
vendor_avg_cost = st.sidebar.number_input("Vendor Avg Historical Cost ($)", min_value=0.0, value=0.0, step=10.0)
transaction_count = st.sidebar.number_input("Vendor Transaction Count", min_value=0, value=0, step=1)

if st.sidebar.button("Analyze Invoice", type="primary"):
    input_data = {
        'Quantity': quantity,
        'UnitPrice': unit_price
    }
    
    if provided_freight > 0:
        input_data['freight_cost'] = provided_freight
        
    if vendor_avg_cost > 0:
        input_data['vendor_avg_cost'] = vendor_avg_cost
        
    if transaction_count > 0:
        input_data['transaction_count'] = transaction_count

    models_dir = 'models/'
    
    if not os.path.exists(os.path.join(models_dir, 'freight_regressor.pkl')):
        st.error("Models not found! Please run `python main.py` first to train and save the models.")
    else:
        with st.spinner("Running AI Analysis..."):
            result = predict_invoice_risk(input_data, models_dir)
        
        st.subheader("📊 Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        
        total_cost = quantity * unit_price
        col1.metric("Total Order Value", f"${total_cost:,.2f}")
        col2.metric("AI Expected Freight", f"${result['expected_freight_cost']:,.2f}")
        
        if provided_freight > 0:
            cost_diff = result['provided_freight_cost'] - result['expected_freight_cost']
            col3.metric("Provided Freight", f"${result['provided_freight_cost']:,.2f}", 
                        delta=f"${cost_diff:,.2f} vs expected", delta_color="inverse")
        else:
            col3.metric("Provided Freight", "N/A")
        
        st.markdown("---")
        if result['risk_flag'] == 1:
            st.error("⚠️ **RISK DETECTED:** This invoice has been flagged as anomalous. The freight overhead is unusually high compared to the total cost and historical data. Please review manually.")
        else:
            st.success("✅ **SAFE:** This invoice appears to be completely within normal parameters.")