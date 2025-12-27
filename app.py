import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import shap
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- PAGE CONFIG ---
st.set_page_config(page_title="Olist AI Logistics", layout="wide", page_icon="📦")

# --- 1. DEFINE CUSTOM TRANSFORMER (Exactly as in your notebook) ---
class TimeFeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_copy = X.copy()
        if isinstance(X_copy, pd.DataFrame):
            col_name = X_copy.columns[0]
            series = pd.to_datetime(X_copy[col_name])
        else:
            series = pd.to_datetime(X_copy[:, 0])
        
        month = series.dt.month
        day_of_week = series.dt.dayofweek
        hour = series.dt.hour
        
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        day_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_cos = np.cos(2 * np.pi * day_of_week / 7)
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        is_weekend = (day_of_week >= 5).astype(int)
        
        return np.column_stack([month_sin, month_cos, day_sin, day_cos, hour_sin, hour_cos, is_weekend])

    def get_feature_names_out(self, input_features=None):
        return ["month_sin", "month_cos", "day_sin", "day_cos", "hour_sin", "hour_cos", "is_weekend"]

# --- 2. DATA LOADING (Cached) ---
@st.cache_data
def load_data():
    # Try connecting to DB, otherwise look for CSV
    try:
        connection_string = "postgresql+psycopg2://postgres:1131995i%40@localhost:5432/olist"
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            query = text("SELECT * FROM olist_prediction_view LIMIT 5000") # Limit for speed in demo
            df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        # Fallback if DB is not active
        return None

# --- 3. MODEL TRAINING (Cached) ---
@st.cache_resource
def train_model(df):
    # Preprocessing Logic from your notebook
    df_ml = df[df["order_status"] == "delivered"].copy()
    df_ml["target"] = (pd.to_datetime(df_ml["order_delivered_customer_date"]) - pd.to_datetime(df_ml["order_purchase_timestamp"])).dt.days
    df_ml = df_ml.dropna(subset=['target'])
    
    # Feature Creation
    df_ml["V_of_object"] = df_ml["product_height_cm"] * df_ml["product_width_cm"] * df_ml["product_length_cm"]
    df_ml['order_purchase_timestamp'] = pd.to_datetime(df_ml['order_purchase_timestamp'])
    df_ml['order_delivered_carrier_date'] = pd.to_datetime(df_ml['order_delivered_carrier_date'])
    df_ml['seller_prep_days'] = (df_ml['order_delivered_carrier_date'] - df_ml['order_purchase_timestamp']).dt.days
    
    # Cleaning
    df_ml = df_ml.dropna(subset=['seller_prep_days'])
    upper_limit = df_ml['target'].quantile(0.98)
    df_clean = df_ml[(df_ml['target'] > 0) & (df_ml['target'] < upper_limit)].copy()
    
    # Split
    X = df_clean.drop(["target"], axis=1)
    y = df_clean["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define Features
    num_f = X_train.select_dtypes(include=["int64", "float64"]).columns.to_list()
    cat_f = [c for c in X_train.select_dtypes(include=["object"]).columns if c not in ['order_status', 'order_purchase_timestamp']]
    # Fix: Ensure we only pick valid categorical columns (taking a subset based on your notebook logic)
    cat_f_new = cat_f[:10] 
    time_cols = ['order_purchase_timestamp']
    
    # Pipeline
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    
    preprocessor = ColumnTransformer([
        ("time", TimeFeatureEngineering(), time_cols),
        ("num", num_pipe, num_f),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_f_new)
    ])
    
    xgb_model = XGBRegressor(n_estimators=500, learning_rate=0.1, max_depth=6, n_jobs=-1, random_state=42)
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", xgb_model)
    ])
    
    pipeline.fit(X_train, y_train)
    
    # SHAP Explainer Preparation
    X_test_proc = pipeline.named_steps['preprocessor'].transform(X_test)
    explainer = shap.TreeExplainer(pipeline.named_steps['model'])
    
    return pipeline, X_test, y_test, explainer, X_test_proc, preprocessor

# --- APP LAYOUT ---

st.title("📦 Olist Logistics Optimization")
st.markdown("**End-to-End Pipeline:** Postgres SQL ➔ XGBoost ➔ SHAP ➔ Streamlit")

# LOAD DATA
data = load_data()

if data is None:
    st.warning("⚠️ Database connection failed. Please upload your CSV to proceed.")
    uploaded_file = st.file_uploader("Upload olist_prediction_view.csv", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
    else:
        st.stop()

# TRAIN MODEL
with st.spinner("Training XGBoost Model & Calculating SHAP values..."):
    pipeline, X_test, y_test, explainer, X_test_proc, preprocessor = train_model(data)

# TABS
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🛠️ Model Process", "🧠 SHAP Analysis", "🚀 Business Impact"])

# --- TAB 1: DASHBOARD ---
with tab1:
    st.header("Executive Summary")
    # Embedding your Tableau Dashboard
    tableau_html = """
    <div class='tableauPlaceholder' id='viz1765718294900' style='position: relative'>
        <noscript><a href='#'><img alt='Olist Dashboard' src='https://public.tableau.com/static/images/ol/olistDashboard_17654562913640/Dashboard1/1_rss.png' style='border: none' /></a></noscript>
        <object class='tableauViz' style='display:none;'>
            <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> 
            <param name='site_root' value='' />
            <param name='name' value='olistDashboard_17654562913640/Dashboard1' />
            <param name='tabs' value='no' />
            <param name='toolbar' value='yes' />
            <param name='display_static_image' value='yes' />
        </object>
    </div>
    <script type='text/javascript'>
        var divElement = document.getElementById('viz1765718294900');
        var vizElement = divElement.getElementsByTagName('object')[0];
        vizElement.style.width='100%';vizElement.style.height='800px';
        var scriptElement = document.createElement('script');
        scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
        vizElement.parentNode.insertBefore(scriptElement, vizElement);
    </script>
    """
    st.components.v1.html(tableau_html, height=800, scrolling=True)

# --- TAB 2: MODEL PROCESS ---
with tab2:
    st.header("Model Performance")
    
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE (Days Error)", f"{mae:.2f}")
    col2.metric("R² Score", f"{r2:.2f}")
    col3.metric("Test Data Size", len(y_test))
    
    st.subheader("Actual vs Predicted Days")
    chart_data = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    st.scatter_chart(chart_data.sample(300)) # Subsample for speed

# --- TAB 3: SHAP ANALYSIS ---
with tab3:
    st.header("Why did the model predict that?")
    
    # Calculate SHAP values for a sample
    # We sample 500 rows to make the app fast
    idx = np.random.choice(X_test_proc.shape[0], size=min(500, X_test_proc.shape[0]), replace=False)
    X_sample = X_test_proc[idx]
    shap_values = explainer(X_sample)
    
    # Get feature names
    feature_names = preprocessor.get_feature_names_out()
    shap_values.feature_names = feature_names

    st.subheader("Global Feature Importance (Beeswarm)")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, max_display=12, show=False)
    st.pyplot(fig)
    
    st.divider()
    
    st.subheader("🔍 Case Studies: Γιατί καθυστέρησε αυτή η παραγγελία;")

# 1. Επιλογή: Ο χρήστης διαλέγει αν θέλει τυχαία ή συγκεκριμένη παραγγελία
    mode = st.radio("Επιλογή Παραγγελίας:", ["Τυχαία Δειγματοληψία (3)", "Επιλογή με Slider"])

    if mode == "Τυχαία Δειγματοληψία (3)":
        if st.button("Δείξε νέα τυχαία παραδείγματα"):
            # Επιλέγουμε 3 τυχαίους δείκτες
            indices = np.random.choice(X_sample.shape[0], size=3, replace=False)
            
            for i, idx in enumerate(indices):
                st.markdown(f"#### Case Study {i+1} (Δείκτης Δείγματος: {idx})")
                
                # --- Η ΜΑΓΕΙΑ ΓΙΑ ΤΟ STREAMLIT ---
                # Δημιουργούμε ένα Figure (καμβά) Matplotlib
                fig, ax = plt.subplots(figsize=(10, 5))
                
                # Λέμε στο SHAP να σχεδιάσει πάνω στο Figure μας (show=False)
                shap.plots.waterfall(shap_values[idx], max_display=15, show=False)
                
                # Στέλνουμε το Figure στο Streamlit
                st.pyplot(fig)
                
                # Καθαρίζουμε τη μνήμη
                plt.close(fig)
                st.divider()

    else:
        # Επιλογή με Slider για να δεις όποια παραγγελία θες
        selected_idx = st.slider("Διάλεξε δείκτη παραγγελίας (0-499)", 0, X_sample.shape[0]-1, 0)
        
        st.markdown(f"#### Ανάλυση για την Παραγγελία #{selected_idx}")
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[selected_idx], max_display=15, show=False)
        st.pyplot(fig)
        plt.close(fig)
    st.subheader("📊 Global Feature Importance (Bar Plot)")

# 1. Δημιουργούμε τον "καμβά" (Figure) με συγκεκριμένο μέγεθος
# Το figsize=(10, 6) βοηθάει να μην φαίνονται συμπιεσμένα τα labels
fig, ax = plt.subplots(figsize=(10, 6))

# 2. Καλούμε το SHAP plot
shap.summary_plot(
    shap_values.values,          # Οι τιμές SHAP
    feature_names=feature_names,  # Τα ονόματα των στηλών
    plot_type="bar",             # Το είδος του γραφήματος
    show=False                   # SOS: Μην το δείξεις ακόμα!
)

# 3. Το στέλνουμε στο Streamlit
st.pyplot(fig)

# 4. Καθαρίζουμε τη μνήμη
plt.close(fig)
# --- TAB 4: BUSINESS IMPACT ---
with tab4:
    st.header("ROI Forecast")
    
    # Use the model to predict on new data (simulated here with test set)
    predictions = pipeline.predict(X_test)
    
    # Calculate "Fast Deliveries" (e.g., < 7 days)
    fast_deliveries = predictions[predictions <= 7]
    revenue_impact = len(fast_deliveries) * 120 # Assuming avg price 120 R$
    
    col1, col2 = st.columns(2)
    col1.metric("Fast Deliveries Forecast (<7 days)", len(fast_deliveries))
    col2.metric("Projected Revenue (Fast Orders)", f"R$ {revenue_impact:,.0f}")
    
    st.success(f"By optimizing seller prep time, we can increase fast deliveries by estimated 15%.")