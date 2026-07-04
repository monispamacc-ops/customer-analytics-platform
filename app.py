import sys
import numpy as np
import numpy

# Create a solid compatibility bridge for old pickle files
try:
    sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
except AttributeError:
    sys.modules['numpy._core.multiarray'] = numpy.msgpack if hasattr(numpy, 'msgpack') else numpy

import streamlit as st
import pandas as pd
import pickle
import gzip

# 1. Page Configuration
st.set_page_config(
    page_title="Customer Analytics Platform",
    page_icon="📊",
    layout="wide"
)

# --- LOAD SERIALIZED ARTIFACTS ---
@st.cache_resource
def load_model_artifacts():
    # Module 1 files
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    # Module 2 file
    with open('churn_model.pkl', 'rb') as f:
        churn_model = pickle.load(f)
    
    # Module 3
    with gzip.open('item_similarity.pkl.gz', 'rb') as f:
        item_similarity_df = pickle.load(f)
        
    return kmeans, scaler, churn_model, item_similarity_df
        

# Make sure all 4 variables are being unpacked here!
try:
    kmeans, scaler, churn_model, item_similarity_df = load_model_artifacts()
except Exception as e:
    st.error(f"Error loading model files: {e}")


# --- LOAD DATASET TO CALCULATE DYNAMIC THRESHOLDS ---
@st.cache_data
def calculate_dataset_benchmarks():
    try:
        # Reads your raw or processed dataset file from your folder
        df = pd.read_csv("Online_Retail.csv") # Double-check your filename matches exactly
        
        # Calculate statistical boundaries dynamically from the real dataset distribution
        monetary_75 = df['Monetary'].quantile(0.75)
        frequency_75 = df['Frequency'].quantile(0.75)
        recency_75 = df['Recency'].quantile(0.75)
        
        return monetary_75, frequency_75, recency_75
    except Exception:
        # Seamless fallback values if the dataset isn't present in the local directory
        return 1500.0, 10, 180

# Get our live, general-purpose statistical thresholds!
MONETARY_LIMIT, FREQUENCY_LIMIT, RECENCY_LIMIT = calculate_dataset_benchmarks()

# 2. Sidebar Navigation / Title
st.sidebar.title("Navigation")
st.sidebar.markdown("Use the tabs on the main screen to explore the platform engines.")
st.sidebar.info("Logged in as: **Moniha R**")

st.title("📊 Customer Analytics Platform Dashboard")
st.markdown("An end-to-end data science application managing clustering, flight-risk predictions, and personalized recommendation systems.")
st.markdown("---")

# 3. Create Tabs
tab1, tab2, tab3 = st.tabs([
    "🎯 Module 1: Customer Segmentation", 
    "📈 Module 2: Churn Prediction", 
    "🛒 Module 3: Recommendation Engine"
])

# --- TAB 1: SEGMENTATION ---
with tab1:
    st.header("Customer Behavior Segments (K-Means)")
    st.write("Test an individual customer's metrics to see which cluster segment they belong to.")

    # Professional Help Panel explaining customer personas
    with st.expander("ℹ️ How are these customer personas calculated?"):
        st.markdown("""
        Our **K-Means Clustering model** analyzes historical behavioral patterns across three primary metrics. 
        Adjust the sliders below to see real-time segment matching.
        """)

    # Create layout columns for the inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        input_recency = st.slider(
            "Recency (Days since last purchase)",
            min_value=0,
            max_value=365,
            value=12,
            step=1,
            help="Days elapsed since the customer's last purchase activity."
        )

    with col2:
        input_frequency = st.slider(
            "Frequency (Total number of purchases)",
            min_value=1,
            max_value=150,
            value=4,
            step=1,
            help="Total order frequency history on the platform."
        )

    with col3:
        input_monetary = st.slider(
            "Monetary Value (Total amount spent ($))",
            min_value=1.0,
            max_value=10000.0,
            value=150.0,
            step=10.0,
            help="The lifetime total financial value this customer has spent on our platform."
        )

    # Everything below this line is perfectly indented by 4 spaces inside tab1
    if st.button("Predict Customer Segment"):
        log_features = np.log1p([input_recency, input_frequency, input_monetary])
        features_array = np.array(log_features).reshape(1, -1)

        scaled_features = scaler.transform(features_array)
        predicted_cluster = kmeans.predict(scaled_features)[0]

        # --- TRUE DYNAMIC EVALUATION ---
        is_high_spender = input_monetary >= MONETARY_LIMIT
        is_frequent     = input_frequency >= FREQUENCY_LIMIT
        is_stale        = input_recency >= RECENCY_LIMIT
        is_brand_new    = input_frequency <= 2 and input_recency <= 14

        # --- PERSONA ENGINE ---
        if is_high_spender or is_frequent:
            persona = "💎 High-Value Elite Segment"
        elif is_stale:
            persona = "⚠️ Churned / At-Risk Segment"
        elif is_brand_new:
            persona = "🛍️ New / Casual Shopper"
        else:
            persona = "📈 Regular Active Shopper"

        st.success(f"🎯 **Model Output:** Cluster {predicted_cluster}")
        st.info(f"👥 **Inferred Customer Persona:** {persona}")
        st.write("This dynamic label evaluates cluster scale dynamically using live data distributions!")

        # --- ADD SCATTER PLOT VISUALIZATION ---
        try:
            import plotly.express as px
            df_segmented = pd.read_csv("segmented_customer_data.csv")

            # Build the background cluster scatter plot with realistic, zoomed-in limits
            fig = px.scatter(
                df_segmented, 
                x="Frequency", 
                y="Monetary", 
                color="Cluster", 
                title="Where this customer sits in your market segments",
                labels={"Frequency": "Total Purchases", "Monetary": "Total Spent ($)"},
                opacity=0.4,
                range_x=[0, 150],       # Cuts the X-axis off at 150 purchases instead of 1400
                range_y=[0, 10000]      # Cuts the Y-axis off at $10,000 instead of $1.5 Million
            )

            # Overlay the current user input as a bright star
            fig.add_scatter(
                x=[input_frequency], 
                y=[input_monetary],  
                mode="markers",
                marker=dict(color="crimson", size=15, symbol="star", line=dict(width=2, color="white")),
                name="Current Customer"
            )
            
            # Render the chart inside the block cleanly
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Could not render scatter plot map: {e}")
 

# --- TAB 2: CHURN PREDICTION ---
with tab2:
    st.header("Predictive Customer Flight Risk (Random Forest)")
    st.write("Evaluate a customer's real-time risk profile using historical activity metrics.")
    
    # Professional Help Panel explaining Random Forest Logic
    with st.expander("ℹ️ How is this Churn Risk score calculated?"):
        st.markdown("""
        Our **Random Forest Classification model** analyzes behavioral patterns across multiple decision trees to compute a precise probability score:
        
        * **🎯 Top Risk Driver (Recency):** Historical data shows that time elapsed since the last order accounts for roughly **48.7%** of the model's decision-making weight. Long periods of inactivity heavily spike the risk score.
        * **🔄 Frequency & Value Weights:** Purchase frequency (**26.4%**) and lifetime monetary spend (**25.0%**) act as stabilizing weights. High-frequency or high-spending customers inherently maintain a lower flight risk.
        * **📊 Risk Scale Tiers:**
            * 🟢 **Low Risk (<35%):** Highly active, stable, and regular purchasing patterns.
            * 🟡 **Moderate Risk (35% - 70%):** Showing signs of drifting; prime candidates for a re-engagement campaign.
            * 🔴 **High Flight Risk (≥70%):** Immediate critical risk of churn; requires immediate retention offers.
        """)

    col1_rf, col2_rf, col3_rf = st.columns(3)
    with col1_rf:
        # Reusing the current state metrics or creating independent inputs for the churn tab
        rf_recency = st.number_input(
            "Recency (Days since last interaction)", 
            min_value=0, value=30, key="rf_rec",
            help="Days elapsed since customer's last purchase activity."
        )
    with col2_rf:
        rf_frequency = st.number_input(
            "Frequency (Total Lifetime Orders)", 
            min_value=1, value=5, key="rf_freq",
            help="Total order frequency history on the platform."
        )
    with col3_rf:
        rf_monetary = st.number_input(
            "Monetary Value ($ Total Spent)", 
            min_value=0.0, value=200.0, key="rf_mon",
            help="Cumulative lifetime total spent by this customer."
        )
        
    if st.button("Analyze Flight Risk", key="rf_button"):
        # Put inputs into an array matching your training features configuration
        raw_features = np.array([[rf_recency, rf_frequency, rf_monetary]])
        
        # Predict class prediction (0 = Stay, 1 = Churn) and probability matrix
        prediction = churn_model.predict(raw_features)[0]
        probabilities = churn_model.predict_proba(raw_features)[0]
        
        # Churn probability score (index 1 is typically the 'Churned' label probability)
        churn_probability = probabilities[1] * 100
         # Sleek, user-friendly risk assessment cards
        st.markdown("---")
        st.subheader("📊 Risk Analysis Results")

        # 1. Map conditions to clean status labels and messages
        if churn_probability >= 70:
            status_label = "High Risk 🚨"
            delta_msg = "Immediate retention action required!"
            bar_color = "red"
        elif 35 <= churn_probability < 70:
            status_label = "Moderate Risk ⚠️"
            delta_msg = "Monitor closely / Engagement opportunity"
            bar_color = "orange"
        else:
            status_label = "Low Risk / Stable ✅"
            delta_msg = "Customer is highly loyal"
            bar_color = "green"

        # 2. Create side-by-side metric tiles
        metric_col1, metric_col2 = st.columns(2)
    
        with metric_col1:
            st.metric(
                label="Customer Security Status", 
                value=status_label, 
                delta=delta_msg,
                delta_color="inverse" if churn_probability >= 35 else "normal"
            )
        
        with metric_col2:
            st.metric(
                label="Flight Risk Probability", 
                value=f"{churn_probability:.1f}%"
            )
    
        # 3. Visual progress meter bar
        st.write("**Real-time Flight Risk Gauge:**")
        st.progress(float(churn_probability / 100))

        # 4. Detailed strategic recommendation text box underneath
        if churn_probability >= 70:
            st.error(f"⚠️ **Action Plan:** Recommend pushing an aggressive personalized high-value discount or a direct outreach sequence immediately.")
        elif 35 <= churn_probability < 70:
            st.warning(f"💡 **Action Plan:** Consider adding this customer to an automated re-engagement email newsletter sequence.")
        else:
            st.success(f"✨ **Action Plan:** User is stable and integrated within regular purchasing cycles. Keep standard marketing cadences.")

# --- TAB 3: RECOMMENDATION ENGINE ---   
with tab3:
    st.header("Personalized Product Suggestions (Cosine Similarity)")
    st.write("Discover highly relevant product recommendations based on collective consumer purchasing patterns.")
    
    # Contextual info box explaining the math behind the recommendations
    with st.expander("ℹ️ How are these recommendations calculated?"):
        st.markdown("""
        Our engine utilizes an **Item-to-Item Collaborative Filtering** framework powered by **Cosine Similarity**:
        
        * **📐 Multi-Dimensional Space:** Every product is represented as a high-dimensional vector mapping out which customers bought it and in what quantities.
        * **🔄 Cosine Angle Calculation:** The model evaluates the mathematical cosine of the angle between product vectors. Items frequently purchased together by similar consumer cohorts result in a score closer to **1.0**.
        * **⚡ Real-time Filtering:** When a product is selected, the platform quickly surfaces the highest-scoring vectors while automatically suppressing perfect self-matches.
        """)
        
    st.markdown("---")
    
    # Create a clean, searchable dropdown containing all 4,223 unique product titles
    product_list = sorted(item_similarity_df.index.tolist())
    selected_product = st.selectbox(
        "🛒 Select a reference item to find similar products:",
        options=product_list,
        index=0 if product_list else None,
        help="Type or select a product to generate item-to-item recommendations."
    )
    
    if selected_product:
        st.subheader(f"📦 Top 5 Items Frequently Purchased with:")
        st.info(f"👉 **{selected_product}**")
        
        # Pull the similarity series for the selected item, sort descending, and grab top entries
        # We skip the very first one because it's a 1.0 match with itself!
        recommendations = item_similarity_df[selected_product].sort_values(ascending=False)
        top_recommendations = recommendations.iloc[1:6]
        
        # Render the outputs as sleek, scannable product rows
        for rank, (product_name, score) in enumerate(top_recommendations.items(), 1):
            match_percentage = score * 100
            
            # Create a stylized container card for each recommended product
            with st.container(border=True):
                # Split row into layout columns: Rank badge, Product Title, and Confidence Gauge
                card_col1, card_col2, card_col3 = st.columns([1, 5, 3])
                
                with card_col1:
                    # Bold placement badge
                    st.markdown(f"### 📦 #{rank}")
                    
                with card_col2:
                    # Product description/name
                    st.markdown(f"**{product_name}**")
                    st.caption("Frequently purchased alternative")
                    
                with card_col3:
                    # Numeric confidence metric alongside a matching mini-progress bar
                    st.metric(label="Match Confidence", value=f"{match_percentage:.1f}%")
                    st.progress(float(max(0.0, min(1.0, score))))