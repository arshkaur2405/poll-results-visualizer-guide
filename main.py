import pandas as pd
import numpy as np
import random
import streamlit as st
import plotly.express as px
from faker import Faker
from textblob import TextBlob

# -----------------------------
# DATA GENERATION
# -----------------------------
def generate_poll_data(n=1000):
    fake = Faker()
    Faker.seed(42)
    np.random.seed(42)

    regions = ['Urban North', 'Rural South', 'Eastern Metro', 'Western Coastal', 'Central']
    age_groups = ['18-24', '25-34', '35-44', '45-54', '55+']
    options = ['Product Alpha', 'Product Beta', 'Product Gamma', 'Product Delta']

    data = []

    for i in range(n):
        reg = random.choice(regions)

        if reg == 'Urban North':
            pref = np.random.choice(options, p=[0.6, 0.1, 0.2, 0.1])
        else:
            pref = random.choice(options)

        data.append({
            'Respondent_ID': i + 1,
            'Timestamp': fake.date_this_year(),
            'Region': reg,
            'Age_Group': random.choice(age_groups),
            'Selected_Option': pref,
            'Satisfaction_Score': random.randint(1, 5),
            'Qualitative_Feedback': fake.sentence(nb_words=12)
        })

    return pd.DataFrame(data)

# -----------------------------
# APPLY WEIGHTS
# -----------------------------
def apply_weights(df):
    targets = {
        'Urban North': 0.30,
        'Rural South': 0.20,
        'Eastern Metro': 0.20,
        'Western Coastal': 0.20,
        'Central': 0.10
    }

    actuals = df['Region'].value_counts(normalize=True).to_dict()

    df['Weight'] = df['Region'].map(lambda x: targets[x] / actuals.get(x, 1))

    return df

# -----------------------------
# MARGIN OF ERROR
# -----------------------------
def get_moe(p, n):
    if n == 0:
        return 0
    return 1.96 * np.sqrt((p * (1 - p)) / n)

# -----------------------------
# STREAMLIT APP
# -----------------------------
def main():
    st.set_page_config(page_title="Poll Visualizer", layout="wide")

    st.title("🗳️ Poll Results Visualizer Dashboard")
    st.markdown("---")

    # Generate & process data
    raw_df = generate_poll_data()
    df = apply_weights(raw_df)

    # -----------------------------
    # SIDEBAR FILTERS
    # -----------------------------
    st.sidebar.header("Filters")

    age_filter = st.sidebar.multiselect(
        "Select Age Group",
        options=df['Age_Group'].unique(),
        default=df['Age_Group'].unique()
    )

    region_filter = st.sidebar.multiselect(
        "Select Region",
        options=df['Region'].unique(),
        default=df['Region'].unique()
    )

    filtered_df = df[
        (df['Age_Group'].isin(age_filter)) &
        (df['Region'].isin(region_filter))
    ].copy()

    # -----------------------------
    # KPI SECTION
    # -----------------------------
    col1, col2 = st.columns(2)

    col1.metric("Total Responses", len(filtered_df))
    col2.metric("Avg Satisfaction", f"{filtered_df['Satisfaction_Score'].mean():.2f} / 5")

    # -----------------------------
    # DOWNLOAD BUTTON
    # -----------------------------
    st.subheader("Download Data")
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Filtered Data", csv, "poll_data.csv", "text/csv")

    # -----------------------------
    # PREFERENCE ANALYSIS
    # -----------------------------
    st.subheader("Preference Analysis")

    pref_summary = (
        filtered_df
        .groupby('Selected_Option')['Weight']
        .sum()
        .reset_index()
    )

    total_weight = pref_summary['Weight'].sum()
    pref_summary['Percentage'] = (pref_summary['Weight'] / total_weight) * 100

    pref_summary['MoE'] = pref_summary['Percentage'].apply(
        lambda x: get_moe(x / 100, len(filtered_df)) * 100
    )

    fig = px.bar(
        pref_summary,
        x='Selected_Option',
        y='Percentage',
        error_y='MoE',
        color='Selected_Option',
        title="Support with 95% Confidence Interval"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # WINNER HIGHLIGHT
    # -----------------------------
    top_option = pref_summary.loc[
        pref_summary['Percentage'].idxmax(),
        'Selected_Option'
    ]

    st.success(f"🏆 Leading Option: {top_option}")

    # -----------------------------
    # TREND ANALYSIS
    # -----------------------------
    st.subheader("Response Trend Over Time")

    trend = filtered_df.groupby('Timestamp').size().reset_index(name='Responses')

    fig_trend = px.line(
        trend,
        x='Timestamp',
        y='Responses',
        title="Daily Response Trend"
    )

    st.plotly_chart(fig_trend, use_container_width=True)

    # -----------------------------
    # REGIONAL ANALYSIS
    # -----------------------------
    st.subheader("Regional Analysis")

    reg_pivot = pd.pivot_table(
        filtered_df,
        values='Weight',
        index='Region',
        columns='Selected_Option',
        aggfunc='sum',
        fill_value=0
    )

    st.dataframe(reg_pivot.style.background_gradient(cmap='Blues'))

    # -----------------------------
    # SENTIMENT ANALYSIS
    # -----------------------------
    st.subheader("Sentiment Analysis")

    filtered_df['Sentiment'] = filtered_df['Qualitative_Feedback'].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity
    )

    fig_sent = px.violin(
        filtered_df,
        y='Sentiment',
        box=True,
        points='all',
        title="Feedback Sentiment Distribution"
    )

    st.plotly_chart(fig_sent, use_container_width=True)

    # -----------------------------
    # FINAL INSIGHTS
    # -----------------------------
    st.subheader("Key Insights")

    st.write(f"✅ Leading product is **{top_option}**")
    st.write("📊 Urban North region shows strong bias toward Product Alpha")
    st.write("📈 Trend chart shows response activity over time")
    st.write("💬 Sentiment analysis reveals emotional tone of responses")

# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    main()
    
    