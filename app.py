
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Social Media Intelligence | Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 2. PROFESSIONAL DARK THEME
# =====================================================

st.markdown("""
<style>
.stApp {
    background: #0B1120;
    color: #E5E7EB;
}

[data-testid="stSidebar"] {
    background: #101827;
    border-right: 1px solid #263449;
}

[data-testid="stSidebar"] * {
    color: #E5E7EB;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #F8FAFC !important;
    font-weight: 700;
}

p, label, span {
    color: #CBD5E1;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #17243A, #111C2E);
    border: 1px solid #293B55;
    border-radius: 14px;
    padding: 18px;
}

[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
}

[data-testid="stMetricValue"] {
    color: #38BDF8 !important;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #293B55;
    border-radius: 10px;
}

.stSelectbox, .stMultiSelect {
    background: #17243A;
}

hr {
    border-color: #293B55;
}

div.stButton > button {
    background: #2563EB;
    color: white;
    border: none;
    border-radius: 8px;
}

div.stButton > button:hover {
    background: #1D4ED8;
    color: white;
}

[data-testid="stAlert"] {
    border-radius: 10px;
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

/* Professional DataFrame Styling */


[data-testid="stDataFrame"] {
    border: 1px solid #38BDF8 !important;
    border-radius: 14px !important;
    overflow: hidden;
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.12);
}

[data-testid="stDataFrame"] [data-testid="glideDataEditor"] {
    border: none !important;
    border-radius: 14px !important;
}

[data-testid="stDataFrame"] iframe {
    background-color: #111C2E !important;
}

[data-testid="stDataFrame"] [role="grid"] {
    background-color: #111C2E !important;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD DATASET
# =====================================================

DATA_PATH = Path(__file__).parent / "Social_Media.csv"


@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)

    # Clean column names
    data.columns = data.columns.str.strip()

    # Convert numeric columns safely
    numeric_columns = [
        "user_followers_count",
        "post_length",
        "like_count",
        "comment_count",
        "share_count",
        "engagement_score",
        "sentiment_score",
        "toxicity_score"
    ]

    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column], errors="coerce"
            )

    # Convert date
    if "posted_datetime" in data.columns:
        data["posted_datetime"] = pd.to_datetime(
            data["posted_datetime"], errors="coerce"
        )

    # Clean text categories
    category_columns = [
        "platform",
        "sentiment_label",
        "emotion_label",
        "topic_category"
    ]

    for column in category_columns:
        if column in data.columns:
            data[column] = (
                data[column].astype("string").str.strip()
            )

    return data


if not DATA_PATH.exists():
    st.error("Social_Media.csv was not found in the project folder.")
    st.info("Keep Social_Media.csv and app.py inside the same folder.")
    st.stop()

try:
    df = load_data(DATA_PATH)
except Exception as error:
    st.error(f"Unable to load dataset: {error}")
    st.stop()

if df.empty:
    st.error("Your dataset is empty.")
    st.stop()


# =====================================================
# 4. HELPER FUNCTIONS
# =====================================================

def heading(title, subtitle=None):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def metric_card(title, value, help_text=None):
    st.metric(
        label=title,
        value=value,
        help=help_text
    )



def make_chart():
    fig, ax = plt.subplots(figsize=(10, 5))

    fig.patch.set_facecolor("#0B1120")
    ax.set_facecolor("#111C2E")

    ax.tick_params(
        axis="both",
        colors="#CBD5E1",
        labelsize=10
    )

    ax.xaxis.label.set_color("#E2E8F0")
    ax.yaxis.label.set_color("#E2E8F0")
    ax.title.set_color("#F8FAFC")

    ax.title.set_fontsize(14)
    ax.title.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_color("#334155")
        spine.set_linewidth(0.8)

    ax.grid(
        axis="y",
        color="#475569",
        linestyle="--",
        alpha=0.25
    )

    ax.set_axisbelow(True)
    plt.tight_layout()

    return fig, ax



def show_bar_chart(data, x, y, title, xlabel="", ylabel="Count"):
    if data.empty:
        st.info("No data available for this chart.")
        return

    fig, ax = make_chart()

    values = pd.to_numeric(data[y], errors="coerce").fillna(0)

    colors = [
        "#38BDF8", "#818CF8", "#A78BFA",
        "#2DD4BF", "#F472B6", "#FBBF24",
        "#60A5FA", "#34D399", "#C084FC"
    ]

    bars = ax.bar(
        data[x].astype(str),
        values,
        color=colors[:len(data)],
        edgecolor="#0B1120",
        linewidth=1,
        width=0.65,
        zorder=3
    )

    ax.set_title(title, fontsize=14, pad=18, fontweight="bold")
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)

    ax.bar_label(
        bars,
        fmt="%.1f" if pd.api.types.is_float_dtype(values) else "%.0f",
        padding=4,
        color="#E2E8F0",
        fontsize=9
    )

    ax.grid(axis="y", alpha=0.2)
    ax.set_ylim(bottom=0)

    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)



def show_donut_chart(data, labels, values, title):
    chart_data = data.dropna(subset=[labels, values])

    if chart_data.empty:
        st.info("No data available for this chart.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    fig.patch.set_facecolor("#0B1120")
    ax.set_facecolor("#111C2E")

    colors = [
        "#38BDF8",
        "#818CF8",
        "#A78BFA",
        "#2DD4BF",
        "#F472B6",
        "#FBBF24",
        "#60A5FA"
    ]

    wedges, texts, autotexts = ax.pie(
        chart_data[values],
        labels=chart_data[labels].astype(str),
        autopct="%1.1f%%",
        startangle=90,
        colors=colors[:len(chart_data)],
        pctdistance=0.78,
        labeldistance=1.12,
        wedgeprops={
            "width": 0.42,
            "edgecolor": "#0B1120",
            "linewidth": 2
        },
        textprops={
            "color": "#E2E8F0",
            "fontsize": 9
        }
    )

    for text in autotexts:
        text.set_color("#FFFFFF")
        text.set_fontweight("bold")
        text.set_fontsize(9)

    ax.set_title(
        title,
        color="#F8FAFC",
        fontsize=14,
        fontweight="bold",
        pad=18
    )

    ax.axis("equal")

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)
    

# =====================================================
# 5. SIDEBAR NAVIGATION
# =====================================================

pages = [
    "Home",
    "Executive Dashboard",
    "Dataset Overview",
    "Sentiment Analysis",
    "Emotion Analysis",
    "Platform Analysis",
    "Engagement Analysis",
    "Trending Topics",
    "Toxicity & Spam Analysis",
    "Key Insights"
]
st.markdown("""
<style>
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.5rem;
}
[data-testid="stSidebar"] {
    margin-top: 60px;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 8px 10px;
    border-radius: 8px;
    transition: 0.2s ease;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background-color: #1E293B;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    font-size: 14px;
    font-weight: 500;
}


[data-testid="stSidebar"] .stMarkdown {
    margin-bottom: -8px;
}

[data-testid="stSidebar"] hr {
    margin-top: 8px;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("# 📊 Social Intelligence")
    st.caption("SOCIAL MEDIA ANALYTICS")

    st.markdown("---")

    selected_page = st.radio(
        "NAVIGATION",
        pages,
        index=0
    )

    st.markdown("---")
    st.markdown("### 🔎 Global Filters")

    available_platforms = sorted(
        df["platform"].dropna().unique().tolist()
    ) if "platform" in df.columns else []

    selected_platforms = st.multiselect(
        "Select Platforms",
        available_platforms,
        default=available_platforms
    )

    available_sentiments = sorted(
        df["sentiment_label"].dropna().unique().tolist()
    ) if "sentiment_label" in df.columns else []

    selected_sentiments = st.multiselect(
        "Select Sentiments",
        available_sentiments,
        default=available_sentiments
    )

    st.markdown("---")
    st.caption("Built with Python & Streamlit")
    # Apply Global Filters to the Dataset


# =====================================================

# 6. APPLY FILTERS

# =====================================================

filtered_df = df.copy()

if "platform" in filtered_df.columns:
    if selected_platforms:
        filtered_df = filtered_df[
            filtered_df["platform"].isin(selected_platforms)
        ]
    else:
        filtered_df = filtered_df.iloc[0:0]

if "sentiment_label" in filtered_df.columns:
    if selected_sentiments:
        filtered_df = filtered_df[
            filtered_df["sentiment_label"].isin(selected_sentiments)
        ]
    else:
        filtered_df = filtered_df.iloc[0:0]


# Common values
total_posts = len(filtered_df)

positive_count = (
    filtered_df["sentiment_label"].astype(str).str.lower().eq("positive").sum()
    if "sentiment_label" in filtered_df.columns else 0
)

negative_count = (
    filtered_df["sentiment_label"].astype(str).str.lower().eq("negative").sum()
    if "sentiment_label" in filtered_df.columns else 0
)

neutral_count = (
    filtered_df["sentiment_label"].astype(str).str.lower().eq("neutral").sum()
    if "sentiment_label" in filtered_df.columns else 0
)

avg_engagement = (
    filtered_df["engagement_score"].mean()
    if "engagement_score" in filtered_df.columns else np.nan
)


# =====================================================
# 7. HOME PAGE
# =====================================================

if selected_page == "Home":

    st.markdown("""
    <div style="
        background:linear-gradient(120deg,#14243B,#1D3B5D);
        padding:42px;
        border-radius:18px;
        border:1px solid #315477;
    ">
        <p style="color:#38BDF8;font-weight:700;letter-spacing:2px;">
            SOCIAL MEDIA INTELLIGENCE
        </p>
        <h1 style="font-size:36px;color:white;">
            Understand Every Conversation.
        </h1>
        <p style="font-size:17px;color:#CBD5E1;">
            Explore public opinion, discover emotions and uncover
            meaningful social media insights through data analytics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    heading(
        "Welcome to Social Media Sentiment Analysis",
        "An interactive platform for exploring social media behaviour and opinions."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card("Total Posts", f"{total_posts:,}")

    with c2:
        metric_card("Platforms", f"{filtered_df['platform'].nunique() if 'platform' in filtered_df.columns else 0}")

    with c3:
        metric_card("Dataset Columns", f"{filtered_df.shape[1]}")

    st.markdown("---")

    heading("What You Can Explore")

    c1, c2 = st.columns(2)

    with c1:
        st.info("😊 **Sentiment Analysis**\n\nUnderstand positive, negative and neutral opinions.")
        st.info("🌐 **Platform Analysis**\n\nCompare activity across social media platforms.")
        st.info("📈 **Engagement Analysis**\n\nExplore likes, comments, shares and engagement.")

    with c2:
        st.info("💭 **Emotion Analysis**\n\nExplore emotions expressed in social media posts.")
        st.info("🔥 **Trending Topics**\n\nDiscover popular topics and trending posts.")
        st.info("🛡️ **Toxicity & Spam Analysis**\n\nExplore toxicity, sarcasm and spam patterns.")

    st.markdown("---")

    heading("Platforms in Dataset")

    if "platform" in filtered_df.columns:
        platform_counts = (
        filtered_df["platform"].value_counts().rename_axis("Platform")
        .reset_index(name="Posts")
    )
        show_bar_chart(
            platform_counts,
            "Platform",
            "Posts",
            "Posts by Platform"
        )

    st.caption("Social Media Sentiment Analysis | Data Analytics Project")


# =====================================================
# 8. EXECUTIVE DASHBOARD
# =====================================================

elif selected_page == "Executive Dashboard":

    heading(
        "Executive Dashboard",
        "A high-level overview of social media activity and sentiment."
    )

    if filtered_df.empty:
        st.warning("No records match the selected filters.")
    else:
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card("Total Posts", f"{total_posts:,}")

        with c2:
            metric_card("Positive Posts", f"{positive_count:,}")

        with c3:
            metric_card("Negative Posts", f"{negative_count:,}")

        with c4:
            metric_card("Average Engagement", f"{avg_engagement:.2f}" if pd.notna(avg_engagement) else "N/A")

        st.markdown("---")

        left, right = st.columns(2)

        with left:
            if "sentiment_label" in filtered_df.columns:
                sentiment_data = (
                    filtered_df["sentiment_label"]
                    .value_counts()
                    .rename_axis("Sentiment")
                    .reset_index(name="Posts")
                )
                show_donut_chart(
                    sentiment_data, "Sentiment", "Posts",
                    "Overall Sentiment Distribution"
                )

        with right:
            if "platform" in filtered_df.columns:
                platform_data = (
                    filtered_df["platform"]
                    .value_counts()
                    .rename_axis("Platform")
                    .reset_index(name="Posts")
                )
                show_bar_chart(
                    platform_data, "Platform", "Posts",
                    "Platform-wise Post Volume"
                )

        if "topic_category" in filtered_df.columns:
            topic_data = (
                filtered_df["topic_category"]
                .value_counts()
                .head(10)
                .rename_axis("Topic")
                .reset_index(name="Posts")
            )
            show_bar_chart(
                topic_data, "Topic", "Posts",
                "Top Topic Categories"
            )
                            # Premium Engagement Trend
        if all(col in filtered_df.columns for col in [
            "posted_datetime", "engagement_score"
        ]):

            trend_df = filtered_df[
                ["posted_datetime", "engagement_score"]
            ].copy()

            trend_df["posted_datetime"] = pd.to_datetime(
                trend_df["posted_datetime"], errors="coerce"
            )

            trend_df["engagement_score"] = pd.to_numeric(
                trend_df["engagement_score"], errors="coerce"
            )

            trend_df = trend_df.dropna()

            if not trend_df.empty:

                daily_engagement = (
                    trend_df.groupby(
                        trend_df["posted_datetime"].dt.date
                    )["engagement_score"]
                    .mean()
                    .reset_index()
                )

                fig, ax = make_chart()

                x = range(len(daily_engagement))
                y = daily_engagement["engagement_score"]

                ax.fill_between(
                    x, y,
                    color="#38BDF8",
                    alpha=0.18
                )

                ax.plot(
                    x, y,
                    color="#38BDF8",
                    linewidth=2.8,
                    marker="o",
                    markersize=5,
                    markerfacecolor="#FFFFFF",
                    markeredgecolor="#38BDF8",
                    markeredgewidth=2
                )

                ax.set_title(
                    "Engagement Performance Over Time",
                    fontsize=15,
                    pad=20
                )

                ax.set_ylabel("Average Engagement")
                ax.set_xlabel("Date")

                step = max(1, len(daily_engagement) // 8)

                ax.set_xticks(list(x)[::step])
                ax.set_xticklabels(
                    daily_engagement["posted_datetime"].astype(str)[::step],
                    rotation=25,
                    ha="right"
                )

                ax.grid(axis="y", alpha=0.15)
                ax.grid(axis="x", visible=False)

                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        st.markdown("---")

        st.markdown("### 📥 Download Dataset")

        csv_data = filtered_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Filtered Dataset",
            data=csv_data,
            file_name="Social_Media_Filtered_Data.csv",
            mime="text/csv",
            use_container_width=True
        )


# =====================================================
# 9. DATASET OVERVIEW
# =====================================================

elif selected_page == "Dataset Overview":

    heading(
        "Dataset Overview",
        "Explore the structure, quality and sample records of your dataset."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card("Total Rows", f"{df.shape[0]:,}")

    with c2:
        metric_card("Total Columns", f"{df.shape[1]}")

    with c3:
        metric_card("Missing Values", f"{int(df.isna().sum().sum()):,}")

    st.markdown("---")
        # Dataset Quality Overview
    
    

    
    heading(
        "Dataset Preview",
        "Explore the first 100 records of your social media dataset."
    )

    st.caption("Showing the first 100 records based on selected filters.")

    
    
    preview_df = filtered_df.head(100)

    table_html = preview_df.to_html(
        index=False,
        border=0,
        classes="premium-table"
    )

    st.markdown("""
    <style>
    
.premium-table {
    width: 100%;
    border-collapse: collapse;
    background-color: #151C2B;
    color: #E2E8F0;
    border: 1px solid #38BDF8;
    border-radius: 12px;
    overflow: hidden;
}

.premium-table th {
    background-color: #263449 !important;
    color: #38BDF8 !important;
    padding: 12px;
    text-align: left;
    border: 1px solid #334155;
}

.premium-table td {
    background-color: #1C293B !important;
    color: #E2E8F0 !important;
    padding: 10px;
    border: 1px solid #334155;
}

    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        f'<div style="height:420px; overflow:auto;">{table_html}</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    heading(
        "Column Information",
        "Details about each column in the dataset."
    )

    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isna().sum().values,
        "Unique Values": df.nunique().values
    })

    info_html = info_df.to_html(
        index=False,
        border=0,
        classes="premium-table"
    )

    st.markdown(
        f'<div style="height:420px; overflow:auto;">{info_html}</div>',
        unsafe_allow_html=True
    )


    
    
    
    
    heading("Missing Data Quality Indicator")

    missing_count = int(df.isna().sum().sum())
    total_count = df.shape[0] * df.shape[1]

    missing_percentage = (
        (missing_count / total_count) * 100
        if total_count > 0 else 0
    )

    complete_percentage = 100 - missing_percentage



    c1, c2 = st.columns(2)

    with c1:
        st.metric("Missing Data", f"{missing_percentage:.2f}%")

    with c2:
        st.metric("Complete Data", f"{complete_percentage:.2f}%")

    st.progress(complete_percentage / 100)
    st.markdown("---")

    heading(
        "Missing Values Details",
        "Identify columns that contain missing data."
    )

    missing_df = df.isna().sum().reset_index()
    missing_df.columns = ["Column", "Missing Values"]

    missing_df = missing_df[
        missing_df["Missing Values"] > 0
    ].sort_values("Missing Values", ascending=False)

    if missing_df.empty:
        st.success("No missing values found.")
    else:
        missing_df["Missing Percentage"] = (
            missing_df["Missing Values"] / len(df) * 100
        ).round(2)

        
        missing_html = missing_df.to_html(
            index=False,
            border=0,
            classes="premium-table"
        )

        st.markdown(
            f'<div style="overflow-x:auto;">{missing_html}</div>',
            unsafe_allow_html=True
        ) 
  # sentiment analysis

elif selected_page == "Sentiment Analysis":

    heading(
        "Sentiment Analysis",
        "Explore positive, negative and neutral opinions across social media."
    )

    if filtered_df.empty:
        st.warning("No records match the selected filters.")

    else:
        sentiment_counts = (
            filtered_df["sentiment_label"]
            .value_counts()
            .rename_axis("Sentiment")
            .reset_index(name="Posts")
        )

        total = len(filtered_df)

        positive = (
            filtered_df["sentiment_label"].str.lower() == "positive"
        ).sum()

        negative = (
            filtered_df["sentiment_label"].str.lower() == "negative"
        ).sum()

        neutral = (
            filtered_df["sentiment_label"].str.lower() == "neutral"
        ).sum()

        st.markdown("### Sentiment Overview")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card("Total Posts", f"{total:,}")

        with c2:
            metric_card("Positive Posts", f"{positive:,}")

        with c3:
            metric_card("Negative Posts", f"{negative:,}")

        with c4:
            metric_card("Neutral Posts", f"{neutral:,}")

        st.markdown("---")

        left, right = st.columns(2)

        with left:
            show_donut_chart(
                sentiment_counts,
                "Sentiment",
                "Posts",
                "Sentiment Distribution"
            )

        with right:
            show_bar_chart(
                sentiment_counts,
                "Sentiment",
                "Posts",
                "Sentiment-wise Post Count",
                ylabel="Number of Posts"
            )

        st.markdown("---")

        if "sentiment_score" in filtered_df.columns:

            score_data = filtered_df[
                ["sentiment_label", "sentiment_score"]
            ].copy()

            score_data["sentiment_score"] = pd.to_numeric(
                score_data["sentiment_score"],
                errors="coerce"
            )

            score_data = (
                score_data.dropna()
                .groupby("sentiment_label")["sentiment_score"]
                .mean()
                .reset_index()
            )

            show_bar_chart(
                score_data,
                "sentiment_label",
                "sentiment_score",
                "Average Sentiment Score",
                ylabel="Average Score"
            )

        if "engagement_score" in filtered_df.columns:

            engagement_data = filtered_df.copy()

            engagement_data["engagement_score"] = pd.to_numeric(
                engagement_data["engagement_score"],
                errors="coerce"
            )

            engagement_data = (
                engagement_data.groupby("sentiment_label")
                ["engagement_score"]
                .mean()
                .reset_index()
            )

            show_bar_chart(
                engagement_data,
                "sentiment_label",
                "engagement_score",
                "Average Engagement by Sentiment",
                ylabel="Average Engagement"
            )

        st.markdown("---")

        heading(
            "Sentiment Insights",
            "Key observations from the selected social media posts."
        )

        if not sentiment_counts.empty:

            dominant_sentiment = sentiment_counts.iloc[0]["Sentiment"]
            dominant_count = sentiment_counts.iloc[0]["Posts"]

            sentiment_percentage = (
                dominant_count / total
            ) * 100

            st.info(
                f"{dominant_sentiment} sentiment is the most common "
                f"category, representing {sentiment_percentage:.1f}% "
                f"of all posts."
            )

            insight1, insight2 = st.columns(2)

            with insight1:
                st.metric(
                    "Most Common Sentiment",
                    str(dominant_sentiment).title()
                )

            with insight2:
                st.metric(
                    "Posts in This Category",
                    f"{dominant_count:,}"
                )

        st.markdown("---")

        heading(
            "Sentiment Summary",
            "A detailed breakdown of sentiment categories."
        )

        summary_df = sentiment_counts.copy()

        summary_df["Percentage"] = (
            summary_df["Posts"] / total * 100
        ).round(2)

        summary_df.columns = [
            "Sentiment",
            "Total Posts",
            "Percentage (%)"
        ]

        summary_html = summary_df.to_html(
            index=False,
            border=0,
            classes="premium-table"
        )

        st.markdown(
            f'<div style="overflow-x:auto;">{summary_html}</div>',
            unsafe_allow_html=True
        )








# =====================================================
#
elif selected_page == "Emotion Analysis":

    heading(
        "Emotion Analysis",
        "Explore the emotions represented in social media posts."
    )

    if filtered_df.empty:
        st.warning("No records match the selected filters.")

    elif "emotion_label" in filtered_df.columns:

        emotion_data = (
            filtered_df["emotion_label"]
            .dropna()
            .value_counts()
            .rename_axis("Emotion")
            .reset_index(name="Posts")
        )

        if emotion_data.empty:
            st.info("No emotion data available.")

        else:
            total_posts = int(emotion_data["Posts"].sum())
            total_emotions = len(emotion_data)

            dominant_emotion = emotion_data.iloc[0]["Emotion"]
            dominant_count = int(emotion_data.iloc[0]["Posts"])

            dominant_percentage = (
                dominant_count / total_posts * 100
                if total_posts > 0 else 0
            )

            st.markdown("### Emotion Overview")

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                metric_card("Total Posts", f"{total_posts:,}")

            with c2:
                metric_card("Different Emotions", f"{total_emotions:,}")

            with c3:
                metric_card(
                    "Top Emotion",
                    str(dominant_emotion).title()
                )

            with c4:
                metric_card(
                    "Top Emotion Share",
                    f"{dominant_percentage:.1f}%"
                )

            st.markdown("---")

            left, right = st.columns(2)

            with left:
                show_donut_chart(
                    emotion_data,
                    "Emotion",
                    "Posts",
                    "Emotion Distribution"
                )

            with right:
                show_bar_chart(
                    emotion_data,
                    "Emotion",
                    "Posts",
                    "Emotion-wise Post Count"
                )

            st.markdown("---")

            # Emotion vs Average Likes
            if "like_count" in filtered_df.columns:

                likes_df = filtered_df[
                    ["emotion_label", "like_count"]
                ].copy()

                likes_df["like_count"] = pd.to_numeric(
                    likes_df["like_count"],
                    errors="coerce"
                )

                emotion_likes = (
                    likes_df.dropna()
                    .groupby("emotion_label")["like_count"]
                    .mean()
                    .reset_index()
                    .sort_values("like_count", ascending=False)
                )

                if not emotion_likes.empty:
                    show_bar_chart(
                        emotion_likes,
                        "emotion_label",
                        "like_count",
                        "Average Likes by Emotion",
                        ylabel="Average Likes"
                    )

            st.markdown("---")

            heading(
                "Key Emotion Insights",
                "Important observations from the selected social media posts."
            )

            st.info(
                f"{str(dominant_emotion).title()} is the most common emotion, "
                f"appearing in {dominant_percentage:.1f}% of the posts."
            )

            st.markdown("---")

            heading(
                "Emotion Summary",
                "A detailed breakdown of emotions and their contribution."
            )

            summary_df = emotion_data.copy()

            summary_df["Percentage"] = (
                summary_df["Posts"] / total_posts * 100
            ).round(2)

            summary_df.columns = [
                "Emotion",
                "Total Posts",
                "Percentage (%)"
            ]

            summary_html = summary_df.to_html(
                index=False,
                border=0,
                classes="premium-table"
            )

            st.markdown(
                f'<div style="overflow-x:auto;">{summary_html}</div>',
                unsafe_allow_html=True
            )

    else:
        st.warning("Emotion column is not available.")




# =====================================================
# 12. PLATFORM ANALYSIS
# =====================================================

elif selected_page == "Platform Analysis":

    heading(
        "Platform Analysis",
        "Compare post volume and sentiment across social media platforms."
    )

    if "platform" in filtered_df.columns:

        platform_data = (
            filtered_df["platform"]
            .value_counts()
            .rename_axis("Platform")
            .reset_index(name="Posts")
        )

        show_bar_chart(
            platform_data, "Platform", "Posts",
            "Posts by Platform"
        )

        if "sentiment_label" in filtered_df.columns:

            cross_data = pd.crosstab(
                filtered_df["platform"],
                filtered_df["sentiment_label"]
            )

            fig, ax = make_chart()

            cross_data.plot(
                kind="bar",
                stacked=True,
                ax=ax,
                colormap="cool"
            )

            ax.set_title("Platform-wise Sentiment Comparison", color="white")
            ax.set_xlabel("Platform")
            ax.set_ylabel("Number of Posts")
            ax.tick_params(colors="#CBD5E1")
            ax.legend(facecolor="#17243A", labelcolor="white")

            plt.xticks(rotation=25, ha="right")
            plt.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

            heading(
                "Platform Sentiment Data",
                "A detailed comparison of sentiment across platforms."
            )

            cross_display = cross_data.reset_index()

            cross_html = cross_display.to_html(
                index=False,
                border=0,
                classes="premium-table"
            )

            st.markdown(
                f'<div style="overflow-x:auto;">{cross_html}</div>',
                unsafe_allow_html=True
            )

# =====================================================
# 13. ENGAGEMENT ANALYSIS
# =====================================================


elif selected_page == "Engagement Analysis":

    heading(
        "Engagement Analysis",
        "Explore how audiences interact with social media content."
    )

    engagement_columns = [
        "like_count",
        "comment_count",
        "share_count"
    ]

    available_engagement = [
        col for col in engagement_columns
        if col in filtered_df.columns
    ]

    if filtered_df.empty:
        st.warning("No records match the selected filters.")

    elif available_engagement:

        engagement_df = filtered_df[available_engagement].apply(
            pd.to_numeric, errors="coerce"
        )

        total_likes = engagement_df["like_count"].sum() if "like_count" in engagement_df.columns else 0
        total_comments = engagement_df["comment_count"].sum() if "comment_count" in engagement_df.columns else 0
        total_shares = engagement_df["share_count"].sum() if "share_count" in engagement_df.columns else 0

        st.markdown("### Engagement Overview")

        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card("Total Likes", f"{total_likes:,.0f}")

        with c2:
            metric_card("Total Comments", f"{total_comments:,.0f}")

        with c3:
            metric_card("Total Shares", f"{total_shares:,.0f}")

        st.markdown("---")

        engagement_summary = pd.DataFrame({
            "Metric": available_engagement,
            "Average": [
                engagement_df[col].mean()
                for col in available_engagement
            ],
            "Total": [
                engagement_df[col].sum()
                for col in available_engagement
            ]
        })

        heading(
            "Interaction Performance",
            "Compare average and total audience interactions."
        )

        avg_values = engagement_summary[["Metric", "Average"]]

        show_bar_chart(
            avg_values,
            "Metric",
            "Average",
            "Average Engagement by Interaction Type",
            ylabel="Average Interactions"
        )

        summary_display = engagement_summary.copy()
        summary_display["Average"] = summary_display["Average"].round(2)
        summary_display["Total"] = summary_display["Total"].round(0).astype("Int64") 
        summary_html = summary_display.to_html(
            index=False,
            border=0,
            classes="premium-table"
        )

        st.markdown(
            f'<div style="overflow-x:auto;">{summary_html}</div>',
            unsafe_allow_html=True
        )

        st.markdown("---")

        # Average Engagement by Platform
        if "platform" in filtered_df.columns:

            platform_engagement = (
                filtered_df.groupby("platform")[available_engagement]
                .mean()
                .reset_index()
            )

            platform_engagement[available_engagement] = (
                platform_engagement[available_engagement].apply(
                    pd.to_numeric, errors="coerce"
                )
            )

            fig, ax = make_chart()

            platform_engagement.set_index("platform").plot(
                kind="bar",
                ax=ax,
                colormap="cool"
            )

            ax.set_title("Average Engagement by Platform", color="white")
            ax.set_xlabel("Platform")
            ax.set_ylabel("Average Interactions")
            ax.legend(facecolor="#17243A", labelcolor="white")
            plt.xticks(rotation=25, ha="right")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("---")

        # Sentiment Score vs Engagement Scatter Plot
        if all(col in filtered_df.columns for col in [
            "sentiment_score", "engagement_score", "sentiment_label"
        ]):

            scatter_data = filtered_df[
                ["sentiment_score", "engagement_score", "sentiment_label"]
            ].copy()

            scatter_data["sentiment_score"] = pd.to_numeric(
                scatter_data["sentiment_score"], errors="coerce"
            )

            scatter_data["engagement_score"] = pd.to_numeric(
                scatter_data["engagement_score"], errors="coerce"
            )

            scatter_data = scatter_data.dropna()

            if not scatter_data.empty:
                fig, ax = make_chart()

                sentiment_colors = {
                    "positive": "#2DD4BF",
                    "negative": "#F87171",
                    "neutral": "#FBBF24"
                }

                for sentiment, group in scatter_data.groupby("sentiment_label"):
                    ax.scatter(
                        group["sentiment_score"],
                        group["engagement_score"],
                        label=sentiment.title(),
                        color=sentiment_colors.get(
                            str(sentiment).lower(), "#38BDF8"
                        ),
                        alpha=0.65,
                        s=35
                    )

                ax.set_title("Sentiment Score vs Engagement Score")
                ax.set_xlabel("Sentiment Score")
                ax.set_ylabel("Engagement Score")
                ax.legend(facecolor="#17243A", labelcolor="white")

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        st.markdown("---")

        heading(
            "Key Engagement Insight",
            "A quick observation from audience interaction data."
        )

        if not engagement_summary.empty:
            highest_metric = engagement_summary.loc[
                engagement_summary["Average"].idxmax()
            ]

            st.info(
                f"{highest_metric['Metric'].replace('_count', '').title()} "
                f"has the highest average interaction "
                f"({highest_metric['Average']:.2f}) among the measured metrics."
            )

    else:
        st.warning("No engagement columns are available.")



# =====================================================
# 14. TRENDING TOPICS
# =====================================================

elif selected_page == "Trending Topics":

    heading(
        "Trending Topics",
        "Explore trending posts and the most discussed topic categories."
    )

    if "is_trending_topic" in filtered_df.columns:

        trending = filtered_df[
            filtered_df["is_trending_topic"].astype(str).str.lower()
            .isin(["true", "1", "yes"])
        ]

        c1, c2 = st.columns(2)

        with c1:
            metric_card("Trending Posts", f"{len(trending):,}")

        with c2:
            percentage = (
                len(trending) / len(filtered_df) * 100
                if len(filtered_df) else 0
            )
            metric_card("Trending Share", f"{percentage:.1f}%")

        if "topic_category" in trending.columns:

            topic_data = (
                trending["topic_category"]
                .value_counts()
                .head(10)
                .rename_axis("Topic")
                .reset_index(name="Trending Posts")
            )

            show_bar_chart(
                topic_data, "Topic", "Trending Posts",
                "Top Trending Categories"
            )

        if "post_text" in trending.columns:
            heading("Trending Post Samples")
            display_columns = [
                col for col in [
                    "platform", "post_text",
                    "topic_category", "engagement_score"
                ] if col in trending.columns
            ]

            trending_display = trending[display_columns].head(20)

            trending_html = trending_display.to_html(
                index=False,
                border=0,
                classes="premium-table"
            )

            st.markdown(
                f'<div style="height:420px; overflow:auto;">{trending_html}</div>',
                unsafe_allow_html=True
            )

    else:
        st.warning("Trending topic column is not available.")


# =====================================================
# 15. TOXICITY & SPAM ANALYSIS
# =====================================================

elif selected_page == "Toxicity & Spam Analysis":

    heading(
        "Toxicity & Spam Analysis",
        "Explore toxicity scores, spam flags and sarcasm detection."
    )

    c1, c2, c3 = st.columns(3)

    if "toxicity_score" in filtered_df.columns:
        avg_toxicity = filtered_df["toxicity_score"].mean()
        with c1:
            metric_card(
                "Average Toxicity",
                f"{avg_toxicity:.2f}" if pd.notna(avg_toxicity) else "N/A"
            )

    if "spam_flag" in filtered_df.columns:
        spam_count = (
            filtered_df["spam_flag"].astype(str).str.lower()
            .isin(["true", "1", "yes"]).sum()
        )
        with c2:
            metric_card("Spam Posts", f"{spam_count:,}")

    if "sarcasm_detected" in filtered_df.columns:
        sarcasm_count = (
            filtered_df["sarcasm_detected"].astype(str).str.lower()
            .isin(["true", "1", "yes"]).sum()
        )
        with c3:
            metric_card("Sarcasm Detected", f"{sarcasm_count:,}")

    left, right = st.columns(2)

    with left:
        if "toxicity_score" in filtered_df.columns:
            toxicity = filtered_df["toxicity_score"].dropna()

            if not toxicity.empty:
                fig, ax = make_chart()
                ax.hist(
                    toxicity,
                    bins=20,
                    color="#E879F9",
                    edgecolor="#0B1120"
                )
                ax.set_title("Toxicity Score Distribution", color="white")
                ax.set_xlabel("Toxicity Score")
                ax.set_ylabel("Number of Posts")
                st.pyplot(fig)
                plt.close(fig)

    with right:
        if "spam_flag" in filtered_df.columns:
            spam_data = (
                filtered_df["spam_flag"].astype(str).str.lower()
                .map(lambda x: "Spam" if x in ["true", "1", "yes"] else "Not Spam")
                .value_counts()
                .rename_axis("Category")
                .reset_index(name="Posts")
            )

            if not spam_data.empty:
              show_donut_chart(
       
        spam_data, "Category", "Posts",
        "Spam vs Non-Spam Posts"
    )
                # Toxicity by Sentiment Box Plot
    if all(col in filtered_df.columns for col in [
        "sentiment_label", "toxicity_score"
    ]):

        box_data = filtered_df[
            ["sentiment_label", "toxicity_score"]
        ].copy()

        box_data["toxicity_score"] = pd.to_numeric(
            box_data["toxicity_score"], errors="coerce"
        )

        box_data = box_data.dropna()

        if not box_data.empty:
            fig, ax = make_chart()

            sentiments = sorted(
                box_data["sentiment_label"].unique()
            )

            values = [
                box_data.loc[
                    box_data["sentiment_label"] == sentiment,
                    "toxicity_score"
                ]
                for sentiment in sentiments
            ]

            ax.boxplot(
                values,
                tick_labels=sentiments,
                patch_artist=True,
                boxprops={"facecolor": "#E879F9"},
                medianprops={"color": "#FBBF24", "linewidth": 2},
                whiskerprops={"color": "#CBD5E1"},
                capprops={"color": "#CBD5E1"}
            )

            ax.set_title("Toxicity by Sentiment")
            ax.set_xlabel("Sentiment")
            ax.set_ylabel("Toxicity Score")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # Spam Summary Table
    if "spam_flag" in filtered_df.columns and not filtered_df.empty:

        heading(
            "Spam Detection Summary",
            "A breakdown of spam and non-spam posts."
        )

        spam_summary = (
            filtered_df["spam_flag"].astype(str).str.lower()
            .map(
                lambda x: "Spam" if x in ["true", "1", "yes"]
                else "Not Spam"
            )
            .value_counts()
            .rename_axis("Category")
            .reset_index(name="Total Posts")
        )

        spam_summary["Percentage (%)"] = (
            spam_summary["Total Posts"] / len(filtered_df) * 100
        ).round(2)

        spam_html = spam_summary.to_html(
            index=False,
            border=0,
            classes="premium-table"
        )

        st.markdown(
            f'<div style="overflow-x:auto;">{spam_html}</div>',
            unsafe_allow_html=True
        )



    st.caption(
        "Note: Toxicity scores are displayed as present in the dataset. "
        "They should not be treated as a clinical or safety classification."
    )


# =====================================================
# 16. KEY INSIGHTS
# =====================================================


# =====================================================
# 16. KEY INSIGHTS
# =====================================================

elif selected_page == "Key Insights":

    heading(
        "Key Insights",
        "Automatically generated observations from the selected dataset records."
    )

    if filtered_df.empty:
        st.warning("No records match the selected filters.")

    else:

        st.markdown("### 📌 Dataset Summary")

        st.write(f"- **Posts analysed:** {len(filtered_df):,}")

        if "platform" in filtered_df.columns:
            top_platform = filtered_df["platform"].value_counts()

            if not top_platform.empty:
                st.write(
                    f"- **Most active platform:** {top_platform.index[0]} "
                    f"({top_platform.iloc[0]:,} posts)"
                )

        if "sentiment_label" in filtered_df.columns:
            sentiment_counts = filtered_df["sentiment_label"].value_counts()

            if not sentiment_counts.empty:
                st.write(
                    f"- **Most common sentiment:** {sentiment_counts.index[0]} "
                    f"({sentiment_counts.iloc[0]:,} posts)"
                )

        if "emotion_label" in filtered_df.columns:
            emotion_counts = filtered_df["emotion_label"].value_counts()

            if not emotion_counts.empty:
                st.write(
                    f"- **Most common emotion:** {emotion_counts.index[0]} "
                    f"({emotion_counts.iloc[0]:,} posts)"
                )

        if "topic_category" in filtered_df.columns:
            topic_counts = filtered_df["topic_category"].value_counts()

            if not topic_counts.empty:
                st.write(
                    f"- **Most common topic:** {topic_counts.index[0]}"
                )

        if "engagement_score" in filtered_df.columns:
            score = pd.to_numeric(
                filtered_df["engagement_score"],
                errors="coerce"
            ).mean()

            if pd.notna(score):
                st.write(
                    f"- **Average engagement score:** {score:.2f}"
                )

        st.markdown("---")

        # Correlation Heatmap
        heading(
            "Correlation Heatmap",
            "Understand relationships between numerical dataset features."
        )

        numeric_cols = [
            "user_followers_count",
            "post_length",
            "like_count",
            "comment_count",
            "share_count",
            "engagement_score",
            "sentiment_score",
            "toxicity_score"
        ]

        available_numeric_cols = [
            col for col in numeric_cols
            if col in filtered_df.columns
        ]

        if len(available_numeric_cols) >= 2:

            corr_data = filtered_df[available_numeric_cols].apply(
                pd.to_numeric,
                errors="coerce"
            )

            corr = corr_data.corr()

            fig, ax = plt.subplots(figsize=(9, 6))

            fig.patch.set_facecolor("#0B1120")
            ax.set_facecolor("#111C2E")

            from matplotlib.colors import LinearSegmentedColormap

            navy_cmap = LinearSegmentedColormap.from_list(
                "navy_theme",
                ["#A9C7E8", "#526F9C", "#17243A", "#080F1E"]
            )

            sns.heatmap(
                corr,
                annot=True,
                cmap=navy_cmap,
                fmt=".2f",
                linewidths=0.5,
                ax=ax,
                annot_kws={"size": 8},
                cbar_kws={
                    "shrink": 0.8,
                    "ticks": [-1, -0.5, 0, 0.5, 1]
                }
            )

            cbar = fig.axes[-1]
            cbar.tick_params(colors="#E2E8F0", labelsize=9)
            cbar.yaxis.label.set_color("#E2E8F0")
            cbar.set_facecolor("#111C2E")

            ax.set_title(
                "Correlation Heatmap",
                color="#F8FAFC",
                fontsize=14,
                fontweight="bold",
                pad=18
            )

            ax.tick_params(colors="#CBD5E1")

            plt.xticks(rotation=35, ha="right")
            plt.yticks(rotation=0)

            plt.tight_layout()

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            

           
    

# =====================================================
# 17. FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#64748B;padding:10px;">
        Social Media Intelligence Dashboard
        <br>
        Built with Python • Pandas • Matplotlib • Seaborn • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)

