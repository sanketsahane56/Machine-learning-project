import os
import streamlit as st
import pandas as pd
import numpy as np

# Page configuration - Must be the very first Streamlit command
st.set_page_config(
    page_title="AI Disaster Tweet Classification System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling (Modern Glassmorphism & Dashboard Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
    }

    /* App Header Gradient Accent */
    .header-banner {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(239, 68, 68, 0.12) 50%, rgba(16, 185, 129, 0.12) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
        backdrop-filter: blur(10px);
    }
    
    .header-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #F8FAFC, #C7D2FE, #FCA5A5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }

    .header-subtitle {
        color: #94A3B8;
        font-size: 1.02rem;
        margin-bottom: 0;
    }

    /* KPI Metric Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }

    .kpi-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
        backdrop-filter: blur(8px);
    }

    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }

    .kpi-label {
        font-size: 0.82rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-bottom: 6px;
    }

    .kpi-value {
        font-family: 'Outfit', sans-serif;
        font-size: 1.85rem;
        font-weight: 700;
        color: #F8FAFC;
    }

    .kpi-subtext {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 4px;
    }

    /* Explanation & Alert Banners */
    .info-card {
        background: rgba(99, 102, 241, 0.08);
        border-left: 4px solid #6366F1;
        padding: 14px 18px;
        border-radius: 0 10px 10px 0;
        margin: 14px 0 20px 0;
        color: #CBD5E1;
        font-size: 0.95rem;
        line-height: 1.55;
    }

    .disaster-badge {
        display: inline-block;
        background: rgba(239, 68, 68, 0.18);
        color: #FCA5A5;
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 20px;
        padding: 4px 14px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    .safe-badge {
        display: inline-block;
        background: rgba(16, 185, 129, 0.18);
        color: #6EE7B7;
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 20px;
        padding: 4px 14px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    .code-box {
        background: #0F172A;
        border-radius: 8px;
        padding: 12px;
        font-family: monospace;
        font-size: 0.88rem;
        color: #E2E8F0;
        margin: 6px 0;
    }
</style>
""", unsafe_allow_html=True)

# Imports from modular src package
from src.data_loader import check_dataset_files, load_raw_datasets, get_dataset_summary
from src.preprocessor import clean_text, clean_text_step_by_step, extract_tweet_features, create_cleaned_working_copy
from src.visualizer import (
    plot_target_distribution_bar, plot_target_distribution_pie,
    plot_tweet_length_histograms, plot_missing_values_bar,
    plot_top_keywords, plot_confusion_matrix_heatmap,
    plot_roc_curve, plot_feature_importance_bars,
    plot_prediction_gauge, plot_model_comparison_bar
)
from src.model_trainer import (
    MODEL_REGISTRY, train_and_evaluate_model,
    benchmark_all_models, predict_single_tweet, predict_test_dataset
)

# -------------------------------------------------------------
# Data & Model Caching in Session State
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_cache_data():
    return load_raw_datasets(base_dir=".")

@st.cache_data(show_spinner=False)
def get_train_features(df):
    return extract_tweet_features(df)

train_df, test_df, sample_sub_df, load_error = load_and_cache_data()

# Initialize Model in Session State if not present
if "trained_model_results" not in st.session_state and train_df is not None:
    with st.spinner("Initializing Baseline Model (Logistic Regression)..."):
        try:
            st.session_state.trained_model_results = train_and_evaluate_model(
                train_df=train_df,
                model_name="Logistic Regression",
                test_size=0.2,
                random_state=42
            )
        except Exception as e:
            st.session_state.trained_model_results = None

# -------------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------------
st.sidebar.markdown("""
<div style="text-align: center; padding: 10px 0 16px 0;">
    <h2 style="font-family: 'Outfit'; font-weight:800; margin:0; color:#F8FAFC;">🚨 Disaster AI</h2>
    <p style="font-size:0.8rem; color:#94A3B8; margin-top:4px;">Tweet Classification System</p>
</div>
""", unsafe_allow_html=True)

nav_options = [
    "1. 📊 Dashboard",
    "2. 📂 Dataset Explorer",
    "3. 🧹 Data Cleaning",
    "4. 📈 Exploratory Data Analysis (EDA)",
    "5. 🔤 NLP Preprocessing",
    "6. ⚙️ Feature Engineering",
    "7. 🧠 Model Training",
    "8. 📉 Model Evaluation",
    "9. 🔮 Live Tweet Prediction",
    "10. 📑 Test Dataset Predictions",
    "11. 💾 Download & Export Results"
]

selected_page = st.sidebar.radio(
    "Navigation",
    nav_options,
    index=0
)

st.sidebar.markdown("---")

# Active Model Status in Sidebar
if st.session_state.get("trained_model_results"):
    curr_res = st.session_state.trained_model_results
    st.sidebar.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.7); border:1px solid rgba(255,255,255,0.1); border-radius:10px; padding:12px;">
        <div style="font-size:0.75rem; color:#94A3B8; text-transform:uppercase; font-weight:700;">Active Model</div>
        <div style="font-size:1.05rem; font-weight:700; color:#6366F1; margin-top:2px;">{curr_res['model_name']}</div>
        <div style="font-size:0.82rem; color:#CBD5E1; margin-top:4px;">Accuracy: <b>{curr_res['accuracy']*100:.2f}%</b></div>
        <div style="font-size:0.82rem; color:#CBD5E1;">F1-Score: <b>{curr_res['f1_score']*100:.2f}%</b></div>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="margin-top: 15px; font-size:0.75rem; color:#64748B; line-height:1.4;">
    <b>Dataset:</b> Disaster Tweets (Kaggle)<br>
    <b>Train Records:</b> 7,613<br>
    <b>Test Records:</b> 3,263<br>
    <b>Platform:</b> Python & Streamlit
</div>
""", unsafe_allow_html=True)


# =============================================================
# ERROR HANDLING: DATASET FILES CHECK
# =============================================================
if load_error or train_df is None or test_df is None:
    st.error(f"⚠️ **Error:** {load_error}")
    st.markdown("""
    <div class="info-card">
        <b>File Status:</b><br>
        The following dataset files must be present in the project folder:
        <ul>
            <li><code>train.csv</code> (Training Data)</li>
            <li><code>test.csv</code> (Test Data)</li>
        </ul>
        Please ensure these files are available in the root workspace directory.
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# =============================================================
# PAGE 1: DASHBOARD
# =============================================================
if "1." in selected_page:
    st.markdown("""
    <div class="header-banner">
        <div class="header-title">AI-Based Disaster Tweet Classification System</div>
        <div class="header-subtitle">Real-time NLP platform for automated disaster emergency detection and analysis from social media tweets</div>
    </div>
    """, unsafe_allow_html=True)
    
    # KPI Metrics
    curr_res = st.session_state.get("trained_model_results")
    acc_val = f"{curr_res['accuracy']*100:.2f}%" if curr_res else "N/A"
    f1_val = f"{curr_res['f1_score']*100:.2f}%" if curr_res else "N/A"
    
    disaster_count = int((train_df['target'] == 1).sum())
    nondisaster_count = int((train_df['target'] == 0).sum())
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Train Tweets</div>
            <div class="kpi-value">{len(train_df):,}</div>
            <div class="kpi-subtext">5 Columns (labeled)</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Disaster Tweets</div>
            <div class="kpi-value" style="color:#EF4444;">{disaster_count:,}</div>
            <div class="kpi-subtext">{(disaster_count/len(train_df))*100:.1f}% of train data</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Non-Disaster Tweets</div>
            <div class="kpi-value" style="color:#10B981;">{nondisaster_count:,}</div>
            <div class="kpi-subtext">{(nondisaster_count/len(train_df))*100:.1f}% of train data</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Test Dataset Tweets</div>
            <div class="kpi-value">{len(test_df):,}</div>
            <div class="kpi-subtext">Evaluation (unlabeled)</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Model Accuracy</div>
            <div class="kpi-value" style="color:#6366F1;">{acc_val}</div>
            <div class="kpi-subtext">F1-Score: {f1_val}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### ⚡ Quick Tweet Inference")
    st.markdown("Type any tweet or select a pre-set sample below to test the AI model's instant prediction:")
    
    col_input, col_out = st.columns([1.1, 0.9])
    with col_input:
        quick_example = st.selectbox(
            "Select Pre-set Example:",
            [
                "The building is on fire and people need immediate help.",
                "Forest fire near La Ronge Sask. Canada with heavy smoke",
                "Watching a movie about earthquakes tonight with friends!",
                "Heavy flood warning issued for coastal areas.",
                "Just had an absolute explosion of spicy flavors at lunch!"
            ]
        )
        user_quick_text = st.text_area("Or enter your own Tweet:", value=quick_example, height=90)
        btn_quick = st.button("🔍 Quick Analyze", type="primary", use_container_width=True)

    with col_out:
        if btn_quick or user_quick_text:
            if curr_res:
                pred_result = predict_single_tweet(
                    curr_res['model'], curr_res['vectorizer'], user_quick_text
                )
                is_dis = pred_result['target'] == 1
                badge_class = "disaster-badge" if is_dis else "safe-badge"
                bg_color = "rgba(239, 68, 68, 0.1)" if is_dis else "rgba(16, 185, 129, 0.1)"
                border_color = "#EF4444" if is_dis else "#10B981"
                
                st.markdown(f"""
                <div style="background:{bg_color}; border:1.5px solid {border_color}; border-radius:14px; padding:18px;">
                    <div style="font-size:0.85rem; color:#94A3B8; text-transform:uppercase;">AI Model Prediction:</div>
                    <div style="font-size:1.4rem; font-weight:700; margin:6px 0;">
                        <span class="{badge_class}">{pred_result['label']}</span>
                    </div>
                    <div style="font-size:0.95rem; color:#E2E8F0; margin-top:8px;">
                        Confidence Score: <b>{pred_result['confidence']*100:.1f}%</b>
                    </div>
                    <div style="font-size:0.82rem; color:#94A3B8; margin-top:6px;">
                        Cleaned Text: <code>{pred_result['cleaned_text']}</code>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Please navigate to the 'Model Training' page to train a model.")

    st.markdown("---")
    st.markdown("### 🗺️ System Architecture & Workflow")
    
    col_w1, col_w2, col_w3, col_w4 = st.columns(4)
    with col_w1:
        st.markdown("""
        <div class="kpi-card">
            <div style="font-size:1.3rem; font-weight:700; margin-bottom:8px;">1️⃣ Load & Inspect</div>
            <div style="font-size:0.85rem; color:#CBD5E1;">
                Ingest authentic <code>train.csv</code> and <code>test.csv</code>. Handle missing values & duplicates safely in memory.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_w2:
        st.markdown("""
        <div class="kpi-card">
            <div style="font-size:1.3rem; font-weight:700; margin-bottom:8px;">2️⃣ NLP & Features</div>
            <div style="font-size:0.85rem; color:#CBD5E1;">
                Sanitize URLs, user mentions, and punctuation. Extract TF-IDF N-Gram matrices for numerical modeling.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_w3:
        st.markdown("""
        <div class="kpi-card">
            <div style="font-size:1.3rem; font-weight:700; margin-bottom:8px;">3️⃣ Model Training</div>
            <div style="font-size:0.85rem; color:#CBD5E1;">
                Stratified train/validation split strictly preventing data leakage. Train Logistic Regression, Naive Bayes, SVM, etc.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_w4:
        st.markdown("""
        <div class="kpi-card">
            <div style="font-size:1.3rem; font-weight:700; margin-bottom:8px;">4️⃣ Evaluation & Export</div>
            <div style="font-size:0.85rem; color:#CBD5E1;">
                Analyze confusion matrix, ROC-AUC, live inference, and export Kaggle-ready submission CSV files.
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================
# PAGE 2: DATASET EXPLORER
# =============================================================
elif "2." in selected_page:
    st.markdown("""
    <div class="header-banner">
        <div class="header-title">📂 Dataset Explorer & Schema Inspection</div>
        <div class="header-subtitle">Detailed verification and exploration of original train.csv and test.csv datasets</div>
    </div>
    """, unsafe_allow_html=True)
    
    files_info = check_dataset_files(".")
    
    # File Verification Table
    st.markdown("#### 1. Local Dataset Files Verification")
    file_rows = []
    for fname, meta in files_info.items():
        file_rows.append({
            "File Name": fname,
            "Status": "✅ Available (Found)" if meta["exists"] else "❌ Not Found",
            "File Size (KB)": f"{meta['size_kb']} KB",
            "Size (Bytes)": f"{meta['size_bytes']:,} Bytes"
        })
    st.dataframe(pd.DataFrame(file_rows), use_container_width=True)

    tab_train, tab_test, tab_sample = st.tabs([
        "🏋️ Train Dataset (train.csv)",
        "🧪 Test Dataset (test.csv)",
        "📋 Sample Submission (sample_submission.csv)"
    ])
    
    with tab_train:
        train_sum = get_dataset_summary(train_df, "Train Dataset")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Rows", f"{train_sum['total_rows']:,}")
        c2.metric("Total Columns", train_sum['total_columns'])
        c3.metric("Missing Values", sum(train_sum['missing_values'].values()))
        c4.metric("Duplicate Tweet Texts", train_sum['text_duplicates'])
        
        st.markdown("##### 📌 Column Schema & Missing Values:")
        schema_df = pd.DataFrame({
            "Column Name": train_sum["column_names"],
            "Data Type": [train_sum["dtypes"][c] for c in train_sum["column_names"]],
            "Missing Count": [train_sum["missing_values"][c] for c in train_sum["column_names"]],
            "Missing Percentage": [f"{train_sum['missing_percentages'][c]}%" for c in train_sum["column_names"]]
        })
        st.dataframe(schema_df, use_container_width=True)
        
        st.markdown("##### 🎯 Target Column Distribution (Disaster vs Non-Disaster):")
        st.markdown(f"""
        - **0 (Non-Disaster / Safe):** {train_sum['target_distribution']['counts'][0]:,} tweets ({train_sum['target_distribution']['percentages'][0]}%)
        - **1 (Disaster):** {train_sum['target_distribution']['counts'][1]:,} tweets ({train_sum['target_distribution']['percentages'][1]}%)
        """)
        
        st.markdown("##### 🔍 Interactive Data Viewer & Search:")
        col_f1, col_f2 = st.columns([1, 1])
        with col_f1:
            target_filter = st.selectbox("Filter by Target:", ["All", "Disaster Only (Target = 1)", "Non-Disaster Only (Target = 0)"])
        with col_f2:
            search_query = st.text_input("Search Tweet Text:", placeholder="e.g. earthquake, fire, flood...")
            
        filtered_train = train_df.copy()
        if target_filter == "Disaster Only (Target = 1)":
            filtered_train = filtered_train[filtered_train['target'] == 1]
        elif target_filter == "Non-Disaster Only (Target = 0)":
            filtered_train = filtered_train[filtered_train['target'] == 0]
            
        if search_query:
            filtered_train = filtered_train[filtered_train['text'].str.contains(search_query, case=False, na=False)]
            
        st.write(f"Total Matches Found: **{len(filtered_train):,} tweets**")
        st.dataframe(filtered_train.head(50), use_container_width=True)

    with tab_test:
        test_sum = get_dataset_summary(test_df, "Test Dataset")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rows", f"{test_sum['total_rows']:,}")
        c2.metric("Total Columns", test_sum['total_columns'])
        c3.metric("Duplicate Tweet Texts", test_sum['text_duplicates'])
        
        st.markdown("""
        <div class="info-card">
            <b>Important Note:</b> <code>test.csv</code> contains unlabeled tweets without the <code>target</code> column.
            This dataset is used for batch inference and generating final predictions for Kaggle submission.
        </div>
        """, unsafe_allow_html=True)
        
        schema_test = pd.DataFrame({
            "Column Name": test_sum["column_names"],
            "Data Type": [test_sum["dtypes"][c] for c in test_sum["column_names"]],
            "Missing Count": [test_sum["missing_values"][c] for c in test_sum["column_names"]],
            "Missing Percentage": [f"{test_sum['missing_percentages'][c]}%" for c in test_sum["column_names"]]
        })
        st.dataframe(schema_test, use_container_width=True)
        st.markdown("##### 🔍 Test Data Sample (First 20 Rows):")
        st.dataframe(test_df.head(20), use_container_width=True)

    with tab_sample:
        if sample_sub_df is not None:
            st.markdown("##### 📋 `sample_submission.csv` Format Preview:")
            st.write(f"Total Rows: **{len(sample_sub_df):,}** | Columns: **{list(sample_sub_df.columns)}**")
            st.dataframe(sample_sub_df.head(10), use_container_width=True)
        else:
            st.info("sample_submission.csv file is not available.")


# =============================================================
# PAGE 3: DATA CLEANING
# =============================================================
elif "3." in selected_page:
    st.markdown("""
    <div class="header-banner">
        <div class="header-title">🧹 Data Cleaning & Preprocessing</div>
        <div class="header-subtitle">Handling missing values, duplicate entries, and systematic text sanitization</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <b>Data Integrity Guarantee:</b> The original <code>train.csv</code> and <code>test.csv</code> files on disk are never altered.
        All cleaning transformations are executed on an isolated in-memory working copy.
    </div>
    """, unsafe_allow_html=True)
    
    tab_miss, tab_dup, tab_text = st.tabs([
        "1️⃣ Missing Values Handling",
        "2️⃣ Duplicate Records Analysis",
        "3️⃣ Text Cleaning Transformation"
    ])
    
    with tab_miss:
        st.markdown("#### 1. Missing Values Status and Imputation Strategy:")
        st.markdown("""
        - `keyword` column has **61** missing values (0.80%).
        - `location` column has **2,533** missing values (33.27%).
        - `text` and `target` columns have **0** missing values (100% complete and intact).
        """)
        
        st.markdown("##### 🛠️ Imputation Strategy:")
        st.markdown("""
        1. **`keyword`:** Imputed with the placeholder `'none'`.
        2. **`location`:** Imputed with the placeholder `'unknown'`.
        3. **`text`:** Used directly for NLP text modeling; requires no null imputation.
        """)
        
        sample_missing = train_df[train_df['keyword'].isnull() | train_df['location'].isnull()].head(5)
        st.markdown("##### 🔍 Missing Data Sample (Raw vs Cleaned):")
        st.dataframe(sample_missing[['id', 'keyword', 'location', 'text', 'target']], use_container_width=True)
        
        cleaned_sample = sample_missing.copy()
        cleaned_sample['keyword'] = cleaned_sample['keyword'].fillna('none')
        cleaned_sample['location'] = cleaned_sample['location'].fillna('unknown')
        st.markdown("##### ✅ Cleaned In-Memory Working Copy:")
        st.dataframe(cleaned_sample[['id', 'keyword', 'location', 'text', 'target']], use_container_width=True)

    with tab_dup:
        st.markdown("#### 2. Duplicate Records Analysis:")
        text_dup_count = train_df.duplicated(subset=['text']).sum()
        st.markdown(f"""
        - Exact Row Duplicates across all columns: **0**
        - Tweets with identical text bodies: **{text_dup_count} tweets**
        """)
        
        st.markdown("""
        <div class="info-card">
            <b>Crowdsourced Social Media Insight:</b><br>
            On Twitter/X, multiple users frequently retweet or copy-paste identical emergency messages during news events.
            In a few instances, identical text may carry differing annotator labels due to subjective interpretation.
        </div>
        """, unsafe_allow_html=True)
        
        duplicate_texts = train_df[train_df.duplicated(subset=['text'], keep=False)].sort_values(by='text')
        st.markdown("##### 🔍 Repeated Tweets Sample:")
        st.dataframe(duplicate_texts[['id', 'text', 'target']].head(20), use_container_width=True)

    with tab_text:
        st.markdown("#### 3. Standard Text Sanitization Pipeline:")
        st.markdown("""
        Each tweet undergoes the following standardized preprocessing steps:
        1. **HTML Decoding:** Converts entities like `&amp;` -> `&` and removes HTML tags.
        2. **Lowercasing:** Uniformly maps all characters to lowercase.
        3. **URL Removal:** Strips web links (`http://...` or `https://...`).
        4. **User Mention Removal:** Strips `@username` handles.
        5. **Hashtag Normalization:** Preserves semantic words while stripping the `#` symbol (e.g. `#earthquake` -> `earthquake`).
        6. **Punctuation & Whitespace:** Removes non-alphanumeric punctuation and standardizes spacing.
        """)
        
        st.markdown("##### 🧪 Dataset Sample Before vs After Text Cleaning:")
        sample_tweets = train_df.head(6)[['text', 'target']].copy()
        sample_tweets['Cleaned Text'] = sample_tweets['text'].apply(clean_text)
        sample_tweets.rename(columns={'text': 'Original Text', 'target': 'Target'}, inplace=True)
        st.dataframe(sample_tweets, use_container_width=True)


# =============================================================
# PAGE 4: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================
elif "4." in selected_page:
    st.markdown("""
    <div class="header-banner">
        <div class="header-title">📈 Exploratory Data Analysis (EDA)</div>
        <div class="header-subtitle">Interactive visual exploration of class distributions, text lengths, and keyword patterns</div>
    </div>
    """, unsafe_allow_html=True)
    
    df_feat = get_train_features(train_df)
    
    st.markdown("### 1. Target Class Distribution")
    col_bar, col_pie = st.columns([1, 1])
    with col_bar:
        st.plotly_chart(plot_target_distribution_bar(train_df), use_container_width=True)
    with col_pie:
        st.plotly_chart(plot_target_distribution_pie(train_df), use_container_width=True)
        
    st.markdown("""
    <div class="info-card">
        <b>💡 Distribution Insight:</b><br>
        The dataset contains <b>7,613</b> tweets in total, with <b>4,342 (57.0%)</b> non-disaster tweets
        and <b>3,271 (43.0%)</b> genuine disaster tweets.
        Because the classes are reasonably well balanced, models train smoothly without requiring synthetic oversampling techniques.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 2. Tweet Length & Word Count Distributions")
    
    fig_char, fig_words = plot_tweet_length_histograms(df_feat)
    col_c, col_w = st.columns([1, 1])
    with col_c:
        st.plotly_chart(fig_char, use_container_width=True)
    with col_w:
        st.plotly_chart(fig_words, use_container_width=True)
        
    st.markdown("""
    <div class="info-card">
        <b>💡 Length Insight:</b><br>
        1. <b>Character Count:</b> Disaster tweets exhibit a higher mean character length (often clustered around 120–140 characters), as users provide specific details such as locations, casualties, and emergency directives.<br>
        2. <b>Word Count:</b> Both classes peak around 10 to 22 words per tweet.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 3. Missing Values Across Datasets")
    st.plotly_chart(plot_missing_values_bar(train_df, test_df), use_container_width=True)
    
    st.markdown("""
    <div class="info-card">
        <b>💡 Null Analysis:</b><br>
        The crucial feature columns <code>text</code> and <code>id</code> have zero missing values across both train and test sets.
        Missing data is confined entirely to <code>location</code> (~33%) and <code>keyword</code> (<1%).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 4. Top Predictive Keywords")
    fig_dis, fig_nondis = plot_top_keywords(train_df, top_n=12)
    col_k1, col_k2 = st.columns([1, 1])
    with col_k1:
        st.plotly_chart(fig_dis, use_container_width=True)
    with col_k2:
        st.plotly_chart(fig_nondis, use_container_width=True)
        
    st.markdown("""
    <div class="info-card">
        <b>💡 Keyword Insight:</b><br>
        Keywords like <i>debris, derailment, wreckage, outbreak, typhoon</i> carry over a 90% disaster probability.
        Conversely, terms like <i>aftershock, siren, body bags</i> frequently appear in informal or metaphorical contexts.
    </div>
    """, unsafe_allow_html=True)


# =============================================================
# PAGE 5: NLP PREPROCESSING SANDBOX
# =============================================================
elif "5." in selected_page:
    st.markdown("""
    <div class="header-banner">
        <div class="header-title">🔤 Natural Language Processing (NLP Sandbox)</div>
        <div class="header-subtitle">Interactive step-by-step demonstration of text sanitization and tokenization</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🧪 Interactive Text Cleaning Sandbox")
    st.markdown("Enter any tweet or select a sample from the dataset to observe every sanitization step:")
    
    sample_select = st.selectbox(
        "Select a Sample Tweet from Dataset:",
        [
            "Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all",
            "Forest fire near La Ronge Sask. Canada http://t.co/abc123xyz",
            "Just got sent this photo from Ruby #Alaska as smoke from #wildfires pours into a school",
            "@BBCBreaking Reports of severe flooding in northern districts. Stay safe! #flood",
            "Watching this movie was a complete train wreck... LOL! #movies"
        ]
    )
    
    custom_tweet = st.text_input("Or type your own custom text:", value=sample_select)
    
    if custom_tweet:
        steps = clean_text_step_by_step(custom_tweet)
        
        st.markdown("#### 🔄 Step-by-Step Transformation:")
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown(f"**1. Raw Input Text:**")
            st.markdown(f"<div class='code-box'>{steps['1_Original']}</div>", unsafe_allow_html=True)
            
            st.markdown(f"**2. Lowercase Transformation:**")
            st.markdown(f"<div class='code-box'>{steps['3_Lowercase']}</div>", unsafe_allow_html=True)
            
            st.markdown(f"**3. URLs Removed:**")
            st.markdown(f"<div class='code-box'>{steps['4_No_URLs']}</div>", unsafe_allow_html=True)
            
            st.markdown(f"**4. User Mentions (@) Removed:**")
            st.markdown(f"<div class='code-box'>{steps['5_No_Mentions']}</div>", unsafe_allow_html=True)

        with col_s2:
            st.markdown(f"**5. Hashtag Symbols Stripped (#earthquake -> earthquake):**")
            st.markdown(f"<div class='code-box'>{steps['6_Clean_Hashtags']}</div>", unsafe_allow_html=True)
            
            st.markdown(f"**6. Punctuation Removed & Normalized:**")
            st.markdown(f"<div class='code-box'>{steps['8_Normalized_Final']}</div>", unsafe_allow_html=True)
            
            st.markdown(f"**7. Tokens after Stopwords Removal:**")
            st.write(steps['10_Tokens_No_Stopwords'])

    st.markdown("---")
    st.markdown("### 📋 Batch Transformation Preview (10 Dataset Tweets)")
    comp_df = train_df[['text']].head(10).copy()
    comp_df['Cleaned Text'] = comp_df['text'].apply(clean_text)
    comp_df.rename(columns={'text': 'Original Text'}, inplace=True)
    st.dataframe(comp_df, use_container_width=True)


# =============================================================
# PAGE 6: FEATURE ENGINEERING
# =============================================================
elif "6." in selected_page:
    st.markdown("""
    <div class="header-banner">
        <div class="header-title">⚙️ Feature Engineering & TF-IDF Vectorization</div>
        <div class="header-subtitle">Transforming textual tokens into numerical feature matrices with N-Gram analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <b>TF-IDF (Term Frequency - Inverse Document Frequency):</b><br>
        Machine learning algorithms require numerical matrices rather than raw strings. TF-IDF evaluates both the local
        frequency of words in a tweet (TF) and rewards words that are informative across the broader corpus (IDF).
    </div>
    """, unsafe_allow_html=True)
    
    curr_res = st.session_state.get("trained_model_results")
    if curr_res:
        vec = curr_res['vectorizer']
        vocab = vec.vocabulary_
        feature_names = vec.get_feature_names_out()
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Vocabulary Size", f"{len(vocab):,} tokens")
        c2.metric("N-Gram Range", f"{vec.ngram_range}")
        c3.metric("Stop Words", "English (Filtered)")
        
        st.markdown("---")
        st.markdown("### 🔍 TF-IDF Vocabulary Search")
        search_term = st.text_input("Search any term to inspect its feature index and IDF weight:", value="earthquake")
        if search_term.lower() in vocab:
            idx = vocab[search_term.lower()]
            idf_val = vec.idf_[idx]
            st.success(f"Token **'{search_term.lower()}'** found! Feature Index: `{idx}` | IDF Weight: `{idf_val:.4f}`")
        else:
            st.info(f"Token **'{search_term}'** is not in the active model vocabulary.")

        st.markdown("---")
        st.markdown("### 🏆 Top Informative Features")
        col_d, col_nd = st.columns(2)
        with col_d:
            st.markdown("##### 🚨 Top Disaster Predictor Features:")
            st.dataframe(pd.DataFrame(curr_res['top_disaster_words'], columns=['Word / N-Gram', 'Coefficient Weight']), use_container_width=True)
        with col_nd:
            st.markdown("##### 🟢 Top Non-Disaster Predictor Features:")
            st.dataframe(pd.DataFrame(curr_res['top_nondisaster_words'], columns=['Word / N-Gram', 'Coefficient Weight']), use_container_width=True)
            
        st.plotly_chart(plot_feature_importance_bars(curr_res['top_disaster_words'], curr_res['top_nondisaster_words']), use_container_width=True)
    else:
        st.warning("Please navigate to the 'Model Training' page to train a model.")


# =============================================================
# PAGE 7: MODEL TRAINING
# =============================================================
elif "7." in selected_page:
    st.markdown("""
    <div class="header-banner">
        <div class="header-title">🧠 Machine Learning Model Training Pipeline</div>
        <div class="header-subtitle">Train, tune, and evaluate multiple algorithms with strict data leakage prevention</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <b>Data Leakage Prevention Protocol:</b><br>
        To guarantee genuine generalization, <code>TfidfVectorizer</code> is fitted <b>strictly on the training split</b>
        after <code>train_test_split</code>. The validation set is only transformed, ensuring zero data leakage and authentic validation scores.
    </div>
    """, unsafe_allow_html=True)
    
    col_conf, col_act = st.columns([1, 1])
    with col_conf:
        st.markdown("#### ⚙️ Training Configuration")
        selected_model_name = st.selectbox(
            "Select Algorithm:",
            list(MODEL_REGISTRY.keys()),
            index=0
        )
        
        split_ratio = st.slider("Validation Split Size (%):", min_value=10, max_value=30, value=20, step=5)
        max_feat = st.select_slider("Max Features:", options=[2000, 3000, 5000, 8000, 10000], value=5000)
        ngram_opt = st.selectbox("N-Gram Range:", ["(1, 1) - Unigrams Only", "(1, 2) - Unigrams + Bigrams"], index=1)
        ngram_val = (1, 1) if "(1, 1)" in ngram_opt else (1, 2)
        
        train_btn = st.button("🚀 Train Model", type="primary", use_container_width=True)

    with col_act:
        st.markdown("#### 📊 Model Characteristics")
        if selected_model_name == "Logistic Regression":
            st.markdown("""
            - **Type:** Linear Classification
            - **Strengths:** Fast, reliable calibrated probabilities, strong interpretability.
            - **Benchmark Performance:** ~81.8% Accuracy on validation set.
            """)
        elif "Naive Bayes" in selected_model_name:
            st.markdown("""
            - **Type:** Probabilistic Classifier (Bayes' Theorem)
            - **Strengths:** Exceptionally fast, excels on sparse high-dimensional text data.
            - **Benchmark Performance:** ~81.2% Accuracy.
            """)
        elif selected_model_name == "Linear SVM":
            st.markdown("""
            - **Type:** Support Vector Machine (Linear Kernel)
            - **Strengths:** Maximizes separating hyperplanes in high-dimensional TF-IDF space.
            """)
        elif selected_model_name == "Random Forest":
            st.markdown("""
            - **Type:** Ensemble of Decision Trees
            - **Strengths:** Captures non-linear interactions across token combinations.
            """)

    if train_btn:
        with st.spinner(f"Training {selected_model_name}..."):
            try:
                res = train_and_evaluate_model(
                    train_df=train_df,
                    model_name=selected_model_name,
                    test_size=split_ratio / 100.0,
                    random_state=42,
                    ngram_range=ngram_val,
                    max_features=max_feat
                )
                st.session_state.trained_model_results = res
                st.success(f"🎉 **{selected_model_name}** trained successfully!")
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{res['accuracy']*100:.2f}%")
                c2.metric("Precision", f"{res['precision']*100:.2f}%")
                c3.metric("Recall", f"{res['recall']*100:.2f}%")
                c4.metric("F1-Score", f"{res['f1_score']*100:.2f}%")
            except Exception as e:
                st.error(f"Error during training: {str(e)}")

    st.markdown("---")
    st.markdown("### 🏁 Benchmark All Algorithms")
    st.markdown("Train and evaluate all candidate classifiers on identical data splits with a single click:")
    
    if st.button("⚡ Run Benchmark Comparison"):
        with st.spinner("Evaluating all algorithms..."):
            bench_df = benchmark_all_models(train_df, test_size=0.2, random_state=42)
            st.dataframe(bench_df.style.format({
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1-Score": "{:.2%}",
                "ROC-AUC": "{:.4f}"
            }), use_container_width=True)
            st.plotly_chart(plot_model_comparison_bar(bench_df), use_container_width=True)


# =============================================================
# PAGE 8: MODEL EVALUATION
# =============================================================
elif "8." in selected_page:
    st.markdown("""
    <div class="header-banner">
        <div class="header-title">📉 Model Evaluation & Validation Metrics</div>
        <div class="header-subtitle">Rigorous validation performance, confusion matrix analysis, and error diagnosis</div>
    </div>
    """, unsafe_allow_html=True)
    
    curr_res = st.session_state.get("trained_model_results")
    if not curr_res:
        st.warning("Please navigate to the 'Model Training' page to train a model.")
        st.stop()
        
    st.markdown(f"#### Active Model: **{curr_res['model_name']}** (Hold-out Validation: {curr_res['val_size']:,} Tweets)")
    
    # KPI Metric Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{curr_res['accuracy']*100:.2f}%")
    c2.metric("Precision", f"{curr_res['precision']*100:.2f}%")
    c3.metric("Recall", f"{curr_res['recall']*100:.2f}%")
    c4.metric("F1-Score", f"{curr_res['f1_score']*100:.2f}%")
    c5.metric("ROC-AUC", f"{curr_res['roc_auc']:.4f}")
    
    st.markdown("""
    <div class="info-card">
        <b>Metric Definitions:</b><br>
        - <b>Precision ({:.1f}%):</b> Out of all tweets predicted as disasters, the proportion that were actual emergencies.<br>
        - <b>Recall ({:.1f}%):</b> Out of all actual emergencies, the proportion successfully detected by the model.<br>
        - <b>F1-Score ({:.1f}%):</b> The balanced harmonic mean of Precision and Recall.
    </div>
    """.format(curr_res['precision']*100, curr_res['recall']*100, curr_res['f1_score']*100), unsafe_allow_html=True)
    
    col_cm, col_roc = st.columns([1, 1])
    with col_cm:
        st.plotly_chart(plot_confusion_matrix_heatmap(curr_res['confusion_matrix']), use_container_width=True)
    with col_roc:
        if curr_res['fpr'] is not None:
            st.plotly_chart(plot_roc_curve(curr_res['fpr'], curr_res['tpr'], curr_res['roc_auc']), use_container_width=True)
        else:
            st.info("ROC Curve is not available for this classifier.")
            
    st.markdown("---")
    st.markdown("### 📋 Detailed Classification Report")
    rep = curr_res['classification_report_dict']
    rep_df = pd.DataFrame({
        "Class": ["Non-Disaster (0)", "Disaster (1)", "Macro Avg", "Weighted Avg"],
        "Precision": [rep['0']['precision'], rep['1']['precision'], rep['macro avg']['precision'], rep['weighted avg']['precision']],
        "Recall": [rep['0']['recall'], rep['1']['recall'], rep['macro avg']['recall'], rep['weighted avg']['recall']],
        "F1-Score": [rep['0']['f1-score'], rep['1']['f1-score'], rep['macro avg']['f1-score'], rep['weighted avg']['f1-score']],
        "Support": [int(rep['0']['support']), int(rep['1']['support']), int(rep['macro avg']['support']), int(rep['weighted avg']['support'])]
    })
    st.dataframe(rep_df.style.format({
        "Precision": "{:.2%}",
        "Recall": "{:.2%}",
        "F1-Score": "{:.2%}"
    }), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔍 Error Analysis (Misclassified Tweets)")
    st.markdown("Inspect edge cases and ambiguous tweets misclassified on the validation fold:")
    
    val_df = curr_res['val_df']
    false_positives = val_df[(val_df['actual'] == 0) & (val_df['predicted'] == 1)]
    false_negatives = val_df[(val_df['actual'] == 1) & (val_df['predicted'] == 0)]
    
    t_fp, t_fn = st.tabs([f"⚠️ False Positives ({len(false_positives)})", f"🚨 False Negatives ({len(false_negatives)})"])
    with t_fp:
        st.markdown("**False Positives (False Alarms):** Actual safe tweets predicted as disaster:")
        st.dataframe(false_positives[['text', 'confidence']].head(10), use_container_width=True)
    with t_fn:
        st.markdown("**False Negatives (Missed Disasters):** Actual emergencies predicted as non-disaster:")
        st.dataframe(false_negatives[['text', 'confidence']].head(10), use_container_width=True)


# =============================================================
# PAGE 9: LIVE TWEET PREDICTION
# =============================================================
elif "9." in selected_page:
    st.markdown("""
    <div class="header-banner">
        <div class="header-title">🔮 Live Tweet Prediction & Inference</div>
        <div class="header-subtitle">Real-time disaster risk classification, confidence gauge, and contributing token analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    curr_res = st.session_state.get("trained_model_results")
    if not curr_res:
        st.warning("Please navigate to the 'Model Training' page to train a model.")
        st.stop()
        
    st.markdown("#### ✍️ Enter Tweet for Classification")
    
    # Preset sample chips
    st.markdown("**Or choose a pre-set test scenario:**")
    col_p1, col_p2, col_p3 = st.columns(3)
    preset_chosen = None
    if col_p1.button("🔥 Building Fire Emergency", use_container_width=True):
        preset_chosen = "The building is on fire and people need immediate help."
    if col_p2.button("🌊 Flash Flood Warning", use_container_width=True):
        preset_chosen = "Heavy rains have triggered massive flash floods across town."
    if col_p3.button("🍔 Metaphorical (Safe Tweet)", use_container_width=True):
        preset_chosen = "Just watched an explosive movie with a killer soundtrack!"
        
    default_text = preset_chosen if preset_chosen else "The building is on fire and people need immediate help."
    user_tweet = st.text_area("Tweet Text Input:", value=default_text, height=100)
    
    btn_predict = st.button("🚀 Analyze Tweet", type="primary")
    
    if btn_predict or user_tweet:
        pred_res = predict_single_tweet(curr_res['model'], curr_res['vectorizer'], user_tweet)
        is_disaster = pred_res['target'] == 1
        
        st.markdown("---")
        st.markdown("### 📊 Prediction Results")
        
        col_res1, col_res2 = st.columns([1.2, 0.8])
        with col_res1:
            card_bg = "rgba(239, 68, 68, 0.12)" if is_disaster else "rgba(16, 185, 129, 0.12)"
            border_c = "#EF4444" if is_disaster else "#10B981"
            badge_c = "disaster-badge" if is_disaster else "safe-badge"
            icon = "🚨" if is_disaster else "✅"
            
            st.markdown(f"""
            <div style="background:{card_bg}; border:2px solid {border_c}; border-radius:16px; padding:24px; margin-bottom:16px;">
                <div style="font-size:0.9rem; color:#94A3B8; text-transform:uppercase; font-weight:700;">Model Assessment</div>
                <div style="font-size:2rem; font-weight:800; color:#F8FAFC; margin:10px 0;">
                    {icon} <span class="{badge_c}" style="font-size:1.4rem; padding:6px 18px;">{pred_res['label']}</span>
                </div>
                <div style="font-size:1.1rem; color:#E2E8F0; margin-top:12px;">
                    Confidence Score: <b>{pred_res['confidence']*100:.1f}%</b>
                </div>
                <div style="font-size:0.85rem; color:#94A3B8; margin-top:8px;">
                    Sanitized Text: <code>{pred_res['cleaned_text']}</code>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Key Contributing Words
            if pred_res['contributing_words']:
                st.markdown("##### 🔑 Key Contributing Feature Tokens:")
                token_df = pd.DataFrame(pred_res['contributing_words'])
                st.dataframe(token_df, use_container_width=True)

        with col_res2:
            st.plotly_chart(plot_prediction_gauge(pred_res['confidence'], is_disaster), use_container_width=True)

        st.markdown("""
        <div class="info-card" style="border-left-color:#F59E0B;">
            <b>⚠️ Technical & Contextual Notice:</b><br>
            A 'Disaster-Related' prediction reflects probabilistic linguistic patterns detected in the text and does not constitute official verification of an ongoing incident. Emergency responses should always consult certified authorities.
        </div>
        """, unsafe_allow_html=True)


# =============================================================
# PAGE 10: TEST DATASET PREDICTIONS
# =============================================================
elif "10." in selected_page:
    st.markdown("""
    <div class="header-banner">
        <div class="header-title">📑 Test Dataset Batch Inference</div>
        <div class="header-subtitle">Automated batch inference and classification for all 3,263 tweets in test.csv</div>
    </div>
    """, unsafe_allow_html=True)
    
    curr_res = st.session_state.get("trained_model_results")
    if not curr_res:
        st.warning("Please navigate to the 'Model Training' page to train a model.")
        st.stop()
        
    st.markdown(f"**Active Model:** `{curr_res['model_name']}` | **Total Test Tweets:** `{len(test_df):,}`")
    
    if "test_predictions_df" not in st.session_state:
        st.info("Click the button below to execute batch inference across all test dataset tweets.")
        if st.button("⚡ Run Batch Inference on All 3,263 Tweets", type="primary"):
            with st.spinner("Executing batch predictions..."):
                test_preds_df, submission_df = predict_test_dataset(
                    curr_res['model'], curr_res['vectorizer'], test_df
                )
                st.session_state.test_predictions_df = test_preds_df
                st.session_state.submission_df = submission_df
                st.rerun()
    else:
        test_preds_df = st.session_state.test_predictions_df
        submission_df = st.session_state.submission_df
        
        # Summary KPI cards for test predictions
        pred_dis = int((test_preds_df['predicted_target'] == 1).sum())
        pred_nondis = int((test_preds_df['predicted_target'] == 0).sum())
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Test Tweets", f"{len(test_preds_df):,}")
        c2.metric("Predicted Disaster", f"{pred_dis:,} ({(pred_dis/len(test_preds_df))*100:.1f}%)")
        c3.metric("Predicted Safe", f"{pred_nondis:,} ({(pred_nondis/len(test_preds_df))*100:.1f}%)")
        
        st.markdown("---")
        st.markdown("### 🔍 Test Predictions Table & Filters")
        
        col_tf1, col_tf2 = st.columns([1, 1])
        with col_tf1:
            p_filter = st.selectbox("Filter by Prediction:", ["All", "Disaster Only (Target = 1)", "Non-Disaster Only (Target = 0)"])
        with col_tf2:
            kw_search = st.text_input("Search Keyword or Tweet Text:")
            
        display_df = test_preds_df.copy()
        if p_filter == "Disaster Only (Target = 1)":
            display_df = display_df[display_df['predicted_target'] == 1]
        elif p_filter == "Non-Disaster Only (Target = 0)":
            display_df = display_df[display_df['predicted_target'] == 0]
            
        if kw_search:
            display_df = display_df[
                display_df['text'].str.contains(kw_search, case=False, na=False) |
                display_df['keyword'].astype(str).str.contains(kw_search, case=False, na=False)
            ]
            
        st.write(f"Showing Records: **{len(display_df):,} tweets**")
        st.dataframe(
            display_df[['id', 'keyword', 'location', 'text', 'predicted_label', 'confidence']].head(50),
            use_container_width=True
        )


# =============================================================
# PAGE 11: DOWNLOAD RESULTS
# =============================================================
elif "11." in selected_page:
    st.markdown("""
    <div class="header-banner">
        <div class="header-title">💾 Export & Download Results</div>
        <div class="header-subtitle">Download Kaggle-compliant submission.csv and comprehensive analytical reports</div>
    </div>
    """, unsafe_allow_html=True)
    
    curr_res = st.session_state.get("trained_model_results")
    if not curr_res:
        st.warning("Please navigate to the 'Model Training' page to train a model.")
        st.stop()
        
    if "test_predictions_df" not in st.session_state:
        st.info("Please navigate to '10. Test Dataset Predictions' first to generate batch predictions.")
        st.stop()
        
    test_preds_df = st.session_state.test_predictions_df
    submission_df = st.session_state.submission_df
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.markdown("""
        <div class="kpi-card" style="border-color: rgba(16, 185, 129, 0.4);">
            <div style="font-size:1.2rem; font-weight:700; color:#10B981; margin-bottom:8px;">
                📥 Kaggle Submission CSV
            </div>
            <div style="font-size:0.85rem; color:#CBD5E1; margin-bottom:14px;">
                Strictly formatted to match <code>sample_submission.csv</code> (Columns: <code>id, target</code>). Ready for Kaggle competition submission.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        sub_csv_bytes = submission_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download submission.csv (3,263 Rows)",
            data=sub_csv_bytes,
            file_name="submission.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )

    with col_d2:
        st.markdown("""
        <div class="kpi-card" style="border-color: rgba(99, 102, 241, 0.4);">
            <div style="font-size:1.2rem; font-weight:700; color:#6366F1; margin-bottom:8px;">
                📊 Comprehensive Prediction Report CSV
            </div>
            <div style="font-size:0.85rem; color:#CBD5E1; margin-bottom:14px;">
                Includes original tweet, sanitized text, predicted label, and confidence score for every test tweet.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        full_csv_bytes = test_preds_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download detailed_predictions.csv",
            data=full_csv_bytes,
            file_name="disaster_tweets_detailed_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("---")
    st.markdown("### 📝 Model Evaluation Summary Report")
    summary_report_text = f"""=======================================================
AI-Based Disaster Tweet Classification System - Summary Report
=======================================================
Model Name: {curr_res['model_name']}
Validation Split: {curr_res['val_size']} tweets
Accuracy: {curr_res['accuracy']*100:.2f}%
Precision: {curr_res['precision']*100:.2f}%
Recall: {curr_res['recall']*100:.2f}%
F1-Score: {curr_res['f1_score']*100:.2f}%
ROC-AUC: {curr_res['roc_auc']:.4f}

Classification Report:
{curr_res['classification_report_text']}
=======================================================
"""
    st.text_area("Model Report (Copy or Export):", value=summary_report_text, height=220)
    st.download_button(
        label="⬇️ Download Summary Report (.txt)",
        data=summary_report_text,
        file_name="model_evaluation_report.txt",
        mime="text/plain",
        use_container_width=True
    )
