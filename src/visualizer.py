import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Cohesive modern color palette
COLOR_PRIMARY = "#6366F1"      # Indigo / Violet
COLOR_DISASTER = "#EF4444"     # Vibrant Coral / Red
COLOR_NON_DISASTER = "#10B981" # Emerald / Teal Green
COLOR_ACCENT = "#F59E0B"       # Amber / Orange
COLOR_BG_DARK = "#1E293B"      # Dark Slate
COLOR_GRID = "rgba(255, 255, 255, 0.08)"

PLOT_LAYOUT_COMMON = dict(
    font=dict(family="Inter, system-ui, sans-serif", size=13),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=40, t=50, b=40),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=12)
    )
)

def plot_target_distribution_bar(train_df):
    """
    Plots a stylish bar chart showing Disaster vs Non-Disaster distribution.
    """
    counts = train_df['target'].value_counts().reset_index()
    counts.columns = ['target', 'count']
    counts['label'] = counts['target'].map({0: 'Non-Disaster (0)', 1: 'Disaster (1)'})
    counts['percentage'] = (counts['count'] / counts['count'].sum()) * 100
    
    fig = go.Figure(data=[
        go.Bar(
            x=counts['label'],
            y=counts['count'],
            text=[f"<b>{c:,}</b> ({p:.1f}%)" for c, p in zip(counts['count'], counts['percentage'])],
            textposition='outside',
            marker=dict(
                color=[COLOR_NON_DISASTER, COLOR_DISASTER],
                line=dict(color='rgba(255,255,255,0.2)', width=1.5),
                cornerradius=8
            ),
            width=0.45
        )
    ])
    
    fig.update_layout(
        **PLOT_LAYOUT_COMMON,
        title=dict(text="<b>Target Distribution (Disaster vs Non-Disaster)</b>", font=dict(size=16)),
        xaxis=dict(title="Category", showgrid=False),
        yaxis=dict(title="Number of Tweets", showgrid=True, gridcolor=COLOR_GRID),
        height=380
    )
    return fig

def plot_target_distribution_pie(train_df):
    """
    Plots a modern donut/pie chart showing class percentage.
    """
    counts = train_df['target'].value_counts()
    labels = ['Non-Disaster (0)', 'Disaster (1)']
    values = [counts.get(0, 0), counts.get(1, 0)]
    
    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker=dict(
                colors=[COLOR_NON_DISASTER, COLOR_DISASTER],
                line=dict(color='#0F172A', width=2)
            ),
            textinfo='label+percent',
            textfont=dict(size=13, color='white'),
            hovertemplate="<b>%{label}</b><br>Total Tweets: %{value:,}<br>Share: %{percent}<extra></extra>"
        )
    ])
    
    fig.update_layout(
        **PLOT_LAYOUT_COMMON,
        title=dict(text="<b>Class Proportion (Percentage Share)</b>", font=dict(size=16)),
        height=380,
        showlegend=False,
        annotations=[
            dict(
                text=f"<b>{len(train_df):,}</b><br><span style='font-size:12px;color:#94A3B8'>Total Tweets</span>",
                x=0.5, y=0.5, font=dict(size=18, color="#F8FAFC"), showarrow=False
            )
        ]
    )
    return fig

def plot_tweet_length_histograms(df_feat):
    """
    Plots side-by-side histograms for character length and word count by target class.
    """
    # Character Length
    fig_char = go.Figure()
    
    fig_char.add_trace(go.Histogram(
        x=df_feat[df_feat['target'] == 0]['char_length'],
        name='Non-Disaster',
        marker_color=COLOR_NON_DISASTER,
        opacity=0.75,
        nbinsx=40
    ))
    
    fig_char.add_trace(go.Histogram(
        x=df_feat[df_feat['target'] == 1]['char_length'],
        name='Disaster',
        marker_color=COLOR_DISASTER,
        opacity=0.75,
        nbinsx=40
    ))
    
    fig_char.update_layout(
        **PLOT_LAYOUT_COMMON,
        barmode='overlay',
        title=dict(text="<b>Character Length Distribution</b>", font=dict(size=15)),
        xaxis=dict(title="Character Count", showgrid=True, gridcolor=COLOR_GRID),
        yaxis=dict(title="Frequency", showgrid=True, gridcolor=COLOR_GRID),
        height=380
    )
    
    # Word Count
    fig_words = go.Figure()
    
    fig_words.add_trace(go.Histogram(
        x=df_feat[df_feat['target'] == 0]['word_count'],
        name='Non-Disaster',
        marker_color=COLOR_NON_DISASTER,
        opacity=0.75,
        nbinsx=30
    ))
    
    fig_words.add_trace(go.Histogram(
        x=df_feat[df_feat['target'] == 1]['word_count'],
        name='Disaster',
        marker_color=COLOR_DISASTER,
        opacity=0.75,
        nbinsx=30
    ))
    
    fig_words.update_layout(
        **PLOT_LAYOUT_COMMON,
        barmode='overlay',
        title=dict(text="<b>Word Count Distribution</b>", font=dict(size=15)),
        xaxis=dict(title="Word Count", showgrid=True, gridcolor=COLOR_GRID),
        yaxis=dict(title="Frequency", showgrid=True, gridcolor=COLOR_GRID),
        height=380
    )
    
    return fig_char, fig_words

def plot_missing_values_bar(train_df, test_df):
    """
    Plots missing values comparison for train and test datasets.
    """
    train_nulls = train_df.isnull().sum().reset_index()
    train_nulls.columns = ['column', 'missing_count']
    train_nulls['dataset'] = 'Train (7,613)'
    train_nulls['pct'] = (train_nulls['missing_count'] / len(train_df)) * 100
    
    test_nulls = test_df.isnull().sum().reset_index()
    test_nulls.columns = ['column', 'missing_count']
    test_nulls['dataset'] = 'Test (3,263)'
    test_nulls['pct'] = (test_nulls['missing_count'] / len(test_df)) * 100
    
    combined = pd.concat([train_nulls, test_nulls], ignore_index=True)
    
    fig = px.bar(
        combined,
        x='column',
        y='missing_count',
        color='dataset',
        barmode='group',
        text=combined.apply(lambda r: f"{r['missing_count']} ({r['pct']:.1f}%)" if r['missing_count'] > 0 else "0", axis=1),
        color_discrete_sequence=[COLOR_PRIMARY, COLOR_ACCENT]
    )
    
    fig.update_traces(textposition='outside')
    fig.update_layout(
        **PLOT_LAYOUT_COMMON,
        title=dict(text="<b>Missing Values Analysis</b>", font=dict(size=16)),
        xaxis=dict(title="Columns", showgrid=False),
        yaxis=dict(title="Missing Values Count", showgrid=True, gridcolor=COLOR_GRID),
        height=380
    )
    return fig

def plot_top_keywords(df, top_n=15):
    """
    Plots top keywords correlated with disaster vs non-disaster tweets.
    """
    kw_df = df[df['keyword'].notnull() & (df['keyword'] != '') & (df['keyword'] != 'none')]
    
    # Keyword target mean & counts
    kw_stats = kw_df.groupby('keyword')['target'].agg(['count', 'mean']).reset_index()
    kw_stats = kw_stats[kw_stats['count'] >= 10] # filter rare
    
    # Top disaster keywords
    top_disaster_kw = kw_stats.sort_values(by='mean', ascending=False).head(top_n)
    
    fig_disaster = go.Figure(go.Bar(
        x=top_disaster_kw['mean'] * 100,
        y=top_disaster_kw['keyword'],
        orientation='h',
        marker=dict(color=COLOR_DISASTER),
        text=[f"{p:.1f}% ({c})" for p, c in zip(top_disaster_kw['mean'] * 100, top_disaster_kw['count'])],
        textposition='outside'
    ))
    
    fig_disaster.update_layout(
        **PLOT_LAYOUT_COMMON,
        title=dict(text=f"<b>Top {top_n} Disaster Keywords</b>", font=dict(size=15)),
        xaxis=dict(title="Disaster Probability (%)", range=[0, 115], showgrid=True, gridcolor=COLOR_GRID),
        yaxis=dict(autorange="reversed"),
        height=420
    )
    
    # Top non-disaster keywords
    top_nondisaster_kw = kw_stats.sort_values(by='mean', ascending=True).head(top_n)
    
    fig_non_disaster = go.Figure(go.Bar(
        x=(1 - top_nondisaster_kw['mean']) * 100,
        y=top_nondisaster_kw['keyword'],
        orientation='h',
        marker=dict(color=COLOR_NON_DISASTER),
        text=[f"{(1-p)*100:.1f}% ({c})" for p, c in zip(top_nondisaster_kw['mean'], top_nondisaster_kw['count'])],
        textposition='outside'
    ))
    
    fig_non_disaster.update_layout(
        **PLOT_LAYOUT_COMMON,
        title=dict(text=f"<b>Top {top_n} Non-Disaster Keywords</b>", font=dict(size=15)),
        xaxis=dict(title="Non-Disaster Probability (%)", range=[0, 115], showgrid=True, gridcolor=COLOR_GRID),
        yaxis=dict(autorange="reversed"),
        height=420
    )
    
    return fig_disaster, fig_non_disaster

def plot_confusion_matrix_heatmap(cm, labels=['Non-Disaster', 'Disaster']):
    """
    Plots an annotated interactive confusion matrix heatmap with TP, FP, TN, FN.
    """
    tn, fp, fn, tp = cm.ravel()
    annotations_text = [
        [f"<b>TN (True Negative)</b><br>{tn:,}<br><span style='font-size:11px'>Correctly Identified Safe</span>",
         f"<b>FP (False Positive)</b><br>{fp:,}<br><span style='font-size:11px'>False Alarm (Incorrect Disaster)</span>"],
        [f"<b>FN (False Negative)</b><br>{fn:,}<br><span style='font-size:11px'>Missed Disaster</span>",
         f"<b>TP (True Positive)</b><br>{tp:,}<br><span style='font-size:11px'>Correctly Identified Disaster</span>"]
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale=[[0, '#0F172A'], [0.5, '#3B82F6'], [1, '#10B981']],
        showscale=False,
        text=annotations_text,
        texttemplate="%{text}",
        textfont=dict(size=13, color='white')
    ))
    
    fig.update_layout(
        **PLOT_LAYOUT_COMMON,
        title=dict(text="<b>Confusion Matrix Heatmap</b>", font=dict(size=16)),
        xaxis=dict(title="<b>Predicted Label</b>", side="bottom"),
        yaxis=dict(title="<b>Actual Label</b>", autorange="reversed"),
        height=380
    )
    return fig

def plot_roc_curve(fpr, tpr, roc_auc):
    """
    Plots ROC curve with AUC score.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.4f})',
        line=dict(color=COLOR_PRIMARY, width=3)
    ))
    
    # Baseline diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Baseline (AUC = 0.50)',
        line=dict(color='#64748B', width=1.5, dash='dash')
    ))
    
    fig.update_layout(
        **PLOT_LAYOUT_COMMON,
        title=dict(text=f"<b>ROC Curve (Receiver Operating Characteristic) - AUC: {roc_auc:.4f}</b>", font=dict(size=16)),
        xaxis=dict(title="False Positive Rate (FPR)", range=[0, 1], showgrid=True, gridcolor=COLOR_GRID),
        yaxis=dict(title="True Positive Rate (TPR / Recall)", range=[0, 1.05], showgrid=True, gridcolor=COLOR_GRID),
        height=380
    )
    return fig

def plot_feature_importance_bars(top_disaster, top_nondisaster):
    """
    Plots horizontal bars showing the strongest predictor tokens for both classes.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[w[1] for w in top_disaster],
        y=[w[0] for w in top_disaster],
        orientation='h',
        marker=dict(color=COLOR_DISASTER),
        name='Disaster Signals'
    ))
    
    fig.add_trace(go.Bar(
        x=[abs(w[1]) for w in top_nondisaster],
        y=[w[0] for w in top_nondisaster],
        orientation='h',
        marker=dict(color=COLOR_NON_DISASTER),
        name='Non-Disaster Signals'
    ))
    
    fig.update_layout(
        **PLOT_LAYOUT_COMMON,
        title=dict(text="<b>Top Contributing Feature Words / N-Grams</b>", font=dict(size=16)),
        xaxis=dict(title="Feature Coefficient Weight", showgrid=True, gridcolor=COLOR_GRID),
        yaxis=dict(autorange="reversed"),
        height=480
    )
    return fig

def plot_prediction_gauge(confidence, is_disaster):
    """
    Plots a gauge meter showing confidence level for prediction.
    """
    color = COLOR_DISASTER if is_disaster else COLOR_NON_DISASTER
    label = "Disaster Risk Confidence" if is_disaster else "Non-Disaster Confidence"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>{label}</b>", 'font': {'size': 16, 'color': 'white'}},
        number={'suffix': "%", 'font': {'size': 32, 'color': 'white'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 1,
            'bordercolor': "rgba(255,255,255,0.2)",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(16, 185, 129, 0.2)'},
                {'range': [50, 75], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [75, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 3},
                'thickness': 0.8,
                'value': confidence * 100
            }
        }
    ))
    
    fig.update_layout(
        **PLOT_LAYOUT_COMMON,
        height=260,
        margin=dict(l=30, r=30, t=40, b=20)
    )
    return fig

def plot_model_comparison_bar(comparison_df):
    """
    Plots multi-metric grouped bar chart comparing algorithms.
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    fig = go.Figure()
    colors = [COLOR_PRIMARY, COLOR_ACCENT, COLOR_NON_DISASTER, COLOR_DISASTER]
    
    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=metric,
            x=comparison_df['Algorithm'],
            y=comparison_df[metric] * 100,
            marker_color=colors[i % len(colors)],
            text=[f"{v*100:.1f}%" for v in comparison_df[metric]],
            textposition='outside'
        ))
        
    fig.update_layout(
        **PLOT_LAYOUT_COMMON,
        barmode='group',
        title=dict(text="<b>Model Performance Benchmark Comparison</b>", font=dict(size=16)),
        xaxis=dict(title="Algorithms", showgrid=False),
        yaxis=dict(title="Score (%)", range=[0, 110], showgrid=True, gridcolor=COLOR_GRID),
        height=400
    )
    return fig
