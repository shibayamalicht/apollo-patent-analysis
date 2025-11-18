# ==================================================================
# --- 1. ライブラリのインポート ---
# ==================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import datetime
import warnings
import io
import unicodedata
import re

# UMAP / HDBSCAN
from umap import UMAP 
import hdbscan 

# 統計マップ用
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import japanize_matplotlib # 日本語化

# 警告を非表示
warnings.filterwarnings('ignore')

# ==================================================================
# --- 2. ヘルパー関数 (Saturn V / ATLAS 共通) ---
# ==================================================================

@st.cache_data
def get_top_tfidf_words(_row_vector, feature_names, top_n=5):
    scores = _row_vector.toarray().flatten() 
    indices = np.argsort(scores)[::-1]
    non_zero_indices = [i for i in indices if scores[i] > 0]
    top_indices = non_zero_indices[:top_n]
    top_words = [feature_names[i] for i in top_indices]
    return ", ".join(top_words)

def update_hover_text(df, col_map):
    hover_texts = []
    for index, row in df.iterrows():
        text = ""
        if col_map['title'] and pd.notna(row[col_map['title']]): text += f"<b>名称:</b> {str(row[col_map['title']])[:50]}...<br>"
        if col_map['app_num'] and pd.notna(row[col_map['app_num']]): text += f"<b>番号:</b> {row[col_map['app_num']]}<br>"
        if col_map['applicant'] and pd.notna(row[col_map['applicant']]): text += f"<b>出願人:</b> {str(row[col_map['applicant']])[:50]}...<br>"
        if 'characteristic_words' in row: text += f"<b>特徴語:</b> {row['characteristic_words']}<br>"
        if 'cluster_label' in row: text += f"<b>クラスタ:</b> {row['cluster_label']}"
        hover_texts.append(text)
    df['hover_text'] = hover_texts
    return df

def update_drill_hover_text(df_subset):
    df_subset['drill_hover_text'] = df_subset.apply(
        lambda row: f"{row['hover_text']}<br><b>サブクラスタ:</b> {row['drill_cluster_label']}", axis=1
    )
    return df_subset

def _create_label_editor_ui(labels_map, placeholder_text, key_prefix):
    widgets_dict = {}
    sorted_ids = sorted([cid for cid in labels_map.keys() if cid != -1])
    
    for cluster_id in sorted_ids:
        auto_label = labels_map.get(cluster_id, "")
        if auto_label == "(該当なし)": continue
        
        col1, col2 = st.columns([2, 3])
        with col1:
            st.markdown(f"`{auto_label}`")
        with col2:
            new_label = st.text_input(
                f"[{auto_label}]", 
                value=auto_label, 
                label_visibility="collapsed",
                key=f"{key_prefix}_{cluster_id}"
            )
            widgets_dict[cluster_id] = new_label
            
    if -1 in labels_map:
        col1, col2 = st.columns([2, 3])
        with col1:
            st.markdown(f"`{labels_map[-1]}`")
        with col2:
            st.text_input(f"noise_label", value=labels_map[-1], disabled=True, key=f"{key_prefix}_noise")
            
    return widgets_dict

def get_date_bin_options(df_filtered, interval_years, year_column='year'):
    if df_filtered.empty or year_column not in df_filtered.columns or df_filtered[year_column].isnull().all():
        return [f"(全期間) ({len(df_filtered)}件)"]

    try:
        min_year = int(df_filtered[year_column].min())
        max_year = int(df_filtered[year_column].max())
        
        bins = list(range(min_year, max_year + interval_years, interval_years))
        if not bins:
             bins = [min_year]
        if bins[-1] <= max_year:
             bins.append(bins[-1] + interval_years)

        labels = [f"{bins[i]}-{bins[i+1] - 1}" for i in range(len(bins)-1)]
        
        df_filtered_copy = df_filtered.copy()
        df_filtered_copy['temp_date_bin'] = pd.cut(
            df_filtered_copy[year_column].fillna(-1).astype(int), 
            bins=bins, 
            labels=labels, 
            right=False, 
            errors='coerce'
        )
        
        date_bin_counts = df_filtered_copy['temp_date_bin'].value_counts()
        
        options = [f"(全期間) ({len(df_filtered_copy)}件)"] + [
            f"{label} ({date_bin_counts.get(label, 0)}件)" 
            for label in labels 
            if date_bin_counts.get(label, 0) > 0
        ]
        return options
    except Exception as e:
        return [f"(全期間) ({len(df_filtered)}件)"]

# --- 統計マップ (Saturn V オリジナル) 描画関数 ---

@st.cache_data
def create_application_trend_chart_internal(df_stats, start_year, end_year):
    if 'year' not in df_stats.columns:
        st.error("エラー: 'year' カラムが見つかりません。")
        return None
            
    yearly_counts = df_stats['year'].value_counts().sort_index()
    if yearly_counts.empty:
        st.warning("有効な出願年データがありません。")
        return None

    plot_data = yearly_counts.reindex(range(int(start_year), int(end_year) + 1), fill_value=0)
    if plot_data.empty:
        st.warning("指定期間にデータがありません。")
        return None
            
    plt.style.use('seaborn-v0_8-talk')
    fig, ax = plt.subplots(figsize=(16, 8))
    bars = ax.bar(plot_data.index, plot_data.values, color='steelblue')
    ax.set_title(f'出願件数時系列推移 ({int(start_year)}年～{int(end_year)}年)', fontsize=20, pad=20)
    ax.set_xlabel('出願年', fontsize=14); ax.set_ylabel('出願件数', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)); plt.setp(ax.get_xticklabels(), rotation=45)
    ax.set_ylim(bottom=0)
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    plt.tight_layout()
    return fig

@st.cache_data
def create_assignee_ranking_map_internal(df_stats, assignee_col, num_to_display, delimiter, start_year, end_year):
    if assignee_col not in df_stats.columns:
        st.error(f"エラー: 出願人カラム '{assignee_col}' が見つかりません。")
        return None
            
    assignee_counts = df_stats[assignee_col].astype(str).str.split(delimiter).explode().str.strip().value_counts()
    data_to_plot = assignee_counts.head(num_to_display).sort_values(ascending=True)

    if data_to_plot.empty:
        st.warning("集計結果がありません。")
        return None
            
    plt.style.use('seaborn-v0_8-talk')
    fig, ax = plt.subplots(figsize=(12, max(5, 0.4 * len(data_to_plot))))
    bars = ax.barh(data_to_plot.index, data_to_plot.values, color='steelblue')
    ax.set_title(f'権利者ランキング ({start_year}年～{end_year}年)', fontsize=20, pad=20)
    ax.set_xlabel('特許件数', fontsize=14); ax.set_ylabel('権利者名', fontsize=14)
    for bar in bars:
        width = bar.get_width()
        ax.text(width + width * 0.01, bar.get_y() + bar.get_height()/2, f'{int(width)}', ha='left', va='center', fontsize=12)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.15)
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig

@st.cache_data
def create_assignee_year_bubble_chart_internal(df_stats, assignee_col, num_to_display, delimiter, start_year, end_year):
    if assignee_col not in df_stats.columns or 'year' not in df_stats.columns:
        st.error(f"エラー: 出願人カラム '{assignee_col}' または 'year' カラムが見つかりません。")
        return None

    assignees_exploded = df_stats.copy()
    assignees_exploded[assignee_col] = assignees_exploded[assignee_col].astype(str).str.split(delimiter)
    assignees_exploded = assignees_exploded.explode(assignee_col)
    assignees_exploded['assignee_parsed'] = assignees_exploded[assignee_col].str.strip()
    top_assignees = assignees_exploded['assignee_parsed'].value_counts().head(num_to_display).index.tolist()
    
    plot_data = assignees_exploded[assignees_exploded['assignee_parsed'].isin(top_assignees)]
    plot_data = plot_data.groupby(['year', 'assignee_parsed']).size().reset_index(name='件数')
    if plot_data.empty:
        st.warning("集計結果が空のため、このマップはスキップします。")
        return None
            
    assignee_rank_map = {name: i for i, name in enumerate(top_assignees[::-1])}
    plot_data['y_rank'] = plot_data['assignee_parsed'].map(assignee_rank_map)
    cmap = plt.get_cmap('Set2', len(top_assignees))

    fig, ax = plt.subplots(figsize=(16, max(8, 0.6 * num_to_display)))
    ax.scatter(x=plot_data['year'], y=plot_data['y_rank'], s=plot_data['件数'] * 40, c=plot_data['y_rank'], cmap=cmap, alpha=0.8)
    for _, row in plot_data.iterrows(): ax.text(row['year'], row['y_rank'], row['件数'], ha='center', va='center', fontsize=9, color='black')
    ax.set_yticks(range(len(top_assignees))); ax.set_yticklabels(top_assignees[::-1])
    ax.set_title(f'出願年別権利者動向 (実数スケール) - {start_year}年～{end_year}年', fontsize=20, pad=20)
    ax.set_xlabel('出願年', fontsize=14); ax.set_ylabel('権利者', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    
    return fig

# ==================================================================
# --- 3. Streamlit UI ---
# ==================================================================

st.set_page_config(
    page_title="APOLLO | Saturn V", 
    page_icon="🚀", 
    layout="wide"
)

st.title("🚀 Saturn V")
st.markdown("SBERT（文脈・意味）に基づき、インタラクティブな技術マップ分析モジュールです。")

# ==================================================================
# --- 4. セッション状態の確認と初期化 ---
# ==================================================================
if not st.session_state.get("preprocess_done", False):
    st.error("分析データがありません。")
    st.warning("先に「Mission Control」（メインページ）でファイルをアップロードし、「分析エンジン起動」を実行してください。")
    st.stop()
else:
    df_main = st.session_state.df_main
    col_map = st.session_state.col_map
    delimiters = st.session_state.delimiters
    sbert_embeddings = st.session_state.sbert_embeddings
    tfidf_matrix = st.session_state.tfidf_matrix
    feature_names = st.session_state.feature_names
    
if "saturnv_sbert_umap_done" not in st.session_state:
    st.session_state.saturnv_sbert_umap_done = False
if "saturnv_cluster_done" not in st.session_state:
    st.session_state.saturnv_cluster_done = False
if "saturnv_labels_map" not in st.session_state:
    st.session_state.saturnv_labels_map = {}
    
if "main_cluster_running" not in st.session_state:
    st.session_state.main_cluster_running = False

# ==================================================================
# --- 5. Saturn V アプリケーション ---
# ==================================================================

# --- 5-1. 初回UMAP計算 (SBERTベース) ---
if not st.session_state.saturnv_sbert_umap_done:
    with st.spinner("Saturn V モジュール初回起動中: UMAPによる次元削減 (SBERTベース) を実行しています..."):
        try:
            reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
            embedding = reducer.fit_transform(sbert_embeddings) 
            st.session_state.df_main['umap_x'] = embedding[:, 0]
            st.session_state.df_main['umap_y'] = embedding[:, 1]
            st.session_state.df_main['characteristic_words'] = [get_top_tfidf_words(tfidf_matrix[i], feature_names) for i in range(tfidf_matrix.shape[0])]
            
            st.session_state.saturnv_sbert_umap_done = True
            st.success("UMAPの初期計算 (SBERTベース) が完了しました。")
            st.rerun()
        except Exception as e:
            st.error(f"UMAPの初期計算中にエラーが発生しました: {e}")
            st.exception(e) 
            st.stop()


# --- 5-2. メインUI ---
tab_main, tab_drill, tab_stats, tab_export = st.tabs([
    "Landscape Map (TELESCOPE)", 
    "Drilldown (PROBE)", 
    "特許マップ (統計分析)", 
    "Data Export"
])

# --- TELESCOPE (メインマップ) ---
with tab_main:
    st.subheader("クラスタリング実行")
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        min_cluster_size_w = st.number_input("最小クラスタサイズ (推奨: 10-50):", min_value=2, value=15, key="main_min_cluster_size")
    with col2:
        min_samples_w = st.number_input("最小サンプル数 (推奨: 5-20):", min_value=1, value=10, key="main_min_samples")
    with col3:
        label_top_n_w = st.number_input("クラスタラベル単語数:", min_value=1, value=3, key="main_label_top_n")
    
    run_cluster_btn = st.button(
        "描画 (再計算)",
        type="primary", 
        key="main_run_cluster",
        disabled=st.session_state.main_cluster_running 
    )
    
    if run_cluster_btn:
        st.session_state.main_cluster_running = True
        run_successful = False 
        with st.spinner("HDBSCANクラスタリングを実行中..."):
            try:
                embedding = st.session_state.df_main[['umap_x', 'umap_y']].values
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=int(min_cluster_size_w),
                    min_samples=int(min_samples_w), 
                    metric='euclidean',
                    cluster_selection_method='eom'
                )
                clustering = clusterer.fit(embedding)
                st.session_state.df_main['cluster'] = clustering.labels_
                
                labels_map = {}
                label_top_n = int(label_top_n_w)
                unique_clusters = sorted(st.session_state.df_main['cluster'].unique())
                
                for cluster_id in unique_clusters:
                    if cluster_id == -1:
                        labels_map[cluster_id] = "ノイズ / 小クラスタ"
                        continue
                    
                    indices = st.session_state.df_main[st.session_state.df_main['cluster'] == cluster_id].index
                    if len(indices) == 0:
                        labels_map[cluster_id] = "(該当なし)"
                        continue

                    cluster_vectors = tfidf_matrix[indices]
                    mean_vector = np.array(cluster_vectors.mean(axis=0)).flatten()
                    top_indices = np.argsort(mean_vector)[::-1][:label_top_n]
                    label = ", ".join([feature_names[i] for i in top_indices])
                    labels_map[cluster_id] = f"[{cluster_id}] {label}"
                
                st.session_state.df_main['cluster_label'] = st.session_state.df_main['cluster'].map(labels_map)
                st.session_state.saturnv_labels_map = labels_map
                
                st.session_state.df_main = update_hover_text(st.session_state.df_main, col_map)
                
                st.session_state.saturnv_cluster_done = True
                st.success("クラスタリングとラベリングが完了しました。")
                run_successful = True

            except Exception as e:
                st.error(f"クラスタリング中にエラーが発生しました: {e}")
                import traceback
                st.exception(traceback.format_exc())
            
            finally:
                st.session_state.main_cluster_running = False
                if run_successful:
                    st.rerun() 

    st.markdown("---")
    
    if st.session_state.saturnv_cluster_done:
        st.subheader("フィルタリング設定 (メインマップ用)")
        st.write("（フィルタを変更すると、下のマップが自動で更新されます）")
        
        def on_main_interval_change():
            if "main_date_filter" in st.session_state:
                del st.session_state.main_date_filter

        col1, col2 = st.columns(2)
        with col1:
            if 'year' in df_main.columns and df_main['year'].notna().any():
                bin_interval_w_val = st.selectbox(
                    "期間の粒度 (年ごと):", 
                    [5, 3, 2, 1], 
                    index=0, 
                    key="main_bin_interval",
                    on_change=on_main_interval_change 
                )
                
                date_bin_options = get_date_bin_options(
                    df_main, 
                    int(bin_interval_w_val), 
                    'year'
                )
                
                date_bin_filter_w = st.selectbox(
                    "表示期間 (フィルタ):", 
                    date_bin_options, 
                    key="main_date_filter" 
                )
            else:
                date_bin_filter_w = "(全期間)"
                st.info("出願日カラムが指定されていないため、期間フィルタは無効です。")
            
            if col_map['applicant'] and col_map['applicant'] in st.session_state.df_main.columns:
                applicants = st.session_state.df_main[col_map['applicant']].fillna('').str.split(delimiters['applicant']).explode().str.strip()
                applicant_counts = applicants.value_counts()
                unique_applicants = sorted([app for app in applicants.unique() if app])
                applicant_options = [(f"(全出願人) ({len(st.session_state.df_main)}件)", "ALL")] + [
                    (f"{app_name} ({applicant_counts.get(app_name, 0)}件)", app_name) 
                    for app_name in unique_applicants
                ]
                applicant_filter_w = st.multiselect(
                    "出願人 (フィルタ):", 
                    applicant_options, 
                    default=[applicant_options[0]],
                    format_func=lambda x: x[0],
                    key="main_applicant_filter"
                )
            else:
                default_tuple = (f"(全出願人) ({len(st.session_state.df_main)}件)", "ALL")
                applicant_filter_w = [default_tuple]
                st.info("出願人カラムが指定されていないため、出願人フィルタは無効です。")

        with col2:
            cluster_counts = st.session_state.df_main['cluster_label'].value_counts()
            cluster_options = [(f"(全クラスタ) ({len(st.session_state.df_main)}件)", "ALL")] + [
                (f"{st.session_state.saturnv_labels_map.get(cid)} ({cluster_counts.get(st.session_state.saturnv_labels_map.get(cid), 0)}件)", cid)
                for cid in sorted(st.session_state.df_main['cluster'].unique())
            ]
            cluster_filter_w = st.multiselect(
                "マップ表示クラスタ (フィルタ):", 
                cluster_options, 
                default=[cluster_options[0]],
                format_func=lambda x: x[0],
                key="main_cluster_filter"
            )

        st.subheader("分析結果 (TELESCOPE メインマップ)")
        show_labels_chk = st.checkbox("マップにラベルを表示する", value=True, key="main_show_labels")
        
        df_visible = st.session_state.df_main.copy()
        
        if not date_bin_filter_w.startswith("(全期間)"):
            date_bin_label = date_bin_filter_w.split(' (')[0].strip() 
            try:
                start_year, end_year = map(int, date_bin_label.split('-'))
                date_mask = (
                    (df_visible['year'].notna()) &
                    (df_visible['year'] >= start_year) & 
                    (df_visible['year'] <= end_year)
                )
            except Exception:
                 date_mask = pd.Series(True, index=df_visible.index) 
        else:
            date_mask = pd.Series(True, index=df_visible.index)
            
        applicant_values = [val[1] for val in applicant_filter_w]
        is_applicant_color_mode = True
        if "ALL" in applicant_values or not applicant_values:
            is_applicant_color_mode = False
            applicant_mask = pd.Series(True, index=df_visible.index)
        else:
            mask_list = []
            for app_name in applicant_values:
                mask_list.append(df_visible[col_map['applicant']].fillna('').str.contains(re.escape(app_name)))
            applicant_mask = pd.concat(mask_list, axis=1).any(axis=1)
        
        cluster_values = [val[1] for val in cluster_filter_w]
        
        if "ALL" in cluster_values or not cluster_values:
            cluster_mask = pd.Series(True, index=df_visible.index) 
        else:
            cluster_mask = df_visible['cluster'].isin(cluster_values) 
            
        final_mask = date_mask & applicant_mask & cluster_mask
        df_visible_final = df_visible[final_mask]
        df_faded_final = df_visible[~final_mask]
        
        fig_main = go.Figure()
        
        # 1. Faded dots (背景)
        fig_main.add_trace(go.Scatter(
            x=df_faded_final['umap_x'], y=df_faded_final['umap_y'], mode='markers',
            marker=dict(color='lightgray', size=3, opacity=0.1),
            hoverinfo='none', name='他'
        ))
        
        # 2. Visible dots (前景)
        if not is_applicant_color_mode:
            # モード1: クラスタ色分け (全出願人)
            fig_main.add_trace(go.Scatter(
                x=df_visible_final['umap_x'], y=df_visible_final['umap_y'], mode='markers',
                marker=dict(
                    color=df_visible_final['cluster'], colorscale='turbo', 
                    showscale=False, size=4, opacity=0.5
                ), 
                hoverinfo='text', hovertext=df_visible_final['hover_text'], name='表示対象'
            ))
        else:
            # モード2: 出願人色分け
            colors = px.colors.qualitative.Plotly 
            for i, app_tuple in enumerate(applicant_filter_w):
                app_name = app_tuple[1]
                if app_name == "ALL": continue
                
                app_mask_final = df_visible_final[col_map['applicant']].fillna('').str.contains(re.escape(app_name))
                df_applicant = df_visible_final[app_mask_final]
                
                if not df_applicant.empty:
                    fig_main.add_trace(go.Scatter(
                        x=df_applicant['umap_x'], y=df_applicant['umap_y'],
                        mode='markers',
                        marker=dict(color=colors[i % len(colors)], size=5, opacity=0.7), 
                        hoverinfo='text', hovertext=df_applicant['hover_text'],
                        name=app_name
                    ))
        
        # 3. ラベル
        annotations = []
        if show_labels_chk:
            visible_cluster_ids = df_visible_final['cluster'].unique()
            for cluster_id, group in df_visible[df_visible['cluster'].isin(visible_cluster_ids)].groupby('cluster'):
                if cluster_id == -1: continue
                mean_pos = group[['umap_x', 'umap_y']].mean()
                label_text = group['cluster_label'].iloc[0]
                if label_text != "(該当なし)":
                    annotations.append(
                        go.layout.Annotation(
                            x=mean_pos['umap_x'], y=mean_pos['umap_y'], 
                            text=label_text, showarrow=False, 
                            font=dict(size=11, color='black'),
                            bgcolor='rgba(255, 255, 255, 0.7)' 
                        )
                    )
        
        fig_main.update_layout(
            title="Saturn V - メインマップ (SBERT UMAP)",
            showlegend=True, width=1000, height=800, template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            annotations=annotations,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=None),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=None)
        )
        st.plotly_chart(fig_main, use_container_width=True)

        st.subheader("ラベル編集")
        st.write("メインマップのクラスタラベルを編集します。編集内容は自動で再描画されます。")
        
        st.session_state.saturnv_labels_map_custom = _create_label_editor_ui(
            st.session_state.saturnv_labels_map, 
            "新しいラベル名", 
            "main_label"
        )
                
        if st.button("ラベルを更新して再描画", key="main_update_labels"):
            if -1 in st.session_state.saturnv_labels_map:
                st.session_state.saturnv_labels_map_custom[-1] = st.session_state.saturnv_labels_map[-1]
                
            st.session_state.df_main['cluster_label'] = st.session_state.df_main['cluster'].map(st.session_state.saturnv_labels_map_custom)
            st.session_state.df_main = update_hover_text(st.session_state.df_main, col_map)
            st.session_state.saturnv_labels_map = st.session_state.saturnv_labels_map_custom
            st.success("ラベルを更新しました。")
            st.rerun()


    # --- PROBE (ドリルダウン) ---
    with tab_drill:
        st.subheader("分析対象クラスタの選択")
        
        drilldown_options = [('(選択してください)', 'NONE')]
        if "saturnv_labels_map" in st.session_state:
            drilldown_options += [
                (f"{label} ({count}件)", cid)
                for cid, label in st.session_state.saturnv_labels_map.items()
                if cid != -1
                for count in [st.session_state.df_main['cluster'].value_counts().get(cid, 0)]
            ]

        selected_drilldown_target_drill = st.selectbox(
            "分析対象クラスタ:",
            options=drilldown_options,
            format_func=lambda x: x[0],
            key="drill_target_select"
        )
        drilldown_target_id = selected_drilldown_target_drill[1] 

        st.subheader("フィルタリング設定 (ドリルダウン用)")
        
        if drilldown_target_id == "NONE":
            df_subset_filter = pd.DataFrame(columns=df_main.columns)
        else:
            df_subset_filter = df_main[df_main['cluster'] == drilldown_target_id].copy()
            
        def on_drill_interval_change():
            if "drill_date_filter_w" in st.session_state:
                del st.session_state.drill_date_filter_w
            
        col1, col2 = st.columns(2)
        with col1:
            if 'year' in df_subset_filter.columns and df_subset_filter['year'].notna().any():
                drill_bin_interval_w_val = st.selectbox(
                    "期間の粒度 (年ごと):", 
                    [5, 3, 2, 1], 
                    index=0, 
                    key="drill_interval_w",
                    on_change=on_drill_interval_change 
                )
                
                drill_date_bin_options = get_date_bin_options(
                    df_subset_filter, 
                    int(drill_bin_interval_w_val),
                    'year'
                )
                
                drill_date_bin_filter_w = st.selectbox(
                    "表示期間 (ドリルダウン用):", 
                    drill_date_bin_options, 
                    key="drill_date_filter_w" 
                )
            else:
                drill_date_bin_filter_w = "(全期間)"
                st.info("出願日カラムが指定されていないため、期間フィルタは無効です。")
        
        with col2:
            drill_applicant_options = [(f"(全出願人) ({len(df_subset_filter)}件)", "ALL")]
            if col_map['applicant'] and col_map['applicant'] in df_subset_filter.columns:
                applicants_drill = df_subset_filter[col_map['applicant']].fillna('').str.split(delimiters['applicant']).explode().str.strip()
                applicant_counts_drill = applicants_drill.value_counts()
                unique_applicants_drill = sorted([app for app in applicants_drill.unique() if app])
                drill_applicant_options += [(f"{app_name} ({applicant_counts_drill.get(app_name, 0)}件)", app_name) for app_name in unique_applicants_drill if applicant_counts_drill.get(app_name, 0) > 0]

            drill_applicant_filter_w = st.multiselect(
                "出願人 (ドリルダウン用):", 
                drill_applicant_options, 
                default=[drill_applicant_options[0]],
                format_func=lambda x: x[0],
                key="drill_applicant_filter_w"
            )

        st.subheader("クラスタリング設定 (ドリルダウン用)")
        col1, col2, col3 = st.columns(3)
        with col1:
            drill_min_cluster_size_w = st.number_input('最小クラスタサイズ:', min_value=2, value=5, key="drill_min_cluster_size_w")
        with col2:
            drill_min_samples_w = st.number_input('最小サンプル数:', min_value=1, value=5, key="drill_min_samples_w")
        with col3:
            drill_label_top_n_w = st.number_input('ラベル単語数:', min_value=1, value=3, key="drill_label_top_n_w")
        
        drill_show_labels_chk = st.checkbox('マップにラベルを表示する', value=True, key="drill_show_labels_chk")

        if st.button("選択クラスタで再マップ (ドリルダウン)", type="primary", key="drill_run_button"):
            if drilldown_target_id == "NONE":
                st.error("F-1で有効なクラスタを選択してください。")
            else:
                with st.spinner(f"クラスタ {drilldown_target_id} のドリルダウンを実行中..."):
                    try:
                        df_subset = df_main[df_main['cluster'] == drilldown_target_id].copy()
                        base_label = df_subset['cluster_label'].iloc[0]
                        
                        if not drill_date_bin_filter_w.startswith("(全期間)"):
                            date_bin_label = drill_date_bin_filter_w.split(' (')[0].strip() 
                            try:
                                start_year, end_year = map(int, date_bin_label.split('-'))
                                date_mask = (
                                    (df_subset['year'].notna()) &
                                    (df_subset['year'] >= start_year) & 
                                    (df_subset['year'] <= end_year)
                                )
                                df_subset = df_subset[date_mask]
                            except Exception:
                                pass 

                        drill_app_values = [val[1] for val in drill_applicant_filter_w]
                        if "ALL" not in drill_app_values and drill_app_values:
                            mask_list_drill = []
                            for app_name in drill_app_values:
                                mask_list_drill.append(df_subset[col_map['applicant']].fillna('').str.contains(re.escape(app_name)))
                            applicant_mask_drill = pd.concat(mask_list_drill, axis=1).any(axis=1)
                            df_subset = df_subset[applicant_mask_drill]
                        
                        if len(df_subset) < 20:
                            st.error(f"フィルタ適用後のデータが少なすぎるため再マップできません (20件未満)。")
                        else:
                            subset_indices = df_subset.index
                            subset_tfidf = tfidf_matrix[subset_indices]
                            subset_sbert_embeddings = sbert_embeddings[subset_indices]
                            
                            subset_indices_pd = pd.Index(subset_indices)

                            n_neighbors_drill = min(10, len(df_subset) - 1)
                            if n_neighbors_drill < 2: n_neighbors_drill = 2
                            
                            reducer_drill = UMAP(n_neighbors=n_neighbors_drill, min_dist=0.1, n_components=2, random_state=42)
                            embedding_drill = reducer_drill.fit_transform(subset_sbert_embeddings) 
                            
                            df_subset['drill_x'] = embedding_drill[:, 0]
                            df_subset['drill_y'] = embedding_drill[:, 1]
                            
                            clusterer_drill = hdbscan.HDBSCAN(
                                min_cluster_size=int(drill_min_cluster_size_w),
                                min_samples=int(drill_min_samples_w),
                                metric='euclidean', cluster_selection_method='eom'
                            )
                            df_subset['drill_cluster'] = clusterer_drill.fit_predict(embedding_drill)
                            
                            drill_labels_map = {}
                            label_top_n = int(drill_label_top_n_w)
                            for cluster_id in sorted(df_subset['drill_cluster'].unique()):
                                if cluster_id == -1:
                                    drill_labels_map[cluster_id] = "ノイズ"
                                    continue
                                indices_drill = df_subset[df_subset['drill_cluster'] == cluster_id].index
                                
                                subset_tfidf_positions = [subset_indices_pd.get_loc(idx) for idx in indices_drill if idx in subset_indices_pd]
                                
                                if not subset_tfidf_positions:
                                    label = "(TF-IDFデータなし)"
                                else:
                                    mean_vector = np.array(subset_tfidf[subset_tfidf_positions].mean(axis=0)).flatten()
                                    top_indices = np.argsort(mean_vector)[::-1][:label_top_n]
                                    label = ", ".join([feature_names[i] for i in top_indices])
                                
                                drill_labels_map[cluster_id] = f"[{cluster_id}] {label}"
                            
                            df_subset['drill_cluster_label'] = df_subset['drill_cluster'].map(drill_labels_map)
                            df_subset = update_drill_hover_text(df_subset)
                            
                            st.session_state.df_drilldown_result = df_subset.copy()
                            st.session_state.drill_labels_map = drill_labels_map
                            st.session_state.drill_base_label = base_label
                            
                            st.success("ドリルダウンマップの計算が完了しました。")
                            st.rerun()

                    except Exception as e:
                        st.error(f"ドリルダウン中にエラー: {e}")
                        import traceback
                        st.exception(traceback.format_exc())

        if "df_drilldown_result" in st.session_state:
            df_drill = st.session_state.df_drilldown_result.copy()
            drill_labels_map = st.session_state.drill_labels_map
            
            st.subheader("サブクラスタ・ラベル編集")
            drill_label_widgets = _create_label_editor_ui(drill_labels_map, "新しいサブクラスタ名", "drill_label")

            if st.button("サブクラスタ・ラベルを更新して再描画", key="drill_update_labels"):
                new_labels_map_drill = {}
                for cluster_id, text_widget_val in drill_label_widgets.items():
                    new_labels_map_drill[cluster_id] = text_widget_val
                if -1 in drill_labels_map:
                    new_labels_map_drill[-1] = drill_labels_map[-1]
                
                df_drill['drill_cluster_label'] = df_drill['drill_cluster'].map(new_labels_map_drill)
                df_drill = update_drill_hover_text(df_drill)
                
                st.session_state.df_drilldown_result = df_drill.copy()
                st.session_state.drill_labels_map = new_labels_map_drill
                st.success("サブクラスタ・ラベルを更新しました。")
                st.rerun()

            st.subheader("ドリルダウンマップ")
            
            fig_drill = go.Figure()
            fig_drill.add_trace(go.Scatter(
                x=df_drill['drill_x'], y=df_drill['drill_y'], mode='markers',
                marker=dict(
                    color=df_drill['drill_cluster'], colorscale='turbo', 
                    showscale=False, size=4, opacity=0.5
                ),
                hoverinfo='text', hovertext=df_drill['drill_hover_text'], name='表示対象'
            ))
            
            annotations_drill = []
            if drill_show_labels_chk:
                for cluster_id, group in df_drill[df_drill['drill_cluster'] != -1].groupby('drill_cluster'):
                    mean_pos = group[['drill_x', 'drill_y']].mean()
                    label_text = group['drill_cluster_label'].iloc[0]
                    annotations_drill.append(go.layout.Annotation(
                        x=mean_pos['drill_x'], y=mean_pos['drill_y'], 
                        text=label_text, showarrow=False, 
                        font=dict(size=11, color='black')
                    ))
            
            fig_drill.update_layout(
                title=f'Saturn V ドリルダウン (PROBE): {st.session_state.drill_base_label}',
                height=800, template='plotly_white',
                annotations=annotations_drill,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=None),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=None)
            )
            st.plotly_chart(fig_drill, use_container_width=True)

    # --- 特許マップ (統計分析) ---
    with tab_stats:
        st.subheader("特許マップ（統計分析）")
        
        with st.container(border=True):
            st.info("ℹ️ **フィルタについて**")
            st.markdown("""
            この「特許マップ」セクションは、**このタブ内の設定のみ**を参照します。
            
            **「Landscape Map (TELESCOPE)」タブのフィルタ（期間、出願人）は適用されません。**
            """)
        
        st.markdown("---")
        st.subheader("特許マップ設定")
        
        # 1. クラスタフィルタを先に配置
        if st.session_state.saturnv_cluster_done:
            cluster_counts_stats = st.session_state.df_main['cluster_label'].value_counts()
            cluster_options_stats = [(f"(全クラスタ) ({len(st.session_state.df_main)}件)", "ALL")] + [
                (f"{st.session_state.saturnv_labels_map.get(cid)} ({cluster_counts_stats.get(st.session_state.saturnv_labels_map.get(cid), 0)}件)", cid)
                for cid in sorted(st.session_state.df_main['cluster'].unique())
            ]
            
            stats_cluster_filter_w = st.multiselect(
                "集計対象クラスタ (フィルタ):", 
                cluster_options_stats, 
                default=[cluster_options_stats[0]],
                format_func=lambda x: x[0],
                key="stats_cluster_filter"
            )
        else:
            stats_cluster_filter_w = []
            st.info("「Landscape Map (TELESCOPE)」タブで「描画 (再計算)」を実行すると、クラスタフィルタが有効になります。")

        # 2. 年入力と件数入力、自動設定ボタンを配置
        col1, col2 = st.columns(2)
        with col1:
            # 年のセッション状態キーを初期化
            if 'stats_start_year' not in st.session_state:
                 st.session_state.stats_start_year = 2010
            if 'stats_end_year' not in st.session_state:
                 st.session_state.stats_end_year = 2024
                 
            stats_start_year = st.number_input(
                '開始年:', 
                key="stats_start_year",
                step=1 
            )
            stats_end_year = st.number_input(
                '終了年:', 
                key="stats_end_year",
                step=1 
            )
        with col2:
            stats_num_assignees = st.number_input('出願人 表示件数:', min_value=1, value=15, key="stats_num_assignees")
            
            def callback_autoset_stats_year():
                selected_values = st.session_state.get("stats_cluster_filter", [])
                stats_cluster_values = [val[1] for val in selected_values]
                
                df_to_calc = st.session_state.df_main.copy() 
                
                if "ALL" not in stats_cluster_values and stats_cluster_values:
                    stats_cluster_mask = df_to_calc['cluster'].isin(stats_cluster_values)
                    df_to_calc = df_to_calc[stats_cluster_mask]
                
                if not df_to_calc.empty and 'year' in df_to_calc.columns and df_to_calc['year'].notna().any():
                    valid_years = df_to_calc['year'].dropna().astype(int)
                    st.session_state.stats_start_year = int(valid_years.min())
                    st.session_state.stats_end_year = int(valid_years.max())

            st.button(
                "（フィルタ結果から期間を自動設定）", 
                key="stats_autoset_year",
                on_click=callback_autoset_stats_year
            )

        if st.button("特許マップを描画", type="primary", key="stats_run_button"):
            if not st.session_state.saturnv_cluster_done:
                st.error("エラー: 「Landscape Map (TELESCOPE)」タブで「描画 (再計算)」を先に実行してください。")
            else:
                df_stats_base = st.session_state.df_main.copy()
                
                # Hタブのクラスタフィルタのみ適用
                stats_cluster_values = [val[1] for val in stats_cluster_filter_w]
                if "ALL" not in stats_cluster_values and stats_cluster_values:
                    stats_cluster_mask = df_stats_base['cluster'].isin(stats_cluster_values)
                    df_stats_base = df_stats_base[stats_cluster_mask]
                
                if 'year' not in df_stats_base.columns or df_stats_base['year'].isnull().all():
                    st.error("エラー: 「出願日カラム」の指定が必須です（Mission Control）。")
                else:
                    # Hタブのセッション状態から値を取得
                    start_year_val = int(st.session_state.stats_start_year) 
                    end_year_val = int(st.session_state.stats_end_year)     
                    num_assignees_val = st.session_state.stats_num_assignees
                    
                    # Hタブの年フィルタを適用
                    df_stats = df_stats_base[
                        (df_stats_base['year'].notna()) & 
                        (df_stats_base['year'] >= start_year_val) & 
                        (df_stats_base['year'] <= end_year_val)
                    ].copy()
                    
                    if df_stats.empty:
                        st.warning("該当するデータがありません。フィルタ条件を見直してください。")
                    else:
                        st.success(f"集計対象件数: {len(df_stats)} 件")
                        
                        st.subheader("出願件数時系列推移")
                        fig1 = create_application_trend_chart_internal(df_stats, start_year_val, end_year_val)
                        if fig1: st.pyplot(fig1)

                        st.subheader("権利者ランキング")
                        fig2 = create_assignee_ranking_map_internal(df_stats, col_map['applicant'], int(num_assignees_val), delimiters['applicant'], start_year_val, end_year_val)
                        if fig2: st.pyplot(fig2)
                        
                        st.subheader("出願年別権利者動向 (実数)")
                        fig3 = create_assignee_year_bubble_chart_internal(df_stats, col_map['applicant'], int(num_assignees_val), delimiters['applicant'], start_year_val, end_year_val)
                        if fig3: st.pyplot(fig3)
                        
    # --- Data Export ---
    with tab_export:
        st.subheader("Data Export")
        st.markdown("---")
        
        st.subheader("TELESCOPE (メインマップ) のエクスポート")
        if st.session_state.saturnv_cluster_done:
            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(encoding='utf-8-sig').encode('utf-8-sig')
            
            cols_to_drop_main = ['hover_text', 'parsed_date', 'drill_cluster', 'drill_cluster_label', 'drill_hover_text', 'drill_x', 'drill_y', 'temp_date_bin']
            df_export_main = st.session_state.df_main.drop(columns=cols_to_drop_main, errors='ignore')
            csv_main = convert_df_to_csv(df_export_main)
            
            st.download_button(
                label="TELESCOPE (メインマップ) 全データをダウンロード",
                data=csv_main,
                file_name="APOLLO_TELESCOPE_MainMap_All.csv",
                mime="text/csv",
            )
        else:
            st.info("「Landscape Map (TELESCOPE)」タブで「描画 (再計算)」を実行すると、ダウンロードボタンが表示されます。")

        st.markdown("---")
        st.subheader("PROBE (ドリルダウン) のエクスポート")
        if "df_drilldown_result" in st.session_state:
            cols_to_drop_drill = ['hover_text', 'parsed_date', 'date_bin', 'drill_hover_text', 'drill_date_bin', 'temp_date_bin']
            df_export_drill = st.session_state.df_drilldown_result.drop(columns=cols_to_drop_drill, errors='ignore')
            csv_drill = convert_df_to_csv(df_export_drill)
            
            if "drill_base_label" in st.session_state:
                target_name = st.session_state.drill_base_label.replace(" ", "_").replace("/", "_")
            else:
                target_name = "unknown_cluster" 
            
            st.download_button(
                label="PROBE (ドリルダウン) 結果をダウンロード",
                data=csv_drill,
                file_name=f"APOLLO_PROBE_Drilldown_{target_name}.csv",
                mime="text/csv",
            )
        else:
            st.info("「Drilldown (PROBE)」タブで「選択クラスタで再マップ」を実行すると、ダウンロードボタンが表示されます。")

# --- 共通サイドバーフッター ---
st.sidebar.markdown("---") 
st.sidebar.caption("ナビゲーション:")
st.sidebar.caption("1. Mission Control でデータをアップロードし、前処理を実行します。")
st.sidebar.caption("2. 左のリストから分析モジュールを選択します。")
st.sidebar.markdown("---")
st.sidebar.caption("© 2025 しばやま")
