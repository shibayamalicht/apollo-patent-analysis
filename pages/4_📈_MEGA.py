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

# UMAP / HDBSCAN
from umap import UMAP
import hdbscan

# 警告を非表示
warnings.filterwarnings('ignore')

# ==================================================================
# --- 2. ヘルパー関数 (MEGA分析ロジック) ---
# ==================================================================
@st.cache_data
def _get_top_words_from_dense_vector(dense_vector, feature_names, top_n=5):
    indices = np.argsort(dense_vector)[::-1]
    top_words = [feature_names[i] for i in indices[:top_n]]
    return ", ".join(top_words)

@st.cache_data
def _calculate_cagr(row, cagr_end_year_val):
    valid_years = row[row > 0].index
    if not any(valid_years):
        return np.nan

    valid_years_in_range = valid_years[valid_years <= cagr_end_year_val]
    if not any(valid_years_in_range):
        return np.nan

    start_year = min(valid_years_in_range)
    end_year = max(valid_years_in_range)

    if start_year >= end_year:
        return np.nan

    start_value = row[start_year]
    end_value = row[end_year]

    num_years = end_year - start_year
    if num_years <= 0:
         return np.nan

    return ((end_value / start_value) ** (1 / num_years)) - 1

@st.cache_data
def _calculate_metrics(pivot_df, cagr_end_year, y_axis_years, current_year, past_offset=0):
    target_cagr_end = cagr_end_year - past_offset
    target_current_year = current_year - past_offset

    y_start = target_current_year - y_axis_years + 1
    y_cols = [col for col in pivot_df.columns if col >= y_start and col <= target_current_year]
    y_axis = pivot_df[y_cols].sum(axis=1) if y_cols else pd.Series(0, index=pivot_df.index)

    if past_offset == 0:
        bubble_size = pivot_df.sum(axis=1)
    else:
        bubble_cols = [col for col in pivot_df.columns if col <= target_current_year]
        bubble_size = pivot_df[bubble_cols].sum(axis=1) if bubble_cols else pd.Series(0, index=pivot_df.index)

    cagr_cols = [col for col in pivot_df.columns if col <= target_cagr_end]
    if not cagr_cols:
        x_axis = pd.Series(np.nan, index=pivot_df.index)
    else:
        x_axis = pivot_df[cagr_cols].apply(
            _calculate_cagr,
            axis=1,
            cagr_end_year_val=target_cagr_end
        )

    return x_axis, y_axis, bubble_size

@st.cache_data
def _prepare_momentum_data(df_main, axis_col):
    df = df_main[['app_num_main', 'year', axis_col]].copy()
    df.dropna(subset=['app_num_main', 'year', axis_col], inplace=True)

    df_exploded = df.explode(axis_col)
    df_exploded[axis_col] = df_exploded[axis_col].str.strip()
    df_exploded.dropna(subset=[axis_col], inplace=True)
    df_exploded = df_exploded[df_exploded[axis_col] != '']

    df_unique = df_exploded.drop_duplicates(subset=['app_num_main', axis_col], keep='first')

    pivot_df = pd.pivot_table(df_unique, index=axis_col, columns='year', aggfunc='size', fill_value=0)

    return pivot_df

def _get_hover_template_mode1(past_offset, is_past=False):
    if is_past:
        return f"""<b>%{{customdata[2]}}</b> (過去)<br>
<br>
X (勢い): %{{x:.1%}}<br>
Y (活動量): %{{customdata[1]:,.0f}}<br>
Bubble (総件数): %{{customdata[0]:,}}<br>
<extra></extra>"""
    else:
        return f"""<b>%{{customdata[5]}}</b> (現在)<br>
<br>
戦略グループ: %{{customdata[0]}}<br>
X (勢い): %{{x:.1%}}<br>
Y (活動量): %{{customdata[4]:,.0f}}<br>
Bubble (総件数): %{{customdata[1]:,}}<br>
<br>---<br>
過去 X ({past_offset}年前): %{{customdata[2]:.1%}}<br>
過去 Y ({past_offset}年前): %{{customdata[3]:,.0f}}<br>
<extra></extra>"""

def _get_hover_template_mode2(past_offset, is_past=False):
    if is_past:
        return ""
    else:
        return f"""<b>%{{hovertext}}</b> (現在)<br>
<br>
戦略グループ: %{{customdata[0]}}<br>
X (勢い): %{{x:.1%}}<br>
Y (活動量): %{{customdata[4]:,.0f}}<br>
Bubble (総件数): %{{customdata[1]:,}}<br>
<br>---<br>
過去 X ({past_offset}年前): %{{customdata[2]:.1%}}<br>
過去 Y ({past_offset}年前): %{{customdata[3]:,.0f}}<br>
<extra></extra>"""

# ==================================================================
# --- 3. Streamlit UI ---
# ==================================================================

st.set_page_config(
    page_title="APOLLO | MEGA",
    page_icon="📈",
    layout="wide"
)

st.title("📈 MEGA")
st.markdown("技術動態（マクロ）と技術ポートフォリオ（ミクロ）を分析します。")

# ==================================================================
# --- 4. セッション状態の確認 ---
# ==================================================================
if not st.session_state.get("preprocess_done", False):
    st.error("分析データがありません。")
    st.warning("先に「Mission Control」（メインページ）でファイルをアップロードし、「分析エンジン起動」を実行してください。")
    st.stop()
else:
    try:
        df_main = st.session_state.df_main
        col_map = st.session_state.col_map
        sbert_embeddings = st.session_state.sbert_embeddings
        tfidf_matrix = st.session_state.tfidf_matrix
        feature_names = st.session_state.feature_names
    except Exception as e:
        st.error(f"セッションデータの読み込みに失敗しました: {e}")
        st.stop()

# ==================================================================
# --- 5. MEGA アプリケーション ---
# ==================================================================

tab_b, tab_c, tab_d = st.tabs([
    "Landscape Analysis (PULSE)",
    "Technology Probe (TELESCOPE)",
    "Data Export"
])

# --- B. 動態分析 ---
with tab_b:
    st.subheader("分析パラメータ")

    col1, col2 = st.columns(2)
    with col1:
        analysis_axis = st.selectbox(
            "分析軸:",
            options=[
                ('出願人', 'applicant_main'),
                ('IPC (メイングループ)', 'ipc_main_group'),
                ('Fターム (テーマコード)', 'fterm_main')
            ],
            format_func=lambda x: x[0], 
            key="mega_analysis_axis"
        )
        yaxis_slider = st.slider("Y軸 (現在) の集計年数:", min_value=1, max_value=10, value=5, key="mega_yaxis")
        cagr_end_year = st.number_input("X軸 (過去の勢い) 計算の最終年:", value=datetime.datetime.now().year - 1, key="mega_cagr_year")

    with col2:
        trajectory_past = st.number_input("軌跡 (過去時点):", min_value=1, value=5, key="mega_trajectory")
        min_patents = st.number_input("最小フィルタ件数 (描画対象):", min_value=1, value=10, key="mega_min_patents")

    st.subheader("ハイライトと軌跡")

    highlight_options = st.session_state.get("mega_highlight_options", [])
    highlight_targets = st.multiselect(
        "注目対象 (複数選択可):",
        options=highlight_options,
        format_func=lambda x: x[0] 
    )

    st.subheader("動態分析マップ実行")

    if st.button("動態分析マップを描画", type="primary", key="mega_run_map"):

        with st.spinner("動態分析マップを計算中..."):
            try:
                # --- 1. パラメータとデータの準備 ---
                axis_col = analysis_axis[1]
                axis_label = analysis_axis[0]
                y_axis_years = int(yaxis_slider)
                past_offset = int(trajectory_past)
                current_year = datetime.datetime.now().year
                min_patents_threshold = int(min_patents)

                # --- 2. 分析軸データの前処理 ---
                pivot_df = _prepare_momentum_data(df_main, axis_col)
                if pivot_df.empty:
                    st.error(f"エラー: 分析軸 ({axis_label}) の有効なデータがありません。")
                    st.stop()

                # --- 4. 分析指標の計算 ---
                x_present, y_present, bubble_present = _calculate_metrics(
                    pivot_df, cagr_end_year, y_axis_years, current_year, past_offset=0
                )
                x_past, y_past, bubble_past = _calculate_metrics(
                    pivot_df, cagr_end_year, y_axis_years, current_year, past_offset=past_offset
                )

                # --- 3. ハイライトUIの更新 ---
                options_with_counts = [
                    (f"{name} ({int(count)}件)", name)
                    for name, count in bubble_present.sort_index().items()
                ]
                st.session_state.mega_highlight_options = options_with_counts

                start_years = pivot_df[pivot_df > 0].apply(lambda row: row.first_valid_index(), axis=1)
                cagr_start_year_min = start_years.min() if not start_years.empty else cagr_end_year
                if pd.isna(cagr_start_year_min):
                    cagr_start_year_min = cagr_end_year

                st.session_state.cagr_start_year_min = cagr_start_year_min
                st.session_state.cagr_end_year_val = cagr_end_year

                # --- 5. 最終データフレームの作成 ---
                df_result = pd.DataFrame({
                    'X_Present': x_present, 'Y_Present': y_present, 'Bubble_Present': bubble_present,
                    'X_Past': x_past, 'Y_Past': y_past, 'Bubble_Past': bubble_past
                })

                df_result = df_result.astype('float')
                df_result.index.name = axis_label
                df_result.replace([np.inf, -np.inf], np.nan, inplace=True)
                df_result.dropna(subset=['X_Present', 'Y_Present'], inplace=True)

                df_result = df_result[df_result['Bubble_Present'] >= min_patents_threshold].copy()

                if df_result.empty:
                    st.error(f"エラー: フィルタリング（最小{min_patents_threshold}件）後の分析結果が0件です。")
                    st.stop()

                # --- 6. 4象限への分類 ---
                x_threshold = df_result['X_Present'].mean() if not df_result.empty else 0
                y_threshold = df_result['Y_Present'].mean() if not df_result.empty else 0
                st.session_state.mega_x_threshold = x_threshold
                st.session_state.mega_y_threshold = y_threshold

                def assign_relative_label(row):
                    if row['Y_Present'] <= 0:
                        return '衰退・ニッチ (Declining/Niche)'
                    if (row['X_Present'] > x_threshold) and (row['Y_Present'] > y_threshold):
                        return 'リーダー (Leaders)'
                    elif (row['X_Present'] > x_threshold) and (row['Y_Present'] <= y_threshold):
                        return '新興・高ポテンシャル (Emerging)'
                    elif (row['X_Present'] <= x_threshold) and (row['Y_Present'] > y_threshold):
                        return '成熟・既存勢力 (Established)'
                    else:
                        return '衰退・ニッチ (Declining/Niche)'

                df_result['Group_Auto'] = df_result.apply(assign_relative_label, axis=1)

                # --- 7. 結果を保存 ---
                st.session_state.df_momentum_result = df_result.copy()
                st.session_state.mega_axis_label = axis_label
                st.session_state.mega_past_offset = past_offset

                df_filtered = df_result[df_result['Group_Auto'] != 'N/A']
                drilldown_options = [('(分析対象を選択)', '(分析対象を選択)')] + [
                    (f"{name} ({int(row['Bubble_Present'])}件)", name)
                    for name, row in df_filtered.sort_index().iterrows()
                ]
                st.session_state.mega_drilldown_options = drilldown_options

                st.success("動態分析マップの計算が完了しました。")
                st.rerun()

            except Exception as e:
                st.error(f"動態分析マップ分析中にエラーが発生しました: {e}")
                import traceback
                st.exception(traceback.format_exc())

    st.subheader("ラベル編集")

    base_color_map = {
        'リーダー (Leaders)': '#28a745',
        '新興・高ポテンシャル (Emerging)': '#ffc107',
        '成熟・既存勢力 (Established)': '#007bff',
        '衰退・ニッチ (Declining/Niche)': '#6c757d',
        'N/A': '#ced4da'
    }

    if "df_momentum_result" in st.session_state:
        df_to_plot = st.session_state.df_momentum_result.copy()

        st.session_state.mega_group_map_custom = {}
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.mega_group_map_custom['リーダー (Leaders)'] = st.text_input(
                "リーダー (Leaders)", "リーダー (Leaders)", key="label_leader"
            )
            st.session_state.mega_group_map_custom['新興・高ポテンシャル (Emerging)'] = st.text_input(
                "新興・高ポテンシャル (Emerging)", "新興・高ポテンシャル (Emerging)", key="label_emerging"
            )
        with col2:
            st.session_state.mega_group_map_custom['成熟・既存勢力 (Established)'] = st.text_input(
                "成熟・既存勢力 (Established)", "成熟・既存勢力 (Established)", key="label_established"
            )
            st.session_state.mega_group_map_custom['衰退・ニッチ (Declining/Niche)'] = st.text_input(
                "衰退・ニッチ (Declining/Niche)", "衰退・ニッチ (Declining/Niche)", key="label_declining"
            )

        df_to_plot['Group_Custom'] = df_to_plot['Group_Auto'].map(st.session_state.mega_group_map_custom).fillna('N/A')

        current_color_map = {
            st.session_state.mega_group_map_custom.get(auto_label, auto_label): color
            for auto_label, color in base_color_map.items()
        }

        axis_label = st.session_state.mega_axis_label
        past_offset = st.session_state.mega_past_offset
        cagr_start_year_min = st.session_state.cagr_start_year_min
        cagr_end_year_val = st.session_state.cagr_end_year_val
        x_threshold = st.session_state.mega_x_threshold
        y_threshold = st.session_state.mega_y_threshold

        title = f"MEGA 動態分析マップ - 分析軸: {axis_label}"
        xaxis_title_label = f"過去の勢い (CAGR, {int(cagr_start_year_min)}-{int(cagr_end_year_val)}年内の活動期間)"

        fig = go.Figure()

        if highlight_targets:
            highlight_values = [t[1] for t in highlight_targets]
            df_highlighted = df_to_plot[df_to_plot.index.isin(highlight_values)].copy()

            if not df_highlighted.empty:
                title += f" (注目対象: {', '.join(df_highlighted.index)})"

                max_bubble_for_scaling = df_to_plot['Bubble_Present'].max()
                if max_bubble_for_scaling == 0: max_bubble_for_scaling = 1
                df_highlighted['Size_Present_Linear'] = (df_highlighted['Bubble_Present'] / max_bubble_for_scaling) * 60
                df_highlighted['Size_Past_Linear'] = (df_highlighted['Bubble_Past'] / max_bubble_for_scaling) * 60
                df_highlighted['Size_Present_Linear'] = df_highlighted['Size_Present_Linear'].clip(lower=5)
                df_highlighted['Size_Past_Linear'] = df_highlighted['Size_Past_Linear'].clip(lower=5)

                df_highlighted['Y_Present_Plot'] = df_highlighted['Y_Present'].replace(0, 0.1)
                df_highlighted['Y_Past_Plot'] = df_highlighted['Y_Past'].replace(0, 0.1)

                palette = px.colors.qualitative.Plotly
                present_year_label = int(cagr_end_year_val)
                past_year_label = int(cagr_end_year_val - past_offset)

                for i, (idx, row) in enumerate(df_highlighted.iterrows()):
                    color = palette[i % len(palette)]
                    if pd.notna(row['X_Past']) and pd.notna(row['Y_Past']):
                        fig.add_trace(go.Scatter(
                            x=[row['X_Past']], y=[row['Y_Past_Plot']], mode='markers',
                            marker=dict(color=color, size=row['Size_Past_Linear'], opacity=0.3),
                            name=f"{idx} ({past_year_label}年)", hovertext="",
                            customdata=np.array([[row['Bubble_Past'], row['Y_Past'], idx]]),
                            hovertemplate=_get_hover_template_mode1(past_offset, is_past=True)
                        ))

                    fig.add_trace(go.Scatter(
                        x=[row['X_Present']], y=[row['Y_Present_Plot']], mode='markers',
                        marker=dict(color=color, size=row['Size_Present_Linear'], opacity=0.8),
                        name=f"{idx} ({present_year_label}年)", hovertext="",
                        customdata=np.array([[row['Group_Custom'], row['Bubble_Present'], row['X_Past'], row['Y_Past'], row['Y_Present'], idx]]),
                        hovertemplate=_get_hover_template_mode1(past_offset, is_past=False)
                    ))

                    if pd.notna(row['X_Past']) and pd.notna(row['Y_Past']):
                        fig.add_trace(go.Scatter(
                            x=[row['X_Past'], row['X_Present']], y=[row['Y_Past_Plot'], row['Y_Present_Plot']],
                            mode='lines', line=dict(color=color, dash='dot', width=1),
                            hoverinfo='none', showlegend=False
                        ))

        else:
            df_to_plot_filtered = df_to_plot[df_to_plot['Group_Custom'] != 'N/A'].copy()
            if not df_to_plot_filtered.empty:
                df_to_plot_filtered = df_to_plot_filtered.reset_index()
                df_to_plot_filtered['Y_Present_Plot'] = df_to_plot_filtered['Y_Present'].replace(0, 0.1)

                category_order_list = sorted(list(current_color_map.keys()))
                custom_data_cols = ['Group_Custom', 'Bubble_Present', 'X_Past', 'Y_Past', 'Y_Present']

                fig = px.scatter(
                    df_to_plot_filtered,
                    x='X_Present', y='Y_Present_Plot',
                    size='Bubble_Present', size_max=60,
                    color='Group_Custom',
                    color_discrete_map=current_color_map,
                    category_orders={'Group_Custom': category_order_list},
                    hover_name=axis_label,
                    hover_data=custom_data_cols,
                    log_y=True,
                )

                fig.update_traces(
                    hoverlabel=dict(bgcolor='white'),
                    hovertemplate=_get_hover_template_mode2(past_offset, is_past=False)
                )

        fig.add_vline(x=x_threshold, line_width=1, line_dash="dash", line_color="gray")
        fig.add_hline(y=y_threshold, line_width=1, line_dash="dash", line_color="gray")

        fig.update_layout(
            title=title,
            xaxis_title=f"← 勢い減速 | {xaxis_title_label} | 勢い加速 → (十字線: {x_threshold:.1%})",
            yaxis_title="← 活動鈍化 | 現在の活動量 | 活動活発 →",
            xaxis_tickformat='.0%',
            yaxis_type="log",
            height=800,
            legend_title_text='戦略グループ' if not highlight_targets else '注目対象'
        )

        st.plotly_chart(fig, use_container_width=True)

        st.session_state.df_momentum_export = df_to_plot.copy()


    # --- C. ドリルダウン分析 ---
    with tab_c:
        st.subheader("分析対象の選択")
        drilldown_options = st.session_state.get("mega_drilldown_options", [('(分析対象を選択)', '(分析対象を選択)')])

        selected_drilldown_target = st.selectbox(
            "ドリルダウン対象:",
            options=drilldown_options,
            format_func=lambda x: x[0],
            key="drill_target"
        )
        drilldown_target = selected_drilldown_target[1]

        st.subheader("マップパラメータ (UMAP / HDBSCAN)")
        col1, col2 = st.columns(2)
        with col1:
            drill_n_neighbors = st.number_input('UMAP 近傍点 (n_neighbors):', min_value=2, value=15, key="drill_n_neighbors")
            drill_min_dist = st.number_input('UMAP 最小距離 (min_dist):', min_value=0.0, value=0.1, format="%.2f", key="drill_min_dist")
        with col2:
            drill_min_cluster_size = st.number_input('HDBSCAN 最小クラスタサイズ:', min_value=2, value=5, key="drill_min_cluster_size")
            drill_min_samples = st.number_input('HDBSCAN 最小サンプル数:', min_value=1, value=5, key="drill_min_samples")

        st.subheader("技術ポートフォリオ分析 (SBERT/UMAP)")

        if st.button("選択対象の技術マップを描画", type="primary", key="drill_run_map"):
            if drilldown_target == '(分析対象を選択)':
                st.error("分析対象を選択してください。")
            else:
                with st.spinner(f"「{drilldown_target}」の技術マップを計算中..."):
                    try:
                        axis_label = st.session_state.mega_axis_label

                        axis_col_name = "applicant_main"
                        if axis_label == "IPC (メイングループ)": axis_col_name = "ipc_main_group"
                        elif axis_label == "Fターム (テーマコード)": axis_col_name = "fterm_main"

                        target_indices_mask = df_main[axis_col_name].apply(lambda l: drilldown_target in l)
                        df_filtered = df_main[target_indices_mask].copy()
                        original_indices = df_main[target_indices_mask].index.tolist()

                        if df_filtered.empty:
                            st.error(f"「{drilldown_target}」に該当するデータがありません。")
                            st.stop()

                        target_embeddings = sbert_embeddings[original_indices]
                        target_tfidf_matrix = tfidf_matrix[original_indices]

                        n_neighbors = min(int(drill_n_neighbors), len(original_indices) - 1)
                        if n_neighbors <= 1: n_neighbors = 1

                        umap_results = UMAP(
                            n_components=2,
                            n_neighbors=n_neighbors,
                            min_dist=float(drill_min_dist),
                            random_state=42
                        ).fit_transform(target_embeddings)
                        df_plot = pd.DataFrame(umap_results, columns=['x', 'y'])

                        clusterer = hdbscan.HDBSCAN(
                            min_cluster_size=int(drill_min_cluster_size),
                            min_samples=int(drill_min_samples),
                            metric='euclidean',
                            cluster_selection_method='eom'
                        )
                        cluster_labels = clusterer.fit_predict(df_plot[['x', 'y']])
                        df_plot['cluster_id'] = cluster_labels

                        df_plot[col_map['app_num']] = df_filtered[col_map['app_num']].values
                        df_plot[col_map['title']] = df_filtered[col_map['title']].values
                        df_plot[col_map['abstract']] = df_filtered[col_map['abstract']].values
                        df_plot[col_map['claim']] = df_filtered[col_map['claim']].values
                        df_plot['year'] = df_filtered['year'].values

                        sbert_sub_cluster_map_auto = {}
                        for cluster_id in sorted(df_plot['cluster_id'].unique()):
                            if cluster_id == -1:
                                sbert_sub_cluster_map_auto[-1] = "ノイズ / 小クラスタ"
                                continue

                            cluster_rows_mask = (df_plot['cluster_id'] == cluster_id)
                            cluster_plot_indices = df_plot[cluster_rows_mask].index.tolist()

                            if not cluster_plot_indices: continue

                            cluster_tfidf_vectors = target_tfidf_matrix[cluster_plot_indices]
                            mean_vector = np.asarray(cluster_tfidf_vectors.mean(axis=0)).flatten()
                            top_words = _get_top_words_from_dense_vector(mean_vector, feature_names, 5)
                            sbert_sub_cluster_map_auto[cluster_id] = f"Cluster {cluster_id}: {top_words}"

                        df_plot['label'] = df_plot['cluster_id'].map(sbert_sub_cluster_map_auto)

                        st.session_state.df_drilldown = df_plot.copy()
                        st.session_state.sbert_sub_cluster_map_auto = sbert_sub_cluster_map_auto
                        st.session_state.sbert_sub_cluster_map_custom = {}
                        st.session_state.drilldown_target_name = drilldown_target

                        st.success("技術マップの計算が完了しました。")
                        st.rerun()

                    except Exception as e:
                        st.error(f"SBERT/UMAP分析中にエラーが発生しました: {e}")
                        import traceback
                        st.exception(traceback.format_exc())

        # --- C-4, C-5, C-6 (描画とフィルタ) ---
        if "df_drilldown" in st.session_state:
            df_to_plot_sbert = st.session_state.df_drilldown.copy()
            sbert_sub_cluster_map_auto = st.session_state.sbert_sub_cluster_map_auto

            st.subheader("サブクラスタ・ラベル編集")
            sbert_sub_cluster_map_custom = {}

            auto_labels_map = sbert_sub_cluster_map_auto
            for cluster_id, auto_label in auto_labels_map.items():
                if cluster_id != -1:
                    new_label = st.text_input(f"[{auto_label}]", value=auto_label, key=f"drill_label_{cluster_id}")
                    sbert_sub_cluster_map_custom[auto_label] = new_label
            if -1 in auto_labels_map:
                noise_label = auto_labels_map[-1]
                sbert_sub_cluster_map_custom[noise_label] = noise_label

            auto_to_custom_map = {
                auto_label: sbert_sub_cluster_map_custom.get(auto_label, auto_label)
                for auto_label in auto_labels_map.values()
            }
            df_to_plot_sbert['label_custom'] = df_to_plot_sbert['label'].map(auto_to_custom_map).fillna('N/A')


            st.subheader("期間フィルタ")
            col1, col2 = st.columns(2)
            with col1:
                sbert_bin_interval = st.selectbox("期間の粒度:", options=[1, 2, 3, 5], index=2, key="drill_interval")

            min_date = df_to_plot_sbert['year'].min()
            max_date = df_to_plot_sbert['year'].max()
            date_bin_options = ["(全期間)"]

            if pd.notna(min_date) and pd.notna(max_date):
                min_date = pd.to_datetime(f"{int(min_date)}-01-01")
                max_date = pd.to_datetime(f"{int(max_date)}-12-31")
                interval = int(sbert_bin_interval)

                bins = pd.date_range(start=min_date, end=max_date + pd.DateOffset(years=interval), freq=f'{interval*12}MS')
                labels = [f"{bins[i].year}-{bins[i+1].year - 1}" for i in range(len(bins)-1)]

                df_dates = pd.to_datetime(df_to_plot_sbert['year'], format='%Y')
                df_to_plot_sbert['date_bin'] = pd.cut(df_dates, bins=bins, labels=labels, right=False)

                date_bin_counts = df_to_plot_sbert['date_bin'].value_counts()
                for label in labels:
                    count = date_bin_counts.get(label, 0)
                    if count > 0:
                        date_bin_options.append(f"{label} ({count}件)")

            with col2:
                selected_date_bin_raw = st.selectbox("表示期間 (フィルタ):", options=date_bin_options, key="drill_date_filter")

            st.subheader("マップ描画")

            title_suffix = ""
            if selected_date_bin_raw == "(全期間)":
                mask_in_range = pd.Series(True, index=df_to_plot_sbert.index)
            else:
                date_bin_label_for_filter = selected_date_bin_raw.split(' (')[0].strip()
                mask_in_range = (df_to_plot_sbert['date_bin'].astype(str) == date_bin_label_for_filter)
                title_suffix = f" ({date_bin_label_for_filter})"

            df_in_range = df_to_plot_sbert[mask_in_range]
            df_out_of_range = df_to_plot_sbert[~mask_in_range]

            fig_sbert = go.Figure()

            palette = px.colors.qualitative.Plotly
            all_unique_labels = sorted(df_to_plot_sbert['label_custom'].unique())
            color_map = {label: palette[i % len(palette)] for i, label in enumerate(all_unique_labels)}

            noise_label = auto_labels_map.get(-1)
            if noise_label and noise_label in color_map:
                color_map[noise_label] = '#ced4da'

            fig_sbert.add_trace(go.Scatter(
                x=df_out_of_range['x'], y=df_out_of_range['y'],
                mode='markers', marker=dict(color='#ced4da', opacity=0.2, size=5),
                name='期間外', showlegend=True,
                customdata=df_out_of_range[[col_map['title'], 'label_custom', col_map['app_num'], 'year']],
                hovertemplate=(
                    f"<b>出願番号: %{{customdata[2]}} (%{{customdata[3]}}年)</b><br>"
                    f"名称: %{{customdata[0]}}<br>"
                    f"クラスタ: %{{customdata[1]}} (期間外)<extra></extra>"
                )
            ))

            in_range_labels = df_in_range['label_custom'].unique()

            for label in all_unique_labels:
                color = color_map[label]

                if label in in_range_labels:
                    df_cluster = df_in_range[df_in_range['label_custom'] == label]
                    fig_sbert.add_trace(go.Scatter(
                        x=df_cluster['x'], y=df_cluster['y'],
                        mode='markers',
                        marker=dict(color=color, size=3 if label == noise_label else 7),
                        name=label,
                        customdata=df_cluster[[col_map['title'], 'label_custom', col_map['app_num'], 'year']],
                        hovertemplate=(
                            f"<b>出願番号: %{{customdata[2]}} (%{{customdata[3]}}年)</b><br>"
                            f"名称: %{{customdata[0]}}<br>"
                            f"クラスタ: %{{customdata[1]}}<extra></extra>"
                        )
                    ))
                else:
                    fig_sbert.add_trace(go.Scatter(
                        x=[None], y=[None], mode='markers',
                        marker=dict(color=color),
                        name=f"{label} (期間外)", showlegend=True
                    ))

            fig_sbert.update_layout(
                title=f"技術ポートフォリオマップ: {st.session_state.drilldown_target_name} (SBERT/UMAP){title_suffix}",
                height=800,
                legend_title_text='サブクラスタ',
                legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5),
                plot_bgcolor='white', paper_bgcolor='white',
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showline=True, linewidth=1, linecolor='black', mirror=True),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showline=True, linewidth=1, linecolor='black', mirror=True, scaleanchor="x", scaleratio=1)
            )

            st.plotly_chart(fig_sbert, use_container_width=True)

            st.session_state.df_drilldown_export = df_to_plot_sbert.copy()

    # --- D. エクスポート ---
    with tab_d:
        st.subheader("動態分析マップ結果をエクスポート")
        if "df_momentum_export" in st.session_state:
            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(encoding='utf-8-sig').encode('utf-8-sig')

            csv_mega = convert_df_to_csv(st.session_state.df_momentum_export)
            st.download_button(
                label="動態分析マップ結果 (CSV) をダウンロード",
                data=csv_mega,
                file_name=f"APOLLO_MEGA_PULSE_{st.session_state.mega_axis_label}.csv",
                mime="text/csv",
            )
        else:
            st.info("Landscape Analysis を実行すると、ダウンロードボタンが表示されます。")

        st.subheader("ドリルダウン結果をエクスポート")
        if "df_drilldown_export" in st.session_state:
            csv_drilldown = convert_df_to_csv(st.session_state.df_drilldown_export)
            target_name = st.session_state.drilldown_target_name.replace(" ", "_").replace("/", "_")
            st.download_button(
                label="ドリルダウン結果 (CSV) をダウンロード",
                data=csv_drilldown,
                file_name=f"APOLLO_MEGA_TELESCOPE_{target_name}.csv",
                mime="text/csv",
            )
        else:
            st.info("Technology Probe を実行すると、ダウンロードボタンが表示されます。")

# --- 共通サイドバーフッター ---
st.sidebar.markdown("---") 
st.sidebar.caption("ナビゲーション:")
st.sidebar.caption("1. Mission Control でデータをアップロードし、前処理を実行します。")
st.sidebar.caption("2. 左のリストから分析モジュールを選択します。")
st.sidebar.markdown("---")
st.sidebar.caption("© 2025 しばやま")
