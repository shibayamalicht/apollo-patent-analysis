import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import japanize_matplotlib
import warnings
import utils
import re

# ==================================================================
# --- 1. 設定・ヘルパー関数 ---
# ==================================================================
warnings.filterwarnings('ignore')



@st.cache_data
def parse_ipc_atlas(ipc, level):
    if not isinstance(ipc, str):
        return ""
    ipc = ipc.strip().upper()
    
    if level == 1:
        return ipc[:4]
    elif level == 2:
        match = re.match(r'([A-H][0-9]{2}[A-Z]\s*[0-9]+)', ipc)
        return f"{match.group(1).strip()}/00" if match else ipc
    else:
        return ipc

@st.cache_data
def create_treemap_data(df_stats, start_year, end_year, mode="ipc"):
    df_target = df_stats.copy()
    
    if mode == "ipc":
        df_exploded = df_target['ipc_normalized'].explode().dropna().astype(str).str.upper()
        data = []
        for ipc in df_exploded:
            if len(ipc) >= 4:
                section = ipc[0]
                ipc_class = ipc[:3]
                subclass = ipc[:4]
                data.append([section, ipc_class, subclass])
        df_tree = pd.DataFrame(data, columns=['Section', 'Class', 'Subclass'])
        df_tree['count'] = 1
        return df_tree
        
    elif mode == "applicant":
        df_exploded = df_target['applicant_main'].explode().dropna()
        df_tree = df_exploded.value_counts().reset_index()
        df_tree.columns = ['Applicant', 'count']
        df_tree = df_tree.head(50)
        df_tree['Root'] = 'Total'
        return df_tree

def update_fig_layout(fig, title, height=600, theme_config=None, show_legend=True):
    if theme_config is None:
        return fig
    
    # タイトルから暗黙的/明示的なHTMLタグを除去してサニタイズ
    if isinstance(title, str):
        title = re.sub(r'<[^>]+>', '', title)

    layout_params = dict(
        template=theme_config["plotly_template"],
        title=dict(text=title, font=dict(size=18, color=theme_config["text_color"], family="Helvetica Neue", weight="normal")),
        paper_bgcolor=theme_config["bg_color"],
        plot_bgcolor=theme_config["bg_color"],
        font_color=theme_config["text_color"],
        height=height
    )
    if not show_legend:
        layout_params['showlegend'] = False
        
    fig.update_layout(**layout_params)
    return fig

# ==================================================================
# --- 2. アプリケーション初期化 & UI構成 ---
# ==================================================================

st.set_page_config(
    page_title="APOLLO | ATLAS",
    page_icon="🌍",
    layout="wide"
)

st.session_state['current_page'] = 'ATLAS'

utils.render_sidebar()

st.title("🌍 ATLAS")
st.markdown("出願年、出願人、IPCなどの基本的な統計グラフを作成します。")

col_theme, _ = st.columns([1, 3])
with col_theme:
    selected_theme = st.selectbox("表示テーマ:", ["APOLLO Standard", "Modern Presentation"], key="atlas_theme_selector")
theme_config = utils.get_theme_config(selected_theme)
st.markdown(f"<style>{theme_config['css']}</style>", unsafe_allow_html=True)

# ==================================================================
# --- 3. データロード & 前処理チェック ---
# ==================================================================

if not st.session_state.get("preprocess_done", False):
    st.error("分析データがありません。")
    st.warning("先に「Mission Control」（メインページ）でファイルをアップロードし、「分析エンジン起動」を実行してください。")
    st.stop()
else:
    try:
        df_main = st.session_state.df_main
        col_map = st.session_state.col_map
        required_cols = ['year', 'applicant_main', 'ipc_normalized']
        if not all(col in df_main.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df_main.columns]
            st.error(f"エラー: 必要なカラム {missing} が見つかりません。Mission Controlで前処理を再実行してください。")
            st.stop()
    except Exception as e:
        st.error(f"データの読み込みに失敗しました: {e}")
        st.stop()

# ==================================================================
# --- 4. 分析アプリケーション ---
# ==================================================================

st.subheader("共通フィルタ設定")
min_year = int(df_main['year'].min())
max_year = int(df_main['year'].max())

col1, col2 = st.columns(2)
with col1:
    stats_start_year = st.number_input('集計開始年:', min_value=min_year, max_value=max_year, value=min_year, key="atlas_start_year")
with col2:
    stats_end_year = st.number_input('集計終了年:', min_value=min_year, max_value=max_year, value=max_year, key="atlas_end_year")

try:
    df_filtered = df_main[
        (df_main['year'] >= int(stats_start_year)) & 
        (df_main['year'] <= int(stats_end_year))
    ].copy()
    st.success(f"集計対象: {int(stats_start_year)}年～{int(stats_end_year)}年 (全 {len(df_filtered)} 件)")
except Exception as e:
    st.error(f"期間フィルタの適用に失敗しました: {e}")
    df_filtered = pd.DataFrame()

st.markdown("---")

tab1, tab1_line, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "件数推移", 
    "件数推移（折れ線）",
    "出願人ランキング", 
    "IPCランキング", 
    "出願人×年 バブル", 
    "IPC×出願人 バブル",
    "構成比マップ (Treemap)",
    "ライフサイクルマップ"
])

# --- ステータスの配色設定 ---
# 全タブで色が統一されるように、ステータスごとの色を固定する
status_color_map = {}
status_col = st.session_state.col_map.get('status')
if status_col:
    # 全てのユニークなステータスを取得（ソートして順序を固定）
    unique_statuses_all = sorted(df_filtered[status_col].dropna().unique().astype(str))
    # Pastel Blue / Distinguishable Palette (User Preferred)
    pastel_blue_palette = [
        "#AEC6CF", # Pastel Blue
        "#779ECB", # Darker Pastel Blue
        "#B39EB5", # Pastel Purple
        "#FFB7B2", # Pastel Red (Soft)
        "#CFCFC4", # Pastel Gray
        "#B0E0E6", # Powder Blue
        "#FFDAC1", # Pastel Peach
        "#E2F0CB", # Pastel Green
        "#FDFD96", # Pastel Yellow
        "#FF6961"  # Pastel Red (Stronger)
    ]
    # 循環的に色を割り当てる
    status_color_map = {s: pastel_blue_palette[i % len(pastel_blue_palette)] for i, s in enumerate(unique_statuses_all)}

# 1. 件数推移
with tab1:
    st.subheader("出願件数時系列推移")
    
    # Status Breakdown Option
    use_status_breakdown = False
    status_col = st.session_state.col_map.get('status')
    if status_col:
        use_status_breakdown = st.checkbox("ステータス内訳を表示", key="atlas_use_status_tab1")

    if st.button("件数推移グラフを描画", key="atlas_run_map1"):
        if df_filtered.empty:
            st.warning("データがありません。")
        else:
            if use_status_breakdown and status_col:
                 # Stacked Bar Chart by Status
                plot_data = df_filtered.groupby(['year', status_col]).size().reset_index(name='count')
                # Use color_discrete_map for consistency
                fig = px.bar(plot_data, x='year', y='count', color=status_col, labels={'year': '出願年', 'count': '出願件数', status_col: 'ステータス'}, 
                             color_discrete_map=status_color_map,
                             category_orders={status_col: sorted(status_color_map.keys())} # 凡例の順序を統一
                            )
            else:
                # Standard Bar Chart
                yearly_counts = df_filtered['year'].value_counts().sort_index()
                plot_data = yearly_counts.reindex(range(int(stats_start_year), int(stats_end_year) + 1), fill_value=0)
                fig = px.bar(x=plot_data.index, y=plot_data.values, labels={'x': '出願年', 'y': '出願件数'}, color_discrete_sequence=[theme_config["color_sequence"][0]])
            
            update_fig_layout(fig, f'出願件数時系列推移 ({int(stats_start_year)}年～{int(stats_end_year)}年)', theme_config=theme_config)
            
            st.session_state['atlas_fig_trend'] = fig
            st.session_state['atlas_data_trend'] = plot_data

    # 永続表示
    if 'atlas_fig_trend' in st.session_state:
        fig = st.session_state['atlas_fig_trend']
        plot_data = st.session_state['atlas_data_trend']
        
        st.plotly_chart(fig, use_container_width=True, config={'editable': False})
        
        # スナップショットボタン
        snap_data = utils.generate_rich_summary(df_filtered, title_col=col_map['title'], abstract_col=col_map['abstract'], n_representatives=0)
        snap_data['module'] = 'ATLAS'
        
        # Optimize Chart Data (Wide Format: Year | Total Only)
        if hasattr(plot_data, 'columns') and 'year' in plot_data.columns and 'count' in plot_data.columns:
             # Group by Year and Sum Count, ignoring Status (Stacked Bar)
             df_snap_safe = plot_data.groupby('year')['count'].sum().reset_index()
             df_snap_safe['year'] = df_snap_safe['year'].astype(int)
        elif hasattr(plot_data, 'reset_index'):
             # Handle Series case (Standard Bar) -> Convert to DataFrame
             df_snap_safe = plot_data.reset_index()
             if df_snap_safe.shape[1] == 2:
                 df_snap_safe.columns = ['year', 'count']
             # Ensure year is int if possible
             if 'year' in df_snap_safe.columns:
                 df_snap_safe['year'] = df_snap_safe['year'].astype(int)
        else:
            df_snap_safe = pd.DataFrame(plot_data)
            
        # Ensure we don't exceed token limits but prioritize showing full year range
        snap_data['chart_data'] = df_snap_safe.head(50).to_string(index=False)
        utils.render_snapshot_button(
            title=f"出願件数推移 ({int(stats_start_year)}-{int(stats_end_year)})",
            description="市場全体の出願動向を示すトレンドグラフ。",
            key="atlas_trend_snap",
            fig=fig,
            data_summary=snap_data
        )

# 1.5 件数推移（折れ線）
with tab1_line:
    st.subheader("件数推移 (折れ線グラフ)")
    
    col_line_1, col_line_2 = st.columns([2, 1])
    
    with col_line_1:
        # モード選択
        line_mode = st.radio("表示モード:", ["全体推移", "出願人比較"], horizontal=True, key="atlas_line_mode")
    
    with col_line_2:
        # ステータス内訳オプション (全体推移モードのみ)
        use_status_breakdown_line = False
        if line_mode == "全体推移" and status_col:
            st.write("") # Spacer
            st.write("")
            use_status_breakdown_line = st.checkbox("ステータス内訳を表示", key="atlas_use_status_line")
    
    target_applicants = []
    
    if line_mode == "出願人比較":
        # Prepare applicant list with counts
        if not df_filtered.empty:
            # Explode and count
            assignees_exploded_line = df_filtered.explode('applicant_main')
            assignees_exploded_line['assignee_parsed'] = assignees_exploded_line['applicant_main'].str.strip()
            
            # Count per applicant
            app_counts = assignees_exploded_line['assignee_parsed'].value_counts()
            
            # Create formatted options: "Name (Count)"
            # Sort is implied by value_counts() which returns descending order
            app_options = [f"{name} ({count})" for name, count in app_counts.items()]
            app_map = {f"{name} ({count})": name for name, count in app_counts.items()}
            
            selected_options = st.multiselect(
                "出願人を選択 (最大5社):", 
                options=app_options,
                max_selections=5,
                key="atlas_line_applicants"
            )
            
            # Map back to raw names
            target_applicants = [app_map[opt] for opt in selected_options]
    
    if st.button("折れ線グラフを描画", key="atlas_run_map1_line"):
        if df_filtered.empty:
            st.warning("データがありません。")
        else:
            fig = None
            plot_data = None
            
            if line_mode == "全体推移":
                if use_status_breakdown_line and status_col:
                     # 積み上げ面グラフ (内訳)
                    plot_data = df_filtered.groupby(['year', status_col]).size().reset_index(name='count')
                    
                    fig = px.area(plot_data, x='year', y='count', color=status_col, markers=True,
                                  labels={'year': '出願年', 'count': '出願件数', status_col: 'ステータス'},
                                  color_discrete_map=status_color_map,
                                  category_orders={status_col: sorted(status_color_map.keys())}
                                 )
                    fig.update_layout(title=dict(text=f'全体件数推移・内訳 ({int(stats_start_year)}年～{int(stats_end_year)}年)', font=dict(size=18)), yaxis=dict(rangemode='tozero'))
                    
                else:
                    # 全体推移 (標準折れ線)
                    yearly_counts = df_filtered['year'].value_counts().sort_index()
                    plot_data = yearly_counts.reindex(range(int(stats_start_year), int(stats_end_year) + 1), fill_value=0).reset_index()
                    plot_data.columns = ['year', 'count']
                    
                    fig = px.line(plot_data, x='year', y='count', markers=True, 
                                  labels={'year': '出願年', 'count': '出願件数'},
                                  color_discrete_sequence=[theme_config["color_sequence"][0]])
                    
                    fig.update_layout(title=dict(text=f'全体件数推移 ({int(stats_start_year)}年～{int(stats_end_year)}年)', font=dict(size=18)), yaxis=dict(rangemode='tozero'))

            else: # 出願人比較
                if not target_applicants:
                    st.warning("出願人を選択してください。")
                else:
                    # Filter data for selected applicants
                    assignees_exploded_line = df_filtered.explode('applicant_main')
                    assignees_exploded_line['assignee_parsed'] = assignees_exploded_line['applicant_main'].str.strip()
                    
                    df_target = assignees_exploded_line[assignees_exploded_line['assignee_parsed'].isin(target_applicants)]
                    
                    if df_target.empty:
                        st.warning("選ばれた出願人のデータが期間内にありません。")
                    else:
                        # Ensure all years are represented for each applicant (fill 0)
                        full_years = range(int(stats_start_year), int(stats_end_year) + 1)
                        plot_data_list = []
                        
                        for app in target_applicants:
                            sub = df_target[df_target['assignee_parsed'] == app]
                            yearly = sub['year'].value_counts().sort_index()
                            yearly = yearly.reindex(full_years, fill_value=0).reset_index()
                            yearly.columns = ['year', 'count']
                            yearly['assignee_parsed'] = app
                            plot_data_list.append(yearly)
                            
                        plot_data = pd.concat(plot_data_list, ignore_index=True)
                        
                        fig = px.line(plot_data, x='year', y='count', color='assignee_parsed', markers=True,
                                      labels={'year': '出願年', 'count': '出願件数', 'assignee_parsed': '出願人'},
                                      color_discrete_sequence=theme_config["color_sequence"])
                        
                        fig.update_layout(title=dict(text='主要出願人の件数推移比較', font=dict(size=18)), yaxis=dict(rangemode='tozero'))
            
            if fig:
                update_fig_layout(fig, '件数推移(折れ線)', theme_config=theme_config)
                
                # Check for session state initialization
                if 'atlas_fig_trend_line' not in st.session_state:
                     st.session_state['atlas_fig_trend_line'] = None
                
                st.session_state['atlas_fig_trend_line'] = fig
                st.session_state['atlas_data_trend_line'] = plot_data

    # Persistent Display
    if 'atlas_fig_trend_line' in st.session_state and st.session_state['atlas_fig_trend_line'] is not None:
        fig = st.session_state['atlas_fig_trend_line']
        data = st.session_state['atlas_data_trend_line']
        
        st.plotly_chart(fig, use_container_width=True, config={'editable': False})
        # Snapshot Button
        snap_data = utils.generate_rich_summary(df_filtered if 'df_target' not in locals() else df_target, title_col=col_map['title'], abstract_col=col_map['abstract'], n_representatives=0)
        snap_data['module'] = 'ATLAS'
        # Optimize Chart Data
        # Optimize Chart Data (Wide Format for Applicants)
        if data is not None and not data.empty:
             if 'assignee_parsed' in data.columns:
                 # Pivot: Year | App A | App B ...
                 df_pivot = data.pivot(index='year', columns='assignee_parsed', values='count').fillna(0).astype(int).reset_index()
                 snap_data['chart_data'] = df_pivot.head(40).to_string(index=False)
             else:
                 snap_data['chart_data'] = data.head(40).to_string(index=False)
        else:
             snap_data['chart_data'] = "No Data"
        utils.render_snapshot_button(
            title="件数推移 (折れ線)",
            description="出願件数の時系列推移（折れ線グラフ）。全体または特定出願人の比較。",
            key="atlas_trend_line_snap",
            fig=fig,
            data_summary=snap_data
        )

# 2. 出願人ランキング
with tab2:
    st.subheader("出願人ランキング")
    col2_1, col2_2 = st.columns([2, 1])
    with col2_1:
         num_to_display_map2 = st.number_input("表示人数:", min_value=1, value=20, key="atlas_num_apps_map2")
    
    # Status Breakdown Option
    use_status_breakdown_tab2 = False
    status_col = st.session_state.col_map.get('status')
    with col2_2:
        if status_col:
            st.write("") # Spacer
            st.write("")
            use_status_breakdown_tab2 = st.checkbox("ステータス内訳を表示", key="atlas_use_status_tab2")

    if st.button("出願人ランキングを描画", key="atlas_run_map2"):
        if df_filtered.empty:
            st.warning("データがありません。")
        else:
            # 1. 上位出願人の特定 (合計件数に基づく)
            assignee_counts = df_filtered['applicant_main'].explode().str.strip().value_counts().head(int(num_to_display_map2)).sort_values(ascending=True)
            top_applicants = assignee_counts.index.tolist()

            if use_status_breakdown_tab2 and status_col:
                # Stacked Bar Chart by Status for Top Applicants
                df_exploded = df_filtered.explode('applicant_main')
                df_exploded['applicant_parsed'] = df_exploded['applicant_main'].str.strip()
                df_top = df_exploded[df_exploded['applicant_parsed'].isin(top_applicants)]
                
                plot_data = df_top.groupby(['applicant_parsed', status_col]).size().reset_index(name='count')
                
                # Ensure sort order matches total count
                fig = px.bar(plot_data, x='count', y='applicant_parsed', color=status_col, orientation='h', 
                             labels={'count': '特許件数', 'applicant_parsed': '出願人', status_col: 'ステータス'}, 
                             color_discrete_map=status_color_map,
                             category_orders={'applicant_parsed': top_applicants[::-1], status_col: sorted(status_color_map.keys())})
            else:
                # Standard Bar Chart
                fig = px.bar(x=assignee_counts.values, y=assignee_counts.index, orientation='h', labels={'x': '特許件数', 'y': '出願人'}, color_discrete_sequence=[theme_config["color_sequence"][1]])
            
            update_fig_layout(fig, f'出願人ランキング ({int(stats_start_year)}年～{int(stats_end_year)}年)', height=max(600, len(assignee_counts)*30), theme_config=theme_config)
            
            st.session_state['atlas_fig_ranking'] = fig
            st.session_state['atlas_data_ranking'] = assignee_counts

    # 永続表示
    if 'atlas_fig_ranking' in st.session_state:
        fig = st.session_state['atlas_fig_ranking']
        assignee_counts = st.session_state['atlas_data_ranking']
        
        st.plotly_chart(fig, use_container_width=True, config={'editable': False})
        
        # スナップショットボタン
        snap_data = utils.generate_rich_summary(df_filtered, title_col=col_map['title'], abstract_col=col_map['abstract'], n_representatives=0)
        snap_data['module'] = 'ATLAS'
        
        # Optimize Chart Data
        df_snap_safe = assignee_counts.head(30).reset_index()
        df_snap_safe.columns = ['Applicant', 'Count']
        df_snap_safe['Applicant'] = df_snap_safe['Applicant'].astype(str).str.slice(0, 50)
        snap_data['chart_data'] = df_snap_safe.to_string(index=False)
        utils.render_snapshot_button(
            title=f"主要出願人ランキング ({int(stats_start_year)}-{int(stats_end_year)})",
            description="特許出願件数に基づく市場の主要プレイヤーランキング。",
            key="atlas_applicant_snap",
            fig=fig,
            data_summary=snap_data
        )

# 3. IPCランキング
with tab3:
    st.subheader("IPCランキング")
    ipc_level_map3 = st.selectbox("IPCレベル:", [(1, "サブクラス (A01B)"), (2, "メイングループ (A01B 1/00)")], format_func=lambda x: x[1], key="atlas_ipc_level_map3")
    num_to_display_map3 = st.number_input("表示IPC数:", min_value=1, value=20, key="atlas_num_ipcs_map3")
    if st.button("IPCランキングを描画", key="atlas_run_map3"):
        if df_filtered.empty:
            st.warning("データがありません。")
        else:
            ipc_exploded = df_filtered['ipc_normalized'].explode().dropna()
            ipc_parsed = ipc_exploded.apply(lambda x: parse_ipc_atlas(x, ipc_level_map3[0]))
            ipc_counts = ipc_parsed.value_counts().head(int(num_to_display_map3)).sort_values(ascending=True)
            fig = px.bar(x=ipc_counts.values, y=ipc_counts.index, orientation='h', labels={'x': '特許件数', 'y': 'IPC分類'}, color_discrete_sequence=[theme_config["color_sequence"][2]])
            update_fig_layout(fig, f'IPCランキング ({ipc_level_map3[1]})', height=max(600, len(ipc_counts)*30), theme_config=theme_config)
            
            st.session_state['atlas_fig_ipc'] = fig
            st.session_state['atlas_data_ipc'] = ipc_counts

    # 永続表示
    if 'atlas_fig_ipc' in st.session_state:
        fig = st.session_state['atlas_fig_ipc']
        data = st.session_state['atlas_data_ipc']
        
        st.plotly_chart(fig, use_container_width=True, config={'editable': False})
        
        # スナップショットボタン
        snap_data = utils.generate_rich_summary(df_filtered, title_col=col_map['title'], abstract_col=col_map['abstract'], n_representatives=0)
        snap_data['module'] = 'ATLAS'
        
        # Optimize Chart Data
        df_snap_safe = data.head(30).reset_index()
        df_snap_safe.columns = ['IPC', 'Count']
        snap_data['chart_data'] = df_snap_safe.to_string(index=False)
        utils.render_snapshot_button(
            title=f"IPCランキング ({ipc_level_map3[1]})",
            description="技術分野 (IPC) 別の上位ランキング。",
            key="atlas_ipc_snap",
            fig=fig,
            data_summary=snap_data
        )

# 4. 出願人×年 バブル
with tab4:
    st.subheader("出願人 × 年 バブルチャート")
    col4_1, col4_2 = st.columns([2, 1])
    with col4_1:
         num_to_display_map4 = st.number_input("表示人数:", min_value=1, value=10, key="atlas_num_apps_map4")
    
    # Status Breakdown Option
    use_status_breakdown_tab4 = False
    status_col = st.session_state.col_map.get('status')
    with col4_2:
        if status_col:
            st.write("") # Spacer
            st.write("")
            use_status_breakdown_tab4 = st.checkbox("ステータス内訳を表示", key="atlas_use_status_tab4")

    if st.button("出願人×年 バブルを描画", key="atlas_run_map4"):
        assignees_exploded = df_filtered.explode('applicant_main')
        assignees_exploded['assignee_parsed'] = assignees_exploded['applicant_main'].str.strip()
        top_assignees = assignees_exploded['assignee_parsed'].value_counts().head(int(num_to_display_map4)).index.tolist()
        
        # Filter for top applicants upfront
        df_target = assignees_exploded[assignees_exploded['assignee_parsed'].isin(top_assignees)].copy()
        
        if df_target.empty:
            st.warning("データがありません。")
        else:
            if use_status_breakdown_tab4 and status_col:
                # --- グリッド状パイチャートの描画 ---
                # 1. グリッド寸法の計算
                start_y = int(stats_start_year)
                end_y = int(stats_end_year)
                
                # Let's align with the filter range for stability
                cols = list(range(start_y, end_y + 1))
                
                # Filter data to this range
                df_target = df_target[df_target['year'].isin(cols)]
                
                if df_target.empty:
                    st.warning("指定期間内のデータがありません。")
                else:
                    # Rows = Applicants, Cols = Years (Linear Sequence)
                    rows = top_assignees 
                    
                    n_rows = len(rows)
                    n_cols = len(cols)
                    
                    fig = go.Figure()
                    
                    # Group by [Applicant, Year, Status]
                    grid_data = df_target.groupby(['assignee_parsed', 'year', status_col]).size().reset_index(name='count')
                    total_counts = df_target.groupby(['assignee_parsed', 'year']).size().reset_index(name='total')
                    max_total = total_counts['total'].max()
                    
                    # Layout Calculation
                    x_margin_l = 0.20 # Increased to 0.20 to prevent label cutoff and align with Standard
                    x_margin_r = 0.02
                    y_margin_b = 0.10 
                    y_margin_t = 0.05
                    
                    plot_width = 1.0 - (x_margin_l + x_margin_r)
                    plot_height = 1.0 - (y_margin_b + y_margin_t)
                    
                    cell_w = plot_width / n_cols
                    cell_h = plot_height / n_rows
                    
                    # Prepare Legend Colors
                    
                    # Filter map to only statuses present in this view for the legend
                    statuses_in_view = sorted(df_target[status_col].dropna().unique().astype(str))
                    
                    # Add Dummy Traces for Legend (Scatter markers)
                    for status in statuses_in_view:
                        fig.add_trace(go.Scatter(
                            x=[None], y=[None],
                            mode='markers',
                            marker=dict(size=10, color=status_color_map.get(status, '#ccc')),
                            name=status,
                            showlegend=True
                        ))
                    
                    # Annotations for Axes
                    annotations = []
                    
                    # Y-Axis Labels (Applicants)
                    for i, applicant in enumerate(rows):
                        y_center = (1.0 - y_margin_t) - (i * cell_h) - (cell_h / 2)
                        
                        annotations.append(dict(
                            x=x_margin_l - 0.01, y=y_center,
                            xref="paper", yref="paper",
                            text="",
                            showarrow=False, xanchor="right", yanchor="middle",
                            font=dict(size=12, color=theme_config["text_color"])
                        ))
                        
                    # X-axis labels are now handled by layout.xaxis
                    annotations = []

                    # Add Pie Traces
                    for i, applicant in enumerate(rows):
                        for j, year in enumerate(cols):
                            cell_df = grid_data[(grid_data['assignee_parsed'] == applicant) & (grid_data['year'] == year)]
                            
                            if not cell_df.empty:
                                total = cell_df['count'].sum()
                                max_r = min(cell_w, cell_h) / 2 * 0.9
                                scale_factor = (total / max_total) ** 0.5
                                # Use sqrt scaling for visual size
                                y_center = (1.0 - y_margin_t) - (i * cell_h) - (cell_h / 2)
                                x_center = x_margin_l + (j * cell_w) + (cell_w / 2)
                                
                                # Domain Calc
                                d_w = cell_w * scale_factor
                                
                                x0 = x_center - (d_w / 2)
                                x1 = x_center + (d_w / 2)
                                y0 = y_center - (scale_factor * cell_h / 2) 
                                y1 = y_center + (scale_factor * cell_h / 2)
                                
                                # Map colors explicitly
                                labels = cell_df[status_col].astype(str).tolist()
                                values = cell_df['count'].tolist()
                                colors = [status_color_map.get(l, '#ccc') for l in labels]
                                
                                fig.add_trace(go.Pie(
                                    labels=labels,
                                    values=values,
                                    marker=dict(colors=colors),
                                    domain=dict(x=[x0, x1], y=[y0, y1]),
                                    showlegend=False, # Use dummy legend instead
                                    textinfo='none',
                                    hoverinfo='label+value',
                                    sort=False 
                                ))
                    
                    # Manual Grid Lines Removed (Handled by yaxis.showgrid)
                    shapes = []
                    
                    # Layout Finalization
                    fig.update_layout(
                        height=max(700, n_rows * 50),
                        annotations=annotations,
                        shapes=shapes,
                        showlegend=True,
                        xaxis=dict(
                            visible=True,
                            domain=[x_margin_l, 1.0 - x_margin_r],
                            # Range: [min_year - 0.5, max_year + 0.5]
                            range=[cols[0] - 0.5, cols[-1] + 0.5],
                            tickmode='auto', 
                            side='bottom',
                            color=theme_config["text_color"],
                            fixedrange=True, 
                            showgrid=False,
                            zeroline=False,
                            showline=False
                        ),
                        yaxis=dict(
                            visible=True,
                            domain=[y_margin_b, 1.0 - y_margin_t],
                            # Map rows (0..N-1) to Y-axis. Top-down order.
                            range=[-0.5, n_rows - 0.5],
                            tickmode='array',
                            tickvals=list(range(n_rows)),
                            ticktext=rows[::-1], # Reverse to put Top Applicant at Top
                            color=theme_config["text_color"],
                            fixedrange=True, 
                            showgrid=True,   
                            gridcolor="#eee",
                            zeroline=False,
                            showline=False
                        ),
                        margin=dict(l=0, r=0, t=40, b=0),
                        paper_bgcolor=theme_config["bg_color"], 
                        plot_bgcolor=theme_config["bg_color"],
                        font_color=theme_config["text_color"],
                        title=dict(text=f'出願年別 出願人動向 (内訳: {status_col})', font=dict(size=18, weight="normal"))
                    )
                    
                # Save to unified state
                st.session_state['atlas_fig_bubble_tab4'] = fig

                # Re-create grid data for state storage since it was local
                grid_data_export = df_target.groupby(['year', 'assignee_parsed', status_col]).size().reset_index(name='count')
                st.session_state['atlas_data_bubble_tab4'] = grid_data_export
            else:
                # Standard Bubble Chart
                plot_data = df_target.groupby(['year', 'assignee_parsed']).size().reset_index(name='件数')
                
                # --- Shared Layout Constants ---
                x_margin_l = 0.20 # Match Breakdown
                x_margin_r = 0.02
                y_margin_b = 0.10
                y_margin_t = 0.05
                
                fig = px.scatter(plot_data, x='year', y='assignee_parsed', size='件数', color='assignee_parsed', 
                                 labels={'year': '出願年', 'assignee_parsed': '出願人', '件数': '件数'}, 
                                 color_discrete_sequence=theme_config["color_sequence"], 
                                 category_orders={"assignee_parsed": top_assignees}) # px handles order
                
                # Apply Strict Layout to Match Breakdown
                update_fig_layout(fig, '出願年別 出願人動向', height=max(700, len(top_assignees)*50), theme_config=theme_config)
                
                fig.update_layout(
                     margin=dict(l=0, r=0, t=40, b=0),
                     xaxis=dict(
                         domain=[x_margin_l, 1.0 - x_margin_r],
                         fixedrange=True,
                         side='bottom'
                     ),
                     yaxis=dict(
                         domain=[y_margin_b, 1.0 - y_margin_t],
                         fixedrange=True,
                         showgrid=True,
                         gridcolor="#eee",
                         visible=True # Ensure visible
                     )
                )
                
                # Save to unified state
                st.session_state['atlas_fig_bubble_tab4'] = fig
                st.session_state['atlas_data_bubble_tab4'] = plot_data

    # Persistent Display (Unified)
    if 'atlas_fig_bubble_tab4' in st.session_state:
        fig = st.session_state['atlas_fig_bubble_tab4']
        data = st.session_state['atlas_data_bubble_tab4']
        
        st.plotly_chart(fig, use_container_width=True, config={'editable': False})
        
        # Snapshot Button
        # Snapshot Button
        snap_data = utils.generate_rich_summary(data, title_col=col_map['title'], abstract_col=col_map['abstract'], n_representatives=0)
        snap_data['module'] = 'ATLAS'
        
        # Optimize Chart Data
        # Optimize Chart Data
        if hasattr(data, 'head'):

             # Copy data to avoid mutating session state
             chart_df = data.copy()
             
             # Normalize column names (件数 -> count)
             if '件数' in chart_df.columns:
                 chart_df.rename(columns={'件数': 'count'}, inplace=True)
             
             # If data is 'grid_data_export' (Year, Applicant, Status, Count) or 'plot_data' (Year, Applicant, Count)
             
             # Filter only necessary columns
             target_cols = [c for c in ['year', 'assignee_parsed', 'count', status_col] if c in chart_df.columns]
             df_snap_safe = chart_df[target_cols].copy()
             
             # Format
             if 'assignee_parsed' in df_snap_safe.columns:
                 df_snap_safe['assignee_parsed'] = df_snap_safe['assignee_parsed'].astype(str).str.slice(0, 30)
             
             # Pivot for readability (Year | App | Count...) is still long.
             # Maybe Pivot: Year vs Applicant (Values = Total Count)
             if 'year' in df_snap_safe.columns and 'assignee_parsed' in df_snap_safe.columns:
                 # Aggregate to remove status if just showing bubble position
                 df_pivot = df_snap_safe.groupby(['year', 'assignee_parsed'])['count'].sum().reset_index()
                 df_pivot = df_pivot.pivot(index='year', columns='assignee_parsed', values='count').fillna(0).astype(int).reset_index()
                 snap_data['chart_data'] = df_pivot.head(40).to_string(index=False)
             else:
                 snap_data['chart_data'] = df_snap_safe.head(40).to_string(index=False)
        else:
             snap_data['chart_data'] = "Data Summary"

        utils.render_snapshot_button(
            title="出願年別 出願人バブルチャート",
            description="主要出願人の時系列活動量 (内訳含む)",
            key="atlas_bubble_tab4_snap",
            fig=fig,
            data_summary=snap_data
        )


# 5. IPC×出願人 バブル
with tab5:
    st.subheader("IPC × 出願人 バブルチャート")
    col1, col2, col3 = st.columns(3)
    with col1: ipc_level_map5 = st.selectbox("IPCレベル:", [(1, "サブクラス"), (2, "メイングループ")], format_func=lambda x: x[1], key="atlas_ipc_level_map5")
    with col2: num_ipcs_map5 = st.number_input("IPC数 (Y軸):", min_value=1, value=15, key="atlas_num_ipcs_map5")
    with col3: num_apps_map5 = st.number_input("出願人数 (X軸):", min_value=1, value=15, key="atlas_num_apps_map5")
    if st.button("IPC×出願人 バブルを描画", key="atlas_run_map5"):
        df_exploded = df_filtered.explode('applicant_main').explode('ipc_normalized')
        df_exploded.dropna(subset=['applicant_main', 'ipc_normalized'], inplace=True)
        df_exploded['assignee_parsed'] = df_exploded['applicant_main'].str.strip()
        df_exploded['ipc_parsed'] = df_exploded['ipc_normalized'].apply(lambda x: parse_ipc_atlas(x, ipc_level_map5[0]))
        top_assignees = df_exploded['assignee_parsed'].value_counts().head(int(num_apps_map5)).index.tolist()
        top_ipcs = df_exploded['ipc_parsed'].value_counts().head(int(num_ipcs_map5)).index.tolist()
        df_top = df_exploded[df_exploded['assignee_parsed'].isin(top_assignees) & df_exploded['ipc_parsed'].isin(top_ipcs)]
        plot_data = df_top.groupby(['assignee_parsed', 'ipc_parsed']).size().reset_index(name='件数')
        if plot_data.empty:
            st.warning("データがありません。")
        else:
            fig = px.scatter(plot_data, x='assignee_parsed', y='ipc_parsed', size='件数', color='ipc_parsed', labels={'assignee_parsed': '出願人', 'ipc_parsed': 'IPC分類', '件数': '件数'}, color_discrete_sequence=theme_config["color_sequence"], category_orders={"ipc_parsed": top_ipcs})
            update_fig_layout(fig, f'IPC ({ipc_level_map5[1]}) × 出願人 ポートフォリオ', height=800, theme_config=theme_config)
            
            st.session_state['atlas_fig_bubble_ipc'] = fig
            st.session_state['atlas_data_bubble_ipc'] = plot_data

    # Persistent Display
    if 'atlas_fig_bubble_ipc' in st.session_state:
        fig = st.session_state['atlas_fig_bubble_ipc']
        data = st.session_state['atlas_data_bubble_ipc']
        
        st.plotly_chart(fig, use_container_width=True, config={'editable': False})
        
        # Snapshot Button
        # Snapshot Button
        snap_data = utils.generate_rich_summary(data, title_col=col_map['title'], abstract_col=col_map['abstract'], n_representatives=0)
        snap_data['module'] = 'ATLAS'
        
        # Optimize Chart Data (IPC Bubble)
        df_snap_safe = data.head(30).copy()
        if 'assignee_parsed' in df_snap_safe.columns:
             df_snap_safe['assignee_parsed'] = df_snap_safe['assignee_parsed'].astype(str).str.slice(0, 50)
        snap_data['chart_data'] = df_snap_safe.to_string(index=False)

        utils.render_snapshot_button(
            title=f"IPC x 出願人 ポートフォリオ",
            description="主要出願人の技術分野（IPC）ごとの注力度合いを示すバブルチャート。",
            key="atlas_bubble_ipc_snap",
            fig=fig,
            data_summary=snap_data
        )

# 6. 構成比マップ
with tab6:
    st.subheader("構成比マップ (Treemap)")
    tree_mode = st.radio("表示モード:", ["IPC階層 (技術分野)", "出願人シェア"], horizontal=True, key="atlas_tree_mode")
    if st.button("ツリーマップを描画", key="atlas_run_treemap"):
        with st.spinner("作成中..."):
            if tree_mode == "IPC階層 (技術分野)":
                df_tree = create_treemap_data(df_filtered, stats_start_year, stats_end_year, mode="ipc")
                if df_tree.empty:
                    st.warning("IPCデータがありません。")
                else:
                    fig = px.treemap(df_tree, path=['Section', 'Class', 'Subclass'], values='count', color='Section', color_discrete_sequence=theme_config["color_sequence"])
                    update_fig_layout(fig, 'IPC階層構造マップ', height=700, theme_config=theme_config)
                    
                    st.session_state['atlas_fig_tree'] = fig
                    st.session_state['atlas_data_tree'] = df_tree

            elif tree_mode == "出願人シェア":
                df_tree = create_treemap_data(df_filtered, stats_start_year, stats_end_year, mode="applicant")
                if df_tree.empty:
                    st.warning("出願人データがありません。")
                else:
                    fig = px.treemap(df_tree, path=['Root', 'Applicant'], values='count', color='count', color_continuous_scale='Blues', labels={'Applicant': '出願人', 'count': '件数', 'Root': '全体'})
                    update_fig_layout(fig, '出願人シェアマップ', height=700, theme_config=theme_config)
                    
                    st.session_state['atlas_fig_tree'] = fig
                    st.session_state['atlas_data_tree'] = df_tree

    # Persistent Display
    if 'atlas_fig_tree' in st.session_state:
        fig = st.session_state['atlas_fig_tree']
        data = st.session_state['atlas_data_tree']
        
        st.plotly_chart(fig, use_container_width=True, config={'editable': False})
        
        # Snapshot Button
        snap_data = utils.generate_rich_summary(df_filtered, title_col=col_map['title'], abstract_col=col_map['abstract'], n_representatives=0)
        snap_data['module'] = 'ATLAS'
        
        # Optimize Chart Data (Treemap)
        df_snap_safe = data.head(30).copy()
        if 'Applicant' in df_snap_safe.columns:
             df_snap_safe['Applicant'] = df_snap_safe['Applicant'].astype(str).str.slice(0, 50)
        snap_data['chart_data'] = df_snap_safe.to_string(index=False)

        utils.render_snapshot_button(
            title="構成比マップ (Treemap)",
            description="技術分野または出願人のシェア構成を示すツリーマップ。",
            key="atlas_tree_snap",
            fig=fig,
            data_summary=snap_data
        )

# 7. ライフサイクルマップ
with tab7:
    st.subheader("技術ライフサイクルマップ")
    st.info("""
    **技術の発展段階（ライフサイクル）を診断します。**
    - 縦軸: 出願人数（参入企業の多さ＝競争の激しさ）
    - 横軸: 出願件数（技術活動の活発さ）
    - プロット: 出願年ごとの生データを曲線で近似して繋いでいます。
    """)
    
    if st.button("ライフサイクルマップを描画", key="atlas_run_lifecycle"):
        with st.spinner("計算中..."):
            df_lc = df_filtered.copy()
            df_lc_applicants = df_lc.explode('applicant_main')
            df_lc_applicants['applicant_main'] = df_lc_applicants['applicant_main'].str.strip()
            df_lc_applicants = df_lc_applicants[df_lc_applicants['applicant_main'] != '']
            
            apps_count = df_lc.groupby('year').size()
            inventors_count = df_lc_applicants.groupby('year')['applicant_main'].nunique()
            
            lifecycle_data = pd.DataFrame({
                'year': apps_count.index,
                'applications': apps_count.values,
                'applicants': inventors_count.reindex(apps_count.index, fill_value=0).values
            })
            
            if lifecycle_data.empty or len(lifecycle_data) < 2:
                st.warning("データ不足のためマップを描画できません（期間を広げてください）。")
            else:
                lifecycle_data['year_label'] = lifecycle_data['year'].apply(lambda y: f"'{str(int(y))[-2:]}")
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=lifecycle_data['applications'],
                    y=lifecycle_data['applicants'],
                    mode='lines',
                    line=dict(shape='spline', smoothing=1.3, width=3, color='#aaaaaa'),
                    opacity=0.5,
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig.add_trace(go.Scatter(
                    x=lifecycle_data['applications'],
                    y=lifecycle_data['applicants'],
                    mode='markers+text',
                    text=lifecycle_data['year_label'],
                    textposition="top center",
                    marker=dict(
                        size=12,
                        color=lifecycle_data['year'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="出願年")
                    ),
                    showlegend=False,
                    hovertemplate="<b>%{text}</b><br>件数: %{x}<br>人数: %{y}<extra></extra>"
                ))
                
                update_fig_layout(fig, '技術ライフサイクル (出願人数 vs 出願件数)', height=700, theme_config=theme_config)
                
                fig.update_layout(
                    xaxis_title="出願件数 (技術活動量)",
                    yaxis_title="出願人数 (参入プレイヤー数)"
                )
                
                st.session_state['atlas_fig_life'] = fig
                st.session_state['atlas_data_life'] = lifecycle_data

    # Persistent Display
    if 'atlas_fig_life' in st.session_state:
        fig = st.session_state['atlas_fig_life']
        data = st.session_state['atlas_data_life']
        
        st.plotly_chart(fig, use_container_width=True, config={'editable': False})
        
        st.markdown("""
        ##### 💡 マップの読み方
        * **右上へ伸びる**: 多くのプレイヤーが参入し、出願も増えている「成長期」。
        * **右下へ向かう**: 出願数は多いが、プレイヤーが減っている（淘汰が進んでいる）「成熟期」。
        * **左下へ戻る**: プレイヤーも出願も減っている「衰退期」または「ニッチ化」。
        """)
        
        snap_data = utils.generate_rich_summary(df_filtered, title_col=col_map['title'], abstract_col=col_map['abstract'], n_representatives=0)
        snap_data['module'] = 'ATLAS'
        
        # Optimize Chart Data (Lifecycle)
        df_snap_safe = data.head(30).copy()
        snap_data['chart_data'] = df_snap_safe.to_string(index=False)
        utils.render_snapshot_button(
            title="技術ライフサイクルマップ",
            description="出願件数と出願人数（参入企業数）の相関から、技術の成熟度を診断するマップ。",
            key="atlas_life_snap",
            fig=fig,
            data_summary=snap_data
        )