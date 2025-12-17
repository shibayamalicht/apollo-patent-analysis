import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import japanize_matplotlib
import warnings
import re

# ==================================================================
# --- 1. 設定・ヘルパー関数 ---
# ==================================================================
warnings.filterwarnings('ignore')

def get_theme_config(theme_name):
    themes = {
        "APOLLO Standard": {
            "bg_color": "#ffffff",
            "text_color": "#333333",
            "sidebar_bg": "#f8f9fa",
            "plotly_template": "plotly_white",
            "color_sequence": px.colors.qualitative.G10,
            "accent_color": "#003366",
            "css": """
                html, body { background-color: #ffffff; color: #333333; }
                [data-testid="stSidebar"] { background-color: #f8f9fa; }
                [data-testid="stHeader"] { background-color: #ffffff; }
                h1, h2, h3 { color: #003366; }
            """
        },
        "Modern Presentation": {
            "bg_color": "#fdfdfd",
            "text_color": "#2c3e50",
            "sidebar_bg": "#eaeaea",
            "plotly_template": "plotly_white",
            "color_sequence": ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#8ab17d"],
            "accent_color": "#264653",
            "css": """
                html, body { background-color: #fdfdfd; color: #2c3e50; font-family: "Helvetica Neue", Arial, sans-serif; }
                [data-testid="stSidebar"] { background-color: #eaeaea; }
                [data-testid="stHeader"] { background-color: #fdfdfd; }
                h1, h2, h3 { color: #264653; font-family: "Georgia", serif; }
                .stButton>button { background-color: #264653; color: white; border-radius: 0px; }
            """
        }
    }
    return themes.get(theme_name, themes["APOLLO Standard"])

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

def update_fig_layout(fig, title, height=600, theme_config=None):
    if theme_config is None:
        return fig
    fig.update_layout(
        template=theme_config["plotly_template"],
        title=title,
        paper_bgcolor=theme_config["bg_color"],
        plot_bgcolor=theme_config["bg_color"],
        font_color=theme_config["text_color"],
        height=height
    )
    return fig

# ==================================================================
# --- 2. アプリケーション初期化 & UI構成 ---
# ==================================================================

st.set_page_config(
    page_title="APOLLO | ATLAS",
    page_icon="🌍",
    layout="wide"
)

st.markdown("""
<style>
    html, body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
    [data-testid="stSidebar"] h1 { color: #003366; font-weight: 900 !important; font-size: 2.5rem !important; }
    [data-testid="stSidebarNav"] { display: none !important; }
    [data-testid="stSidebar"] .block-container { padding-top: 2rem; padding-bottom: 1rem; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stButton>button { font-weight: 600; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #f0f2f6; border-radius: 8px 8px 0 0; padding: 10px 15px; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-bottom: 2px solid #003366; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("APOLLO") 
    st.markdown("Advanced Patent & Overall Landscape-analytics Logic Orbiter")
    st.markdown("**v.3**")
    st.markdown("---")
    st.subheader("Home")
    st.page_link("Home.py", label="Mission Control", icon="🛰️")
    st.subheader("Modules")
    st.page_link("pages/1_🌍_ATLAS.py", label="ATLAS", icon="🌍")
    st.page_link("pages/2_💡_CORE.py", label="CORE", icon="💡")
    st.page_link("pages/3_🚀_Saturn_V.py", label="Saturn V", icon="🚀")
    st.page_link("pages/4_📈_MEGA.py", label="MEGA", icon="📈")
    st.page_link("pages/5_🧭_Explorer.py", label="Explorer", icon="🧭")
    st.page_link("pages/6_🔗_CREW.py", label="CREW", icon="🔗")
    st.markdown("---")
    st.caption("ナビゲーション:\n1. Mission Control でデータをアップロードし、前処理を実行します。\n2. 上のリストから分析モジュールを選択します。")
    st.markdown("---")
    st.caption("© 2025 しばやま")

st.title("🌍 ATLAS")
st.markdown("出願年、出願人、IPCなどの基本的な統計グラフを作成します。")

col_theme, _ = st.columns([1, 3])
with col_theme:
    selected_theme = st.selectbox("表示テーマ:", ["APOLLO Standard", "Modern Presentation"], key="atlas_theme_selector")
theme_config = get_theme_config(selected_theme)
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

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "件数推移", 
    "出願人ランキング", 
    "IPCランキング", 
    "出願人×年 バブル", 
    "IPC×出願人 バブル",
    "構成比マップ (Treemap)",
    "ライフサイクルマップ"
])

# 1. 件数推移
with tab1:
    st.subheader("出願件数時系列推移")
    if st.button("件数推移グラフを描画", key="atlas_run_map1"):
        if df_filtered.empty:
            st.warning("データがありません。")
        else:
            yearly_counts = df_filtered['year'].value_counts().sort_index()
            plot_data = yearly_counts.reindex(range(int(stats_start_year), int(stats_end_year) + 1), fill_value=0)
            fig = px.bar(x=plot_data.index, y=plot_data.values, labels={'x': '出願年', 'y': '出願件数'}, color_discrete_sequence=[theme_config["color_sequence"][0]])
            update_fig_layout(fig, f'出願件数時系列推移 ({int(stats_start_year)}年～{int(stats_end_year)}年)', theme_config=theme_config)
            st.plotly_chart(fig, use_container_width=True)

# 2. 出願人ランキング
with tab2:
    st.subheader("出願人ランキング")
    num_to_display_map2 = st.number_input("表示人数:", min_value=1, value=20, key="atlas_num_apps_map2")
    if st.button("出願人ランキングを描画", key="atlas_run_map2"):
        if df_filtered.empty:
            st.warning("データがありません。")
        else:
            assignee_counts = df_filtered['applicant_main'].explode().str.strip().value_counts().head(int(num_to_display_map2)).sort_values(ascending=True)
            fig = px.bar(x=assignee_counts.values, y=assignee_counts.index, orientation='h', labels={'x': '特許件数', 'y': '出願人'}, color_discrete_sequence=[theme_config["color_sequence"][1]])
            update_fig_layout(fig, f'出願人ランキング ({int(stats_start_year)}年～{int(stats_end_year)}年)', height=max(600, len(assignee_counts)*30), theme_config=theme_config)
            st.plotly_chart(fig, use_container_width=True)

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
            st.plotly_chart(fig, use_container_width=True)

# 4. 出願人×年 バブル
with tab4:
    st.subheader("出願人 × 年 バブルチャート")
    num_to_display_map4 = st.number_input("表示人数:", min_value=1, value=10, key="atlas_num_apps_map4")
    if st.button("出願人×年 バブルを描画", key="atlas_run_map4"):
        assignees_exploded = df_filtered.explode('applicant_main')
        assignees_exploded['assignee_parsed'] = assignees_exploded['applicant_main'].str.strip()
        top_assignees = assignees_exploded['assignee_parsed'].value_counts().head(int(num_to_display_map4)).index.tolist()
        plot_data = assignees_exploded[assignees_exploded['assignee_parsed'].isin(top_assignees)].groupby(['year', 'assignee_parsed']).size().reset_index(name='件数')
        if plot_data.empty:
            st.warning("データがありません。")
        else:
            fig = px.scatter(plot_data, x='year', y='assignee_parsed', size='件数', color='assignee_parsed', labels={'year': '出願年', 'assignee_parsed': '出願人', '件数': '件数'}, color_discrete_sequence=theme_config["color_sequence"], category_orders={"assignee_parsed": top_assignees})
            update_fig_layout(fig, '出願年別 出願人動向', height=700, theme_config=theme_config)
            st.plotly_chart(fig, use_container_width=True)

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
            st.plotly_chart(fig, use_container_width=True)

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
                    st.plotly_chart(fig, use_container_width=True)
            elif tree_mode == "出願人シェア":
                df_tree = create_treemap_data(df_filtered, stats_start_year, stats_end_year, mode="applicant")
                if df_tree.empty:
                    st.warning("出願人データがありません。")
                else:
                    fig = px.treemap(df_tree, path=['Root', 'Applicant'], values='count', color='count', color_continuous_scale='Blues', labels={'Applicant': '出願人', 'count': '件数', 'Root': '全体'})
                    update_fig_layout(fig, '出願人シェアマップ', height=700, theme_config=theme_config)
                    st.plotly_chart(fig, use_container_width=True)

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
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                ##### 💡 マップの読み方
                * **右上へ伸びる**: 多くのプレイヤーが参入し、出願も増えている「成長期」。
                * **右下へ向かう**: 出願数は多いが、プレイヤーが減っている（淘汰が進んでいる）「成熟期」。
                * **左下へ戻る**: プレイヤーも出願も減っている「衰退期」または「ニッチ化」。
                """)