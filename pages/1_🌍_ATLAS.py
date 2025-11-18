%%writefile pages/1_🌍_ATLAS.py
# ==================================================================
# --- 1. ライブラリのインポート ---
# ==================================================================
import streamlit as st
import pandas as pd
import numpy as np
import warnings
import re

# グラフ描画
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import plotly.express as px
import japanize_matplotlib # 日本語化

# 警告を非表示
warnings.filterwarnings('ignore')

# ==================================================================
# --- 2. ATLAS専用ヘルパー関数 ---
# ==================================================================

@st.cache_data
def parse_ipc_atlas(ipc, level):
    """IPCコードを指定されたレベルに解析する内部関数"""
    ipc = str(ipc).strip().upper()
    
    if level == 1:  # サブクラス
        return ipc[:4]
    elif level == 2:  # メイングループ
        match = re.match(r'([A-H][0-9]{2}[A-Z]\s*[0-9]+)', ipc)
        return f"{match.group(1).strip()}/00" if match else ipc
    else:  # サブグループ
        return ipc

@st.cache_data
def create_application_trend_chart(df_stats, start_year, end_year):
    """(MAP 1) 出願件数時系列推移"""
    yearly_counts = df_stats['year'].value_counts().sort_index()
    if yearly_counts.empty:
        return "有効な出願年データがありません。"
    plot_data = yearly_counts.reindex(range(int(start_year), int(end_year) + 1), fill_value=0)
    if plot_data.empty:
        return "指定期間にデータがありません。"
    
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
def create_assignee_ranking_map(df_stats, num_to_display, start_year, end_year):
    """(MAP 2) 出願人ランキング"""
    assignee_counts = df_stats['applicant_main'].explode().str.strip().value_counts()
    data_to_plot = assignee_counts.head(int(num_to_display)).sort_values(ascending=True)

    if data_to_plot.empty:
        return "集計結果がありません。"
    
    plt.style.use('seaborn-v0_8-talk')
    fig, ax = plt.subplots(figsize=(12, max(5, 0.4 * len(data_to_plot))))
    bars = ax.barh(data_to_plot.index, data_to_plot.values, color='steelblue')
    ax.set_title(f'出願人ランキング ({int(start_year)}年～{int(end_year)}年)', fontsize=20, pad=20)
    ax.set_xlabel('特許件数', fontsize=14); ax.set_ylabel('出願人名', fontsize=14)
    for bar in bars:
        width = bar.get_width()
        ax.text(width + width * 0.01, bar.get_y() + bar.get_height()/2, f'{int(width)}', ha='left', va='center', fontsize=12)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.15)
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig

@st.cache_data
def create_ipc_ranking_map(df_stats, ipc_level_tuple, num_to_display, start_year, end_year):
    """(MAP 3) IPCランキング"""
    ipc_level, level_name = ipc_level_tuple
    ipc_exploded = df_stats['ipc_normalized'].explode().dropna()
    ipc_parsed = ipc_exploded.apply(lambda x: parse_ipc_atlas(x, ipc_level))
    ipc_counts = ipc_parsed.value_counts()
    data_to_plot = ipc_counts.head(int(num_to_display)).sort_values(ascending=True)

    if data_to_plot.empty:
        return "集計結果がありません。"
        
    plt.style.use('seaborn-v0_8-talk')
    fig, ax = plt.subplots(figsize=(12, max(5, 0.4 * len(data_to_plot))))
    bars = ax.barh(data_to_plot.index, data_to_plot.values, color='darkgreen')
    ax.set_title(f'IPCランキング ({level_name}レベル, {int(start_year)}年～{int(end_year)}年)', fontsize=20, pad=20)
    ax.set_xlabel('特許件数', fontsize=14); ax.set_ylabel('IPC', fontsize=14)
    for bar in bars:
        width = bar.get_width()
        ax.text(width + width * 0.01, bar.get_y() + bar.get_height()/2, f'{int(width)}', ha='left', va='center', fontsize=12)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.15)
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig

@st.cache_data
def create_assignee_year_bubble(df_stats, num_to_display, start_year, end_year):
    """(MAP 4) 出願人×年 バブル（デュアル表示）"""
    assignees_exploded = df_stats.explode('applicant_main')
    assignees_exploded['assignee_parsed'] = assignees_exploded['applicant_main'].str.strip()
    top_assignees = assignees_exploded['assignee_parsed'].value_counts().head(int(num_to_display)).index.tolist()
    
    plot_data = assignees_exploded[assignees_exploded['assignee_parsed'].isin(top_assignees)]
    plot_data = plot_data.groupby(['year', 'assignee_parsed']).size().reset_index(name='件数')

    if plot_data.empty:
        return "集計結果が空のため、このマップはスキップします。", None

    assignee_rank_map = {name: i for i, name in enumerate(top_assignees[::-1])}
    plot_data['y_rank'] = plot_data['assignee_parsed'].map(assignee_rank_map)
    cmap = plt.get_cmap('Set2', len(top_assignees))
    
    # 対数スケール
    fig1, ax1 = plt.subplots(figsize=(16, max(8, 0.6 * len(top_assignees))))
    ax1.scatter(x=plot_data['year'], y=plot_data['y_rank'], s=np.log1p(plot_data['件数']) * 80, c=plot_data['y_rank'], cmap=cmap, alpha=0.8)
    for _, row in plot_data.iterrows(): ax1.text(row['year'], row['y_rank'], row['件数'], ha='center', va='center', fontsize=9, color='black')
    ax1.set_yticks(range(len(top_assignees))); ax1.set_yticklabels(top_assignees[::-1])
    ax1.set_title(f'出願年別 出願人動向 (対数スケール) - {int(start_year)}年～{int(end_year)}年', fontsize=20, pad=20)
    ax1.set_xlabel('出願年', fontsize=14); ax1.set_ylabel('出願人', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7); ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    
    # 実数スケール
    fig2, ax2 = plt.subplots(figsize=(16, max(8, 0.6 * len(top_assignees))))
    ax2.scatter(x=plot_data['year'], y=plot_data['y_rank'], s=plot_data['件数'] * 40, c=plot_data['y_rank'], cmap=cmap, alpha=0.8)
    for _, row in plot_data.iterrows(): ax2.text(row['year'], row['y_rank'], row['件数'], ha='center', va='center', fontsize=9, color='black')
    ax2.set_yticks(range(len(top_assignees))); ax2.set_yticklabels(top_assignees[::-1])
    ax2.set_title(f'出願年別 出願人動向 (実数スケール) - {int(start_year)}年～{int(end_year)}年', fontsize=20, pad=20)
    ax2.set_xlabel('出願年', fontsize=14); ax2.set_ylabel('出願人', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7); ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    
    return fig1, fig2

@st.cache_data
def create_ipc_assignee_bubble(df_stats, ipc_level_tuple, num_ipcs, num_assignees, start_year, end_year):
    """(MAP 5) IPC×出願人 バブル（デュアル表示）"""
    ipc_level, level_name = ipc_level_tuple

    df_exploded = df_stats.explode('applicant_main').explode('ipc_normalized')
    df_exploded.dropna(subset=['applicant_main', 'ipc_normalized'], inplace=True)
    
    df_exploded['assignee_parsed'] = df_exploded['applicant_main'].str.strip()
    df_exploded['ipc_parsed'] = df_exploded['ipc_normalized'].apply(lambda x: parse_ipc_atlas(x, ipc_level))
    
    top_assignees = df_exploded['assignee_parsed'].value_counts().head(int(num_assignees)).index.tolist()
    top_ipcs = df_exploded['ipc_parsed'].value_counts().head(int(num_ipcs)).index.tolist()

    df_top = df_exploded[
        df_exploded['assignee_parsed'].isin(top_assignees) & 
        df_exploded['ipc_parsed'].isin(top_ipcs)
    ]
    
    plot_data = df_top.groupby(['assignee_parsed', 'ipc_parsed']).size().reset_index(name='件数')
    if plot_data.empty:
        return "集計結果が空のため、このマップはスキップします。", None

    ipc_rank_map = {ipc: i for i, ipc in enumerate(top_ipcs)}
    assignee_rank_map = {name: i for i, name in enumerate(top_assignees[::-1])}
    plot_data['x_rank'] = plot_data['ipc_parsed'].map(ipc_rank_map)
    plot_data['y_rank'] = plot_data['assignee_parsed'].map(assignee_rank_map)
    cmap = plt.get_cmap('Set2', len(top_assignees))

    # 対数スケール
    fig1, ax1 = plt.subplots(figsize=(max(16, 0.8 * len(top_ipcs)), max(8, 0.5 * len(top_assignees))))
    ax1.scatter(x=plot_data['x_rank'], y=plot_data['y_rank'], s=np.log1p(plot_data['件数']) * 100, c=plot_data['y_rank'], cmap=cmap, alpha=0.8)
    for _, row in plot_data.iterrows(): ax1.text(row['x_rank'], row['y_rank'], row['件数'], ha='center', va='center', fontsize=9, color='black')
    ax1.set_xticks(range(len(top_ipcs))); ax1.set_xticklabels(top_ipcs, rotation=90)
    ax1.set_yticks(range(len(top_assignees))); ax1.set_yticklabels(top_assignees[::-1])
    ax1.set_title(f'IPC × 出願人 ポートフォリオ (対数スケール) - {int(start_year)}年～{int(end_year)}年', fontsize=20, pad=20)
    ax1.set_xlabel(f'IPC ({level_name})', fontsize=14); ax1.set_ylabel('出願人', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 実数スケール
    fig2, ax2 = plt.subplots(figsize=(max(16, 0.8 * len(top_ipcs)), max(8, 0.5 * len(top_assignees))))
    ax2.scatter(x=plot_data['x_rank'], y=plot_data['y_rank'], s=plot_data['件数'] * 40, c=plot_data['y_rank'], cmap=cmap, alpha=0.8)
    for _, row in plot_data.iterrows(): ax2.text(row['x_rank'], row['y_rank'], row['件数'], ha='center', va='center', fontsize=9, color='black')
    ax2.set_xticks(range(len(top_ipcs))); ax2.set_xticklabels(top_ipcs, rotation=90)
    ax2.set_yticks(range(len(top_assignees))); ax2.set_yticklabels(top_assignees[::-1])
    ax2.set_title(f'IPC × 出願人 ポートフォリオ (実数スケール) - {int(start_year)}年～{int(end_year)}年', fontsize=20, pad=20)
    ax2.set_xlabel(f'IPC ({level_name})', fontsize=14); ax2.set_ylabel('出願人', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return fig1, fig2


# ==================================================================
# --- 3. Streamlit UI ---
# ==================================================================
st.set_page_config(
    page_title="APOLLO | ATLAS",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 ATLAS")
st.markdown("出願年、出願人、IPCなどの基本的な統計グラフを作成します。")

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
        delimiters = st.session_state.delimiters
    except Exception as e:
        st.error(f"セッションデータの読み込みに失敗しました: {e}")
        st.stop()
        
required_cols = ['year', 'applicant_main', 'ipc_normalized']
if not all(col in df_main.columns for col in required_cols):
    st.error("エラー: Mission Controlでの前処理（出願年、出願人、IPCの正規化）が完了していないようです。")
    st.info(f"不足カラム: {[col for col in required_cols if col not in df_main.columns]}")
    st.stop()

# ==================================================================
# --- 5. ATLAS アプリケーション ---
# ==================================================================

ATLAS_FIG_KEYS = [
    'atlas_fig_map1', 
    'atlas_fig_map2', 
    'atlas_fig_map3',
    'atlas_fig_map4a', 
    'atlas_fig_map4b',
    'atlas_fig_map5a',
    'atlas_fig_map5b'
]

for key in ATLAS_FIG_KEYS:
    if key not in st.session_state:
        st.session_state[key] = None

def clear_all_atlas_figs():
    for key in ATLAS_FIG_KEYS:
        if key in st.session_state:
            st.session_state[key] = None

def clear_specific_atlas_fig(key):
    if key in st.session_state:
        st.session_state[key] = None

def clear_specific_atlas_figs(keys_list):
    for key in keys_list:
        if key in st.session_state:
            st.session_state[key] = None

# --- A. 共通フィルタ ---
st.subheader("共通フィルタ設定")
st.info("ここで設定した期間が、以下の全てのタブの集計対象となります。")

min_year = int(df_main['year'].min())
max_year = int(df_main['year'].max())

col1, col2 = st.columns(2)
with col1:
    stats_start_year = st.number_input(
        '集計開始年:', 
        min_value=min_year, 
        max_value=max_year, 
        value=min_year, 
        key="atlas_start_year",
        on_change=clear_all_atlas_figs 
    )
with col2:
    stats_end_year = st.number_input(
        '集計終了年:', 
        min_value=min_year, 
        max_value=max_year, 
        value=max_year, 
        key="atlas_end_year",
        on_change=clear_all_atlas_figs 
    )

try:
    df_filtered = df_main[
        (df_main['year'] >= int(stats_start_year)) & 
        (df_main['year'] <= int(stats_end_year))
    ].copy()
    st.success(f"集計対象: {int(stats_start_year)}年～{int(stats_end_year)}年 (全 {len(df_filtered)} 件)")
except Exception as e:
    st.error(f"期間の絞り込みに失敗しました: {e}")
    df_filtered = pd.DataFrame() 

st.markdown("---")


# --- B. 各グラフ用のタブ ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "件数推移", 
    "出願人ランキング", 
    "IPCランキング", 
    "出願人×年 バブル", 
    "IPC×出願人 バブル"
])

# --- 件数推移 ---
with tab1:
    st.subheader("出願件数時系列推移")
    
    if st.button("件数推移グラフを描画", key="atlas_run_map1"):
        if df_filtered.empty:
            st.warning("該当するデータがありません。")
            st.session_state.atlas_fig_map1 = None
        else:
            with st.spinner("グラフを作成中..."):
                fig = create_application_trend_chart(df_filtered, stats_start_year, stats_end_year)
                st.session_state.atlas_fig_map1 = fig
    
    if st.session_state.atlas_fig_map1:
        if isinstance(st.session_state.atlas_fig_map1, str):
            st.warning(st.session_state.atlas_fig_map1)
        else:
            st.pyplot(st.session_state.atlas_fig_map1)

# --- 出願人ランキング ---
with tab2:
    st.subheader("出願人ランキング")
    num_to_display_map2 = st.number_input(
        "表示する出願人数:", 
        min_value=1, 
        value=20, 
        key="atlas_num_apps_map2",
        on_change=clear_specific_atlas_fig, args=('atlas_fig_map2',)
    )

    if st.button("出願人ランキングを描画", key="atlas_run_map2"):
        if df_filtered.empty:
            st.warning("該当するデータがありません。")
            st.session_state.atlas_fig_map2 = None
        else:
            with st.spinner("グラフを作成中..."):
                fig = create_assignee_ranking_map(df_filtered, num_to_display_map2, stats_start_year, stats_end_year)
                st.session_state.atlas_fig_map2 = fig
    
    if st.session_state.atlas_fig_map2:
        if isinstance(st.session_state.atlas_fig_map2, str):
            st.warning(st.session_state.atlas_fig_map2)
        else:
            st.pyplot(st.session_state.atlas_fig_map2)

# --- IPCランキング ---
with tab3:
    st.subheader("IPCランキング")
    
    ipc_level_map3 = st.selectbox(
        "IPC集計レベル:", 
        options=[(1, "サブクラス (A01B)"), (2, "メイングループ (A01B 1/00)"), (3, "サブグループ (A01B 1/02)")], 
        format_func=lambda x: x[1],
        key="atlas_ipc_level_map3",
        on_change=clear_specific_atlas_fig, args=('atlas_fig_map3',)
    )
    num_to_display_map3 = st.number_input(
        "表示するIPC数:", 
        min_value=1, 
        value=20, 
        key="atlas_num_ipcs_map3",
        on_change=clear_specific_atlas_fig, args=('atlas_fig_map3',)
    )

    if st.button("IPCランキングを描画", key="atlas_run_map3"):
        if df_filtered.empty:
            st.warning("該当するデータがありません。")
            st.session_state.atlas_fig_map3 = None
        else:
            with st.spinner("グラフを作成中..."):
                fig = create_ipc_ranking_map(df_filtered, ipc_level_map3, num_to_display_map3, stats_start_year, stats_end_year)
                st.session_state.atlas_fig_map3 = fig

    if st.session_state.atlas_fig_map3:
        if isinstance(st.session_state.atlas_fig_map3, str):
            st.warning(st.session_state.atlas_fig_map3)
        else:
            st.pyplot(st.session_state.atlas_fig_map3)


# --- 出願人×年 バブル ---
with tab4:
    st.subheader("出願人 × 年 バブルチャート")
    num_to_display_map4 = st.number_input(
        "表示する出願人数:", 
        min_value=1, 
        value=10, 
        key="atlas_num_apps_map4",
        on_change=clear_specific_atlas_figs, args=(['atlas_fig_map4a', 'atlas_fig_map4b'],)
    )

    if st.button("出願人×年 バブルを描画", key="atlas_run_map4"):
        if df_filtered.empty:
            st.warning("該当するデータがありません。")
            st.session_state.atlas_fig_map4a = None
            st.session_state.atlas_fig_map4b = None
        else:
            with st.spinner("グラフを作成中..."):
                fig1, fig2 = create_assignee_year_bubble(df_filtered, num_to_display_map4, stats_start_year, stats_end_year)
                st.session_state.atlas_fig_map4a = fig1
                st.session_state.atlas_fig_map4b = fig2

    if st.session_state.atlas_fig_map4a:
        if isinstance(st.session_state.atlas_fig_map4a, str):
            st.warning(st.session_state.atlas_fig_map4a)
        else:
            st.subheader("対数スケール")
            st.pyplot(st.session_state.atlas_fig_map4a)
            st.subheader("実数スケール")
            st.pyplot(st.session_state.atlas_fig_map4b)


# --- IPC×出願人 バブル ---
with tab5:
    st.subheader("IPC × 出願人 バブルチャート")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        ipc_level_map5 = st.selectbox(
            "IPC集計レベル (Y軸):", 
            options=[(1, "サブクラス (A01B)"), (2, "メイングループ (A01B 1/00)"), (3, "サブグループ (A01B 1/02)")], 
            format_func=lambda x: x[1],
            key="atlas_ipc_level_map5",
            on_change=clear_specific_atlas_figs, args=(['atlas_fig_map5a', 'atlas_fig_map5b'],)
        )
    with col2:
        num_ipcs_map5 = st.number_input(
            "IPC表示件数 (Y軸):", 
            min_value=1, 
            value=15, 
            key="atlas_num_ipcs_map5",
            on_change=clear_specific_atlas_figs, args=(['atlas_fig_map5a', 'atlas_fig_map5b'],)
        )
    with col3:
        num_apps_map5 = st.number_input(
            "出願人表示件数 (X軸):", 
            min_value=1, 
            value=15, 
            key="atlas_num_apps_map5",
            on_change=clear_specific_atlas_figs, args=(['atlas_fig_map5a', 'atlas_fig_map5b'],)
        )

    if st.button("IPC×出願人 バブルを描画", key="atlas_run_map5"):
        if df_filtered.empty:
            st.warning("該当するデータがありません。")
            st.session_state.atlas_fig_map5a = None
            st.session_state.atlas_fig_map5b = None
        else:
            with st.spinner("グラフを作成中..."):
                fig1, fig2 = create_ipc_assignee_bubble(
                    df_filtered, 
                    ipc_level_map5, 
                    num_ipcs_map5, 
                    num_apps_map5, 
                    stats_start_year, 
                    stats_end_year
                )
                st.session_state.atlas_fig_map5a = fig1
                st.session_state.atlas_fig_map5b = fig2

    if st.session_state.atlas_fig_map5a:
        if isinstance(st.session_state.atlas_fig_map5a, str):
            st.warning(st.session_state.atlas_fig_map5a)
        else:
            st.subheader("対数スケール")
            st.pyplot(st.session_state.atlas_fig_map5a)
            st.subheader("実数スケール")
            st.pyplot(st.session_state.atlas_fig_map5b)

# --- 共通サイドバーフッター ---
st.sidebar.markdown("---") 
st.sidebar.caption("ナビゲーション:")
st.sidebar.caption("1. Mission Control でデータをアップロードし、前処理を実行します。")
st.sidebar.caption("2. 左のリストから分析モジュールを選択します。")
st.sidebar.markdown("---")
st.sidebar.caption("© 2025 しばやま")