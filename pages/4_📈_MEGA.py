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
import platform
import os
import string
from collections import Counter
from itertools import combinations

# 機械学習・自然言語処理
from umap import UMAP
import hdbscan
from wordcloud import WordCloud
from janome.tokenizer import Tokenizer
import networkx as nx
from scipy.spatial import ConvexHull

# 描画用
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import japanize_matplotlib
import utils

# 警告を非表示
warnings.filterwarnings('ignore')

# ==================================================================
# --- 1. フォント設定 ---
# ==================================================================


FONT_PATH = utils.get_japanese_font_path()
if FONT_PATH:
    try:
        prop = fm.FontProperties(fname=FONT_PATH)
        plt.rcParams['font.family'] = prop.get_name()
    except:
        pass

# ==================================================================
# --- 2. ページ設定 ---
# ==================================================================
st.set_page_config(
    page_title="APOLLO | MEGA",
    page_icon="📈",
    layout="wide"
)

# ==================================================================
# --- 3. テキスト処理・ストップワード ---
# ==================================================================
@st.cache_resource
def load_tokenizer_mega():
    return Tokenizer()

t = load_tokenizer_mega()

# ストップワード定義
if "stopwords" in st.session_state and st.session_state["stopwords"]:
    stopwords = st.session_state["stopwords"]
else:
    stopwords = utils.get_stopwords()

_ngram_rows = [
    ("参照符号付き要素", r"[一-龥ぁ-んァ-ンA-Za-z0-9／\-＋・]+?(?:部|層|面|体|板|孔|溝|片|部材|要素|機構|装置|手段|電極|端子|領域|基板|回路|材料|工程)\s*[（(]\s*[0-9０-９A-Za-z]+[A-Za-z]?\s*[）)]", "regex", 1),
    ("参照符号付き要素", r"(?:上記|前記)?[一-龥ぁ-んァ-ンA-Za-z0-9／\-＋・]+?(?:部|層|面|体|板|孔|溝|片|部材|要素|機構|装置|手段|電極|端子|領域|基板|回路|材料|工程)\s*[0-9０-９A-Za-z]+[A-Za-z]?", "regex", 1),
    ("参照符号付き要素", r"[A-Z]+[0-9]+", "regex", 1),
    ("見出し・章句","一実施形態において","literal",1), ("見出し・章句","他の実施形態において","literal",1), ("見出し・章句","別の実施形態において","literal",1),
    ("見出し・章句","本明細書において","literal",1), ("見出し・章句","本明細書では","literal",1), ("見出し・章句","本発明の一側面","literal",1),
    ("見出し・章句","一実施例において","literal",1), ("見出し・章句","他の実施例において","literal",1), ("見出し・章句","好ましい態様として","literal",2),
    ("見出し・章句","好適には","literal",2), ("見出し・章句","用語の定義","literal",2), ("見出し・章句","図示しない","literal",2),
    ("図表参照", r"図[ 　]*[０-９0-9]+に示す", "regex", 1), ("図表参照", r"表[ 　]*[０-９0-9]+に示す", "regex", 1),
    ("図表参照", r"式[ 　]*[０-９0-9]+に示す", "regex", 1), ("図表参照", r"請求項[ 　]*[０-９0-9]+", "regex", 1),
    ("図表参照", r"(?:【|\[)\s*[０-９0-9]{4,5}\s*(?:】|\])", "regex", 1), ("図表参照", r"[（(][０-９0-9]+[）)]", "regex", 2),
    ("図表参照", r"第\s*[０-９0-9]+の?実施形態", "regex", 2), ("図表参照", r"段落\s*[０-９0-9]+", "regex", 2),
    ("図表参照", r"図[ 　]*[０-９0-9]+[A-Za-z]?", "regex", 2), ("定義導入", r"以下、[^、。]+を[^、。]+と称する", "regex", 1),
    ("定義導入", r"以下、[^、。]+を[^、。]+という", "regex", 1), ("機能句","してもよい","literal",1), ("機能句","であってもよい","literal",1),
    ("機能句","することができる","literal",1), ("機能句","行うことができる","literal",1), ("機能句","に限定されない","literal",1),
    ("機能句","に限られない","literal",1), ("機能句","一例として","literal",2), ("機能句","例示的には","literal",2),
    ("参照句","前述のとおり","literal",2), ("参照句","前述の通り","literal",2), ("参照句","後述するように","literal",2),
    ("参照句","後述のとおり","literal",2), ("範囲表現", r"少なくとも(?:一|１)つ", "regex", 2), ("範囲表現", "少なくとも一部", "literal", 2),
    ("範囲表現", r"複数の(?:実施形態|構成|要素)", "regex", 3), ("課題句", r"(?:上記|前記)の?課題", "regex", 1),
    ("接続・論理","一方で","literal",3), ("接続・論理","他方で","literal",3), ("接続・論理","すなわち","literal",3),
    ("接続・論理","したがって","literal",3), ("接続・論理","しかしながら","literal",3), ("接続・論理","例えば","literal",3),
    ("接続・論理","具体的には","literal",3), ("補助句","以下に説明する","literal",3), ("補助句","前記のとおり","literal",3),
    ("補助句","これにより","literal",3), ("補助句","このように","literal",3)
]

_ngram_compiled = sorted(_ngram_rows, key=lambda x: (x[3], -len(x[1]) if x[2]=="literal" else -50))
_ngram_compiled = [(cat, re.compile(pat) if ptype == "regex" else pat, ptype, pri) for cat, pat, ptype, pri in _ngram_compiled]

def normalize_text(text):
    if not isinstance(text, str): text = "" if pd.isna(text) else str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("µ", "μ")
    text = re.sub(r"\s+", " ", text)
    return text

def apply_ngram_filters(text):
    for cat, pat, ptype, pri in _ngram_compiled:
        if ptype == "literal":
            if pat in text: text = text.replace(pat, "")
        else:
            text = pat.sub("", text)
    return text

@st.cache_data
def extract_compound_nouns(text):
    text = normalize_text(text)
    text = apply_ngram_filters(text)
    text = re.sub(r'【.*?】', '', text)
    text = re.sub(r'[!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', ' ', text)

    tokens = t.tokenize(text)
    words, compound_word = [], ''
    for token in tokens:
        pos = token.part_of_speech.split(',')[0]
        if pos == '名詞':
            compound_word += token.surface
        else:
            if (len(compound_word) > 1 and
                compound_word not in stopwords and
                not re.fullmatch(r'[\d０-９]+', compound_word) and
                not re.fullmatch(r'(図|表|式|第)[\d０-９]+.*', compound_word) and
                not re.match(r'^(上記|前記|本開示|当該|該)', compound_word) and
                not re.search(r'[0-9０-９]+[)）]?$', compound_word) and
                not re.match(r'[0-9０-９]+[a-zA-Zａ-ｚＡ-Ｚ]', compound_word)):
                words.append(compound_word)
            compound_word = ''
            
    if (len(compound_word) > 1 and
        compound_word not in stopwords and
        not re.fullmatch(r'[\d０-９]+', compound_word) and
        not re.fullmatch(r'(図|表|式|第)[\d０-９]+.*', compound_word) and
        not re.match(r'^(上記|前記|本開示|当該|該)', compound_word) and
        not re.search(r'[0-9０-９]+[)）]?$', compound_word) and
        not re.match(r'[0-9０-９]+[a-zA-Zａ-ｚＡ-Ｚ]', compound_word)):
        words.append(compound_word)
    return words

def generate_wordcloud_and_list(words, title, top_n=20, font_path=None):
    if not words:
        st.subheader(title)
        st.warning("キーワードが見つからなかったため、表示をスキップしました。")
        return None

    word_freq = Counter(words)

    try:
        wc = WordCloud(
            width=800, height=400, background_color='white',
            font_path=font_path, collocations=False,
            max_words=100
        ).generate_from_frequencies(word_freq)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation='bilinear')
        ax.set_title(title, fontsize=20)
        ax.axis('off')
        st.pyplot(fig)
        
        st.markdown(f"**上位キーワード (Top {top_n})**")
        list_data = { "キーワード": [], "出現頻度": [] }
        for word, freq in word_freq.most_common(top_n):
            list_data["キーワード"].append(word)
            list_data["出現頻度"].append(freq)
        st.dataframe(pd.DataFrame(list_data), height=200)
        
    except Exception as e:
        st.error(f"ワードクラウドの描画に失敗しました: {e}")
        if font_path is None:
            st.warning("日本語フォントが見つかりませんでした。")

# ==================================================================
# --- 4. 共通デザイン設定 (CSS) ---
# ==================================================================


# ==================================================================
# --- 5. デザインテーマ管理 ---
# ==================================================================





# ==================================================================
# --- 6. ヘルパー関数 (MEGA分析ロジック) ---
# ==================================================================

@st.cache_data
def _get_top_words_filtered(dense_vector, feature_names, top_n=5):
    """TF-IDFベクトルから上位語を抽出（ストップワード除外）"""
    indices = np.argsort(dense_vector)[::-1]
    top_words = []
    for i in indices:
        word = feature_names[i]
        if word not in stopwords and not re.fullmatch(r'[\d０-９]+', word) and len(word) > 1:
            top_words.append(word)
        if len(top_words) >= top_n:
            break
    return ", ".join(top_words)

@st.cache_data
def _calculate_cagr(row, cagr_end_year_val):
    valid_years = row[row > 0].index
    if not any(valid_years): return np.nan
    valid_years_in_range = valid_years[valid_years <= cagr_end_year_val]
    if not any(valid_years_in_range): return np.nan
    start_year = min(valid_years_in_range)
    end_year = max(valid_years_in_range)
    if start_year >= end_year: return np.nan
    start_value = row[start_year]
    end_value = row[end_year]
    num_years = end_year - start_year
    try: return ((end_value / start_value) ** (1 / num_years)) - 1
    except: return np.nan

@st.cache_data
def _calculate_metrics(pivot_df, cagr_end_year, y_axis_years, current_year, past_offset=0):
    target_cagr_end = cagr_end_year - past_offset
    target_current_year = current_year - past_offset
    y_start = target_current_year - y_axis_years + 1
    y_cols = [col for col in pivot_df.columns if col >= y_start and col <= target_current_year]
    y_axis = pivot_df[y_cols].sum(axis=1) if y_cols else pd.Series(0, index=pivot_df.index)
    
    if past_offset == 0: bubble_size = pivot_df.sum(axis=1)
    else:
        bubble_cols = [col for col in pivot_df.columns if col <= target_current_year]
        bubble_size = pivot_df[bubble_cols].sum(axis=1) if bubble_cols else pd.Series(0, index=pivot_df.index)

    cagr_cols = [col for col in pivot_df.columns if col <= target_cagr_end]
    if not cagr_cols: x_axis = pd.Series(np.nan, index=pivot_df.index)
    else: x_axis = pivot_df[cagr_cols].apply(_calculate_cagr, axis=1, cagr_end_year_val=target_cagr_end)
    return x_axis, y_axis, bubble_size

def _calculate_single_point_metrics(row_data, year_point, cagr_base_year, y_axis_years):
    y_start = year_point - y_axis_years + 1
    y_val = 0
    for y in range(y_start, year_point + 1):
        if y in row_data.index: y_val += row_data[y]
    size_val = 0
    for y in row_data.index:
        if y <= year_point: size_val += row_data[y]
    x_val = _calculate_cagr(row_data, year_point)
    return x_val, y_val, size_val

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
    pivot_df.columns = pivot_df.columns.astype(int)
    return pivot_df

def _hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip('#')
    return f'rgba({int(hex_color[0:2], 16)},{int(hex_color[2:4], 16)},{int(hex_color[4:6], 16)},{alpha})'

def _get_hover_template_mode2(is_past=False):
    return f"""<b>%{{hovertext}}</b><br><br>戦略グループ: %{{customdata[0]}}<br>X (勢い): %{{x:.1%}}<br>Y (活動量): %{{y:,.0f}}<br>Bubble (総件数): %{{marker.size}}<br><extra></extra>"""

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(encoding='utf-8-sig').encode('utf-8-sig')

# ==================================================================
# --- 7. Streamlit UI構成 ---
# ==================================================================

utils.render_sidebar()

st.title("📈 MEGA")
st.markdown("技術動態（マクロ）と技術ポートフォリオ（ミクロ）を分析します。")

col_theme, col_dummy = st.columns([1, 3])
with col_theme:
    selected_theme = st.selectbox("表示テーマ:", ["APOLLO Standard", "Modern Presentation"], key="mega_theme_selector")
theme_config = utils.get_theme_config(selected_theme)
st.markdown(f"<style>{theme_config['css']}</style>", unsafe_allow_html=True)

# ==================================================================
# --- 8. データロード & 初期化 ---
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
# --- 9. MEGA アプリケーション ---
# ==================================================================

tab_b, tab_c, tab_d = st.tabs([
    "Landscape Analysis (PULSE)",
    "Technology Probe (TELESCOPE)",
    "Data Export"
])

# --- A. 動態分析 (PULSE) ---
with tab_b:
    st.subheader("分析パラメータ")
    col1, col2 = st.columns(2)
    with col1:
        axis_options = [('出願人', 'applicant_main'), ('IPC (メイングループ)', 'ipc_main_group')]
        if st.session_state.col_map.get('fterm'): axis_options.append(('Fターム (テーマコード)', 'fterm_main'))
        analysis_axis = st.selectbox("分析軸:", options=axis_options, format_func=lambda x: x[0], key="mega_analysis_axis")
        yaxis_slider = st.slider("Y軸 (現在) の集計年数:", min_value=1, max_value=10, value=5, key="mega_yaxis")
        cagr_end_year = st.number_input("X軸 (過去の勢い) 計算の最終年:", value=datetime.datetime.now().year - 1, key="mega_cagr_year")
    with col2:
        trajectory_past = st.number_input("軌跡 (過去への遡り年数):", min_value=1, value=5, key="mega_trajectory")
        min_patents = st.number_input("最小フィルタ件数 (描画対象):", min_value=1, value=10, key="mega_min_patents")

    st.subheader("ハイライトと軌跡")
    highlight_options = st.session_state.get("mega_highlight_options", [])
    highlight_targets = st.multiselect("注目対象 (軌跡を表示):", options=highlight_options, format_func=lambda x: x[0])

    st.subheader("動態分析マップ実行")
    if st.button("動態分析マップを描画", type="primary", key="mega_run_map"):
        with st.spinner("動態分析マップを計算中..."):
            try:
                axis_col, axis_label = analysis_axis[1], analysis_axis[0]
                y_axis_years, past_offset, current_year = int(yaxis_slider), int(trajectory_past), datetime.datetime.now().year
                min_patents_threshold = int(min_patents)

                pivot_df = _prepare_momentum_data(df_main, axis_col)
                st.session_state.mega_pivot_df = pivot_df 
                if pivot_df.empty:
                    st.error(f"エラー: 分析軸 ({axis_label}) の有効なデータがありません。")
                    st.stop()

                x_present, y_present, bubble_present = _calculate_metrics(pivot_df, cagr_end_year, y_axis_years, current_year, past_offset=0)
                
                # 件数が多い順にソート
                options_with_counts = [(f"{name} ({int(count)}件)", name) for name, count in bubble_present.sort_values(ascending=False).items()]
                st.session_state.mega_highlight_options = options_with_counts

                start_years = pivot_df[pivot_df > 0].apply(lambda row: row.first_valid_index(), axis=1)
                cagr_start_year_min = start_years.min() if not start_years.empty else cagr_end_year
                st.session_state.cagr_start_year_min = cagr_start_year_min
                st.session_state.cagr_end_year_val = cagr_end_year

                df_result = pd.DataFrame({'X_Present': x_present, 'Y_Present': y_present, 'Bubble_Present': bubble_present}).astype('float')
                df_result.index.name = axis_label
                df_result.replace([np.inf, -np.inf], np.nan, inplace=True)
                df_result.dropna(subset=['X_Present', 'Y_Present'], inplace=True)
                df_result = df_result[df_result['Bubble_Present'] >= min_patents_threshold].copy()

                if df_result.empty:
                    st.error("エラー: フィルタリング後の結果が0件です。")
                    st.stop()

                x_threshold, y_threshold = df_result['X_Present'].mean(), df_result['Y_Present'].mean()
                st.session_state.mega_x_threshold = x_threshold
                st.session_state.mega_y_threshold = y_threshold

                def assign_relative_label(row):
                    if row['Y_Present'] <= 0: return '衰退・ニッチ (Declining/Niche)'
                    if (row['X_Present'] > x_threshold) and (row['Y_Present'] > y_threshold): return 'リーダー (Leaders)'
                    elif (row['X_Present'] > x_threshold) and (row['Y_Present'] <= y_threshold): return '新興・高ポテンシャル (Emerging)'
                    elif (row['X_Present'] <= x_threshold) and (row['Y_Present'] > y_threshold): return '成熟・既存勢力 (Established)'
                    else: return '衰退・ニッチ (Declining/Niche)'

                df_result['Group_Auto'] = df_result.apply(assign_relative_label, axis=1)
                st.session_state.df_momentum_result = df_result.copy()
                st.session_state.mega_axis_label = axis_label
                st.session_state.mega_past_offset = past_offset
                st.session_state.mega_y_axis_years = y_axis_years

                df_filtered = df_result[df_result['Group_Auto'] != 'N/A']
                
                # 件数が多い順にソート
                drilldown_options = [('(分析対象を選択)', '(分析対象を選択)')] + [
                    (f"{name} ({int(row['Bubble_Present'])}件)", name) 
                    for name, row in df_filtered.sort_values('Bubble_Present', ascending=False).iterrows()
                ]
                st.session_state.mega_drilldown_options = drilldown_options
                st.success("完了")
                st.rerun()
            except Exception as e:
                st.error(f"エラー: {e}")

    # --- ラベル編集 & 描画 (PULSE) ---
    st.subheader("ラベル編集")
    base_color_map = {'リーダー (Leaders)': '#28a745', '新興・高ポテンシャル (Emerging)': '#ffc107', '成熟・既存勢力 (Established)': '#007bff', '衰退・ニッチ (Declining/Niche)': '#6c757d', 'N/A': '#ced4da'}

    if "df_momentum_result" in st.session_state:
        df_to_plot = st.session_state.df_momentum_result.copy()
        st.session_state.mega_group_map_custom = {}
        c1, c2 = st.columns(2)
        with c1:
            st.session_state.mega_group_map_custom['リーダー (Leaders)'] = st.text_input("リーダー", "リーダー (Leaders)", key="label_leader")
            st.session_state.mega_group_map_custom['新興・高ポテンシャル (Emerging)'] = st.text_input("新興", "新興・高ポテンシャル (Emerging)", key="label_emerging")
        with c2:
            st.session_state.mega_group_map_custom['成熟・既存勢力 (Established)'] = st.text_input("成熟", "成熟・既存勢力 (Established)", key="label_established")
            st.session_state.mega_group_map_custom['衰退・ニッチ (Declining/Niche)'] = st.text_input("衰退", "衰退・ニッチ (Declining/Niche)", key="label_declining")

        df_to_plot['Group_Custom'] = df_to_plot['Group_Auto'].map(st.session_state.mega_group_map_custom).fillna('N/A')
        current_color_map = {st.session_state.mega_group_map_custom.get(k, k): v for k, v in base_color_map.items()}
        
        axis_label = st.session_state.mega_axis_label
        cagr_start = int(st.session_state.get('cagr_start_year_min', 2000))
        cagr_end = int(st.session_state.get('cagr_end_year_val', datetime.datetime.now().year))
        xaxis_title_label = f"過去の勢い (CAGR, {cagr_start}-{cagr_end}年内の活動期間)"
        
        fig = go.Figure()
        
        # 軌跡描画
        if highlight_targets:
            highlight_values = [t[1] for t in highlight_targets]
            palette = theme_config["color_sequence"]
            pivot_df = st.session_state.mega_pivot_df
            max_bubble = df_to_plot['Bubble_Present'].max()
            
            for i, target in enumerate(highlight_values):
                if target not in pivot_df.index: continue
                row = pivot_df.loc[target]
                base_color = palette[i % len(palette)]
                traj_x, traj_y, traj_s, traj_t, traj_c, traj_yr = [], [], [], [], [], []
                
                yr_range = list(range(datetime.datetime.now().year - st.session_state.mega_past_offset, datetime.datetime.now().year + 1))
                for idx, y in enumerate(yr_range):
                    xv, yv, sv = _calculate_single_point_metrics(row, y, y, st.session_state.mega_y_axis_years)
                    if pd.notna(xv) and pd.notna(yv):
                        traj_x.append(xv); traj_y.append(yv); traj_yr.append(y)
                        scaled_s = (sv / max_bubble) * 60 if max_bubble > 0 else 10
                        traj_s.append(max(5, scaled_s))
                        traj_t.append(f"'{str(y)[-2:]}")
                        alpha = 0.2 + 0.8 * (idx / max(1, len(yr_range)-1))
                        traj_c.append(_hex_to_rgba(base_color, alpha))
                
                if traj_x:
                    fig.add_trace(go.Scatter(x=traj_x, y=traj_y, mode='lines', line=dict(color=base_color, width=1), opacity=0.5, showlegend=False, hoverinfo='skip'))
                    fig.add_trace(go.Scatter(x=traj_x, y=traj_y, mode='markers+text', name=target, marker=dict(size=traj_s, color=traj_c, line=dict(width=1, color='white')), text=traj_t, textposition="top center", textfont=dict(size=10, color=base_color)))
        else:
            # 通常モード
            df_filt = df_to_plot[df_to_plot['Group_Custom'] != 'N/A'].copy()
            if not df_filt.empty:
                df_filt['Y_Present_Plot'] = df_filt['Y_Present'].replace(0, 0.1)
                
                fig = px.scatter(
                    df_filt.reset_index(), 
                    x='X_Present', 
                    y='Y_Present_Plot',
                    size='Bubble_Present', 
                    size_max=60, 
                    color='Group_Custom', 
                    color_discrete_map=current_color_map, 
                    hover_name=st.session_state.mega_axis_label, 
                    log_y=True
                )
                fig.update_traces(hovertemplate=_get_hover_template_mode2())

        fig.add_vline(x=st.session_state.mega_x_threshold, line_width=1, line_dash="dash", line_color="gray")
        fig.add_hline(y=st.session_state.mega_y_threshold, line_width=1, line_dash="dash", line_color="gray")
        
        utils.update_fig_layout(fig, "MEGA 動態分析マップ", height=800, theme_config=theme_config, show_axes=True, show_legend=False)
        fig.update_layout(
            xaxis_title=f"← 勢い減速 | {xaxis_title_label} | 勢い加速 → (十字線: {st.session_state.mega_x_threshold:.1%})",
            yaxis_title="← 活動鈍化 | 現在の活動量 | 活動活発 →",
            xaxis_tickformat='.0%', 
            yaxis_type="log",
            xaxis=dict(showgrid=True, zeroline=True, showline=True),
            yaxis=dict(showgrid=True, zeroline=False, showline=True)
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'editable': False})
        st.session_state.df_momentum_export = df_to_plot.copy()


# --- C. ドリルダウン分析 (TELESCOPE) ---
with tab_c:
    st.subheader("分析対象の選択")
    drilldown_options = st.session_state.get("mega_drilldown_options", [('(分析対象を選択)', '(分析対象を選択)')])
    drilldown_target = st.selectbox("ドリルダウン対象:", options=drilldown_options, format_func=lambda x: x[0], key="drill_target")[1]

    # --- クラスタリング設定 ---
    st.subheader("クラスタリング設定 (ドリルダウン用)")
    
    col1, col2, col3 = st.columns(3)
    with col1: drill_min_cluster_size = st.number_input('最小クラスタサイズ:', min_value=2, value=5, key="drill_min_cluster_size")
    with col2: drill_min_samples = st.number_input('最小サンプル数:', min_value=1, value=5, key="drill_min_samples")
    with col3: drill_label_top_n = st.number_input('ラベル単語数:', min_value=1, value=3, key="drill_label_top_n")

    if st.button("選択対象の技術マップを描画", type="primary", key="drill_run_map"):
        if drilldown_target == '(分析対象を選択)': st.error("選択してください。")
        else:
            with st.spinner("計算中..."):
                try:
                    axis_col = "applicant_main" if st.session_state.mega_axis_label == "出願人" else "ipc_main_group" if "IPC" in st.session_state.mega_axis_label else "fterm_main"
                    mask = df_main[axis_col].apply(lambda l: drilldown_target in l)
                    df_filtered = df_main[mask].copy()
                    
                    if df_filtered.empty: st.error("データなし"); st.stop()
                    
                    emb = sbert_embeddings[df_main[mask].index]
                    tfidf = tfidf_matrix[df_main[mask].index]
                    original_indices = df_main[mask].index.tolist()

                    n_neighbors = min(10, len(original_indices) - 1)
                    if n_neighbors < 2: n_neighbors = 2
                    
                    umap_res = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42).fit_transform(emb)
                    df_plot = pd.DataFrame(umap_res, columns=['x', 'y'])
                    
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=int(drill_min_cluster_size), 
                        min_samples=int(drill_min_samples),
                        metric='euclidean',
                        cluster_selection_method='eom'
                    )
                    cluster_labels = clusterer.fit_predict(df_plot[['x', 'y']].values)
                    
                    df_plot['cluster_id'] = cluster_labels
                    df_plot['year'] = df_filtered['year'].values
                    df_plot[col_map['title']] = df_filtered[col_map['title']].values
                    if col_map['abstract']: df_plot[col_map['abstract']] = df_filtered[col_map['abstract']].values
                    
                    label_map = {}
                    for cid in sorted(df_plot['cluster_id'].unique()):
                        if cid == -1: label_map[cid] = "ノイズ"
                        else:
                            vecs = tfidf[(df_plot['cluster_id'] == cid).values]
                            mean_vector = np.asarray(vecs.mean(axis=0)).flatten()
                            top_words = _get_top_words_filtered(mean_vector, feature_names, top_n=int(drill_label_top_n))
                            label_map[cid] = f"[{cid}] {top_words}"
                    
                    df_plot['label'] = df_plot['cluster_id'].map(label_map)
                    st.session_state.df_drilldown = df_plot
                    st.session_state.sbert_sub_cluster_map_auto = label_map
                    st.session_state.drilldown_target_name = drilldown_target
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.exception(traceback.format_exc())

    # --- 描画 ---
    if "df_drilldown" in st.session_state:
        df_d = st.session_state.df_drilldown.copy()
        
        map_mode = st.radio("表示モード:", ["散布図 (Scatter)", "密度マップ (Density)", "クラスタ領域 (Clusters)"], horizontal=True, key="mega_map_mode")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            mesh_size = st.number_input("メッシュサイズ (Grid)", 40, 200, 40, step=5, key="mega_mesh")
            use_abs = st.checkbox("密度スケール固定", False, key="mega_abs") if map_mode == "密度マップ (Density)" else False
        with c2:
            no_noise = st.checkbox("ノイズを除く", False, key="mega_noise")
        with c3:
            show_label = st.checkbox("ラベルを表示", True, key="mega_label")
            
        c_t1, c_t2 = st.columns(2)
        with c_t1: interval = st.selectbox("期間の粒度:", [1, 2, 3, 5], index=2, key="mega_interval")
        
        min_y, max_y = df_d['year'].min(), df_d['year'].max()
        bins = pd.date_range(start=f"{int(min_y)}-01-01", end=f"{int(max_y)}-12-31", freq=f'{interval}YE')
        labels = [f"{bins[i].year}-{bins[i+1].year}" for i in range(len(bins)-1)]
        df_d['year_bin'] = pd.cut(df_d['year'], bins=[b.year for b in bins], labels=labels, include_lowest=True)
        
        bin_opts = ["(全期間)"] + sorted([l for l in labels if l in df_d['year_bin'].unique()])
        with c_t2: date_filter = st.selectbox("表示期間:", bin_opts, key="mega_date")

        if no_noise: df_d = df_d[df_d['cluster_id'] != -1]
        
        if date_filter == "(全期間)":
            df_in, df_out = df_d, pd.DataFrame()
            title_s = ""
        else:
            df_in = df_d[df_d['year_bin'] == date_filter]
            df_out = df_d[df_d['year_bin'] != date_filter]
            title_s = f" ({date_filter})"

        fig = go.Figure()
        
        # 密度マップ
        if map_mode == "密度マップ (Density)" and not df_in.empty:
            colors = [[0, "rgba(255,255,255,0)"], [0.1, "rgba(225,245,254,0.3)"], [1, "rgba(2,119,189,0.9)"]]
            fig.add_trace(go.Histogram2dContour(
                x=df_in['x'], y=df_in['y'], colorscale=colors, nbinsx=mesh_size, nbinsy=mesh_size,
                contours=dict(coloring='fill', showlines=True), 
                line=dict(width=0.5, color='rgba(0,0,0,0.2)'),
                showscale=False,
                hoverinfo='skip'
            ))

        # クラスタ領域
        if map_mode == "クラスタ領域 (Clusters)" and not df_in.empty:
            colors = theme_config["color_sequence"]
            u_cls = sorted(df_in['cluster_id'].unique())
            for i, cid in enumerate(u_cls):
                if cid == -1: continue
                pts = df_in[df_in['cluster_id'] == cid][['x', 'y']].values
                if len(pts) >= 3:
                    try:
                        hull = ConvexHull(pts)
                        h_pts = pts[hull.vertices]
                        h_pts = np.append(h_pts, [h_pts[0]], axis=0)
                        col = colors[i % len(colors)]
                        fig.add_trace(go.Scatter(x=h_pts[:,0], y=h_pts[:,1], mode='lines', fill='toself', fillcolor=col, opacity=0.1, line=dict(color=col, width=2), showlegend=False, hoverinfo='skip'))
                    except: pass

        # Ghost
        if not df_out.empty:
            fig.add_trace(go.Scattergl(x=df_out['x'], y=df_out['y'], mode='markers', marker=dict(color='#cccccc', size=3, opacity=0.5), name='期間外', hoverinfo='skip'))

        # Focus Scatter
        m_line = dict(width=1, color='white') if map_mode == "密度マップ (Density)" else dict(width=0)
        colorscale = theme_config["color_sequence"] if isinstance(theme_config["color_sequence"], str) else 'turbo'
        
        fig.add_trace(go.Scattergl(
            x=df_in['x'], y=df_in['y'], mode='markers', 
            marker=dict(color=df_in['cluster_id'], colorscale=colorscale, size=5, line=m_line),
            hovertext=df_in['label'] + "<br>" + df_in[col_map['title']], name='期間内'
        ))

        # Labels
        if show_label:
            u_cls = sorted(df_in['cluster_id'].unique())
            colors = theme_config["color_sequence"]
            all_cls = sorted(df_d['cluster_id'].unique())
            
            for cid in u_cls:
                if cid == -1: continue
                grp = df_in[df_in['cluster_id'] == cid]
                if grp.empty: continue
                
                mx, my = grp['x'].mean(), grp['y'].mean()
                label_txt = st.session_state.sbert_sub_cluster_map_auto.get(cid, str(cid))
                
                try: b_col = colors[all_cls.index(cid) % len(colors)]
                except: b_col = "#333"
                
                fig.add_annotation(x=mx, y=my, text=label_txt, showarrow=False, font=dict(size=10, color='black'), bgcolor='rgba(255,255,255,0.8)', bordercolor=b_col, borderwidth=2, borderpad=4)

        utils.update_fig_layout(fig, f"技術ポートフォリオ: {st.session_state.drilldown_target_name}{title_s}", height=1000, width=800, theme_config=theme_config, show_axes=False)
        st.plotly_chart(fig, use_container_width=True, config={
            'editable': True,
            'edits': {
                'annotationPosition': True,
                'annotationText': False,
                'axisTitleText': False,
                'legendPosition': False,
                'legendText': False,
                'shapePosition': False,
                'titleText': False
            }
        })
        st.session_state.df_drilldown_export = df_d

        st.subheader("クラスタ・ラベル編集")
        st.markdown("AIを活用してクラスタのラベルを自動提案できます。")
        # MEGA用のキーprefix: mega_drill_labels_map
        if "mega_drill_labels_map" not in st.session_state:
             st.session_state.mega_drill_labels_map = st.session_state.sbert_sub_cluster_map_auto.copy()
        
        utils.render_ai_label_assistant(df_d, 'cluster_id', "mega_drill_labels_map", col_map, tfidf_matrix, feature_names, widget_key_prefix="mega_drill_label")

        # 手動編集UI
        st.markdown("**手動編集**")
        if "mega_drill_labels_map_original" not in st.session_state:
             st.session_state.mega_drill_labels_map_original = st.session_state.mega_drill_labels_map.copy()
        
        widget_dict = utils.create_label_editor_ui(st.session_state.mega_drill_labels_map_original, st.session_state.mega_drill_labels_map, "mega_drill_label")
        
        if st.button("ラベルを更新", key="mega_update_labels"):
             for cid, val in widget_dict.items():
                 st.session_state.mega_drill_labels_map[cid] = val
             st.session_state.sbert_sub_cluster_map_auto = st.session_state.mega_drill_labels_map
             st.session_state.df_drilldown['label'] = st.session_state.df_drilldown['cluster_id'].map(st.session_state.mega_drill_labels_map)
             st.rerun()

        st.markdown("---")
        st.subheader("クラスタ・テキスト分析 (Text Mining)")
        
        col_tm1, col_tm2 = st.columns(2)
        with col_tm1:
            cooc_top_n = st.slider("共起: 上位単語数", 30, 100, 50, key="mega_cooc_top_n")
            cooc_threshold = st.slider("共起: Jaccard係数 閾値", 0.01, 0.3, 0.05, 0.01, key="mega_cooc_threshold")
        
        if st.button("テキスト分析を実行", key="mega_run_text_mining"):
            with st.spinner("分析中..."):
                all_text = ""
                for _, row in df_in.iterrows():
                    if col_map['title'] and pd.notna(row[col_map['title']]): all_text += str(row[col_map['title']]) + " "
                    if col_map.get('abstract') and col_map['abstract'] in row and pd.notna(row[col_map['abstract']]): 
                        all_text += str(row[col_map['abstract']]) + " "
                
                words = extract_compound_nouns(all_text)
                
                if not words: st.warning("有効なキーワードなし")
                else:
                    st.markdown("##### 1. ワードクラウド")
                    generate_wordcloud_and_list(words, f"対象: {st.session_state.drilldown_target_name}{title_s}", 30, FONT_PATH)
                    
                    st.markdown("##### 2. 共起ネットワーク")
                    word_freq = Counter(words)
                    top_words = [w for w, c in word_freq.most_common(cooc_top_n)]
                    pair_counts = Counter()
                    
                    for _, row in df_in.iterrows():
                        dt = ""
                        if col_map['title']: dt += str(row[col_map['title']]) + " "
                        if col_map.get('abstract') and col_map['abstract'] in row: 
                            dt += str(row[col_map['abstract']]) + " "
                        dw = set(extract_compound_nouns(dt))
                        dw = {w for w in dw if w in top_words}
                        if len(dw) >= 2:
                            for pair in combinations(sorted(list(dw)), 2): pair_counts[pair] += 1
                    
                    G = nx.Graph()
                    for w in top_words: G.add_node(w, count=word_freq[w])
                    for (w1, w2), c in pair_counts.items():
                        jac = c / (word_freq[w1] + word_freq[w2] - c)
                        if jac >= cooc_threshold: G.add_edge(w1, w2, weight=jac)
                    
                    G.remove_nodes_from(list(nx.isolates(G)))
                    if G.number_of_nodes() == 0: st.warning("共起ペアなし")
                    else:
                        pos = nx.spring_layout(G, k=0.5, seed=42)
                        edge_x, edge_y = [], []
                        for edge in G.edges():
                            x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
                            edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
                        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
                        
                        node_x, node_y, node_text, node_size = [], [], [], []
                        for node in G.nodes():
                            x, y = pos[node]; node_x.append(x); node_y.append(y)
                            c = G.nodes[node]['count']
                            node_text.append(f"{node} ({c})")
                            node_size.append(np.log(c+1)*10)
                        
                        node_trace = go.Scatter(
                            x=node_x, y=node_y, mode='markers+text', hoverinfo='text', text=list(G.nodes()), textposition="top center",
                            marker=dict(showscale=True, colorscale='YlGnBu', size=node_size, color=node_size, line_width=2)
                        )
                        fig_net = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(title='共起ネットワーク', showlegend=False, hovermode='closest', margin=dict(b=20,l=5,r=5,t=40), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
                        utils.update_fig_layout(fig_net, '共起ネットワーク', theme_config=theme_config, show_axes=False)
                        fig_net.update_xaxes(visible=False); fig_net.update_yaxes(visible=False)
                        st.plotly_chart(fig_net, use_container_width=True)

# --- D. エクスポート ---
with tab_d:
    st.subheader("データエクスポート")
    if "df_momentum_export" in st.session_state:
        st.download_button("動態分析データ (CSV)", convert_df_to_csv(st.session_state.df_momentum_export), "MEGA_PULSE.csv", "text/csv")
    if "df_drilldown_export" in st.session_state:
        st.download_button("ドリルダウンデータ (CSV)", convert_df_to_csv(st.session_state.df_drilldown_export), "MEGA_TELESCOPE.csv", "text/csv")