import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
import re
import datetime
import unicodedata
import string
from collections import Counter
from itertools import combinations
import networkx as nx
from scipy.ndimage import label as nd_label
from scipy.spatial import ConvexHull
from sklearn.feature_extraction.text import TfidfVectorizer
import utils
import utils_ai
import utils_spatial
import patiroha
from umap import UMAP
import hdbscan
from wordcloud import WordCloud
from janome.tokenizer import Tokenizer
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 警告を非表示
warnings.filterwarnings('ignore')

# ページ設定
st.set_page_config(page_title="APOLLO v8 | EAGLE", page_icon="🦅", layout="wide")

st.session_state['current_page'] = 'EAGLE'

# フォント設定
FONT_PATH = utils.get_japanese_font_path()
if FONT_PATH:
    try:
        prop = fm.FontProperties(fname=FONT_PATH)
        plt.rcParams['font.family'] = prop.get_name()
    except: pass

# サイドバー
utils.render_sidebar()

st.title("🦅 EAGLE")
st.markdown("投げ縄ツールで技術マップ上の任意の領域を手動選択し、独自のクラスタを構築・分析します。")

# ==================================================================
# --- テキスト処理設定 ---
# ==================================================================
@st.cache_resource
def load_tokenizer_eagle():
    return Tokenizer()

t = load_tokenizer_eagle()
if "stopwords" in st.session_state and st.session_state["stopwords"]:
    stopwords = st.session_state["stopwords"]
else:
    stopwords = patiroha.get_stopwords()

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
    ("範囲表現", r"(?:以上|以下|未満|超|以内)", "regex", 2)
]
_ngram_compiled = [(cat, (re.compile(pat) if ptype == "regex" else pat), ptype, pri) for cat, pat, ptype, pri in _ngram_rows]

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
def extract_compound_nouns(text, stopwords_list):
    # 防御層: 異常入力と超長文で Janome の IndexError を避ける
    if not isinstance(text, str) or not text.strip():
        return []
    if len(text) > 8000:
        text = text[:8000]

    text = normalize_text(text)
    text = apply_ngram_filters(text)
    text = re.sub(r'【.*?】', '', text)
    text = re.sub(r'[!\"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~]', ' ', text)

    try:
        tokens = t.tokenize(text)
    except Exception:
        return []
    words, compound_word = [], ''
    for token in tokens:
        pos = token.part_of_speech.split(',')[0]
        if pos == '名詞':
            compound_word += token.surface
        else:
            if (len(compound_word) > 1 and
                compound_word not in stopwords_list and
                not re.fullmatch(r'[\d０-９]+', compound_word) and
                not re.fullmatch(r'(図|表|式|第)[\d０-９]+.*', compound_word) and
                not re.match(r'^(上記|前記|本開示|当該|該)', compound_word) and
                not re.search(r'[0-9０-９]+[)）]?$', compound_word) and
                not re.match(r'[0-9０-９]+[a-zA-Zａ-ｚＡ-Ｚ]', compound_word)):
                words.append(compound_word)
            compound_word = ''
            
    if (len(compound_word) > 1 and
        compound_word not in stopwords_list and
        not re.fullmatch(r'[\d０-９]+', compound_word) and
        not re.fullmatch(r'(図|表|式|第)[\d０-９]+.*', compound_word) and
        not re.match(r'^(上記|前記|本開示|当該|該)', compound_word) and
        not re.search(r'[0-9０-９]+[)）]?$', compound_word) and
        not re.match(r'[0-9０-９]+[a-zA-Zａ-ｚＡ-Ｚ]', compound_word)):
        words.append(compound_word)
    return words

def generate_wordcloud_and_list(words, title, top_n=20, font_path=None, capcom_key=None):
    if not words: return None
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

        # CAPCOM: ワードクラウドデータ保存
        if capcom_key:
            try:
                import capcom
                if capcom.is_active():
                    import io
                    wc_data = {
                        "metadata": {"module": "EAGLE", "title": title, "top_n": top_n},
                        "word_frequencies": {w: c for w, c in word_freq.most_common(100)}
                    }
                    capcom.save_data(f"{capcom_key}_wordcloud.json", wc_data)
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                    buf.seek(0)
                    capcom.save_snapshot_image(f"{capcom_key}_wordcloud", buf.read())
            except Exception:
                pass

        # VOYAGERスナップショットボタン
        if capcom_key:
            utils.render_snapshot_button(
                title=f"ワードクラウド: {title}",
                description=f"TF-IDFワードクラウド（上位{top_n}語）",
                key=f"{capcom_key}_wordcloud",
                fig=fig,
                data_summary={
                    "module": "EAGLE",
                    "type": "wordcloud",
                    "title": title,
                    "top_words": [{"word": w, "freq": c} for w, c in word_freq.most_common(top_n)]
                }
            )
    except Exception as e:
        st.error(f"ワードクラウドの描画に失敗しました: {e}")

def get_date_bin_options(df_filtered, interval_years, year_column='year'):
    if df_filtered is None or df_filtered.empty: return [f"(データなし)"]
    if year_column not in df_filtered.columns: return [f"(全期間) ({len(df_filtered)}件)"]
    
    df_filtered = df_filtered.copy()
    df_filtered[year_column] = pd.to_numeric(df_filtered[year_column], errors='coerce')
    if df_filtered[year_column].isnull().all(): return [f"(全期間) ({len(df_filtered)}件)"]

    try:
        min_year = int(df_filtered[year_column].min())
        max_year = int(df_filtered[year_column].max())
        if min_year == max_year: return [f"{min_year} ({len(df_filtered)}件)"]
        
        bins = list(range(min_year, max_year + interval_years, interval_years))
        if not bins: bins = [min_year]
        if bins[-1] <= max_year: bins.append(bins[-1] + interval_years)

        labels = [f"{bins[i]}-{bins[i+1] - 1}" for i in range(len(bins)-1)]
        df_filtered['temp_date_bin'] = pd.cut(df_filtered[year_column], bins=bins, labels=labels, right=False, include_lowest=True)
        date_bin_counts = df_filtered['temp_date_bin'].value_counts()
        
        options = [f"(全期間) ({len(df_filtered)}件)"] + [f"{label} ({date_bin_counts.get(label, 0)}件)" for label in labels if date_bin_counts.get(label, 0) > 0]
        return options
    except Exception as e:
        return [f"Error: {str(e)}"]

def update_drill_hover_text(df_subset):
    df_subset['drill_hover_text'] = df_subset.apply(
        lambda row: f"{row['hover_text']}<br><b>サブクラスタ:</b> {row['drill_cluster_label']}", axis=1
    )
    return df_subset

def get_top_tfidf_words(row_vector, feature_names, top_n=5):
    scores = row_vector.toarray().flatten() 
    indices = np.argsort(scores)[::-1]
    non_zero_indices = [i for i in indices if scores[i] > 0]
    top_indices = non_zero_indices[:top_n]
    top_words = [feature_names[i] for i in top_indices]
    return ", ".join(top_words)

# ヘルパー: ホバーテキスト更新 (EAGLE用)
def update_hover_text_eagle(df, col_map, labels_map=None, cluster_col='eagle_cluster'):
    hover_texts = []
    for index, row in df.iterrows():
        text = ""
        if col_map['title'] and pd.notna(row[col_map['title']]): text += f"<b>名称:</b> {str(row[col_map['title']])[:50]}...<br>"
        if col_map['app_num'] and pd.notna(row[col_map['app_num']]): text += f"<b>番号:</b> {row[col_map['app_num']]}<br>"
        if col_map['applicant'] and pd.notna(row[col_map['applicant']]): text += f"<b>出願人:</b> {str(row[col_map['applicant']])[:50]}...<br>"
        if 'characteristic_words' in row: text += f"<b>特徴語:</b> {row['characteristic_words']}<br>"
        hover_texts.append(text)
    return hover_texts

# データ読み込み
if not st.session_state.get("preprocess_done", False):
    st.error("分析データがありません。Mission Controlでデータをロードしてください。")
    st.stop()

df_main = st.session_state.df_main
sbert_embeddings = st.session_state.sbert_embeddings
tfidf_matrix = st.session_state.tfidf_matrix
feature_names = st.session_state.feature_names
col_map = st.session_state.col_map
delimiters = {'applicant': ';', 'inventor': ';', 'ipc': ';', 'fi': ';', 'f_term': ';'}

# UMAP座標の存在確認 (Saturn Vと共有)
if 'umap_x' not in df_main.columns or 'umap_y' not in df_main.columns:
    with st.spinner("UMAP座標を算出中 (Saturn Vと共有)..."):
        reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        embedding = reducer.fit_transform(sbert_embeddings)
        st.session_state.df_main['umap_x'] = embedding[:, 0]
        st.session_state.df_main['umap_y'] = embedding[:, 1]
        st.session_state.saturnv_sbert_umap_done = True
        df_main = st.session_state.df_main

# EAGLE用セッション状態初期化
if "eagle_cluster_map" not in st.session_state: st.session_state.eagle_cluster_map = {}
if "eagle_labels_map" not in st.session_state: st.session_state.eagle_labels_map = {}
if "df_eagle" not in st.session_state: 
    st.session_state.df_eagle = df_main.copy()
    st.session_state.df_eagle['eagle_cluster'] = -1
    # Check for lost label column if re-init
    if 'eagle_cluster' in st.session_state.df_eagle.columns:
         st.session_state.df_eagle['eagle_cluster'] = st.session_state.df_eagle['eagle_cluster'].fillna(-1).astype(int)

# 特徴語の初期化/検証
if 'characteristic_words' not in st.session_state.df_eagle.columns:
    with st.spinner("特徴語を抽出中..."):
        # df_eagleはdf_mainのコピーであり、インデックスがTF-IDF行列と整合していることを前提とする
        kw_list = []
        # 最適化のためdf_mainに既に存在するか確認
        if 'characteristic_words' in df_main.columns:
             st.session_state.df_eagle['characteristic_words'] = df_main['characteristic_words']
        else:
             # 再利用のため先にdf_mainで計算
             st.session_state.df_main['characteristic_words'] = [get_top_tfidf_words(tfidf_matrix[i], feature_names) for i in range(tfidf_matrix.shape[0])]
             st.session_state.df_eagle['characteristic_words'] = st.session_state.df_main['characteristic_words']

# hover_textの存在確認
if 'hover_text' not in st.session_state.df_eagle.columns or 'characteristic_words' not in st.session_state.df_eagle['hover_text'].iloc[0]:
    st.session_state.df_eagle['hover_text'] = update_hover_text_eagle(st.session_state.df_eagle, col_map)

# ヘルパー: ラベル生成（単一クラスタ用、投げ縄選択時に使用）
def generate_label_for_cluster(df_sub, tfidf_mat, feat_names, top_n=3):
    """単一クラスタのラベルを生成する。patiroha.auto_labelのc-TF-IDFと同等。"""
    if df_sub.empty:
        return "Empty"
    texts = (df_sub[col_map.get('title', '')].fillna('') + ' ' +
             df_sub[col_map.get('abstract', '')].fillna(''))
    # 全件を同一クラスタ(0)として扱い、c-TF-IDFでラベル生成
    import numpy as np
    dummy_labels = np.zeros(len(df_sub), dtype=int)
    label_map = patiroha.auto_label(texts, dummy_labels, method='c-tfidf', top_n=top_n)
    # "[0] term1, term2, term3" から "[0] " を除去して返す
    raw = label_map.get(0, "Empty")
    return raw.split("] ", 1)[-1] if "] " in raw else raw

# ヘルパー: Utilsを使用してレイアウト更新
def update_fig_eagle(fig, title, show_legend=False):
    utils.update_fig_layout(fig, title, height=1000, show_axes=False, show_legend=show_legend)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    if not show_legend:
        fig.update_layout(showlegend=False) # Hide legend box if not requested
    return fig

# ヘルパー: 密度トレース取得
def get_density_trace(x, y, mesh_size):
    custom_density_colorscale = [
        [0.0, "rgba(255, 255, 255, 0)"], 
        [0.1, "rgba(225, 245, 254, 0.3)"],
        [0.4, "rgba(129, 212, 250, 0.6)"],
        [1.0, "rgba(2, 119, 189, 0.9)"]
    ]
    return go.Histogram2dContour(
        x=x, y=y, 
        colorscale=custom_density_colorscale, 
        showscale=False, 
        nbinsx=mesh_size, nbinsy=mesh_size,
        contours=dict(coloring='fill', showlines=True),
        line=dict(width=0.5, color='rgba(0, 0, 0, 0.2)'),
        hoverinfo='skip'
    )

# --- 共通設定 ---
col_common, _ = st.columns([1, 2])
with col_common:
    resolution = st.number_input("メッシュサイズ (Grid)", min_value=10, max_value=200, value=30, step=5, key="eagle_resolution_common")

st.markdown("---")

# --- フィルタリングとデータレイヤリング (Saturn Vアーキテクチャ) ---
st.subheader("フィルタリング設定")

# 1. Universe (全体)
df_universe = st.session_state.df_eagle.copy()

# 絶対密度スケール用の全体ZMax計算
# ユニバース上のメッシュ密度を計算し、全体最大密度を取得
try:
    _H, _x, _y = np.histogram2d(df_universe['umap_x'], df_universe['umap_y'], bins=resolution)
    eagle_global_zmax = _H.max()
except:
    eagle_global_zmax = None

# フィルタUI
col_f1, col_f2 = st.columns(2)
def on_eagle_interval_change():
    if "eagle_main_date_filter" in st.session_state: del st.session_state.eagle_main_date_filter

with col_f1:
    # 日付ビニング
    if 'year' in df_universe.columns and df_universe['year'].notna().any():
        bin_interval_val = st.selectbox("期間の粒度:", [5, 3, 2, 1], index=0, key="eagle_main_bin_interval", on_change=on_eagle_interval_change)
        date_bin_opts = get_date_bin_options(df_universe, int(bin_interval_val), 'year')
        date_filter_val = st.selectbox("表示期間:", date_bin_opts, key="eagle_main_date_filter")
    else:
        date_filter_val = "(全期間)"
        st.info("年データ (year) がありません")

# 2. Trend (期間ごとの地形)
df_trend = df_universe.copy()
if not date_filter_val.startswith("(全期間)"):
    try:
        date_label = date_filter_val.split(' (')[0].strip()
        s_year, e_year = map(int, date_label.split('-'))
        df_trend = df_trend[(df_trend['year'] >= s_year) & (df_trend['year'] <= e_year)]
    except: pass

with col_f2:
    # 出願人フィルタ (フォーカス作成のためにトレンドに適用)
    if 'applicant_main' in df_trend.columns:
        apps = df_trend['applicant_main'].explode().dropna()
    elif col_map['applicant'] and col_map['applicant'] in df_trend.columns:
        apps = df_trend[col_map['applicant']].fillna('').str.split(delimiters['applicant']).explode().str.strip()
    else:
        apps = pd.Series([])

    if not apps.empty:
        app_counts = apps.value_counts()
        uniq_apps = app_counts.index.tolist()
        app_opts = [(f"(全出願人) ({len(df_trend)}件)", "ALL")] + \
                   [(f"{a} ({app_counts[a]}件)", a) for a in uniq_apps]
        
        app_filter_val = st.multiselect(
            "出願人:", app_opts, default=[app_opts[0]], 
            format_func=lambda x: x[0], key="eagle_main_app_filter"
        )
    else:
        app_filter_val = [(f"(全出願人) ({len(df_trend)}件)", "ALL")]
        st.info("出願人データがありません")

# 3. Focus (注目データ)
df_focus = df_trend.copy()
selected_apps = [x[1] for x in app_filter_val]
if "ALL" not in selected_apps:
    mask_list = [df_focus[col_map['applicant']].fillna('').str.contains(re.escape(a)) for a in selected_apps]
    if mask_list:
        df_focus = df_focus[pd.concat(mask_list, axis=1).any(axis=1)]
    else:
        df_focus = df_focus.iloc[0:0]

# 4. Ghost (Universe - Focus)
try:
    df_ghost = df_universe.drop(df_focus.index, errors='ignore')
except:
    df_ghost = pd.DataFrame()

st.markdown(f"**表示データ数: {len(df_focus)} / {len(df_universe)}**")
st.markdown("---")

# --- メイン分析: Lassoクラスタリング ---
st.subheader("手動選択クラスタリング")

# クラスタ管理UI
c_mgmt1, c_mgmt2 = st.columns([1, 1])
with c_mgmt1:
    edit_mode = st.radio("モード:", ["編集中 (Edit)", "閲覧中 (FIX)"], horizontal=True, key="eagle_edit_mode")

is_editing = (edit_mode == "編集中 (Edit)")

if is_editing:
    st.markdown("グラフ上の「Lasso Select」等で範囲を選択し、新規クラスタを作成してください。<br>不要なクラスタは下部から削除できます。", unsafe_allow_html=True)
else:
    st.markdown("クラスタリングはロックされています。修正する場合は「編集中」に切り替えてください。", unsafe_allow_html=True)

# コントロール (ラベル & 密度固定)
col_ctrl1, col_ctrl2 = st.columns([1, 2])
with col_ctrl1:
    show_labels_chk = st.checkbox("マップにラベルを表示する", value=True, key="eagle_main_show_labels")
with col_ctrl2:
    fix_density_chk = st.checkbox("密度マップを固定 (全体基準)", value=True, key="eagle_fix_density")

# 現在のクラスタを表示
fig_lasso = go.Figure()

# 1. 密度背景 (トレンドに基づく)
# fix_density_chkがONの場合、絶対スケール比較にglobal zmaxを利用
if not df_trend.empty:
    density_trace = get_density_trace(df_trend['umap_x'], df_trend['umap_y'], resolution)
    if fix_density_chk and eagle_global_zmax is not None:
        density_trace.update(zauto=False, zmin=0, zmax=eagle_global_zmax)
    fig_lasso.add_trace(density_trace)

# 2. ゴーストポイント (除外データ)
if not df_ghost.empty:
    fig_lasso.add_trace(go.Scattergl(
        x=df_ghost['umap_x'], y=df_ghost['umap_y'], mode='markers',
        marker=dict(color='#dddddd', size=3, opacity=0.3),
        name='その他 (Ghost)',
        hoverinfo='skip'
    ))

# 3. フォーカスポイント (クラスタリング対象)
uniq = sorted(df_focus['eagle_cluster'].unique())
color_seq = utils.APOLLO_COLORS

is_applicant_filtered = "ALL" not in selected_apps

# 編集モード用マーカー枠線
marker_border = dict(width=1, color='#333333') if is_editing else dict(width=0)

if is_applicant_filtered:
    # 出願人着色モード (Saturn Vスタイル)
    palette = px.colors.qualitative.Bold
    
    for i, app_name in enumerate(selected_apps):
        # この出願人でフィルタ
        mask = df_focus[col_map['applicant']].fillna('').str.contains(re.escape(app_name))
        d_app = df_focus[mask]
        
        if not d_app.empty:
                # 動的ホバーテキスト構築
                # 内部クラスタIDをラベルにマッピング
                current_labels = d_app['eagle_cluster'].map(lambda x: st.session_state.eagle_labels_map.get(x, str(x)) if x != -1 else "")
                dynamic_hover = d_app['hover_text'] + d_app['eagle_cluster'].apply(lambda x: f"<b>クラスタ:</b> {st.session_state.eagle_labels_map.get(x, str(x))}" if x != -1 else "")
                
                fig_lasso.add_trace(go.Scattergl(
                    x=d_app['umap_x'], y=d_app['umap_y'], mode='markers',
                    marker=dict(color=palette[i % len(palette)], size=6, opacity=0.9, line=marker_border),
                    name=app_name,
                    customdata=d_app.index,
                    hoverinfo='text',
                    hovertext=dynamic_hover,
                    showlegend=True
                ))
else:
    # クラスタ着色モード (オリジナル)
    for i, c in enumerate(uniq):
        d = df_focus[df_focus['eagle_cluster'] == c]
        if d.empty: continue
        name = st.session_state.eagle_labels_map.get(c, str(c))
        color = '#dddddd' if c == -1 else color_seq[i % len(color_seq)]
        opacity = 0.3 if c == -1 else 0.8
        
        # クラスタモードの場合、d内の全点はクラスタc(name)に属する
        dynamic_hover_c = d['hover_text'] + (f"<b>クラスタ:</b> {name}" if c != -1 else "")

        fig_lasso.add_trace(go.Scattergl(
            x=d['umap_x'], y=d['umap_y'], mode='markers',
            marker=dict(color=color, size=5, opacity=opacity, line=marker_border),
            name=name,
            customdata=d.index,
            hoverinfo='text',
            hovertext=dynamic_hover_c,
            showlegend=False
        ))

# 3. アノテーション
annotations_main = []
if show_labels_chk:
    for c in uniq:
        if c == -1: continue
        d = df_focus[df_focus['eagle_cluster'] == c]
        if d.empty: continue
        
        mean_x = d['umap_x'].mean()
        mean_y = d['umap_y'].mean()
        label_text = st.session_state.eagle_labels_map.get(c, str(c))
        
        try:
            c_idx_strict = uniq.index(c)
            border_color = color_seq[c_idx_strict % len(color_seq)]
        except: 
            border_color = "#333333"

        annotations_main.append(go.layout.Annotation(
            x=mean_x, y=mean_y, text=label_text, showarrow=False, 
            font=dict(size=11, color='black', family="Helvetica"), 
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor=border_color,
            borderwidth=1,
            borderpad=3
        ))

fig_lasso.update_layout(annotations=annotations_main)
update_fig_eagle(fig_lasso, "Current Clusters", show_legend=False)

# インタラクティブロジック
if is_editing:
    fig_lasso.update_layout(dragmode='lasso', clickmode='event+select')
    selection = st.plotly_chart(fig_lasso, use_container_width=True, on_select="rerun", config={
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
    
    selected_indices = []
    if selection and "selection" in selection:
        points = selection["selection"]["points"]
        selected_indices = [p["customdata"] for p in points]
    
    st.write(f"選択中: {len(selected_indices)} 件")
    
    # 新規クラスタ作成
    if selected_indices:
        col_l1, col_l2 = st.columns(2)
        with col_l1:
            all_ids = st.session_state.df_eagle['eagle_cluster'].unique()
            max_id = max(all_ids) if len(all_ids) > 0 else 0
            if max_id < 0: max_id = 0
            rec_id = max(max_id + 1, 1)
            new_id = st.number_input("新規クラスタID", min_value=1, value=int(rec_id))
        with col_l2:
            if st.button("選択範囲を新規クラスタにする"):
                st.session_state.df_eagle.loc[selected_indices, 'eagle_cluster'] = new_id
                sub_df = st.session_state.df_eagle.loc[selected_indices]
                lbl = generate_label_for_cluster(sub_df, tfidf_matrix, feature_names)
                st.session_state.eagle_labels_map[new_id] = f"[{new_id}] {lbl}"
                # CAPCOM: patents.csvにeagle_cluster列を更新
                try:
                    import capcom
                    if capcom.is_active():
                        capcom.save_patents_csv()
                except Exception:
                    pass
                st.success(f"ID {new_id} を作成しました！")
                st.rerun()

    # クラスタ削除UI
    st.markdown("#### クラスタ削除")
    del_ids = [c for c in sorted(st.session_state.df_eagle['eagle_cluster'].unique()) if c != -1]
    if del_ids:
        col_d1, col_d2 = st.columns([1, 1])
        with col_d1:
            del_target_id = st.selectbox("削除するクラスタID:", del_ids, key="eagle_delete_target")
        with col_d2:
            if st.button("削除実行"):
                # Reset to -1
                st.session_state.df_eagle.loc[st.session_state.df_eagle['eagle_cluster'] == del_target_id, 'eagle_cluster'] = -1
                if del_target_id in st.session_state.eagle_labels_map:
                    del st.session_state.eagle_labels_map[del_target_id]
                # CAPCOM: patents.csvにeagle_cluster列を更新
                try:
                    import capcom
                    if capcom.is_active():
                        capcom.save_patents_csv()
                except Exception:
                    pass
                st.success(f"ID {del_target_id} を削除しました")
                st.rerun()

else:
    # 固定モード
    fig_lasso.update_layout(dragmode='pan') # 選択ロック
    st.plotly_chart(fig_lasso, use_container_width=True, config={
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

    # エクスポート & インサイトボタン
    snap_data = utils.generate_rich_summary(df_focus, title_col=col_map['title'], abstract_col=col_map['abstract'])
    snap_data['module'] = 'EAGLE'
    
    # 統計情報の追加
    try:
         cluster_counts_snap = df_focus['eagle_cluster'].value_counts()
         cluster_summary_lines = []
         
         # クラスタごとの代表を抽出
         cluster_reps = utils.get_cluster_representatives(df_focus, cluster_col='eagle_cluster', n_reps=3)

         for cid in sorted(df_focus['eagle_cluster'].unique()):
             if cid == -1: continue
             label = st.session_state.eagle_labels_map.get(cid, f"Cluster {cid}")
             count = cluster_counts_snap.get(cid, 0)
             cluster_summary_lines.append(f"- {label} ({count}件)")
             
             # 代表を追加
             if cid in cluster_reps:
                 for rep in cluster_reps[cid]:
                     cluster_summary_lines.append(rep)

         snap_data['cluster_summary'] = "設定クラスタ構成 (Lasso):\n" + "\n".join(cluster_summary_lines)
    except: pass

    # AIインサイト (メイン)

    # AIインサイトコンテキスト準備
    insight_context = f"""
    **マップタイプ**: 技術ランドスケープ (EAGLE - Telescope)
    **分析対象**: 全体俯瞰マップ。
    **手法**: SBERT (文章ベクトル化) + UMAP (次元圧縮) + Lasso (手動クラスタ探索)。
    **視覚的エンコーディング**:
    - **点**: 個々の特許/文献。距離が近いほど意味的に類似しています。
    - **クラスタ**: 色分けされたグループは、自動検出された技術領域を表します。
    - **配置**: マップ全体の「形状」が技術空間の広がりを表します。
    """
    insight_role = "あなたはシニア特許アナリストです。技術俯瞰図から戦略的な示唆を導きます。"
    insight_instruction = """
    ランドスケープの構造を分析してください：
    1. **主要テーマ**: どのような技術クラスタが形成されていますか？
    2. **技術の関係性**: どのクラスタとどのクラスタが近接していますか？そこから読み取れる技術的シナジーは？
    3. **注目領域**: フィルタリングされた領域の特徴は何ですか？
    **重要**: 回答は箇条書きで、技術的な洞察を深掘りしてください。
    """
    
    # 空間情報
    spatial_info = utils_spatial.generate_spatial_cluster_summary(
        df_focus, 'eagle_cluster', 'umap_x', 'umap_y', label_map=st.session_state.eagle_labels_map
    )

    # スナップショット用に統合
    full_ai_context = f"""
### AI Insight Context (Auto-Generated)
{insight_context}

### Spatial Context
{spatial_info}

### Analyst Instructions
{insight_instruction}
"""
    snap_data['ai_insight_context'] = full_ai_context



    # --- Snapshot: メインランドスケープ ---
    utils.render_snapshot_button(
        title="EAGLE: メインランドスケープ",
        description="SBERT+UMAPによる技術ランドスケープ（手動クラスタリング）。",
        key="eagle_main_snap",
        fig=fig_lasso,
        data_summary=snap_data
    )

    main_prompt = utils_ai.generate_ai_insight_prompt(
        insight_role, insight_context, snap_data, insight_instruction,
        extra_content=f"\n# 空間配置情報 (Spatial Context)\n{spatial_info}"
    )
    utils_ai.render_ai_insight_button(main_prompt, "eagle_main_insight")

    # CAPCOM data/ JSON出力（EAGLE クラスタ）
    try:
        import capcom
        if capcom.is_active():
            eagle_clusters_json = []
            cluster_counts_eagle = df_focus['eagle_cluster'].value_counts()
            for cid in sorted(df_focus['eagle_cluster'].unique()):
                if cid == -1:
                    continue
                label = st.session_state.eagle_labels_map.get(cid, f"Cluster {cid}")
                count = int(cluster_counts_eagle.get(cid, 0))
                cid_mask = df_focus['eagle_cluster'] == cid
                cx = float(df_focus.loc[cid_mask, 'umap_x'].mean()) if cid_mask.any() else 0
                cy = float(df_focus.loc[cid_mask, 'umap_y'].mean()) if cid_mask.any() else 0
                reps_raw = cluster_reps.get(cid, []) if 'cluster_reps' in dir() and cid in cluster_reps else []
                eagle_clusters_json.append({
                    "cluster_id": int(cid),
                    "label": label,
                    "count": count,
                    "centroid": [round(cx, 4), round(cy, 4)],
                    "representative_patents": reps_raw
                })
            noise_count = int((df_focus['eagle_cluster'] == -1).sum()) if -1 in df_focus['eagle_cluster'].values else 0
            eagle_json = {
                "metadata": {
                    "module": "EAGLE",
                    "mode": "manual_lasso",
                    "n_clusters": len(eagle_clusters_json),
                    "noise_count": noise_count,
                    "total_patents": len(df_focus)
                },
                "clusters": eagle_clusters_json,
                "spatial_context": spatial_info if 'spatial_info' in dir() else ""
            }
            capcom.save_data("eagle_clusters.json", eagle_json)
    except Exception as e:
        pass

    # --- クラスタ動態マップ ---
    if 'eagle_cluster' in df_focus.columns and 'year' in df_focus.columns:
        eagle_labels = st.session_state.get('eagle_labels_map', {})
        if eagle_labels and df_focus['eagle_cluster'].nunique() > 1:
            dyn_data = utils.render_cluster_dynamics_section(
                df_focus, 'eagle_cluster', eagle_labels,
                year_col='year', cagr_window=5,
                unique_key='eagle_dynamics',
                module_name='EAGLE',
            )
            if dyn_data:
                try:
                    import capcom
                    if capcom.is_active():
                        capcom.save_data('eagle_cluster_dynamics', {'cluster_dynamics': dyn_data})
                except Exception:
                    pass

# --- ラベルエディタ ---
st.markdown("---")
st.subheader("クラスタ・ラベル編集")

if "eagle_labels_map_original" not in st.session_state:
    st.session_state.eagle_labels_map_original = st.session_state.eagle_labels_map.copy()

if len(st.session_state.eagle_labels_map) != len(st.session_state.eagle_labels_map_original):
     st.session_state.eagle_labels_map_original = st.session_state.eagle_labels_map.copy()

utils.render_ai_label_assistant(st.session_state.df_eagle, 'eagle_cluster', "eagle_labels_map", col_map, tfidf_matrix, feature_names, widget_key_prefix="eagle_ai")
label_widgets = utils.create_label_editor_ui(st.session_state.eagle_labels_map_original, st.session_state.eagle_labels_map, "eagle_manual")

if st.button("ラベルを更新", key="eagle_update_labels"):
    for c, v in label_widgets.items(): st.session_state.eagle_labels_map[c] = v
    st.success("ラベルを更新しました")
    st.rerun()

st.markdown("---")
st.subheader("分析結果のエクスポート")
with st.expander("CSVダウンロードオプション", expanded=True):
    st.markdown("現在のクラスタリング結果（ラベル、特徴語を含む）をCSV形式でダウンロードします。")
    if st.button("エクスポート用データを生成", key="eagle_gen_export"):
        with st.spinner("CSVを生成中..."):
            df_export = st.session_state.df_eagle.copy()
            # Map labels
            df_export['cluser_id'] = df_export['eagle_cluster']
            df_export['cluster_label'] = df_export['eagle_cluster'].map(lambda x: st.session_state.eagle_labels_map.get(x, "") if x != -1 else "")
            
            # Ensure characteristic_words exists (it should, but just in case)
            if 'characteristic_words' not in df_export.columns:
                 # Try to recover from df_main if missing
                 if 'characteristic_words' in st.session_state.df_main.columns:
                     df_export['characteristic_words'] = st.session_state.df_main['characteristic_words']
            
            csv_data = df_export.to_csv(index=False).encode('utf-8-sig')
            st.session_state.eagle_export_csv = csv_data
            st.success("生成完了")

    if "eagle_export_csv" in st.session_state:
        st.download_button(
            label="CSVをダウンロード",
            data=st.session_state.eagle_export_csv,
            file_name="eagle_clustering_result.csv",
            mime="text/csv",
            key='eagle_download_csv_btn'
        )

# ==================================================================
# --- ドリルダウン分析 (Saturn Vより) ---
# ==================================================================
st.markdown("---")
st.subheader("ドリルダウン分析 / 詳細分析")

# カウント付きクラスタ選択
c_counts = st.session_state.df_eagle['eagle_cluster'].value_counts()
sorted_cids = sorted(st.session_state.df_eagle['eagle_cluster'].unique())
cluster_opts = [(f"(未選択)", "NONE")] + \
               [(f"{st.session_state.eagle_labels_map.get(c, str(c))} ({c_counts.get(c, 0)}件)", c) for c in sorted_cids if c != -1]

drilldown_target_id = st.selectbox("分析対象クラスタを選択:", options=[x[1] for x in cluster_opts], format_func=lambda x: [o[0] for o in cluster_opts if o[1] == x][0])

if drilldown_target_id != "NONE":
    df_subset_filter = st.session_state.df_eagle[st.session_state.df_eagle['eagle_cluster'] == drilldown_target_id].copy()
    
    col1, col2 = st.columns(2)
    with col1:
        if 'year' in df_subset_filter.columns and df_subset_filter['year'].notna().any():
            def on_drill_interval_change(): pass # minimal
            drill_bin_interval_w_val = st.selectbox("期間の粒度:", [5, 3, 2, 1], index=0, key="eagle_drill_interval_w", on_change=on_drill_interval_change)
            drill_date_bin_options = get_date_bin_options(df_subset_filter, int(drill_bin_interval_w_val), 'year')
            drill_date_bin_filter_w = st.selectbox("表示期間:", drill_date_bin_options, key="eagle_drill_date_filter_w")
        else:
            drill_date_bin_filter_w = "(全期間)"
    with col2:
        if 'applicant_main' in df_subset_filter.columns:
            applicants_drill = df_subset_filter['applicant_main'].explode().dropna()
        elif col_map['applicant'] and col_map['applicant'] in df_subset_filter.columns:
            applicants_drill = df_subset_filter[col_map['applicant']].fillna('').str.split(delimiters['applicant']).explode().str.strip()
        else:
            applicants_drill = pd.Series([])

        if not applicants_drill.empty:
            app_counts_drill = applicants_drill.value_counts()
            unique_applicants_drill = app_counts_drill.index.tolist()
            drill_applicant_options = [(f"(全出願人) ({len(df_subset_filter)}件)", "ALL")] + \
                                      [(f"{app} ({app_counts_drill[app]}件)", app) for app in unique_applicants_drill]
            
            drill_applicant_filter_w = st.multiselect(
                "出願人:", 
                drill_applicant_options, 
                default=[drill_applicant_options[0]], 
                format_func=lambda x: x[0], 
                key="eagle_drill_applicant_filter_w"
            )
        else:
            drill_applicant_filter_w = [(f"(全出願人) ({len(df_subset_filter)}件)", "ALL")]

    st.subheader("詳細クラスタリングモード")
    drill_method = st.radio("手法を選択:", ["自動 (HDBSCAN)", "手動 (Lasso)"], horizontal=True, key="eagle_drill_method")

    if drill_method == "自動 (HDBSCAN)":
        c1, c2, c3 = st.columns(3)
        with c1: drill_min_cluster_size_w = st.number_input('最小クラスタサイズ:', min_value=2, value=5, key="eagle_drill_min_cluster_size_w")
        with c2: drill_min_samples_w = st.number_input('最小サンプル数:', min_value=1, value=5, key="eagle_drill_min_samples_w")
        with c3: drill_label_top_n_w = st.number_input('ラベル単語数:', min_value=1, value=3, key="eagle_drill_label_top_n_w")
    else:
        drill_min_cluster_size_w, drill_min_samples_w, drill_label_top_n_w = 0, 0, 3 # Dummy

    drill_show_labels_chk = st.checkbox('マップにラベルを表示する', value=True, key="eagle_drill_show_labels_chk")

    if st.button("選択クラスタで詳細マップ作成", type="primary", key="eagle_drill_run_button"):
        with st.spinner(f"クラスタ {drilldown_target_id} の詳細分析を実行中..."):
            try:
                # 独立した状態としてeagle_drilldown_resultを使用
                df_subset = st.session_state.df_eagle[st.session_state.df_eagle['eagle_cluster'] == drilldown_target_id].copy()
                # ラベルはカスタマイズされている可性あり
                base_label = st.session_state.eagle_labels_map.get(drilldown_target_id, str(drilldown_target_id))
                
                # フィルタ
                if not drill_date_bin_filter_w.startswith("(全期間)"):
                    try:
                        date_bin_label = drill_date_bin_filter_w.split(' (')[0].strip() 
                        start_year, end_year = map(int, date_bin_label.split('-'))
                        df_subset = df_subset[(df_subset['year'] >= start_year) & (df_subset['year'] <= end_year)]
                    except: pass 

                drill_app_values = [val[1] for val in drill_applicant_filter_w]
                if "ALL" not in drill_app_values:
                    mask_list_drill = [df_subset[col_map['applicant']].fillna('').str.contains(re.escape(app)) for app in drill_app_values]
                    df_subset = df_subset[pd.concat(mask_list_drill, axis=1).any(axis=1)]
                
                if len(df_subset) < 3: # 制限緩和
                    st.warning(f"データが少なすぎます ({len(df_subset)}件)。再分割できません。")
                else:
                    subset_indices = df_subset.index
                    subset_tfidf = tfidf_matrix[subset_indices]
                    subset_sbert = sbert_embeddings[subset_indices]
                    subset_indices_pd = pd.Index(subset_indices)

                    n_neighbors = min(10, len(df_subset) - 1)
                    if n_neighbors < 2: n_neighbors = 2
                    
                    reducer_drill = UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=2, random_state=42)
                    embedding_drill = reducer_drill.fit_transform(subset_sbert) 
                    df_subset['drill_x'] = embedding_drill[:, 0]
                    df_subset['drill_y'] = embedding_drill[:, 1]
                    
                    drill_labels_map = {}
                    
                    if drill_method == "自動 (HDBSCAN)":
                        clusterer_drill = hdbscan.HDBSCAN(min_cluster_size=int(drill_min_cluster_size_w), min_samples=int(drill_min_samples_w), metric='euclidean', cluster_selection_method='eom')
                        df_subset['drill_cluster'] = clusterer_drill.fit_predict(embedding_drill)
                    else:
                        # Manual Mode: Initialize as unclassified (-1)
                        df_subset['drill_cluster'] = -1
                        drill_labels_map[-1] = "未分類"

                    # patiroha.auto_label で c-TF-IDF ラベリング（EAGLEドリルダウン）
                    if drill_method == "自動 (HDBSCAN)":
                        drill_texts = (
                            df_subset[col_map['title']].fillna('') + ' ' +
                            df_subset[col_map['abstract']].fillna('')
                        )
                        drill_labels_map = patiroha.auto_label(
                            drill_texts,
                            df_subset['drill_cluster'].values,
                            method='c-tfidf',
                            top_n=int(drill_label_top_n_w),
                        )
                    
                    df_subset['drill_cluster_label'] = df_subset['drill_cluster'].map(drill_labels_map)
                    
                    # Ensure hover_text exists before update
                    if 'hover_text' not in df_subset.columns:
                        df_subset['hover_text'] = update_hover_text_eagle(df_subset, col_map)

                    df_subset = update_drill_hover_text(df_subset)
                    st.session_state.eagle_drilldown_result = df_subset.copy()
                    st.session_state.eagle_drill_labels_map = drill_labels_map.copy()
                    st.session_state.eagle_drill_labels_map_original = drill_labels_map.copy()
                    st.session_state.eagle_drill_base_label = base_label
                    st.success("詳細マップ作成完了。")
                    st.rerun()

            except Exception as e:
                st.error(f"エラー: {e}")

    # --- Drill-down Results UI ---
    if "eagle_drilldown_result" in st.session_state:
        df_drill = st.session_state.eagle_drilldown_result.copy()
        drill_labels_map = st.session_state.eagle_drill_labels_map
        
        tab_drill_map, tab_drill_net, tab_drill_stats, tab_drill_export = st.tabs(["詳細マップ (Map)", "共起分析 (Word)", "統計マップ (Stats)", "エクスポート (Export)"])

        with tab_drill_map:
            st.subheader("ドリルダウンマップ")
            
            drill_map_mode = st.radio("表示モード:", ["散布図 (Scatter)", "密度マップ (Density)", "クラスタ領域 (Clusters)"], horizontal=True, key="eagle_drill_map_mode_radio")
            
            d_c1, d_c2, d_c3 = st.columns(3)
            with d_c1:
                drill_mesh_size = st.number_input("メッシュサイズ", value=40, min_value=10, max_value=200, step=5, key="eagle_drill_mesh_size")
            with d_c2:
                drill_remove_noise_chk = st.checkbox("ノイズを除く", value=False, key="eagle_drill_remove_noise")
            with d_c3: pass

            if drill_remove_noise_chk:
                df_drill_plot = df_drill[df_drill['drill_cluster'] != -1]
            else:
                df_drill_plot = df_drill

            fig_drill = go.Figure()
            
            custom_density_colorscale_d = [
                [0.0, "rgba(255, 255, 255, 0)"], 
                [0.1, "rgba(225, 245, 254, 0.3)"],
                [0.4, "rgba(129, 212, 250, 0.6)"],
                [1.0, "rgba(2, 119, 189, 0.9)"]
            ]

            if drill_map_mode == "密度マップ (Density)":
                contour_d = dict(
                    x=df_drill_plot['drill_x'], y=df_drill_plot['drill_y'], 
                    colorscale=custom_density_colorscale_d, 
                    reversescale=False, xaxis='x', yaxis='y', showscale=False, name="密度", 
                    nbinsx=drill_mesh_size, nbinsy=drill_mesh_size, 
                    contours=dict(coloring='fill', showlines=True),
                    line=dict(width=0.5, color='rgba(0, 0, 0, 0.2)')
                )
                fig_drill.add_trace(go.Histogram2dContour(**contour_d))
                
            if drill_map_mode == "クラスタ領域 (Clusters)":
                color_sequence = utils.APOLLO_COLORS
                unique_clusters_d = sorted(df_drill_plot['drill_cluster'].unique())
                for i, cid in enumerate(unique_clusters_d):
                    if cid == -1: continue
                    points = df_drill_plot[df_drill_plot['drill_cluster'] == cid][['drill_x', 'drill_y']].values
                    if len(points) >= 3:
                        try:
                            hull = ConvexHull(points)
                            hull_points = points[hull.vertices]
                            hull_points = np.append(hull_points, [hull_points[0]], axis=0)
                            c_color = color_sequence[i % len(color_sequence)]
                            fig_drill.add_trace(go.Scatter(
                                x=hull_points[:, 0], y=hull_points[:, 1], mode='lines', fill='toself',
                                fillcolor=c_color, opacity=0.1, line=dict(color=c_color, width=2),
                                hoverinfo='skip', showlegend=False
                            ))
                        except: pass

            marker_line_d = dict(width=1, color='white') if drill_map_mode == "密度マップ (Density)" else dict(width=0)
            
            # --- Drill-down Scatter with Manual Selection Support ---
            # Group by cluster to have separate traces for coloring, BUT for lasso we ideally want one trace or careful handling.
            # However, to color by cluster, we need separate traces or a color array. 
            # Lasso in Plotly returns selected points indices. 
            
            uniq_d = sorted(df_drill_plot['drill_cluster'].unique())
            color_sequence = utils.APOLLO_COLORS
            
            for i, cid in enumerate(uniq_d):
                 d_sub = df_drill_plot[df_drill_plot['drill_cluster'] == cid]
                 if d_sub.empty: continue
                 
                 c_color = '#dddddd' if cid == -1 else color_sequence[i % len(color_sequence)]
                 c_name = drill_labels_map.get(cid, str(cid))
                 
                 fig_drill.add_trace(go.Scattergl(
                    x=d_sub['drill_x'], y=d_sub['drill_y'], mode='markers',
                    marker=dict(color=c_color, size=5, opacity=0.8, line=marker_line_d),
                    hoverinfo='text', hovertext=d_sub['drill_hover_text'], name=c_name,
                    customdata=d_sub.index
                ))

            annotations_drill = []
            if drill_show_labels_chk:
                color_sequence = utils.APOLLO_COLORS
                sorted_unique_cids_d = sorted(df_drill_plot['drill_cluster'].unique())
                
                for cid, grp in df_drill_plot[df_drill_plot['drill_cluster'] != -1].groupby('drill_cluster'):
                    mean_pos = grp[['drill_x', 'drill_y']].mean()
                    try:
                        color_idx = sorted_unique_cids_d.index(cid)
                        border_color = color_sequence[color_idx % len(color_sequence)]
                    except: border_color = "#333333"

                    annotations_drill.append(go.layout.Annotation(
                        x=mean_pos['drill_x'], y=mean_pos['drill_y'], text=drill_labels_map.get(cid, ""), showarrow=False, 
                        font=dict(size=10, color='black', family="Helvetica"), 
                        bgcolor='rgba(255,255,255,0.8)', bordercolor=border_color, borderwidth=2, borderpad=4
                    ))
            fig_drill.update_layout(annotations=annotations_drill)
            utils.update_fig_layout(fig_drill, f'EAGLE 詳細: {st.session_state.eagle_drill_base_label}', height=1000)
            fig_drill.update_layout(dragmode='lasso', clickmode='event+select', showlegend=False) # Enable Lasso
            
            selection_drill = st.plotly_chart(fig_drill, use_container_width=True, on_select="rerun", config={'editable': False})
            
            # Export & Insight (Drill-down)
            snap_data_d = utils.generate_rich_summary(df_drill, title_col=col_map['title'], abstract_col=col_map['abstract'])
            snap_data_d['module'] = 'EAGLE Drill-down'
            
            # Sub-cluster summary for Voyager
            try:
                 cluster_counts_snap_d = df_drill['drill_cluster'].value_counts()
                 cluster_summary_lines_d = []
                 
                 # Extract representatives
                 cluster_reps_d = utils.get_cluster_representatives(df_drill, cluster_col='drill_cluster', n_reps=3)

                 for cid in sorted(df_drill['drill_cluster'].unique()):
                     if cid == -1: continue
                     label = drill_labels_map.get(cid, f"Sub-Cluster {cid}")
                     count = cluster_counts_snap_d.get(cid, 0)
                     cluster_summary_lines_d.append(f"- {label} ({count}件)")
                     
                     if cid in cluster_reps_d:
                         for rep in cluster_reps_d[cid]:
                             cluster_summary_lines_d.append(rep)

                 snap_data_d['cluster_summary'] = f"サブクラスタ構成 ({st.session_state.eagle_drill_base_label}):\n" + "\n".join(cluster_summary_lines_d)
            except: pass


            # Prepare AI Insight Context (Drill)
            drill_insight_context = f"""
            **マップタイプ**: 局所ドリルダウンマップ (EAGLE)
            **分析対象**: クラスタ「{st.session_state.eagle_drill_base_label}」
            **手法**: 再計算されたUMAP。サブクラスタは自動(HDBSCAN)または手動で識別されます。
            **目的**: 選択された上位クラスタ（親分類）の内部にある、詳細なサブ構造を分析すること。
            """
            drill_insight_role = "あなたは高度なIPランドスケープアナリストです。技術動向と競合状況を深く読み解く専門家です。"
            drill_insight_instruction = """
            この特定技術領域（クラスタ）の内部構造を分析してください：
            1. **サブテーマの構成**: この領域はどのような細かいサブテーマ（サブクラスタ）に分かれていますか？
            2. **詳細な内容**: 代表的な特許/文献から、具体的にどのような技術課題や解決策が議論されているか要約してください。
            """
            
            d_spatial_info = utils_spatial.generate_spatial_cluster_summary(
                df_drill, 'drill_cluster', 'drill_x', 'drill_y', label_map=drill_labels_map
            )

            # Combine for Snapshot
            full_drill_context = f"""
### AI Insight Context (Auto-Generated)
{drill_insight_context}

### Spatial Context
{d_spatial_info}

### Analyst Instructions
{drill_insight_instruction}
"""
            snap_data_d['ai_insight_context'] = full_drill_context



            # --- Snapshot: ドリルダウンマップ ---
            utils.render_snapshot_button(
                title="EAGLE: ドリルダウンマップ",
                description="選択クラスタの詳細分析マップ（サブクラスタリング）。",
                key="eagle_drill_snap",
                fig=fig_drill,
                data_summary=snap_data_d
            )

            drill_prompt = utils_ai.generate_ai_insight_prompt(
                drill_insight_role, drill_insight_context, snap_data_d, drill_insight_instruction,
                extra_content=f"\n# 空間配置情報 (Spatial Context)\n{d_spatial_info}"
            )
            utils_ai.render_ai_insight_button(drill_prompt, "eagle_drill_insight")


            # --- Manual Lasso Logic for Drill-down ---
            s_indices_d = []
            if selection_drill and "selection" in selection_drill:
                s_indices_d = [p["customdata"] for p in selection_drill["selection"]["points"]]
            
            if s_indices_d:
                st.write(f"サブクラスタ選択中: {len(s_indices_d)} 件")
                c_l1, c_l2 = st.columns(2)
                with c_l1:
                     # Calculate next available ID
                     curr_ids = st.session_state.eagle_drilldown_result['drill_cluster'].unique()
                     max_id_d = max(curr_ids) if len(curr_ids) > 0 else 0
                     if max_id_d < 0: max_id_d = 0
                     new_id_d = st.number_input("新規サブクラスタID", min_value=1, value=int(max_id_d + 1), key="eagle_drill_new_id")
                with c_l2:
                    if st.button("選択範囲を新規サブクラスタにする", key="eagle_drill_apply_lasso"):
                        st.session_state.eagle_drilldown_result.loc[s_indices_d, 'drill_cluster'] = new_id_d
                        
                        # c-TF-IDF でサブクラスタのラベルを生成
                        sub_df_d = st.session_state.eagle_drilldown_result.loc[s_indices_d]
                        lbl = generate_label_for_cluster(sub_df_d, tfidf_matrix, feature_names, top_n=3)
                        st.session_state.eagle_drill_labels_map[new_id_d] = f"[{new_id_d}] {lbl}"
                        
                        # Update labels map and column
                        st.session_state.eagle_drilldown_result['drill_cluster_label'] = st.session_state.eagle_drilldown_result['drill_cluster'].map(st.session_state.eagle_drill_labels_map)
                        st.session_state.eagle_drilldown_result = update_drill_hover_text(st.session_state.eagle_drilldown_result)
                        st.success(f"サブクラスタ ID {new_id_d} を作成しました")
                        st.rerun()
            
            st.subheader("サブクラスタ・ラベル編集")
            utils.render_ai_label_assistant(df_drill, 'drill_cluster', "eagle_drill_labels_map", col_map, tfidf_matrix, feature_names, widget_key_prefix="eagle_drill_label")
            if "eagle_drill_labels_map_original" not in st.session_state:
                 st.session_state.eagle_drill_labels_map_original = drill_labels_map.copy()
            drill_label_widgets = utils.create_label_editor_ui(st.session_state.eagle_drill_labels_map_original, st.session_state.eagle_drill_labels_map, "eagle_drill_label")
            if st.button("サブクラスタ・ラベルを更新", key="eagle_drill_update_labels"):
                for cid, val in drill_label_widgets.items(): drill_labels_map[cid] = val
                df_drill['drill_cluster_label'] = df_drill['drill_cluster'].map(drill_labels_map)
                st.session_state.eagle_drilldown_result = update_drill_hover_text(df_drill)
                st.session_state.eagle_drill_labels_map = drill_labels_map
                st.rerun()

        # Word Cloud & Network
        with tab_drill_net:
            st.subheader("クラスタ・テキスト分析 (Text Mining)")
            col_tm1, col_tm2 = st.columns(2)
            with col_tm1:
                cooc_top_n = st.slider("共起: 上位単語数", 30, 100, 70, key="eagle_cooc_top_n")
                cooc_threshold = st.slider("共起: Jaccard係数 閾値", 0.01, 0.3, 0.03, 0.01, key="eagle_cooc_threshold")
            
            if st.button("テキスト分析を実行", key="eagle_run_text_mining"):
                with st.spinner("分析中..."):
                    all_text = ""
                    for _, row in df_drill.iterrows():
                        if col_map['title'] and pd.notna(row[col_map['title']]): all_text += row[col_map['title']] + " "
                        if col_map['abstract'] and pd.notna(row[col_map['abstract']]): all_text += row[col_map['abstract']] + " "
                    words = extract_compound_nouns(all_text, stopwords)
                    
                    if not words: st.warning("有効なキーワードなし")
                    else:
                        st.markdown("##### 1. ワードクラウド")
                        generate_wordcloud_and_list(words, f"クラスタ: {st.session_state.eagle_drill_base_label}", 30, FONT_PATH, capcom_key="eagle_drill")
                        
                        st.markdown("##### 2. 共起ネットワーク")
                        word_freq = Counter(words)
                        top_words = [w for w, c in word_freq.most_common(cooc_top_n)]
                        pair_counts = Counter()
                        for _, row in df_drill.iterrows():
                            dt = ""
                            if col_map['title']: dt += str(row[col_map['title']]) + " "
                            if col_map['abstract']: dt += str(row[col_map['abstract']]) + " "
                            dw = set(extract_compound_nouns(dt, stopwords))
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
                            utils.update_fig_layout(fig_net, '共起ネットワーク', show_axes=False)
                            fig_net.update_xaxes(visible=False); fig_net.update_yaxes(visible=False)
                            st.plotly_chart(fig_net, use_container_width=True)
                            
                            # AI Insight (Network)
                            net_nodes_list = [f"{n} ({G.nodes[n]['count']})" for n in G.nodes()]
                            net_edges_list = [f"{u} - {v} (J={d['weight']:.2f})" for u, v, d in G.edges(data=True)]
                            
                            # Extract Keyword-Centric Representatives for Insight
                            net_reps = utils.get_keyword_centric_representatives(df_drill, top_words, n_reps=10)
                            rep_lines_net = []
                            for i, r in enumerate(net_reps):
                                rep_lines_net.append(f"{i+1}. 【{r['title']}】 ({r['applicant']}) - {r['abstract'][:80]}...")

                            net_data_summary = {
                                "Total Nodes": G.number_of_nodes(),
                                "Total Edges": G.number_of_edges(),
                                "Top Words": ", ".join(net_nodes_list[:30]),
                                "Strongest Edges": ", ".join(sorted(net_edges_list, key=lambda x: float(x.split('J=')[1][:-1]), reverse=True)[:20]),
                                "Representative Patents (Keyword-Centric)": "\n".join(rep_lines_net)
                            }
                            net_context = f"""
                            **チャートタイプ**: 共起ネットワーク (テキストマイニング)
                            **対象データ**: クラスタ「{st.session_state.eagle_drill_base_label}」内の文書。
                            **手法**: 複合名詞のJaccard係数による共起分析。
                            **視覚的エンコーディング**:
                            - **ノード**: キーワード。サイズは出現頻度。
                            - **エッジ**: 共起関係。太さ/有無はJaccard係数 > {cooc_threshold} で定義。
                            **目的**: 技術用語同士の意味的なつながりや、複合技術の構造を理解すること。
                            """
                            net_role = "あなたはテキストマイニングの専門家です。キーワードの共起関係から技術的な文脈を読み解きます。"
                            net_inst = """
                            共起ネットワークの構造を分析してください：
                            1. **中核的な概念**: 中心にある、または最もつながりの多いキーワードは何ですか？
                            2. **技術の組み合わせ**: 強く結びついている単語のペア（エッジ）から、どのような技術要素が組み合わされているか推測してください。
                            3. **文脈**: このクラスタは具体的に何をする技術（What/How）に関するものだと考えられますか？
                            """
                            net_prompt = utils_ai.generate_ai_insight_prompt(net_role, net_context, net_data_summary, net_inst)
                            utils_ai.render_ai_insight_button(net_prompt, "eagle_net_insight")


        with tab_drill_stats:
            st.subheader("特許マップ（統計分析）")
            c1, c2 = st.columns(2)
            with c1:
                auto_min_year = 2000
                auto_max_year = datetime.datetime.now().year
                if 'year' in df_drill.columns:
                     try:
                         valid_years = df_drill['year'].dropna()
                         if not valid_years.empty:
                             auto_min_year, auto_max_year = int(valid_years.min()), int(valid_years.max())
                     except: pass
                s_year = st.number_input('開始年:', min_value=1900, max_value=2100, value=auto_min_year, key="eagle_stats_start_year", step=1)
                e_year = st.number_input('終了年:', min_value=1900, max_value=2100, value=auto_max_year, key="eagle_stats_end_year", step=1)
            with c2:
                n_apps = st.number_input('表示人数:', min_value=1, value=15, key="eagle_stats_num_assignees")
            
            if st.button("特許マップを描画", key="eagle_stats_run_button"):
                df_s = df_drill[(df_drill['year'] >= s_year) & (df_drill['year'] <= e_year)]
                if df_s.empty: st.warning("データなし")
                else:
                    yc = df_s['year'].value_counts().sort_index().reindex(range(s_year, e_year+1), fill_value=0)
                    fig1 = px.bar(x=yc.index, y=yc.values, labels={'x':'年', 'y':'件数'}, color_discrete_sequence=[utils.APOLLO_COLORS[0]])
                    utils.update_fig_layout(fig1, '出願推移', show_axes=True)
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    if 'applicant_main' in df_s.columns:
                        ac = df_s['applicant_main'].explode().value_counts().head(n_apps).sort_values(ascending=True)
                        fig2 = px.bar(x=ac.values, y=ac.index, orientation='h', labels={'x':'件数', 'y':'出願人'}, color_discrete_sequence=[utils.APOLLO_COLORS[1]])
                        utils.update_fig_layout(fig2, '出願人ランキング', height=max(600, len(ac)*30), show_axes=True)
                        st.plotly_chart(fig2, use_container_width=True)

        with tab_drill_export:
            st.subheader("データエクスポート")
            df_drill_export = df_drill.copy()
            
            # Ensure characteristic_words exists
            if 'characteristic_words' not in df_drill_export.columns and 'characteristic_words' in st.session_state.df_main.columns:
                 # Need to align by index
                 common_indices = df_drill_export.index.intersection(st.session_state.df_main.index)
                 df_drill_export.loc[common_indices, 'characteristic_words'] = st.session_state.df_main.loc[common_indices, 'characteristic_words']

            cols_drop_d = ['hover_text', 'parsed_date', 'date_bin', 'drill_hover_text', 'drill_date_bin', 'temp_date_bin']
            csv_d = df_drill_export.drop(columns=cols_drop_d, errors='ignore').to_csv(encoding='utf-8-sig', index=False).encode('utf-8-sig')
            st.download_button("ドリルダウン結果 (CSV)", csv_d, "EAGLE_Drilldown.csv", "text/csv")
