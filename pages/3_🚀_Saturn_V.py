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
import json
from sklearn.metrics.pairwise import euclidean_distances

# 機械学習・自然言語処理
from umap import UMAP 
import hdbscan
from sentence_transformers import SentenceTransformer

# 共通ユーティリティ
import utils
import utils_ai
import utils_spatial
from wordcloud import WordCloud
from janome.tokenizer import Tokenizer
import patiroha
import networkx as nx
from scipy.spatial import ConvexHull

# 描画用
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
utils.configure_matplotlib_font()


# 警告を非表示
warnings.filterwarnings('ignore')

# ==================================================================
# --- 1. ページ設定 ---
# ==================================================================
st.set_page_config(
    page_title="APOLLO v8 | Saturn V", 
    page_icon="🚀", 
    layout="wide"
)

st.session_state['current_page'] = 'Saturn V'

# ==================================================================
# --- 2. フォント設定 ---
# ==================================================================


FONT_PATH = utils.get_japanese_font_path()
if FONT_PATH:
    try:
        prop = fm.FontProperties(fname=FONT_PATH)
        plt.rcParams['font.family'] = prop.get_name()
    except:
        pass

# ==================================================================
# --- 3. 共通デザイン設定 (CSS) ---
# ==================================================================


# ==================================================================
# --- 4. デザインテーマ管理 ---
# ==================================================================





# ==================================================================
# --- 5. テキスト処理関数 ---
# ==================================================================

@st.cache_resource
def load_tokenizer_saturn():
    return Tokenizer()

t = load_tokenizer_saturn()

# ストップワード定義
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
    ("参照句","後述のとおり","literal",2), ("範囲表現", r"少なくとも(?:一|１)つ", "regex", 2), ("範囲表現", "少なくとも一部", "literal", 2),

    ("範囲表現", r"複数の(?:実施形態|構成|要素)", "regex", 3), ("課題句", r"(?:上記|前記)の?課題", "regex", 1),
    ("範囲表現", r"(?:以上|以下|未満|超|以内)", "regex", 2),
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
def extract_compound_nouns(text, stopwords_list):
    # 防御層: 異常入力と超長文で Janome の IndexError を避ける
    if not isinstance(text, str) or not text.strip():
        return []
    if len(text) > 8000:
        text = text[:8000]

    text = normalize_text(text)
    text = apply_ngram_filters(text)
    text = re.sub(r'【.*?】', '', text)
    text = re.sub(r'[!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', ' ', text)

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
                        "metadata": {"module": "Saturn V", "title": title, "top_n": top_n},
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
                    "module": "Saturn V",
                    "type": "wordcloud",
                    "title": title,
                    "top_words": [{"word": w, "freq": c} for w, c in word_freq.most_common(top_n)]
                }
            )
    except Exception as e:
        st.error(f"ワードクラウドの描画に失敗しました: {e}")

def get_top_tfidf_words(row_vector, feature_names, top_n=5):
    scores = row_vector.toarray().flatten() 
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

# ==================================================================
# --- 6. UI構成 ---
# ==================================================================

# --- サイドバー ---
utils.render_sidebar()

st.title("🚀 Saturn V")
st.markdown("特許テキストの意味的類似性から技術マップを自動生成。クラスタ構造・ノイズ（萌芽技術）・成長動態を一望します。")

# ==================================================================
# --- 7. データロード & 初期化 ---
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
    
if "saturnv_sbert_umap_done" not in st.session_state: st.session_state.saturnv_sbert_umap_done = False
if "saturnv_cluster_done" not in st.session_state: st.session_state.saturnv_cluster_done = False
if "saturnv_labels_map" not in st.session_state: st.session_state.saturnv_labels_map = {}
if "main_cluster_running" not in st.session_state: st.session_state.main_cluster_running = False
if "saturnv_global_zmax" not in st.session_state: st.session_state.saturnv_global_zmax = None

# ==================================================================
# --- 8. Saturn V アプリケーション ---
# ==================================================================

# --- 初回UMAP計算 ---
if not st.session_state.saturnv_sbert_umap_done:
    with st.spinner("Saturn V モジュール初回起動中: UMAPによる次元削減 (SBERTベース) を実行しています..."):
        try:
            reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
            embedding = reducer.fit_transform(sbert_embeddings) 
            st.session_state.df_main['umap_x'] = embedding[:, 0]
            st.session_state.df_main['umap_y'] = embedding[:, 1]
            st.session_state.df_main['characteristic_words'] = [get_top_tfidf_words(tfidf_matrix[i], feature_names) for i in range(tfidf_matrix.shape[0])]
            
            try:
                H, _, _ = np.histogram2d(embedding[:, 0], embedding[:, 1], bins=50)
                st.session_state.saturnv_global_zmax = H.max()
            except:
                st.session_state.saturnv_global_zmax = None
            
            st.session_state.saturnv_sbert_umap_done = True
            st.success("UMAPの初期計算が完了しました。")
            st.rerun()
        except Exception as e:
            st.error(f"UMAPの初期計算中にエラーが発生しました: {e}")
            st.stop()

# --- メインUI ---
tab_main, tab_drill, tab_stats, tab_export = st.tabs([
    "Landscape Map (TELESCOPE)", 
    "Drilldown (PROBE)", 
    "特許マップ (統計分析)", 
    "Data Export"
])

# --- TELESCOPE ---
with tab_main:
    st.subheader("クラスタリング実行")
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1: min_cluster_size_w = st.number_input("最小クラスタサイズ (推奨: 10-50):", min_value=2, value=15, key="main_min_cluster_size")
    with col2: min_samples_w = st.number_input("最小サンプル数 (推奨: 5-20):", min_value=1, value=10, key="main_min_samples")
    with col3: label_top_n_w = st.number_input("クラスタラベル単語数:", min_value=1, value=3, key="main_label_top_n")
    
    if st.button("描画 (再計算)", type="primary", key="main_run_cluster", disabled=st.session_state.main_cluster_running):
        st.session_state.main_cluster_running = True
        with st.spinner("HDBSCANクラスタリングを実行中..."):
            try:
                embedding = st.session_state.df_main[['umap_x', 'umap_y']].values
                clusterer = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size_w), min_samples=int(min_samples_w), metric='euclidean', cluster_selection_method='eom')
                clustering = clusterer.fit(embedding)
                st.session_state.df_main['cluster'] = clustering.labels_
                
                # patiroha.auto_label で c-TF-IDF ラベリング
                label_top_n = int(label_top_n_w)
                texts_for_label = (
                    st.session_state.df_main[col_map['title']].fillna('') + ' ' +
                    st.session_state.df_main[col_map['abstract']].fillna('')
                )
                labels_map = patiroha.auto_label(
                    texts_for_label,
                    st.session_state.df_main['cluster'].values,
                    method='c-tfidf',
                    top_n=label_top_n,
                )
                
                st.session_state.df_main['cluster_label'] = st.session_state.df_main['cluster'].map(labels_map)
                st.session_state.saturnv_labels_map = labels_map.copy()
                st.session_state.saturnv_labels_map_original = labels_map.copy()
                st.session_state.df_main = update_hover_text(st.session_state.df_main, col_map)
                st.session_state.saturnv_cluster_done = True
                # CAPCOM: patents.csvにクラスタ情報を追加更新
                try:
                    import capcom
                    if capcom.is_active():
                        capcom.save_patents_csv()
                except Exception:
                    pass
                st.success("クラスタリング完了")
                st.rerun()
            except Exception as e:
                st.error(f"エラー: {e}")
            finally:
                st.session_state.main_cluster_running = False

    st.markdown("---")
    
    if st.session_state.saturnv_cluster_done:
        st.subheader("フィルタリング設定 (メイン用)")
        def on_main_interval_change():
            if "main_date_filter" in st.session_state: del st.session_state.main_date_filter

        col1, col2 = st.columns(2)
        with col1:
            if 'year' in df_main.columns and df_main['year'].notna().any():
                bin_interval_w_val = st.selectbox("期間の粒度:", [5, 3, 2, 1], index=0, key="main_bin_interval", on_change=on_main_interval_change)
                date_bin_options = get_date_bin_options(df_main, int(bin_interval_w_val), 'year')
                date_bin_filter_w = st.selectbox("表示期間:", date_bin_options, key="main_date_filter")
            else:
                date_bin_filter_w = "(全期間)"
        
        with col2:
            if 'applicant_main' in st.session_state.df_main.columns:
                applicants = st.session_state.df_main['applicant_main'].explode().dropna()
            elif col_map['applicant'] and col_map['applicant'] in st.session_state.df_main.columns:
                applicants = st.session_state.df_main[col_map['applicant']].fillna('').str.split(delimiters['applicant']).explode().str.strip()
            else:
                applicants = pd.Series([])

            if not applicants.empty:
                applicant_counts = applicants.value_counts()
                unique_applicants = applicant_counts.index.tolist()
                applicant_options = [(f"(全出願人) ({len(st.session_state.df_main)}件)", "ALL")] + \
                                    [(f"{app} ({applicant_counts[app]}件)", app) for app in unique_applicants]
                
                applicant_filter_w = st.multiselect(
                    "出願人:", 
                    applicant_options, 
                    default=[applicant_options[0]], 
                    format_func=lambda x: x[0], 
                    key="main_applicant_filter"
                )
            else:
                applicant_filter_w = [(f"(全出願人) ({len(st.session_state.df_main)}件)", "ALL")]

        cluster_counts = st.session_state.df_main['cluster_label'].value_counts()
        cluster_options = [(f"(全クラスタ) ({len(st.session_state.df_main)}件)", "ALL")] + [
            (f"{st.session_state.saturnv_labels_map.get(cid)} ({cluster_counts.get(st.session_state.saturnv_labels_map.get(cid), 0)}件)", cid)
            for cid in sorted(st.session_state.df_main['cluster'].unique())
        ]
        cluster_filter_w = st.multiselect("マップ表示クラスタ:", cluster_options, default=[cluster_options[0]], format_func=lambda x: x[0], key="main_cluster_filter")

        st.subheader("分析結果 (TELESCOPE メインマップ)")
        
        # --- UIレイアウト ---
        map_mode = st.radio("表示モード:", ["散布図 (Scatter)", "密度マップ (Density)", "クラスタ領域 (Clusters)"], horizontal=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**密度マップ設定**")
            main_mesh_size = st.number_input("メッシュサイズ (Grid)", value=30, min_value=10, max_value=200, step=5, key="main_mesh_size")
            use_abs_scale = False
            if map_mode == "密度マップ (Density)":
                use_abs_scale = st.checkbox("密度スケールを固定 (絶対評価)", value=False, key="main_abs_scale")
        with c2:
            st.markdown("**フィルタ**")
            remove_noise_chk = st.checkbox("ノイズを除く (Exclude Noise)", value=False, key="main_remove_noise")
        with c3:
            st.markdown("**表示オプション**")
            show_labels_chk = st.checkbox("マップにラベルを表示する", value=True, key="main_show_labels")
        
        # --- ハイブリッド・レイヤー構造 ---
        
        # 1. Universe (第1層: 背景/Ghost用)
        df_universe = st.session_state.df_main.copy()
        if remove_noise_chk:
            df_universe = df_universe[df_universe['cluster'] != -1]

        # 2. Trend (第2層: 地形用)
        df_trend = df_universe.copy()
        if not date_bin_filter_w.startswith("(全期間)"):
            try:
                date_bin_label = date_bin_filter_w.split(' (')[0].strip()
                start_year, end_year = map(int, date_bin_label.split('-'))
                df_trend = df_trend[(df_trend['year'] >= start_year) & (df_trend['year'] <= end_year)]
            except: pass

        # 3. Focus (第3層: 注目用)
        df_focus = df_trend.copy()
        
        # 出願人フィルタ
        applicant_values = [val[1] for val in applicant_filter_w]
        if "ALL" not in applicant_values:
             mask_list = [df_focus[col_map['applicant']].fillna('').str.contains(re.escape(app)) for app in applicant_values]
             if mask_list:
                 df_focus = df_focus[pd.concat(mask_list, axis=1).any(axis=1)]
             else:
                 df_focus = df_focus.iloc[0:0]

        # クラスタフィルタ
        cluster_values = [val[1] for val in cluster_filter_w]
        if "ALL" not in cluster_values:
            df_focus = df_focus[df_focus['cluster'].isin(cluster_values)]

        # 4. Ghost (Universe - Focus)
        try:
            df_ghost = df_universe.drop(df_focus.index, errors='ignore')
        except:
            df_ghost = pd.DataFrame()

        # --- 描画ロジック ---
        fig_main = go.Figure()
        
        # 密度マップ
        if not df_trend.empty and map_mode == "密度マップ (Density)":
            custom_density_colorscale = [
                [0.0, "rgba(255, 255, 255, 0)"], 
                [0.1, "rgba(225, 245, 254, 0.3)"],
                [0.4, "rgba(129, 212, 250, 0.6)"],
                [1.0, "rgba(2, 119, 189, 0.9)"]
            ]
            
            contour_params = dict(
                x=df_trend['umap_x'], y=df_trend['umap_y'], 
                colorscale=custom_density_colorscale, 
                reversescale=False, xaxis='x', yaxis='y', 
                showscale=False, name="密度", 
                nbinsx=main_mesh_size, nbinsy=main_mesh_size,
                contours=dict(coloring='fill', showlines=True),
                line=dict(width=0.5, color='rgba(0, 0, 0, 0.2)')
            )
            if use_abs_scale and st.session_state.saturnv_global_zmax:
                contour_params.update(dict(zauto=False, zmin=0, zmax=st.session_state.saturnv_global_zmax))
            else: 
                contour_params.update(dict(zauto=True))
            
            fig_main.add_trace(go.Histogram2dContour(**contour_params))

        # クラスタ領域
        if map_mode == "クラスタ領域 (Clusters)" and not df_universe.empty:
            unique_clusters = sorted(df_universe['cluster'].unique())
            color_sequence = utils.APOLLO_COLORS
            for i, cid in enumerate(unique_clusters):
                if cid == -1: continue
                points = df_universe[df_universe['cluster'] == cid][['umap_x', 'umap_y']].values
                if len(points) >= 3:
                    try:
                        hull = ConvexHull(points)
                        hull_points = points[hull.vertices]
                        hull_points = np.append(hull_points, [hull_points[0]], axis=0)
                        cluster_color = color_sequence[i % len(color_sequence)]
                        fig_main.add_trace(go.Scatter(
                            x=hull_points[:, 0], y=hull_points[:, 1], mode='lines', fill='toself',
                            fillcolor=cluster_color, opacity=0.1, line=dict(color=cluster_color, width=2),
                            hoverinfo='skip', showlegend=False
                        ))
                    except: pass

        # Ghost (Universe背景)
        if not df_ghost.empty:
            ghost_color = '#dddddd'
            ghost_opacity = 0.4
            fig_main.add_trace(go.Scatter(
                x=df_ghost['umap_x'], y=df_ghost['umap_y'], mode='markers', 
                marker=dict(color=ghost_color, size=3, opacity=ghost_opacity, line=dict(width=0)), 
                hoverinfo='skip', name='その他 (Ghost)'
            ))

        # Focus (注目)
        if not df_focus.empty:
            marker_line = dict(width=1, color='white') if map_mode == "密度マップ (Density)" else dict(width=0)
            is_applicant_filtered = "ALL" not in applicant_values
            
            if is_applicant_filtered:
                # Applicant Drill Down Mode
                palette = px.colors.qualitative.Bold
                for i, app_name in enumerate(applicant_values):
                    mask = df_focus[col_map['applicant']].fillna('').str.contains(re.escape(app_name))
                    df_app = df_focus[mask]
                    if not df_app.empty:
                        fig_main.add_trace(go.Scatter(
                            x=df_app['umap_x'], y=df_app['umap_y'], mode='markers',
                            marker=dict(color=palette[i % len(palette)], size=6, opacity=0.9, line=marker_line),
                            hoverinfo='text', hovertext=df_app['hover_text'], name=app_name
                        ))
            else:
                # Standard Mode (Cluster Coloring)

                if 'cluster' in df_focus.columns:
                    df_focus_valid = df_focus[df_focus['cluster'] != -1]
                    df_focus_noise = df_focus[df_focus['cluster'] == -1]
                else:
                    df_focus_valid = df_focus
                    df_focus_noise = pd.DataFrame()

                # Plot Valid Clusters
                if not df_focus_valid.empty:
                    fig_main.add_trace(go.Scatter(
                        x=df_focus_valid['umap_x'], y=df_focus_valid['umap_y'], mode='markers', 
                        marker=dict(
                            color=df_focus_valid['cluster'], 
                            colorscale=utils.APOLLO_COLORS, 
                            showscale=False, 
                            size=5, 
                            opacity=0.8, 
                            line=marker_line
                        ), 
                        hoverinfo='text', hovertext=df_focus_valid['hover_text'], name='特許 (Valid)'
                    ))
                
                # Plot Noise (Separate Trace)
                if not df_focus_noise.empty:
                    fig_main.add_trace(go.Scatter(
                        x=df_focus_noise['umap_x'], y=df_focus_noise['umap_y'], mode='markers', 
                        marker=dict(
                            color='#999999', 
                            size=3, 
                            opacity=0.3, 
                            line=dict(width=0)
                        ), 
                        hoverinfo='text', hovertext=df_focus_noise['hover_text'], name='Noise'
                    ))

        # ラベル追加
        if show_labels_chk:
            label_data_source = df_universe
            target_cids = cluster_values if "ALL" not in cluster_values else label_data_source['cluster'].unique()
            color_sequence = utils.APOLLO_COLORS
            sorted_unique_cids = sorted(df_universe['cluster'].unique()) 
            
            # Filter Noise from labels
            valid_label_data = label_data_source[label_data_source['cluster'] != -1]
            
            for cid, grp in valid_label_data[valid_label_data['cluster'].isin(target_cids)].groupby('cluster'):
                mean_pos = grp[['umap_x', 'umap_y']].mean()
                label_txt = grp['cluster_label'].iloc[0]
                try:
                    color_idx = sorted_unique_cids.index(cid)
                    border_color = color_sequence[color_idx % len(color_sequence)]
                except: border_color = "#333333"

                fig_main.add_annotation(
                    x=mean_pos['umap_x'], y=mean_pos['umap_y'], 
                    text=label_txt, showarrow=False, 
                    font=dict(size=11, color='black', family="Helvetica"), 
                    bgcolor='rgba(255,255,255,0.8)', bordercolor=border_color, borderwidth=2, borderpad=4
                )

        norm_msg = " (絶対評価)" if use_abs_scale and map_mode == "密度マップ (Density)" else ""
        utils.update_fig_layout(fig_main, f"Saturn V - メインマップ{norm_msg}", height=1200, show_legend=False)
        

        # 1. アスペクト比: 歪みを防ぐため1:1を強制
        # 2. フォーカス: 有効なクラスタにズーム（ノイズ除外）
        
        if not df_focus.empty and 'cluster' in df_focus.columns:
             # Use only VALID clusters for bounds calculation
             target_df = df_focus[df_focus['cluster'] != -1]
        else:
             target_df = df_focus if not df_focus.empty else df_universe
             if 'cluster' in target_df.columns:
                 target_df = target_df[target_df['cluster'] != -1]

        if not target_df.empty:
             # Calculate bounds
             x_min, x_max = target_df['umap_x'].min(), target_df['umap_x'].max()
             y_min, y_max = target_df['umap_y'].min(), target_df['umap_y'].max()
             
             # Calculate spread
             x_range = x_max - x_min
             y_range = y_max - y_min
             
             # Add Padding (2%)
             pad_factor = 0.02
             x_pad = x_range * pad_factor if x_range > 0 else 1.0
             y_pad = y_range * pad_factor if y_range > 0 else 1.0
             
             # Apply new ranges with Fixed Aspect Ratio matching
             fig_main.update_layout(
                height=1200,
                xaxis=dict(range=[x_min - x_pad, x_max + x_pad], autorange=False),
                yaxis=dict(
                    range=[y_min - y_pad, y_max + y_pad], 
                    autorange=False,
                    scaleanchor="x", 
                    scaleratio=1
                )
             )

        st.plotly_chart(fig_main, use_container_width=True, config={
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

        # スナップショットボタン
        # 安全なサマリーを作成
        summary_cols = ['cluster_label']
        if 'year' in df_universe.columns: summary_cols.append('year')
        if col_map.get('title') and col_map['title'] in df_universe.columns:
            summary_cols.append(col_map['title'])
        if col_map.get('applicant') and col_map['applicant'] in df_universe.columns:
            summary_cols.append(col_map['applicant'])
            

        df_summary_source = df_universe[df_universe['cluster'] != -1] if 'cluster' in df_universe.columns else df_universe
            
        snap_data = utils.generate_rich_summary(df_summary_source, title_col=col_map['title'], abstract_col=col_map['abstract'])
        snap_data['module'] = 'Saturn V'
        
        # チャートデータの最適化（トークンオーバーフロー防止）
        df_snap_safe = df_summary_source[summary_cols].head(30).copy()
        # テキストの切り捨て
        for c in summary_cols:
            if df_snap_safe[c].dtype == object:
                df_snap_safe[c] = df_snap_safe[c].astype(str).str.slice(0, 50) + "..."
        
        snap_data['chart_data'] = df_snap_safe.to_string(index=False)
        

        try:
             cluster_counts_snap = df_universe['cluster'].value_counts()
             cluster_summary_lines = []
             
             # クラスタごとの代表特許を抽出
             cluster_reps = utils.get_cluster_representatives(df_universe, cluster_col='cluster', n_reps=3)

             for cid in sorted(df_universe['cluster'].unique()):
                 if cid == -1: continue
                 label = st.session_state.saturnv_labels_map.get(cid, f"Cluster {cid}")
                 count = cluster_counts_snap.get(cid, 0)
                 cluster_summary_lines.append(f"- {label} ({count}件)")
                 
                # 代表特許を追加
                 if cid in cluster_reps:
                     for rep in cluster_reps[cid]:
                         cluster_summary_lines.append(rep)

             snap_data['cluster_summary'] = "全クラスタ構成（上位から）:\n" + "\n".join(cluster_summary_lines)
        except: pass
        
        # AI Insight コンテキストの準備（スナップショット生成前）
        if st.session_state.saturnv_cluster_done:
            insight_context = f"""
            **マップタイプ**: 技術ランドスケープ (Saturn V - Telescope)
            **分析対象**: 全体俯瞰マップ。
            **手法**: SBERT (文章ベクトル化) + UMAP (次元圧縮) + HDBSCAN (クラスタリング)。
            **視覚的エンコーディング**:
            - **点**: 個々の特許/文献。距離が近いほど意味的に類似しています。
            - **クラスタ**: 色分けされたグループは、自動検出された技術領域を表します。
            - **配置**: マップ全体の「形状」が技術空間の広がりを表します。
            **目的**: マクロな視点で技術全体の構造を把握し、主要なテーマ（クラスタ）や未開拓領域（空白地帯）を発見すること。
            """
            
            insight_role = "あなたはシニア特許アナリストです。技術俯瞰図から戦略的な示唆を導きます。"
            
            insight_instruction = """
            ランドスケープの構造を分析してください：
            1. **主要テーマ**: どのような技術クラスタが形成されていますか？（ラベル参照）
            2. **技術の関係性**: どのクラスタとどのクラスタが近接していますか？そこから読み取れる技術的なシナジーや関連性は？
            3. **注目領域**: ユーザーの関心（フィルタ結果など）に基づき、特に注目すべき領域はどこですか？
            **重要**: 回答は箇条書きで、技術的な洞察を深掘りしてください。
            """

            # 空間情報の計算
            spatial_info = utils_spatial.generate_spatial_cluster_summary(
                df_universe, 'cluster', 'umap_x', 'umap_y', label_map=st.session_state.saturnv_labels_map
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

        # スナップショットボタン
        try:
            fig_main_snap = fig_main 
            utils.render_snapshot_button(
                title=f"Saturn V: Landscape Map ({date_bin_filter_w})",
                description="Global technology landscape map showing cluster distribution and proximity based on semantic similarity.",
                fig=fig_main_snap,
                data_summary=snap_data,
                key="saturn_main_snap"
            )
        except Exception as e:
             st.error(f"スナップショット生成エラー: {e}")

        if st.session_state.saturnv_cluster_done:
            # AIインサイト (メインマップ)
            insight_context = f"""
            **マップタイプ**: 技術ランドスケープ (Saturn V - Telescope)
            **分析対象**: 全体俯瞰マップ。
            **手法**: SBERT (文章ベクトル化) + UMAP (次元圧縮) + HDBSCAN (クラスタリング)。
            **視覚的エンコーディング**:
            - **点**: 個々の特許/文献。距離が近いほど意味的に類似しています。
            - **クラスタ**: 色分けされたグループは、自動検出された技術領域を表します。
            - **配置**: マップ全体の「形状」が技術空間の広がりを表します。
            **目的**: マクロな視点で技術全体の構造を把握し、主要なテーマ（クラスタ）や未開拓領域（空白地帯）を発見すること。
            """
            
            insight_role = "あなたはシニア特許アナリストです。技術俯瞰図から戦略的な示唆を導きます。"
            
            insight_instruction = """
            ランドスケープの構造を分析してください：
            1. **主要テーマ**: どのような技術クラスタが形成されていますか？（ラベル参照）
            2. **技術の関係性**: どのクラスタとどのクラスタが近接していますか？そこから読み取れる技術的なシナジーや関連性は？
            3. **注目領域**: ユーザーの関心（フィルタ結果など）に基づき、特に注目すべき領域はどこですか？
            **重要**: 回答は箇条書きで、技術的な洞察を深掘りしてください。
            """

            # 空間情報の計算
            spatial_info = utils_spatial.generate_spatial_cluster_summary(
                df_universe, 'cluster', 'umap_x', 'umap_y', label_map=st.session_state.saturnv_labels_map
            )

            # 生成 (空間情報を追加コンテキストとして渡す)
            prompt = utils_ai.generate_ai_insight_prompt(
                insight_role, insight_context, snap_data, insight_instruction, 
                extra_content=f"\n# 空間配置情報 (Spatial Context)\n{spatial_info}"
            )
            
            utils_ai.render_ai_insight_button(prompt, "saturn_main_insight")

            # CAPCOM data/ JSON出力（Saturn V TELESCOPEクラスタ）
            try:
                import capcom
                if capcom.is_active():
                    clusters_json = []
                    for cid in sorted(df_universe['cluster'].unique()):
                        if cid == -1:
                            continue
                        label = st.session_state.saturnv_labels_map.get(cid, f"Cluster {cid}")
                        auto_label = st.session_state.get('saturnv_labels_map_original', {}).get(cid, label)
                        count = int(cluster_counts_snap.get(cid, 0))
                        # 重心座標
                        cid_mask = df_universe['cluster'] == cid
                        cx = float(df_universe.loc[cid_mask, 'umap_x'].mean()) if cid_mask.any() else 0
                        cy = float(df_universe.loc[cid_mask, 'umap_y'].mean()) if cid_mask.any() else 0
                        # TF-IDF上位語
                        tfidf_terms = []
                        try:
                            cid_indices = df_universe[cid_mask].index.tolist()
                            valid_idx = [i for i in cid_indices if i < tfidf_matrix.shape[0]]
                            if valid_idx:
                                mean_tfidf = tfidf_matrix[valid_idx].mean(axis=0).A1
                                top_idx = mean_tfidf.argsort()[::-1][:10]
                                tfidf_terms = [feature_names[i] for i in top_idx]
                        except:
                            pass

                        # 代表特許（各エントリ200文字で截断）
                        reps_raw = []
                        if cid in cluster_reps:
                            for rep_str in cluster_reps[cid]:
                                reps_raw.append(rep_str[:200] if len(rep_str) > 200 else rep_str)

                        clusters_json.append({
                            "cluster_id": int(cid),
                            "label": label,
                            "auto_label": auto_label,
                            "count": count,
                            "centroid": [round(cx, 4), round(cy, 4)],
                            "tfidf_top_terms": tfidf_terms,
                            "representative_patents": reps_raw
                        })

                    noise_count = int((df_universe['cluster'] == -1).sum()) if -1 in df_universe['cluster'].values else 0

                    # ノイズ特許のデータ抽出（上限200件）
                    noise_patents = []
                    if noise_count > 0:
                        df_noise = df_universe[df_universe['cluster'] == -1]
                        for _, row in df_noise.head(200).iterrows():
                            noise_entry = {
                                "umap_x": round(float(row['umap_x']), 4),
                                "umap_y": round(float(row['umap_y']), 4),
                            }
                            title_col = col_map.get('title')
                            if title_col and title_col in row.index and pd.notna(row[title_col]):
                                noise_entry["title"] = str(row[title_col])[:100]
                            if 'applicant_main' in row.index:
                                val = row['applicant_main']
                                noise_entry["applicant"] = val[0] if isinstance(val, list) and val else str(val)[:50]
                            if 'year' in row.index and pd.notna(row['year']):
                                noise_entry["year"] = int(row['year'])
                            noise_patents.append(noise_entry)

                    # ノイズ率解釈
                    _noise_ratio = round(noise_count / len(df_universe), 4) if len(df_universe) > 0 else 0
                    if _noise_ratio < 0.05:
                        _noise_interp = "成熟・均質な技術領域（ノイズ率 < 5%）"
                    elif _noise_ratio < 0.15:
                        _noise_interp = "標準的・安定構造（ノイズ率 5-15%）"
                    elif _noise_ratio < 0.30:
                        _noise_interp = "多様・融合活発（ノイズ率 15-30%）"
                    else:
                        _noise_interp = "萌芽・黎明期（ノイズ率 > 30%）"

                    # ノイズ時系列分布
                    _noise_year_dist = {}
                    _noise_temporal_pattern = ""
                    if noise_count > 0 and 'year' in df_universe.columns:
                        _df_noise_capcom = df_universe[df_universe['cluster'] == -1]
                        _year_counts = _df_noise_capcom['year'].dropna().value_counts().sort_index()
                        _noise_year_dist = {str(int(k)): int(v) for k, v in _year_counts.items()}
                        if not _year_counts.empty:
                            _recent_years = _year_counts.index.max() - 3
                            _recent_ratio = _year_counts[_year_counts.index > _recent_years].sum() / noise_count
                            if _recent_ratio > 0.6:
                                _noise_temporal_pattern = "近年集中（新興テーマの可能性）"
                            elif _recent_ratio < 0.2:
                                _noise_temporal_pattern = "過去集中（歴史的バリエーション）"
                            else:
                                _noise_temporal_pattern = "均一分布（永続的ニッチ）"

                    # ノイズ出願人分布
                    _noise_top_applicants = []
                    if noise_count > 0 and 'applicant_main' in df_universe.columns:
                        _df_noise_capcom = df_universe[df_universe['cluster'] == -1]
                        _all_apps = []
                        for _apps in _df_noise_capcom['applicant_main']:
                            if isinstance(_apps, list):
                                _all_apps.extend(_apps)
                            elif isinstance(_apps, str):
                                _all_apps.append(_apps)
                        if _all_apps:
                            _app_counter = Counter(_all_apps)
                            _noise_top_applicants = [{"applicant": a, "count": c} for a, c in _app_counter.most_common(10)]

                    # ノイズ萌芽キーワード
                    _noise_keywords = []
                    if noise_count >= 5:
                        try:
                            _title_col = col_map.get('title', '')
                            _abstract_col = col_map.get('abstract', '')
                            if _title_col and _title_col in df_universe.columns:
                                _df_noise_capcom = df_universe[df_universe['cluster'] == -1]
                                _noise_texts = (_df_noise_capcom[_title_col].fillna('') + ' ' +
                                               _df_noise_capcom.get(_abstract_col, pd.Series([''] * len(_df_noise_capcom))).fillna(''))
                                _noise_kws = _noise_texts.apply(utils.extract_keywords)
                                _all_kws = []
                                for _kw_list in _noise_kws:
                                    _all_kws.extend(_kw_list)
                                _kw_freq = Counter(_all_kws).most_common(20)
                                _noise_keywords = [{"keyword": k, "frequency": f} for k, f in _kw_freq]
                        except Exception:
                            pass

                    saturnv_json = {
                        "metadata": {
                            "module": "Saturn V",
                            "mode": "TELESCOPE",
                            "n_clusters": len(clusters_json),
                            "noise_count": noise_count,
                            "noise_ratio": _noise_ratio,
                            "total_patents": len(df_universe)
                        },
                        "clusters": clusters_json,
                        "noise_patents": noise_patents,
                        "noise_analysis": {
                            "noise_count": noise_count,
                            "noise_ratio": _noise_ratio,
                            "noise_interpretation": _noise_interp,
                            "temporal_pattern": _noise_temporal_pattern,
                            "year_distribution": _noise_year_dist,
                            "top_applicants": _noise_top_applicants,
                            "emerging_keywords": _noise_keywords,
                        },
                        "spatial_context": spatial_info if 'spatial_info' in dir() else ""
                    }
                    # クラスタ動態データを追加（前回実行時に session_state に保存されたものを使用）
                    if 'saturnv_dynamics_data' in st.session_state:
                        saturnv_json['cluster_dynamics'] = st.session_state['saturnv_dynamics_data']
                    capcom.save_data("saturnv_clusters.json", saturnv_json)
            except Exception as e:
                pass

        st.subheader("ラベル編集")
        utils.render_ai_label_assistant(st.session_state.df_main, 'cluster', "saturnv_labels_map", col_map, tfidf_matrix, feature_names, widget_key_prefix="main_label")



        if "saturnv_labels_map_original" not in st.session_state: st.session_state.saturnv_labels_map_original = st.session_state.saturnv_labels_map.copy()
        st.session_state.saturnv_labels_map_custom = utils.create_label_editor_ui(st.session_state.saturnv_labels_map_original, st.session_state.saturnv_labels_map, "main_label")
        if st.button("ラベルを更新", key="main_update_labels"):
            st.session_state.df_main['cluster_label'] = st.session_state.df_main['cluster'].map(st.session_state.saturnv_labels_map_custom)
            st.session_state.df_main = update_hover_text(st.session_state.df_main, col_map)
            st.session_state.saturnv_labels_map = st.session_state.saturnv_labels_map_custom
            st.rerun()

        # --- ノイズ分析セクション ---
        df_main = st.session_state.df_main
        if df_main is not None and 'cluster' in df_main.columns:
            noise_mask = df_main['cluster'] == -1
            noise_count = noise_mask.sum()
            total_count = len(df_main)
            noise_ratio = noise_count / total_count if total_count > 0 else 0

            with st.expander(f"🔍 ノイズ分析 ({noise_count}件 / {noise_ratio:.1%})", expanded=False):
                # 1. ノイズ率の解釈
                if noise_ratio < 0.05:
                    noise_interp = "成熟・均質な技術領域（ノイズ率 < 5%）"
                elif noise_ratio < 0.15:
                    noise_interp = "標準的・安定構造（ノイズ率 5-15%）"
                elif noise_ratio < 0.30:
                    noise_interp = "多様・融合活発（ノイズ率 15-30%）"
                else:
                    noise_interp = "萌芽・黎明期（ノイズ率 > 30%）"

                st.info(f"**ノイズ率解釈**: {noise_interp}")

                if noise_count > 0:
                    df_noise = df_main[noise_mask].copy()

                    # 2. 時系列分析
                    st.markdown("##### 時系列分布")
                    if 'year' in df_noise.columns:
                        year_counts = df_noise['year'].dropna().value_counts().sort_index()
                        if not year_counts.empty:
                            recent_years = year_counts.index.max() - 3
                            recent_ratio = year_counts[year_counts.index > recent_years].sum() / noise_count
                            if recent_ratio > 0.6:
                                st.markdown("📈 **近年集中**: ノイズの多くが直近3年に集中 → **新興テーマの可能性**")
                            elif recent_ratio < 0.2:
                                st.markdown("📊 **過去集中**: ノイズの多くが過去に分布 → **歴史的バリエーション**")
                            else:
                                st.markdown("📉 **均一分布**: ノイズが期間全体に分布 → **永続的ニッチ**")

                            import plotly.express as px
                            fig_noise_year = px.bar(x=year_counts.index, y=year_counts.values,
                                                   labels={'x': '年', 'y': 'ノイズ件数'})
                            fig_noise_year.update_layout(height=300, title='ノイズ特許の年別分布')
                            st.plotly_chart(fig_noise_year, use_container_width=True)

                    # 3. 出願人分析
                    st.markdown("##### 出願人分布")
                    if 'applicant_main' in df_noise.columns:
                        all_applicants = []
                        for apps in df_noise['applicant_main']:
                            if isinstance(apps, list):
                                all_applicants.extend(apps)
                            elif isinstance(apps, str):
                                all_applicants.append(apps)
                        if all_applicants:
                            from collections import Counter
                            app_counts = Counter(all_applicants)
                            top_apps = app_counts.most_common(10)
                            top1_share = top_apps[0][1] / noise_count if top_apps else 0
                            if top1_share > 0.3:
                                st.markdown(f"🎯 **集中**: 上位出願人 '{top_apps[0][0]}' がノイズの{top1_share:.0%}を占める → **戦略的ニッチ**")
                            else:
                                st.markdown("🌐 **分散**: 多数の出願人にノイズが分布 → **標準化前段階**")

                            st.dataframe(
                                pd.DataFrame(top_apps, columns=['出願人', 'ノイズ件数']),
                                use_container_width=True, hide_index=True
                            )

                    # 4. 萌芽テーマ抽出（TF-IDFキーワード）
                    st.markdown("##### 萌芽テーマ候補")
                    if noise_count >= 5:
                        col_map = st.session_state.get('col_map', {})
                        title_col = col_map.get('title', '')
                        abstract_col = col_map.get('abstract', '')
                        if title_col and title_col in df_noise.columns:
                            noise_texts = (df_noise[title_col].fillna('') + ' ' +
                                          df_noise.get(abstract_col, pd.Series([''] * len(df_noise))).fillna(''))
                            noise_kws = noise_texts.apply(utils.extract_keywords)

                            from collections import Counter
                            all_kws = []
                            for kw_list in noise_kws:
                                all_kws.extend(kw_list)
                            kw_freq = Counter(all_kws).most_common(20)

                            if kw_freq:
                                st.markdown("ノイズ特許から抽出した頻出キーワード（萌芽技術の候補）:")
                                kw_df = pd.DataFrame(kw_freq, columns=['キーワード', '出現頻度'])
                                st.dataframe(kw_df, use_container_width=True, hide_index=True)

                    # CAPCOM出力
                    try:
                        import capcom
                        if capcom.is_active():
                            noise_data = {
                                'noise_count': int(noise_count),
                                'noise_ratio': round(noise_ratio, 4),
                                'noise_interpretation': noise_interp,
                            }
                            # existing saturnv data will be updated elsewhere
                            st.caption("📡 ノイズ分析データはCAPCOMに蓄積されます")
                    except Exception:
                        pass

        # --- クラスタ動態マップ (Saturn V TELESCOPE) ---
        if df_main is not None and 'cluster' in df_main.columns and 'year' in df_main.columns:
            labels_map = st.session_state.get('saturnv_labels_map', {})
            if labels_map and df_main['cluster'].nunique() > 1:
                # 戻り値 dyn_data を取得して session_state に保存（CAPCOM JSON 出力用）
                dyn_data = utils.render_cluster_dynamics_section(
                    df_main, 'cluster', labels_map,
                    year_col='year', cagr_window=5,
                    unique_key='saturnv_dynamics',
                    module_name='Saturn V',
                )
                if dyn_data:
                    st.session_state['saturnv_dynamics_data'] = dyn_data

    # --- PROBE (ドリルダウン) ---
    with tab_drill:
        st.subheader("分析対象クラスタの選択")
        drilldown_options = [('(選択してください)', 'NONE')]
        if "saturnv_labels_map" in st.session_state:
            drilldown_options += [(f"{label} ({count}件)", cid) for cid, label in st.session_state.saturnv_labels_map.items() if cid != -1 for count in [st.session_state.df_main['cluster'].value_counts().get(cid, 0)]]
        
        selected_drilldown_target_drill = st.selectbox("分析対象クラスタ:", options=drilldown_options, format_func=lambda x: x[0], key="drill_target_select")
        drilldown_target_id = selected_drilldown_target_drill[1] 

        st.subheader("フィルタリング設定 (ドリルダウン用)")
        if drilldown_target_id == "NONE":
            df_subset_filter = pd.DataFrame(columns=df_main.columns)
            st.info("👆 上のメニューで「分析対象クラスタ」を選択すると、フィルタメニューが表示されます。")
        else:
            df_subset_filter = df_main[df_main['cluster'] == drilldown_target_id].copy()
            
        def on_drill_interval_change():
            if "drill_date_filter_w" in st.session_state: del st.session_state.drill_date_filter_w
            
        if drilldown_target_id != "NONE":
            col1, col2 = st.columns(2)
            with col1:
                if 'year' in df_subset_filter.columns and df_subset_filter['year'].notna().any():
                    drill_bin_interval_w_val = st.selectbox("期間の粒度:", [5, 3, 2, 1], index=0, key="drill_interval_w", on_change=on_drill_interval_change)
                    drill_date_bin_options = get_date_bin_options(df_subset_filter, int(drill_bin_interval_w_val), 'year')
                    drill_date_bin_filter_w = st.selectbox("表示期間:", drill_date_bin_options, key="drill_date_filter_w")
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
                        key="drill_applicant_filter_w"
                    )
                else:
                    drill_applicant_filter_w = [(f"(全出願人) ({len(df_subset_filter)}件)", "ALL")]

        st.subheader("クラスタリング設定 (ドリルダウン用)")
        col1, col2, col3 = st.columns(3)
        with col1: drill_min_cluster_size_w = st.number_input('最小クラスタサイズ:', min_value=2, value=5, key="drill_min_cluster_size_w")
        with col2: drill_min_samples_w = st.number_input('最小サンプル数:', min_value=1, value=5, key="drill_min_samples_w")
        with col3: drill_label_top_n_w = st.number_input('ラベル単語数:', min_value=1, value=3, key="drill_label_top_n_w")
        drill_show_labels_chk = st.checkbox('マップにラベルを表示する', value=True, key="drill_show_labels_chk")

        if st.button("選択クラスタで再マップ", type="primary", key="drill_run_button"):
            if drilldown_target_id == "NONE":
                st.error("エラー: 分析対象クラスタを選択してください。")
            else:
                with st.spinner(f"クラスタ {drilldown_target_id} のドリルダウンを実行中..."):
                    try:
                        df_subset = df_main[df_main['cluster'] == drilldown_target_id].copy()
                        base_label = df_subset['cluster_label'].iloc[0]
                        
                        if not drill_date_bin_filter_w.startswith("(全期間)"):
                            try:
                                date_bin_label = drill_date_bin_filter_w.split(' (')[0].strip() 
                                start_year, end_year = map(int, date_bin_label.split('-'))
                                df_subset = df_subset[(df_subset['year'] >= start_year) & (df_subset['year'] <= end_year)]
                            except: pass 

                        # 出願人でフィルタリングを実行
                        drill_app_values = [val[1] for val in drill_applicant_filter_w]
                        if "ALL" not in drill_app_values:
                            mask_list_drill = [df_subset[col_map['applicant']].fillna('').str.contains(re.escape(app)) for app in drill_app_values]
                            df_subset = df_subset[pd.concat(mask_list_drill, axis=1).any(axis=1)]
                        
                        if len(df_subset) < 10:
                            st.warning(f"データが少なすぎます ({len(df_subset)}件)。")
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
                            
                            clusterer_drill = hdbscan.HDBSCAN(min_cluster_size=int(drill_min_cluster_size_w), min_samples=int(drill_min_samples_w), metric='euclidean', cluster_selection_method='eom')
                            df_subset['drill_cluster'] = clusterer_drill.fit_predict(embedding_drill)
                            
                            # patiroha.auto_label で c-TF-IDF ラベリング（ドリルダウン）
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
                            df_subset = update_drill_hover_text(df_subset)
                            st.session_state.df_drilldown_result = df_subset.copy()
                            st.session_state.drill_labels_map = drill_labels_map.copy()
                            st.session_state.drill_labels_map_original = drill_labels_map.copy()
                            st.session_state.drill_base_label = base_label
                            st.success("完了しました。")
                            st.rerun()

                    except Exception as e:
                        st.error(f"エラー: {e}")

        if "df_drilldown_result" in st.session_state:
            df_drill = st.session_state.df_drilldown_result.copy()
            drill_labels_map = st.session_state.drill_labels_map
            


            st.subheader("ドリルダウンマップ")
            
            # --- UIレイアウト ---
            drill_map_mode = st.radio("表示モード:", ["散布図 (Scatter)", "密度マップ (Density)", "クラスタ領域 (Clusters)"], horizontal=True, key="drill_map_mode_radio")
            
            d_c1, d_c2, d_c3 = st.columns(3)
            with d_c1:
                st.markdown("**密度マップ設定**")
                drill_mesh_size = st.number_input("メッシュサイズ (Grid)", value=40, min_value=10, max_value=200, step=5, key="drill_mesh_size")
            with d_c2:
                st.markdown("**フィルタ**")
                drill_remove_noise_chk = st.checkbox("ノイズを除く (Exclude Noise)", value=False, key="drill_remove_noise")
            with d_c3:
                st.empty()

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
            
            # Split Data into Signal (Valid) and Noise
            df_drill_valid = df_drill_plot[df_drill_plot['drill_cluster'] != -1]
            df_drill_noise = df_drill_plot[df_drill_plot['drill_cluster'] == -1]
            
            # 1. Plot Noise (Grey Background) - Always plot if present and not filtered out
            if not df_drill_noise.empty:
                fig_drill.add_trace(go.Scattergl(
                    x=df_drill_noise['drill_x'], y=df_drill_noise['drill_y'], mode='markers',
                    marker=dict(color='#dddddd', size=4, opacity=0.4, line=dict(width=0)),
                    hoverinfo='text', hovertext=df_drill_noise['drill_hover_text'], name='Noise'
                ))

            # 2. Plot Valid Sub-clusters (Colored)
            if not df_drill_valid.empty:
                fig_drill.add_trace(go.Scattergl(
                    x=df_drill_valid['drill_x'], y=df_drill_valid['drill_y'], mode='markers',
                    marker=dict(
                        color=df_drill_valid['drill_cluster'], 
                        colorscale=utils.APOLLO_COLORS if isinstance(utils.APOLLO_COLORS, str) else 'turbo', 
                        showscale=False, 
                        size=6, 
                        opacity=0.9, 
                        line=marker_line_d
                    ),
                    hoverinfo='text', hovertext=df_drill_valid['drill_hover_text'], name='Sub-clusters'
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
                    except:
                        border_color = "#333333"

                    annotations_drill.append(go.layout.Annotation(
                        x=mean_pos['drill_x'], y=mean_pos['drill_y'], text=drill_labels_map.get(cid, ""), showarrow=False, 
                        font=dict(size=10, color='black', family="Helvetica"), 
                        bgcolor='rgba(255,255,255,0.8)',
                        bordercolor=border_color,
                        borderwidth=2,
                        borderpad=4
                    ))
            fig_drill.update_layout(annotations=annotations_drill)
            utils.update_fig_layout(fig_drill, f'Saturn V ドリルダウン: {st.session_state.drill_base_label}', height=1000, show_legend=False)
            st.plotly_chart(fig_drill, use_container_width=True, config={
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
            
            # Create safe summary
            summary_cols_d = ['drill_cluster_label']
            if 'year' in df_drill.columns: summary_cols_d.append('year')
            if col_map.get('title') and col_map['title'] in df_drill.columns:
                summary_cols_d.append(col_map['title'])
            if col_map.get('applicant') and col_map['applicant'] in df_drill.columns:
                summary_cols_d.append(col_map['applicant'])
                
            snap_data = utils.generate_rich_summary(df_drill, title_col=col_map['title'], abstract_col=col_map['abstract'])
            snap_data['module'] = 'Saturn V'
            
            # Optimize Chart Data (Prevent Token Overflow)
            df_snap_safe_d = df_drill[summary_cols_d].head(30).copy()
            # Text Truncation
            for c in summary_cols_d:
                if df_snap_safe_d[c].dtype == object:
                    df_snap_safe_d[c] = df_snap_safe_d[c].astype(str).str.slice(0, 50) + "..."
            
            snap_data['chart_data'] = df_snap_safe_d.to_string(index=False)


            try:
                cluster_counts_snap_d = df_drill['drill_cluster'].value_counts()
                cluster_summary_lines_d = []
                
                # Extract representatives for sub-clusters
                cluster_reps_d = utils.get_cluster_representatives(df_drill, cluster_col='drill_cluster', n_reps=3)

                for cid in sorted(df_drill['drill_cluster'].unique()):
                    if cid == -1: continue
                    label = drill_labels_map.get(cid, f"Sub-Cluster {cid}")
                    count = cluster_counts_snap_d.get(cid, 0)
                    cluster_summary_lines_d.append(f"- {label} ({count}件)")
                    
                    # Append representatives
                    if cid in cluster_reps_d:
                        for rep in cluster_reps_d[cid]:
                            cluster_summary_lines_d.append(rep)
                 
                snap_data['cluster_summary'] = "ドリルダウン・クラスタ構成 (詳細):\n" + "\n".join(cluster_summary_lines_d)
            except: pass

            # Draw Snapshot Button with Context
            
            # Prepare AI Insight Context (Drill)
            drill_insight_context = f"""
            **マップタイプ**: 技術ドリルダウン (Saturn V - Probe)
            **分析対象**: クラスタ「{st.session_state.drill_base_label}」の詳細マップ。
            **手法**: 再計算されたUMAP (局所的構造) + HDBSCAN (サブクラスタ)。
            **視覚的エンコーディング**:
            - **点**: 特定クラスタ内の特許/文献。
            - **色**: サブクラスタ (詳細テーマ)。
            **目的**: 選択された上位クラスタ（親分類）の内部にある、詳細なサブ構造を分析すること。
            """
            insight_role = "あなたはシニア特許アナリストです。技術俯瞰図から戦略的な示唆を導きます。"
            drill_insight_instruction = """
            提供されたデータを元に、この特定技術領域（クラスタ）の内部構造を分析してください：
            1. **サブテーマの構成**: この技術領域はどのような細かいサブテーマ（サブクラスタ）に分かれていますか？
            2. **詳細な内容**: 代表的な特許/文献から、具体的にどのような技術課題や解決策が議論されているか要約してください。
            3. **出願傾向**: (もし年次情報があれば) 最近のトレンドはどうなっていますか？
            """
            
            # Spatial Info (Drill)
            drill_spatial_info = utils_spatial.generate_spatial_cluster_summary(
                df_drill, 'drill_cluster', 'drill_x', 'drill_y', label_map=drill_labels_map
            )
            
            # Combine for Snapshot
            full_drill_context = f"""
### AI Insight Context (Auto-Generated)
{drill_insight_context}

### Spatial Context
{drill_spatial_info}

### Analyst Instructions
{drill_insight_instruction}
"""
            snap_data['ai_insight_context'] = full_drill_context

            # Render Snapshot Button
            utils.render_snapshot_button(
                title=f"Saturn V: Drilldown - {st.session_state.drill_base_label}",
                description=f"Detailed analysis of cluster: {st.session_state.drill_base_label}",
                fig=fig_drill, # Kept fig_drill as it was in the original code
                data_summary=snap_data,
                key="saturn_drill_snap"
            )

            st.markdown("---")
            
            # AI Insight Button
            drill_prompt = utils_ai.generate_ai_insight_prompt(
                insight_role, drill_insight_context, snap_data, drill_insight_instruction, 
                extra_content=f"\n# 空間配置情報 (Spatial Context)\n{drill_spatial_info}"
            )
            utils_ai.render_ai_insight_button(drill_prompt, "saturn_drill_insight")

            # CAPCOM data/ JSON出力（Saturn V PROBEドリルダウン）
            try:
                import capcom
                if capcom.is_active():
                    drill_clusters_json = []
                    drill_counts = df_drill['drill_cluster'].value_counts()
                    drill_reps = utils.get_cluster_representatives(df_drill, cluster_col='drill_cluster', n_reps=3)
                    for cid in sorted(df_drill['drill_cluster'].unique()):
                        if cid == -1:
                            continue
                        label = drill_labels_map.get(cid, f"Sub-Cluster {cid}")
                        count = int(drill_counts.get(cid, 0))
                        cid_mask = df_drill['drill_cluster'] == cid
                        cx = float(df_drill.loc[cid_mask, 'drill_x'].mean()) if cid_mask.any() else 0
                        cy = float(df_drill.loc[cid_mask, 'drill_y'].mean()) if cid_mask.any() else 0
                        reps_full = drill_reps.get(cid, []) if cid in drill_reps else []
                        reps_raw = [r[:200] if len(r) > 200 else r for r in reps_full]
                        drill_clusters_json.append({
                            "cluster_id": int(cid),
                            "label": label,
                            "count": count,
                            "centroid": [round(cx, 4), round(cy, 4)],
                            "representative_patents": reps_raw
                        })
                    drill_json = {
                        "metadata": {
                            "module": "Saturn V",
                            "mode": "PROBE",
                            "parent_cluster": st.session_state.get('drill_base_label', ''),
                            "n_clusters": len(drill_clusters_json),
                            "total_patents": len(df_drill)
                        },
                        "clusters": drill_clusters_json,
                        "spatial_context": drill_spatial_info if 'drill_spatial_info' in dir() else ""
                    }
                    capcom.save_data("saturnv_drilldown.json", drill_json)
            except Exception as e:
                pass

            st.subheader("サブクラスタ・ラベル編集")
            utils.render_ai_label_assistant(df_drill, 'drill_cluster', "drill_labels_map", col_map, tfidf_matrix, feature_names, widget_key_prefix="drill_label")



            if "drill_labels_map_original" not in st.session_state:
                 st.session_state.drill_labels_map_original = drill_labels_map.copy()
            drill_label_widgets = utils.create_label_editor_ui(st.session_state.drill_labels_map_original, st.session_state.drill_labels_map, "drill_label")
            if st.button("サブクラスタ・ラベルを更新", key="drill_update_labels"):
                for cid, val in drill_label_widgets.items(): drill_labels_map[cid] = val
                df_drill['drill_cluster_label'] = df_drill['drill_cluster'].map(drill_labels_map)
                st.session_state.df_drilldown_result = update_drill_hover_text(df_drill)
                st.session_state.drill_labels_map = drill_labels_map
                st.rerun()
            
            # --- テキストマイニング ---
            st.markdown("---")
            st.subheader("クラスタ・テキスト分析 (Text Mining)")
            col_tm1, col_tm2 = st.columns(2)
            with col_tm1:
                cooc_top_n = st.slider("共起: 上位単語数", 30, 100, 70, key="cooc_top_n")
                cooc_threshold = st.slider("共起: Jaccard係数 閾値", 0.01, 0.3, 0.03, 0.01, key="cooc_threshold")
            
            if st.button("テキスト分析を実行", key="run_text_mining"):
                with st.spinner("分析中..."):
                    all_text = ""
                    for _, row in df_drill.iterrows():
                        if col_map['title'] and pd.notna(row[col_map['title']]): all_text += row[col_map['title']] + " "
                        if col_map['abstract'] and pd.notna(row[col_map['abstract']]): all_text += row[col_map['abstract']] + " "
                    words = extract_compound_nouns(all_text, stopwords)
                    
                    if not words: st.warning("有効なキーワードなし")
                    else:
                        st.markdown("##### 1. ワードクラウド")
                        generate_wordcloud_and_list(words, f"クラスタ: {st.session_state.drill_base_label}", 30, FONT_PATH, capcom_key="saturnv_drill")

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
                            fig_net.update_xaxes(visible=False)
                            fig_net.update_yaxes(visible=False)
                            st.plotly_chart(fig_net, use_container_width=True)
                            
                            # AI Insight (Co-occurrence Network)
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
                                "Top Words (Nodes)": ", ".join(net_nodes_list[:30]),
                                "Strongest Connections (Edges)": ", ".join(sorted(net_edges_list, key=lambda x: float(x.split('J=')[1][:-1]), reverse=True)[:20]),
                                "Representative Patents (Keyword-Centric)": "\n".join(rep_lines_net)
                            }
                            
                            net_insight_context = f"""
                            **チャートタイプ**: 共起ネットワーク (テキストマイニング)
                            **対象データ**: クラスタ「{st.session_state.drill_base_label}」内の文書。
                            **手法**: 複合名詞のJaccard係数による共起分析。
                            **視覚的エンコーディング**:
                            - **ノード**: キーワード。サイズは出現頻度。
                            - **エッジ**: 共起関係。太さ/有無はJaccard係数 > {cooc_threshold} で定義。
                            **目的**: 技術用語同士の意味的なつながりや、複合技術の構造を理解すること。
                            """
                            net_role = "あなたはテキストマイニングの専門家です。キーワードの共起関係から技術的な文脈を読み解きます。"
                            net_instruction = """
                            共起ネットワークの構造を分析してください：
                            1. **中核的な概念**: 中心にある、または最もつながりの多いキーワードは何ですか？
                            2. **技術の組み合わせ**: 強く結びついている単語のペア（エッジ）から、どのような技術要素が組み合わされているか推測してください。
                            3. **文脈**: このクラスタは具体的に何をする技術（What/How）に関するものだと考えられますか？
                            """
                            net_prompt = utils_ai.generate_ai_insight_prompt(net_role, net_insight_context, net_data_summary, net_instruction)
                            utils_ai.render_ai_insight_button(net_prompt, "saturn_net_insight")

    # --- C. 特許マップ (統計分析) ---
    with tab_stats:
        st.subheader("特許マップ（統計分析）")
        if st.session_state.saturnv_cluster_done:
            cluster_counts_stats = st.session_state.df_main['cluster_label'].value_counts()
            cluster_options_stats = [(f"(全クラスタ) ({len(st.session_state.df_main)}件)", "ALL")] + [
                (f"{st.session_state.saturnv_labels_map.get(cid)} ({cluster_counts_stats.get(st.session_state.saturnv_labels_map.get(cid), 0)}件)", cid)
                for cid in sorted(st.session_state.df_main['cluster'].unique())
            ]
            stats_cluster_filter_w = st.multiselect("集計対象クラスタ:", cluster_options_stats, default=[cluster_options_stats[0]], format_func=lambda x: x[0], key="stats_cluster_filter")
            
            c1, c2 = st.columns(2)
            with c1:
                auto_min_year = 2000
                auto_max_year = datetime.datetime.now().year
                if 'year' in st.session_state.df_main.columns:
                    try:
                        valid_years = st.session_state.df_main['year'].dropna()
                        if not valid_years.empty:
                            auto_min_year = int(valid_years.min())
                            auto_max_year = int(valid_years.max())
                    except:
                        pass

                if 'stats_start_year' not in st.session_state: st.session_state.stats_start_year = auto_min_year
                if 'stats_end_year' not in st.session_state: st.session_state.stats_end_year = auto_max_year
                
                s_year = st.number_input('開始年:', min_value=1900, max_value=2100, key="stats_start_year", step=1)
                e_year = st.number_input('終了年:', min_value=1900, max_value=2100, key="stats_end_year", step=1)
            with c2:
                n_apps = st.number_input('表示人数:', min_value=1, value=15, key="stats_num_assignees")
            
            if st.button("特許マップを描画", key="stats_run_button"):
                df_s = st.session_state.df_main.copy()
                vals = [v[1] for v in stats_cluster_filter_w]
                if "ALL" not in vals: df_s = df_s[df_s['cluster'].isin(vals)]
                df_s = df_s[(df_s['year'] >= s_year) & (df_s['year'] <= e_year)]
                
                if df_s.empty: st.warning("データなし")
                else:
                    # 1. 時系列
                    yc = df_s['year'].value_counts().sort_index().reindex(range(s_year, e_year+1), fill_value=0)
                    fig1 = px.bar(x=yc.index, y=yc.values, labels={'x':'年', 'y':'件数'}, color_discrete_sequence=[utils.APOLLO_COLORS[0]])
                    utils.update_fig_layout(fig1, '出願推移', show_axes=True)
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # 2. ランキング
                    ac = df_s['applicant_main'].explode().value_counts().head(n_apps).sort_values(ascending=True)
                    fig2 = px.bar(x=ac.values, y=ac.index, orientation='h', labels={'x':'件数', 'y':'出願人'}, color_discrete_sequence=[utils.APOLLO_COLORS[1]])
                    utils.update_fig_layout(fig2, '出願人ランキング', height=max(600, len(ac)*30), show_axes=True)
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # 3. バブル
                    ae = df_s.explode('applicant_main')
                    ae['ap'] = ae['applicant_main'].astype(str).str.strip()
                    top_a = ae['ap'].value_counts().head(n_apps).index.tolist()
                    pd_plot = ae[ae['ap'].isin(top_a)].groupby(['year', 'ap']).size().reset_index(name='count')
                    
                    if not pd_plot.empty:
                        fig3 = px.scatter(pd_plot, x='year', y='ap', size='count', color='ap', labels={'year':'出願年', 'ap':'出願人', 'count':'件数'}, category_orders={'ap': top_a})
                        utils.update_fig_layout(fig3, '出願年別動向', height=700, show_axes=True)
                        fig3.update_layout(
                            legend=dict(
                                orientation="v", 
                                yanchor="top", y=1, 
                                xanchor="left", x=1.02, 
                                borderwidth=0
                            )
                        )
                        st.plotly_chart(fig3, use_container_width=True)

    # --- D. エクスポート ---
    with tab_export:
        st.subheader("データエクスポート")
        if st.session_state.saturnv_cluster_done:
            cols_drop = ['hover_text', 'parsed_date', 'drill_cluster', 'drill_cluster_label', 'drill_hover_text', 'drill_x', 'drill_y', 'temp_date_bin']
            csv = df_main.drop(columns=cols_drop, errors='ignore').to_csv(encoding='utf-8-sig', index=False).encode('utf-8-sig')
            st.download_button("メインマップ全データ (CSV)", csv, "APOLLO_SaturnV_Main.csv", "text/csv")

            # Parameter Export
            param_content = f"APOLLO Saturn V Analysis Parameters\n"
            param_content += f"-----------------------------------\n"
            param_content += f"Min Cluster Size: {min_cluster_size_w}\n"
            param_content += f"Min Samples: {min_samples_w}\n"
            param_content += f"Label Word Count: {label_top_n_w}\n"
            st.download_button("パラメータ設定 (TXT)", param_content, "APOLLO_SaturnV_Params.txt", "text/plain")
        
        if "df_drilldown_result" in st.session_state:
            cols_drop_d = ['hover_text', 'parsed_date', 'date_bin', 'drill_hover_text', 'drill_date_bin', 'temp_date_bin']
            csv_d = st.session_state.df_drilldown_result.drop(columns=cols_drop_d, errors='ignore').to_csv(encoding='utf-8-sig', index=False).encode('utf-8-sig')
            st.download_button("ドリルダウン結果 (CSV)", csv_d, "APOLLO_SaturnV_Drill.csv", "text/csv")