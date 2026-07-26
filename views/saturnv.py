# ==================================================================
# APOLLO — Saturn V（俯瞰図分析）
# 旧 APOLLO pages/3_:material/rocket_launch:_Saturn_V.py の移植版。
# 特許テキストの意味的類似性から技術マップ（俯瞰図）を自動生成し、
# クラスタ構造・ノイズ（萌芽技術の芽）・成長動態を一望する。
# 分析ロジック・session_state キー・CAPCOM 出力スキーマは旧版から不変。
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
import apollo_ui
import flight_recorder
from wordcloud import WordCloud
from janome.tokenizer import Tokenizer
import patiroha
import networkx as nx

# 描画用
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
utils.configure_matplotlib_font()


# 警告を非表示
warnings.filterwarnings('ignore')

# ==================================================================
# --- フォント設定（モジュールレベルに維持） ---
# ==================================================================
FONT_PATH = utils.get_japanese_font_path()
if FONT_PATH:
    try:
        prop = fm.FontProperties(fname=FONT_PATH)
        plt.rcParams['font.family'] = prop.get_name()
    except:
        pass

# ==================================================================
# --- テキスト処理関数（モジュールレベル） ---
# ==================================================================

@st.cache_resource
def load_tokenizer_saturn():
    return Tokenizer()

t = load_tokenizer_saturn()


def _get_stopwords():
    """ストップワードを毎ランで取得する（session_state のユーザー定義があれば優先）"""
    if "stopwords" in st.session_state and st.session_state["stopwords"]:
        return st.session_state["stopwords"]
    return patiroha.get_stopwords()

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
    text = re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]', ' ', text)

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
        wc_array = utils.compute_wordcloud_array(tuple(sorted(word_freq.items())), font_path)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc_array, interpolation='bilinear')
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
            except Exception as e:
                st.caption(f":material/warning: ワードクラウド の CAPCOM 保存に失敗しました（要確認）: {e}")

        # スナップショットボタン
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
# --- 画面描画 ---
# ==================================================================
def _terra_adjust_intent(mode):
    """「しっくりこない」ボタンの共通コールバック。

    パラメータ名をユーザーに見せず、目標クラスタ数などを内部で調整して
    次の実行フラグ（terra_intent_rerun）を立てる。コールバックは
    ウィジェット生成前に走るため、ここでの session_state 書き換えが
    そのまま詳細設定ウィジェットの値として反映される。

    設計上の要点（同じ解の再選択を防ぐ）:
    - 掃引は目標の半分〜2倍という広い許容範囲から品質(DBCV)優先で選ぶため、
      目標を少し動かすだけでは同じ構成が再選択されうる。
      → finer/coarser では「今の実測クラスタ数より細かい/粗い解のみ許容」の
        制約(k_min/k_max)を terra_intent_ctx で実行側に渡し、同じ解の再選択を禁じる。
    - 基準を実測kだけにすると、結果が動かない限り目標も動かない不動点に陥る。
      → 目標は「現在の目標値と実測kの前進側」から計算し、クリックごとに必ず前進させる。
    - 変化の有無は実行側が組み立て、意図ボタンカードがインライン表示する
      （terra_intent_ctx に前回状態を持たせる）。
    """
    ar = st.session_state.get('main_auto_result') or {}
    # 実測クラスタ数・未分類率は df から直接数える（手動モード実行後は main_auto_result が無いため）
    achieved_k = None
    prev_noise = None
    try:
        _lab = st.session_state.df_main['cluster']
        achieved_k = int(len(set(_lab.unique()) - {-1}))
        prev_noise = float((_lab == -1).mean())
    except Exception:
        pass
    cur_target = st.session_state.get('main_auto_target_k') or achieved_k or 12

    if mode == 'finer':
        if achieved_k is not None and achieved_k >= 80:
            st.session_state['terra_intent_feedback'] = {
                'msg': "すでに上限（80クラスタ）です。これ以上は細かくできません", 'changed': False}
            return
        base_k = max(int(cur_target), int(achieved_k or 0))
        new_k = min(80, max(base_k + 2, round(base_k * 1.4)))
        st.session_state['main_auto_hdbscan'] = True
        st.session_state['main_auto_target_k'] = int(new_k)
        _k_floor = (int(achieved_k) + 1) if achieved_k else None
        st.session_state['terra_intent_ctx'] = {
            'mode': mode, 'prev_k': achieved_k, 'prev_noise': prev_noise,
            'k_min': _k_floor, 'k_max': max(int(new_k) * 2, (_k_floor or 2) + 1)}
    elif mode == 'coarser':
        if achieved_k is not None and achieved_k <= 2:
            st.session_state['terra_intent_feedback'] = {
                'msg': "すでに最小（2クラスタ）です。これ以上は大きくまとめられません", 'changed': False}
            return
        base_k = min(int(cur_target), int(achieved_k or 10 ** 9))
        new_k = max(2, min(base_k - 2, round(base_k * 0.65)))
        st.session_state['main_auto_hdbscan'] = True
        st.session_state['main_auto_target_k'] = int(new_k)
        _k_ceil = max(2, int(achieved_k) - 1) if achieved_k else None
        st.session_state['terra_intent_ctx'] = {
            'mode': mode, 'prev_k': achieved_k, 'prev_noise': prev_noise,
            'k_min': 2, 'k_max': _k_ceil}
    elif mode == 'less_noise':
        # 自動決定済みのパラメータを土台に、核の判定（最小サンプル数）だけ緩めて
        # 未分類を減らす。掃引はスキップするため手動モードで1回実行する。
        mcs = ar.get('mcs') or st.session_state.get('main_min_cluster_size', 15)
        ms = int(ar.get('ms') or st.session_state.get('main_min_samples', 10))
        new_ms = max(1, int(round(ms * 0.5)))
        if new_ms == ms:
            _nz_txt = f"（現在 {prev_noise * 100:.0f}%）" if prev_noise is not None else ""
            st.session_state['terra_intent_feedback'] = {
                'msg': f"未分類の判定はすでに最も緩い設定です{_nz_txt}。これ以上は自動では減らせません",
                'changed': False}
            return
        st.session_state['main_auto_hdbscan'] = False
        st.session_state['main_min_cluster_size'] = int(mcs)
        st.session_state['main_min_samples'] = new_ms
        st.session_state['terra_intent_ctx'] = {
            'mode': mode, 'prev_k': achieved_k, 'prev_noise': prev_noise}
    st.session_state['terra_intent_rerun'] = True


def _terra_drill_adjust_intent(mode):
    """ドリルダウン版「しっくりこない」ボタンの共通コールバック（メインマップと同じ設計）。

    実測サブクラスタ数より細かい/粗い解のみ許容する制約を terra_drill_intent_ctx で
    実行側に渡し、目標はクリックごとに前進させる。結果の通知はカード内にインライン表示。
    """
    achieved_k = None
    prev_noise = None
    try:
        _lab = st.session_state.df_drilldown_result['drill_cluster']
        achieved_k = int(len(set(_lab.unique()) - {-1}))
        prev_noise = float((_lab == -1).mean())
    except Exception:
        pass
    _tsel = st.session_state.get('drill_target_select')
    _tid = _tsel[1] if isinstance(_tsel, tuple) and len(_tsel) > 1 else None
    if _tid in (None, 'NONE'):
        return
    _k_key = f"drill_target_k_w_{_tid}"
    cur_target = st.session_state.get(_k_key) or achieved_k or 8

    if mode == 'finer':
        if achieved_k is not None and achieved_k >= 80:
            st.session_state['terra_drill_intent_feedback'] = {
                'msg': "すでに上限（80サブクラスタ）です。これ以上は細かくできません", 'changed': False}
            return
        base_k = max(int(cur_target), int(achieved_k or 0))
        new_k = min(80, max(base_k + 2, round(base_k * 1.4)))
        st.session_state['drill_auto_hdbscan'] = True
        st.session_state[_k_key] = int(new_k)
        _k_floor = (int(achieved_k) + 1) if achieved_k else None
        st.session_state['terra_drill_intent_ctx'] = {
            'mode': mode, 'prev_k': achieved_k, 'prev_noise': prev_noise,
            'k_min': _k_floor, 'k_max': max(int(new_k) * 2, (_k_floor or 2) + 1)}
    elif mode == 'coarser':
        if achieved_k is not None and achieved_k <= 2:
            st.session_state['terra_drill_intent_feedback'] = {
                'msg': "すでに最小（2サブクラスタ）です。これ以上は大きくまとめられません", 'changed': False}
            return
        base_k = min(int(cur_target), int(achieved_k or 10 ** 9))
        new_k = max(2, min(base_k - 2, round(base_k * 0.65)))
        st.session_state['drill_auto_hdbscan'] = True
        st.session_state[_k_key] = int(new_k)
        _k_ceil = max(2, int(achieved_k) - 1) if achieved_k else None
        st.session_state['terra_drill_intent_ctx'] = {
            'mode': mode, 'prev_k': achieved_k, 'prev_noise': prev_noise,
            'k_min': 2, 'k_max': _k_ceil}
    elif mode == 'less_noise':
        # 直前が自動決定ならその値を土台に、核の判定（最小サンプル数）だけ緩める
        _dar = st.session_state.get('saturnv_drill_auto_result') or {}
        if _dar.get('target_id') == _tid:
            mcs = int(_dar.get('mcs') or st.session_state.get('drill_min_cluster_size_w', 5))
            ms = int(_dar.get('ms') or st.session_state.get('drill_min_samples_w', 5))
        else:
            mcs = int(st.session_state.get('drill_min_cluster_size_w', 5))
            ms = int(st.session_state.get('drill_min_samples_w', 5))
        new_ms = max(1, int(round(ms * 0.5)))
        if new_ms == ms:
            _nz_txt = f"（現在 {prev_noise * 100:.0f}%）" if prev_noise is not None else ""
            st.session_state['terra_drill_intent_feedback'] = {
                'msg': f"未分類の判定はすでに最も緩い設定です{_nz_txt}。これ以上は自動では減らせません",
                'changed': False}
            return
        st.session_state['drill_auto_hdbscan'] = False
        st.session_state['drill_min_cluster_size_w'] = mcs
        st.session_state['drill_min_samples_w'] = new_ms
        st.session_state['terra_drill_intent_ctx'] = {
            'mode': mode, 'prev_k': achieved_k, 'prev_noise': prev_noise}
    st.session_state['terra_drill_intent_rerun'] = True


def render():
    apollo_ui.page_header(
        "俯瞰図分析", "Saturn V",
        "文埋め込み（SBERT）× UMAP × HDBSCAN による技術ランドスケープ。クラスタ構造と空白領域を同定します",
    )
    # スナップショットのモジュール記録キー（CAPCOM スキーマ互換のため値も旧名のまま）
    st.session_state['current_page'] = 'Saturn V'

    # --- データガード ---
    apollo_ui.require_data()

    df_main = st.session_state.df_main
    col_map = st.session_state.col_map
    delimiters = st.session_state.delimiters
    sbert_embeddings = st.session_state.sbert_embeddings
    tfidf_matrix = st.session_state.tfidf_matrix
    feature_names = st.session_state.feature_names

    # ストップワード（ユーザー定義があれば優先）
    stopwords = _get_stopwords()

    if "saturnv_sbert_umap_done" not in st.session_state: st.session_state.saturnv_sbert_umap_done = False
    if "saturnv_cluster_done" not in st.session_state: st.session_state.saturnv_cluster_done = False
    if "saturnv_labels_map" not in st.session_state: st.session_state.saturnv_labels_map = {}
    if "main_cluster_running" not in st.session_state: st.session_state.main_cluster_running = False
    if "saturnv_global_zmax" not in st.session_state: st.session_state.saturnv_global_zmax = None

    # --- 初回UMAP計算 ---
    if not st.session_state.saturnv_sbert_umap_done:
        with st.spinner("俯瞰図分析の初回起動中: 特許を2次元の地図に配置しています（UMAP・SBERTベース）..."):
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
                st.success("地図の下ごしらえ（UMAPの初期計算）が完了しました。")
                st.rerun()
            except Exception as e:
                st.error(f"UMAPの初期計算中にエラーが発生しました: {e}")
                st.stop()

    # --- メインUI ---
    tab_main, tab_drill, tab_stats, tab_export = st.tabs([
        "俯瞰図マップ",
        "ドリルダウン",
        "統計",
        "エクスポート"
    ])

    # --- 俯瞰図マップ（旧 TELESCOPE） ---
    with tab_main:
        # --- 実行バー: 結論の粒度を左右するパラメータは分析者の手元に置く ---
        with st.container(border=True):
            bar_l, bar_r = st.columns([3, 1])
            with bar_l:
                st.markdown(":material/tune: **クラスタリングの条件** — 粒度は自動掃引と手動指定のどちらでも指定できます")
                st.caption("初回は計算に時間がかかることがあります。条件やフィルタを変えたら、もう一度実行すると再構築します。")
                if not col_map.get('abstract'):
                    st.caption("このデータには要約がないため、発明の名称のみを対象に分析します。クラスタの分離とラベルの精度は要約ありのデータより低下します。")
            with bar_r:
                run_clicked = st.button(
                    "俯瞰図を生成", type="primary", key="main_run_cluster",
                    icon=":material/map:",
                    use_container_width=True,
                    disabled=st.session_state.main_cluster_running)

            # 自動最適化モード: HDBSCAN の2パラメータを母集団に合わせて掃引する（既定ON）
            auto_hdbscan = st.checkbox(
                "粒度を自動掃引する（最小クラスタサイズ・最小サンプル数を自動決定）",
                value=True,
                key="main_auto_hdbscan",
                help="母集団に合わせて HDBSCAN の2パラメータ（最小クラスタサイズ・最小サンプル数）を自動で掃引し、"
                     "クラスタ数が少なすぎ(2,3)も多すぎもしない設定を自動選択します。"
                     "外すと下の2つの値がそのまま使われます。",
            )

            _p1, _p2, _p3, _p4 = st.columns(4)
            with _p1: min_cluster_size_w = st.number_input("最小クラスタサイズ", min_value=2, value=15, key="main_min_cluster_size", disabled=auto_hdbscan, help="1つのクラスタとして認める最小の特許件数です（クラスタの粒度設定）。小さくすると細かい技術テーマまで分かれてクラスタ数が増えますが、どこにも属さないノイズ（外れ値）も増えます。大きくすると少数の大まかなクラスタにまとまり安定しますが、細部は埋もれます。目安は母集団の約1〜2%（例: 2,000件なら20前後）。推奨10〜50。")
            with _p2: min_samples_w = st.number_input("最小サンプル数", min_value=1, value=10, key="main_min_samples", disabled=auto_hdbscan, help="クラスタの「核」と認める密度の厳しさです。大きいほど判定が厳しくなり、ノイズ（外れ値）が増えてクラスタは密な中心部だけになります。小さいほど緩くなり、多くの点がクラスタに取り込まれます。通常は最小クラスタサイズ以下に設定します。推奨5〜20。")
            with _p3:
                _n_pop = len(st.session_state.df_main) if st.session_state.df_main is not None else 0
                target_k_w = st.number_input(
                    "目標クラスタ数", min_value=2, max_value=80,
                    value=utils.suggest_target_k(_n_pop), key="main_auto_target_k",
                    disabled=not auto_hdbscan,
                    help="この数に近づくよう2パラメータを掃引します（母集団規模から自動初期化）。"
                         "結果のクラスタ数や粒度に満足できない場合は、この値を増減して再実行してください。"
                         "実際の値は密度構造に依存するため、目標ちょうどにならないこともあります（品質 DBCV を優先して選びます）。")
            with _p4:
                umap_seed_w = st.number_input(
                    "乱数シード", min_value=0, max_value=99999, value=42, key="main_umap_seed",
                    help="座標配置（UMAP）を決める乱数の種です。同じ値なら何度実行しても同じ配置になります。"
                         "値を変えて再実行すると、配置が変わっても同じ塊が現れるか＝結論が配置に依存していないかを検証できます。"
                         "変更すると座標を再計算します。")

            # 品質指標は結果の隣に常時出す（詳細設定の奥に畳まない）
            _ar = st.session_state.get('main_auto_result')
            if _ar:
                _rv = _ar.get('validity')
                _q1, _q2, _q3, _q4 = st.columns(4)
                _q1.metric("クラスタ数", f"{_ar['k']}")
                _q2.metric("未分類（ノイズ）", f"{_ar['noise'] * 100:.1f}%")
                _q3.metric("地図の品質 DBCV", f"{_rv:.2f}" if isinstance(_rv, (int, float)) else "—")
                _q4.metric("採用した粒度", f"{_ar['mcs']} / {_ar['ms']}",
                           help="最小クラスタサイズ / 最小サンプル数")
                st.caption(
                    f"目標≈{_ar['target_k']} に対する自動決定の結果です。"
                    "粒度が意図と合わない場合は「目標クラスタ数」を変えて再実行してください。")

        # --- 詳細設定（描き方・ラベルなど。既定で開く） ---
        with st.expander("詳細設定（ラベル・地図の描き方）", expanded=True, icon=":material/tune:"):
            col3, = st.columns(1)
            with col3: label_top_n_w = st.number_input("クラスタラベル単語数:", min_value=1, value=3, key="main_label_top_n", help="各クラスタの自動命名に使う特徴語の数です。クラスタを特徴づける語を上位から何語ラベルに並べるかを決めます。多いほど内容を詳しく表せますが冗長になり、少ないほど簡潔になります。")
            if auto_hdbscan:
                utils.render_dbcv_help()

            st.markdown("---")
            st.markdown("**地図の描き方**")
            map_mode = st.radio("表示モード:", ["クラスタ領域 (Clusters)", "密度マップ (Density)", "散布図 (Scatter)"], horizontal=True, help="俯瞰図の描き方を切り替えます。クラスタ領域＝各クラスタを色付きの領域で表示、密度マップ＝点の混み具合をヒートマップで表示、散布図＝個々の特許を点で表示。全体像は領域、混雑度は密度、個別確認は散布図が見やすいです。")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**密度マップ設定**")
                main_mesh_size = st.number_input("メッシュサイズ (Grid)", value=30, min_value=10, max_value=200, step=5, key="main_mesh_size", help="密度マップ（ヒートマップ）を描くときの格子の細かさです。大きいほど細かい格子になり局所的な濃淡が見えますが、点がまばらだと粗く見えます。小さいほど滑らかで大まかな分布になります。")
                use_abs_scale = False
                if map_mode == "密度マップ (Density)":
                    use_abs_scale = st.checkbox("密度スケールを固定 (絶対評価)", value=False, key="main_abs_scale", help="密度の色の濃さの基準を全体共通に固定します。オンにすると期間やフィルタを変えても色の意味（濃さ＝件数）が一定になり期間どうしを公平に比較できます。オフにすると表示中のデータ内で色を相対的に割り当てます。")
            with c2:
                st.markdown("**フィルタ**")
                remove_noise_chk = st.checkbox("ノイズを除く (Exclude Noise)", value=False, key="main_remove_noise", help="どのクラスタにも属さないノイズ（外れ値）の特許をマップから隠します。オンにすると主要なクラスタ構造がすっきり見えますが、萌芽技術の候補となるノイズは見えなくなります。")
            with c3:
                st.markdown("**表示オプション**")
                show_labels_chk = st.checkbox("マップにラベルを表示する", value=True, key="main_show_labels", help="各クラスタの自動命名ラベルをマップ上に重ねて表示します。オフにすると点の分布だけが見えるため、ラベルが重なって読みにくいときに使います。")
                # ステータス列があれば色分け基準を選べる（クラスタ / ステータス・パステル）
                _status_col_sv = col_map.get('status')
                if _status_col_sv and _status_col_sv in st.session_state.df_main.columns:
                    color_by = st.radio("色分け基準:", ["クラスタ", "ステータス"], horizontal=True, key="main_color_by", help="マップの点を何で色分けするかを選びます。基準により見える構造が変わります（クラスタ＝技術テーマ別、ステータス＝権利状況別）。")
                else:
                    color_by = "クラスタ"

        # --- クラスタリング実行（ボタンは上のおすすめバー・パラメータは詳細設定） ---
        # terra_intent_rerun: 「しっくりこない」ボタン（意図ベース調整）による自動再実行フラグ
        # terra_intent_ctx: 同ボタンの制約（今より細かい/粗い解のみ許容）と前回状態（変化トースト用）
        _intent_rerun = st.session_state.pop('terra_intent_rerun', False)
        _intent_ctx = st.session_state.pop('terra_intent_ctx', None)
        if not _intent_rerun:
            _intent_ctx = None
        if run_clicked or _intent_rerun:
            st.session_state.main_cluster_running = True
            try:
                # 乱数シードが変わっていれば座標配置から計算し直す
                _seed = int(umap_seed_w)
                if st.session_state.get('saturnv_umap_seed_used', 42) != _seed:
                    with st.spinner(f"乱数シード {_seed} で座標配置を再計算しています（UMAP）..."):
                        _reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=_seed)
                        _emb = _reducer.fit_transform(st.session_state.sbert_embeddings)
                        st.session_state.df_main['umap_x'] = _emb[:, 0]
                        st.session_state.df_main['umap_y'] = _emb[:, 1]
                        st.session_state['saturnv_umap_seed_used'] = _seed
                embedding = st.session_state.df_main[['umap_x', 'umap_y']].values
                if auto_hdbscan:
                    _pb = st.progress(0.0, text="クラスタの粒度を自動調整中（HDBSCAN パラメータ掃引）...")
                    # 意図ボタン経由なら「今の粒度と同じ解の再選択」を範囲制約で禁じる
                    _sweep_kwargs = {}
                    if _intent_ctx and _intent_ctx.get('k_min'):
                        _sweep_kwargs['k_min'] = int(_intent_ctx['k_min'])
                    if _intent_ctx and _intent_ctx.get('k_max'):
                        _sweep_kwargs['k_max'] = int(_intent_ctx['k_max'])
                    _sweep = utils.sweep_hdbscan_params(
                        embedding, target_k=int(target_k_w),
                        progress_callback=lambda f: _pb.progress(min(f, 1.0), text="クラスタの粒度を自動調整中（HDBSCAN パラメータ掃引）..."),
                        **_sweep_kwargs)
                    _pb.empty()
                    cluster_labels = _sweep['labels']
                    st.session_state['main_auto_result'] = {
                        'mcs': _sweep['min_cluster_size'], 'ms': _sweep['min_samples'],
                        'k': _sweep['n_clusters'], 'noise': _sweep['noise_ratio'],
                        'target_k': _sweep['target_k'], 'validity': _sweep.get('validity')}
                else:
                    with st.spinner("クラスタリングを実行中（HDBSCAN）..."):
                        clusterer = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size_w), min_samples=int(min_samples_w), metric='euclidean', cluster_selection_method='eom')
                        cluster_labels = clusterer.fit_predict(embedding)
                    st.session_state.pop('main_auto_result', None)

                with st.spinner("クラスタのラベリング中..."):
                    st.session_state.df_main['cluster'] = cluster_labels

                    # utils.safe_auto_label で c-TF-IDF ラベリング (Janome 例外耐性付き)
                    label_top_n = int(label_top_n_w)
                    texts_for_label = utils.combine_text_columns(
                        st.session_state.df_main, col_map, keys=('title', 'abstract'))
                    labels_map = utils.safe_auto_label(
                        texts_for_label,
                        st.session_state.df_main['cluster'].values,
                        method='c-tfidf',
                        top_n=label_top_n,
                    )

                    st.session_state.df_main['cluster_label'] = st.session_state.df_main['cluster'].map(labels_map)
                    st.session_state.saturnv_labels_map = labels_map.copy()
                    st.session_state.saturnv_labels_map_original = labels_map.copy()

                    # 飛行記録: 結論の粒度がどの条件で出たかを残す
                    try:
                        _n_all = len(cluster_labels)
                        _n_noise = int((np.asarray(cluster_labels) == -1).sum())
                        _auto = st.session_state.get('main_auto_result') or {}
                        flight_recorder.record(
                            "Saturn V", "俯瞰図の生成",
                            judgment={
                                "クラスタ数": int(len({c for c in cluster_labels if c != -1})),
                                "未分類率(%)": round(_n_noise / _n_all * 100, 1) if _n_all else 0,
                                "最小クラスタサイズ": int(_auto.get('mcs') or min_cluster_size_w),
                                "最小サンプル数": int(_auto.get('ms') or min_samples_w),
                                "粒度の決め方": "自動掃引" if auto_hdbscan else "手動指定",
                                "地図の品質 DBCV": _auto.get('validity'),
                            },
                            repro={
                                "近傍数 n_neighbors": 15,
                                "最小距離 min_dist": 0.1,
                                "乱数シード": int(st.session_state.get('saturnv_umap_seed_used', 42)),
                                "ラベル語数": int(label_top_n_w),
                                "対象件数": int(_n_all),
                            })
                    except Exception:
                        pass
                    st.session_state.df_main = update_hover_text(st.session_state.df_main, col_map)
                    st.session_state.saturnv_cluster_done = True
                    # CAPCOM: patents.csvにクラスタ情報を追加更新
                    try:
                        import capcom
                        if capcom.is_active():
                            capcom.save_patents_csv()
                    except Exception as e:
                        st.caption(f":material/warning: クラスタ付きデータ の CAPCOM 保存に失敗しました（要確認）: {e}")
                # 意図ボタン経由の場合: 変化の有無の通知を組み立てて session_state に預ける
                # （直後の st.rerun() で消えるため、ここでは表示せず、
                #   意図ボタンカードの描画時にインライン表示する。変わらない場合も無反応に見せない）
                if _intent_ctx:
                    _new_k = int(len(set(cluster_labels.tolist()) - {-1}))
                    _new_nz = float((cluster_labels == -1).mean()) if len(cluster_labels) else 0.0
                    _im = _intent_ctx.get('mode')
                    _pk = _intent_ctx.get('prev_k')
                    _pn = _intent_ctx.get('prev_noise')
                    _fb = None
                    if _im == 'finer':
                        if _pk and _new_k > _pk:
                            _fb = {'msg': f"クラスタ数が {_pk} → {_new_k} に増えました", 'changed': True}
                        elif _pk:
                            _fb = {'msg': f"これ以上細かく分かれませんでした（クラスタ数 {_new_k} のまま）。"
                                          "データの密度構造の限界です", 'changed': False}
                    elif _im == 'coarser':
                        if _pk and _new_k < _pk:
                            _fb = {'msg': f"クラスタ数が {_pk} → {_new_k} にまとまりました", 'changed': True}
                        elif _pk:
                            _fb = {'msg': f"これ以上大きなくくりにできませんでした（クラスタ数 {_new_k} のまま）",
                                   'changed': False}
                    elif _im == 'less_noise':
                        if _pn is not None and _new_nz < _pn - 1e-9:
                            _fb = {'msg': f"未分類が {_pn * 100:.0f}% → {_new_nz * 100:.0f}% に減りました",
                                   'changed': True}
                        elif _pn is not None:
                            _fb = {'msg': f"未分類は減りませんでした（{_new_nz * 100:.0f}%）。"
                                          "もう一度押すと、さらに条件を緩めて試します", 'changed': False}
                    if _fb:
                        st.session_state['terra_intent_feedback'] = _fb
                st.success("クラスタリング完了")
                st.rerun()
            except Exception as e:
                st.error(f"エラー: {e}")
            finally:
                st.session_state.main_cluster_running = False

        st.markdown("---")

        if st.session_state.saturnv_cluster_done:
            st.subheader("フィルタリング設定（俯瞰図マップ用）")
            def on_main_interval_change():
                if "main_date_filter" in st.session_state: del st.session_state.main_date_filter

            col1, col2 = st.columns(2)
            with col1:
                if 'year' in df_main.columns and df_main['year'].notna().any():
                    bin_interval_w_val = st.selectbox("期間の粒度:", [5, 3, 2, 1], index=0, key="main_bin_interval", on_change=on_main_interval_change, help="表示期間を区切る年数の幅です。大きくすると5年単位などの大まかな期間に、小さくすると1年単位など細かい期間に分かれます（下の「表示期間」の選択肢が変わります）。")
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
            cluster_filter_w = st.multiselect("マップ表示クラスタ:", cluster_options, default=[cluster_options[0]], format_func=lambda x: x[0], key="main_cluster_filter", help="分析・表示の対象にするクラスタを選びます。特定の技術テーマだけに絞って詳しく見たいときに使います。")

            st.subheader("分析結果（俯瞰図マップ）")

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
            # fig_main をフィルタ・表示条件・クラスタリング結果が変わったときだけ再構築し、
            # それ以外のリラン（他ウィジェット操作等）では session_state のキャッシュを再利用する。
            _main_fig_key = (
                map_mode, main_mesh_size, use_abs_scale, remove_noise_chk, show_labels_chk, color_by,
                date_bin_filter_w,
                tuple(tuple(v) for v in applicant_filter_w),
                tuple(tuple(v) for v in cluster_filter_w),
                tuple(sorted(st.session_state.saturnv_labels_map.items())),
            )
            if (st.session_state.get('saturnv_main_fig_key') == _main_fig_key
                    and 'saturnv_main_fig' in st.session_state):
                fig_main = st.session_state['saturnv_main_fig']
            else:
                fig_main = go.Figure()

                # クラスタ→固定色の対応（点・勢力圏・ラベルで共通。フィルタに依らず
                # 全体集合 df_universe 基準で割り当てるため、絞り込んでも色が変わらない）
                _land_cmap = utils.landscape_color_map(df_universe['cluster']) if 'cluster' in df_universe.columns else {}
                _is_density_mode = (map_mode == "密度マップ (Density)")

                # 密度マップ（クラスタ別ソフト密度 = 技術の地形図スタイル）
                if not df_trend.empty and _is_density_mode:
                    # 全クラスタ共通のメッシュ定義（メッシュ細かさ設定を全域基準で適用・端は余白ビンで見切れ防止）
                    _bx, _by = utils.landscape_density_bins(df_trend['umap_x'], df_trend['umap_y'], main_mesh_size)
                    # 絶対評価ONは全クラスタ共通の濃さ基準（初回全体の最大ビン件数）を使う
                    _zmax = st.session_state.saturnv_global_zmax if use_abs_scale else None
                    for _cid, _grp in df_trend[df_trend['cluster'] != -1].groupby('cluster'):
                        utils.add_landscape_density(
                            fig_main, _grp['umap_x'], _grp['umap_y'],
                            _land_cmap.get(_cid, '#8C93A6'), zmax=_zmax, xbins=_bx, ybins=_by)

                # クラスタ領域（平滑化した勢力圏・点と同じ固定色）
                if map_mode == "クラスタ領域 (Clusters)" and not df_universe.empty and 'cluster' in df_universe.columns:
                    for _cid, _grp in df_universe[df_universe['cluster'] != -1].groupby('cluster'):
                        utils.add_landscape_region(
                            fig_main, _grp[['umap_x', 'umap_y']].values, _land_cmap.get(_cid, '#8C93A6'))

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
                    # 点は常時白フチ（重なっても粒が読める）。密度モードは地形が主役なので点を小さく薄く。
                    marker_line = dict(width=0.6 if _is_density_mode else 0.8, color='white')
                    _pt_size = 4 if _is_density_mode else utils.LANDSCAPE_POINT_SIZE
                    _pt_opacity = 0.65 if _is_density_mode else utils.LANDSCAPE_POINT_OPACITY
                    is_applicant_filtered = "ALL" not in applicant_values

                    if is_applicant_filtered:
                        # Applicant Drill Down Mode（企業識別色も俯瞰図パレットで統一）
                        for i, app_name in enumerate(applicant_values):
                            mask = df_focus[col_map['applicant']].fillna('').str.contains(re.escape(app_name))
                            df_app = df_focus[mask]
                            if not df_app.empty:
                                fig_main.add_trace(go.Scatter(
                                    x=df_app['umap_x'], y=df_app['umap_y'], mode='markers',
                                    marker=dict(color=utils.landscape_color(i), size=max(_pt_size, 6), opacity=0.9, line=marker_line),
                                    hoverinfo='text', hovertext=df_app['hover_text'], customdata=df_app.index, name=app_name
                                ))
                    else:
                        # Standard Mode (Cluster or Status Coloring)

                        if color_by == "ステータス" and _status_col_sv and _status_col_sv in df_focus.columns:
                            # ステータス別にパステル着色（個別特許点・クリック対応）
                            for _stt, _grp in df_focus.groupby(df_focus[_status_col_sv].fillna('(未設定)').astype(str)):
                                if not _grp.empty:
                                    fig_main.add_trace(go.Scatter(
                                        x=_grp['umap_x'], y=_grp['umap_y'], mode='markers',
                                        marker=dict(color=utils.status_color_for(_stt, vivid=True), size=max(_pt_size, 6),
                                                    opacity=0.9, line=marker_line),
                                        hoverinfo='text', hovertext=_grp['hover_text'], customdata=_grp.index,
                                        name=str(_stt), showlegend=True
                                    ))
                        else:
                            if 'cluster' in df_focus.columns:
                                df_focus_valid = df_focus[df_focus['cluster'] != -1]
                                df_focus_noise = df_focus[df_focus['cluster'] == -1]
                            else:
                                df_focus_valid = df_focus
                                df_focus_noise = pd.DataFrame()

                            # Plot Valid Clusters（勢力圏・ラベルと同じ固定色を点にも使う）
                            if not df_focus_valid.empty:
                                fig_main.add_trace(go.Scatter(
                                    x=df_focus_valid['umap_x'], y=df_focus_valid['umap_y'], mode='markers',
                                    marker=dict(
                                        color=[_land_cmap.get(_c, '#8C93A6') for _c in df_focus_valid['cluster']],
                                        size=_pt_size,
                                        opacity=_pt_opacity,
                                        line=marker_line
                                    ),
                                    hoverinfo='text', hovertext=df_focus_valid['hover_text'], customdata=df_focus_valid.index, name='特許 (Valid)'
                                ))

                            # Plot Noise (Separate Trace・背景に沈める)
                            if not df_focus_noise.empty:
                                fig_main.add_trace(go.Scatter(
                                    x=df_focus_noise['umap_x'], y=df_focus_noise['umap_y'], mode='markers',
                                    marker=dict(line=dict(width=0), **utils.LANDSCAPE_NOISE_STYLE),
                                    hoverinfo='text', hovertext=df_focus_noise['hover_text'], customdata=df_focus_noise.index, name='Noise'
                                ))

                # ラベル追加（ピル型・点/勢力圏と同色のドット付き・件数上位クラスタは強調）
                if show_labels_chk:
                    label_data_source = df_universe
                    target_cids = cluster_values if "ALL" not in cluster_values else label_data_source['cluster'].unique()

                    # Filter Noise from labels
                    valid_label_data = label_data_source[label_data_source['cluster'] != -1]
                    _sizes_by_cid = valid_label_data['cluster'].value_counts().to_dict()
                    _top_cids = utils.landscape_top_clusters(_sizes_by_cid)
                    # 端のクラスタのラベルが図の外に見切れないよう、表示範囲を渡してアンカーを調整
                    _lab_xr = (float(valid_label_data['umap_x'].min()), float(valid_label_data['umap_x'].max()))
                    _lab_yr = (float(valid_label_data['umap_y'].min()), float(valid_label_data['umap_y'].max()))

                    for cid, grp in valid_label_data[valid_label_data['cluster'].isin(target_cids)].groupby('cluster'):
                        mean_pos = grp[['umap_x', 'umap_y']].mean()
                        label_txt = grp['cluster_label'].iloc[0]
                        utils.add_landscape_label(
                            fig_main, mean_pos['umap_x'], mean_pos['umap_y'], label_txt,
                            _land_cmap.get(cid, '#8C93A6'), emphasized=(cid in _top_cids),
                            x_range=_lab_xr, y_range=_lab_yr)

                # 図内タイトルは置かない（ページ側に見出しがあり、スナップショットの
                # タイトルはレポート/スライド側で付与される設計のため）
                utils.update_fig_layout(fig_main, "", height=1200, show_legend=False)
                fig_main.update_layout(margin=dict(l=16, r=16, t=24, b=16))

                # 出願人フィルタ中（企業別カラー）／ステータス表示のときは凡例を表示する。
                # 凡例はマップ「上に重ねて」配置するため、マップ本体の大きさは変わらない。
                _show_legend_sv = ("ALL" not in applicant_values) or (color_by == "ステータス")
                if _show_legend_sv:
                    fig_main.update_layout(
                        showlegend=True,
                        legend=dict(
                            x=0.01, y=0.99, xanchor='left', yanchor='top',
                            bgcolor='rgba(255,255,255,0.78)', bordercolor='#cccccc', borderwidth=1,
                            font=dict(size=13),
                        ),
                    )


                # 1. アスペクト比: 歪みを防ぐため1:1を強制
                # 2. フォーカス: 有効なクラスタにズーム（ノイズ除外）

                # マップの表示範囲（ズーム）はフィルタに依存せず一定にする。絞り込み後の df_focus では
                # なく、全体集合 df_universe（有効クラスタ）から範囲を算出する。これにより「マップ表示
                # クラスタ」「出願人」等のフィルタを変えても俯瞰図の枠（位置・ズーム）が動かない。
                if 'cluster' in df_universe.columns:
                    target_df = df_universe[df_universe['cluster'] != -1]
                    if target_df.empty:
                        target_df = df_universe
                else:
                    target_df = df_universe

                if not target_df.empty:
                     # Calculate bounds
                     x_min, x_max = target_df['umap_x'].min(), target_df['umap_x'].max()
                     y_min, y_max = target_df['umap_y'].min(), target_df['umap_y'].max()

                     # Calculate spread
                     x_range = x_max - x_min
                     y_range = y_max - y_min

                     # Add Padding（密度モードは等高線のすそ野が点群の外に広がるため広めに取る）
                     pad_factor = 0.06 if _is_density_mode else 0.02
                     x_pad = x_range * pad_factor if x_range > 0 else 1.0
                     y_pad = y_range * pad_factor if y_range > 0 else 1.0

                     # Apply new ranges with Fixed Aspect Ratio matching
                     # constrain="domain": 範囲を厳守し、1:1アスペクトはプロット領域側で調整する。
                     # これにより、ラベル(注釈)の有無で表示範囲が広がって俯瞰図が縮む現象を防ぐ。
                     fig_main.update_layout(
                        height=1200,
                        xaxis=dict(range=[x_min - x_pad, x_max + x_pad], autorange=False, constrain="domain"),
                        yaxis=dict(
                            range=[y_min - y_pad, y_max + y_pad],
                            autorange=False,
                            scaleanchor="x",
                            scaleratio=1,
                            constrain="domain"
                        )
                     )

                st.session_state['saturnv_main_fig'] = fig_main
                st.session_state['saturnv_main_fig_key'] = _main_fig_key

            # 絶対評価スケール中は、密度の濃さ基準が全表示で共通であることを図の下に明示する
            if map_mode == "密度マップ (Density)" and use_abs_scale and st.session_state.saturnv_global_zmax:
                st.caption("絶対評価スケールで表示中 — 期間などを絞り込んでも、濃さの基準は全体マップと共通です。")

            @st.fragment
            def _saturnv_click_main():
                utils.disable_selection_fade(fig_main)
                selection_main = st.plotly_chart(fig_main, use_container_width=True,
                    on_select="rerun", selection_mode="points", key="saturnv_main_map", config={
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
                # クリックした特許点 → 詳細ポップアップ（クラスタ領域/ラベルは customdata 無しのため反応しない）
                utils.handle_map_click(selection_main, "saturnv_main", title="クリックした特許")
            _saturnv_click_main()

            # 読み方カード（ランドスケープ図の直下）
            apollo_ui.howto(
                "<b>近くにある点</b>ほど内容が似た特許です。<b>同じ色のかたまり</b>がひとつの技術グループ（クラスタ）で、"
                "<b>等高線や色の濃さ</b>は特許が密集している場所（激戦区）を表します。"
                "<b>灰色の点</b>はどのグループにも入らなかった未分類（ノイズ）— 新技術の芽が隠れていることがあります。"
            )

            # --- しっくりこないときの調整（意図ベース・パラメータ名を出さない） ---
            with st.container(border=True):
                # 現在地は df から実測（手動モード実行後は main_auto_result が無いため）
                try:
                    _lab_now = st.session_state.df_main['cluster']
                    _k_now = int(len(set(_lab_now.unique()) - {-1}))
                    _k_txt = f"（現在 {_k_now} クラスタ・未分類 {float((_lab_now == -1).mean()) * 100:.0f}%）"
                except Exception:
                    _k_txt = ""
                st.markdown(
                    f"**このマップ、しっくりきましたか？**{_k_txt} — "
                    "ボタンを押すとその場で描き直します。何回でも試せます。")
                # 直前の意図ボタン実行の結果通知。st.toast は直後の rerun をまたぐと
                # 表示されないことがあるため使わず、カード内のインライン表示で知らせる
                # （pop により表示は1描画限り＝次の操作で自然に消える）
                _fb = st.session_state.pop('terra_intent_feedback', None)
                if _fb:
                    if _fb.get('changed'):
                        st.success(_fb['msg'], icon=":material/check_circle:")
                    else:
                        st.info(_fb['msg'], icon=":material/info:")
                _ib1, _ib2, _ib3 = st.columns(3)
                _ib1.button(":material/search: もっと細かく分ける", key="terra_intent_finer", use_container_width=True,
                            on_click=_terra_adjust_intent, args=('finer',),
                            disabled=st.session_state.main_cluster_running,
                            help="クラスタの数を増やして、より細かい技術テーマまで分けます")
                _ib2.button(":material/map: もっと大きなくくりに", key="terra_intent_coarser", use_container_width=True,
                            on_click=_terra_adjust_intent, args=('coarser',),
                            disabled=st.session_state.main_cluster_running,
                            help="クラスタの数を減らして、大まかな技術グループにまとめます")
                _ib3.button(":material/label: 未分類（灰色）を減らす", key="terra_intent_noise", use_container_width=True,
                            on_click=_terra_adjust_intent, args=('less_noise',),
                            disabled=st.session_state.main_cluster_running,
                            help="どのグループにも入らなかった特許を、できるだけ近くのグループに取り込みます")
                st.caption("細かい調整（クラスタの粒度・目標数の直接指定）は上の「詳細設定」からもできます。")

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

            snap_data['chart_data'] = utils_ai.df_to_markdown(df_snap_safe, index=False)


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
                **マップタイプ**: 技術ランドスケープ（俯瞰図分析・メインマップ）
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
                    title=f"俯瞰図マップ ({date_bin_filter_w})",
                    description="Global technology landscape map showing cluster distribution and proximity based on semantic similarity.",
                    fig=fig_main_snap,
                    data_summary=snap_data,
                    key="saturn_main_snap"
                )
            except Exception as e:
                 st.error(f"スナップショット生成エラー: {e}")
            apollo_ui.snapshot_hint()

            # :material/image: 整理版ランドスケープ（スライド用・上位クラスタのみ）
            if st.checkbox(":material/image: 整理版ランドスケープ（スライド/レポート用・上位クラスタのみ）を表示", key="saturn_curated_show"):
                st.caption("全クラスタを密にラベルすると重なるため、件数上位クラスタだけを大きく示したスライド向けの俯瞰図です。")
                _cl_topn = st.slider("ラベル表示するクラスタ数（件数上位）", 3, 15, 8, key="saturn_curated_topn", help="整理版（スライド/レポート用）の俯瞰図で、件数が多い上位何クラスタにだけ大きなラベルを付けるかです。全クラスタに付けると重なって読めないため、主要クラスタだけを強調します。")
                _cl_style_lbl = st.radio("表示モード:", ["クラスタ領域 (Clusters)", "密度マップ (Density)", "散布図 (Scatter)"],
                                         horizontal=True, key="saturn_curated_style", help="俯瞰図の描き方を切り替えます。クラスタ領域＝各クラスタを色付きの領域で表示、密度マップ＝点の混み具合をヒートマップで表示、散布図＝個々の特許を点で表示。全体像は領域、混雑度は密度、個別確認は散布図が見やすいです。")
                _cl_style = {"クラスタ領域 (Clusters)": "hull", "密度マップ (Density)": "density", "散布図 (Scatter)": "points"}.get(_cl_style_lbl, "hull")
                try:
                    _cl_fig = utils.build_curated_landscape(
                        df_universe, cluster_col='cluster', label_col='cluster_label',
                        x_col='umap_x', y_col='umap_y', top_n=_cl_topn, region_style=_cl_style,
                        density_bins=int(st.session_state.get('main_mesh_size', 30)))
                    st.plotly_chart(_cl_fig, use_container_width=True, config={'editable': False})
                    _n_all = int(df_universe[df_universe['cluster'] != -1]['cluster'].nunique()) if 'cluster' in df_universe.columns else 0
                    # CAPCOMアクティブ時はクリーンPNGをZIPに同梱（スライド/レポート用の整理版を下流へ）
                    utils.save_curated_to_capcom(
                        _cl_fig, snap_id="saturnv_curated_landscape",
                        cache_token=f"{_cl_topn}_{_cl_style}")
                    utils.render_report_png_button(
                        _cl_fig, key="saturn_curated_report",
                        default_title="技術ランドスケープ：主要クラスタ",
                        default_subtitle=f"出願 {len(df_universe):,}件 / 上位{_cl_topn}クラスタ（全{_n_all}クラスタ）",
                        default_caption="",
                        label=":material/palette: 整理版をスライド/レポート用PNGに書き出す")
                except Exception as _e:
                    st.warning(f"整理版ランドスケープの生成に失敗しました: {_e}")

            if st.session_state.saturnv_cluster_done:
                # AIインサイト (メインマップ)
                insight_context = f"""
                **マップタイプ**: 技術ランドスケープ（俯瞰図分析・メインマップ）
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

                # クラスタ動態（前回実行時に session_state へ保存済み）とノイズ（萌芽技術）を insight に反映。
                # これらは本 insight より後段で計算されるため、前回ラン分を参照する（CAPCOM JSON出力と同方式）。
                _sv_noise = int((df_universe['cluster'] == -1).sum()) if -1 in df_universe['cluster'].values else 0
                _sv_map_extra, _sv_map_inst = utils_ai.build_map_dynamics_noise_addon(
                    dynamics_data=st.session_state.get('saturnv_dynamics_data'),
                    noise_count=_sv_noise, total_count=len(df_universe))

                # 生成 (空間情報・クラスタ動態・ノイズを追加コンテキストとして渡す)
                prompt = utils_ai.generate_ai_insight_prompt(
                    insight_role, insight_context, snap_data, insight_instruction + _sv_map_inst,
                    extra_content=f"\n# 空間配置情報 (Spatial Context)\n{spatial_info}\n{_sv_map_extra}"
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
                            _cnt = cluster_counts_snap.get(cid, 0)
                            count = int(_cnt) if pd.notna(_cnt) else 0
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
                                "cluster_id": int(cid) if pd.notna(cid) else -1,
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
                                                   _df_noise_capcom.get(_abstract_col, pd.Series([''] * len(_df_noise_capcom), index=_df_noise_capcom.index)).fillna(''))
                                    _noise_kws = _noise_texts.apply(lambda x: utils.extract_keywords(x, stopwords=stopwords))
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
                    st.caption(f":material/warning: クラスタ（俯瞰図） の CAPCOM 保存に失敗しました（要確認）: {e}")

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

                with st.expander(f":material/search: ノイズ（未分類）分析 ({noise_count}件 / {noise_ratio:.1%})", expanded=False):
                    # 1. ノイズ率の解釈（自動解釈は howto カード形式で表示・内容は旧版のまま）
                    if noise_ratio < 0.05:
                        noise_interp = "成熟・均質な技術領域（ノイズ率 < 5%）"
                    elif noise_ratio < 0.15:
                        noise_interp = "標準的・安定構造（ノイズ率 5-15%）"
                    elif noise_ratio < 0.30:
                        noise_interp = "多様・融合活発（ノイズ率 15-30%）"
                    else:
                        noise_interp = "萌芽・黎明期（ノイズ率 > 30%）"

                    apollo_ui.howto(
                        f"どのクラスタにも入らなかった特許（ノイズ）の割合は <b>{noise_ratio:.1%}</b>。"
                        f"この母集団は <b>{noise_interp}</b> と読み取れます。",
                        title="ノイズ率の自動解釈")

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
                                    apollo_ui.howto(
                                        "<span class=msym>trending_up</span> <b>近年集中</b>: ノイズの多くが直近3年に集中 → <b>新興テーマの可能性</b>",
                                        title="時系列パターンの自動解釈")
                                elif recent_ratio < 0.2:
                                    apollo_ui.howto(
                                        "<span class=msym>bar_chart</span> <b>過去集中</b>: ノイズの多くが過去に分布 → <b>歴史的バリエーション</b>",
                                        title="時系列パターンの自動解釈")
                                else:
                                    apollo_ui.howto(
                                        "<span class=msym>trending_down</span> <b>均一分布</b>: ノイズが期間全体に分布 → <b>永続的ニッチ</b>",
                                        title="時系列パターンの自動解釈")

                                # 注: ここで px を再importしない（関数内importはrender()全体で名前を
                                # ローカル化し、これより前の使用箇所が UnboundLocalError になる）
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
                                app_counts = Counter(all_applicants)
                                top_apps = app_counts.most_common(10)
                                top1_share = top_apps[0][1] / noise_count if top_apps else 0
                                if top1_share > 0.3:
                                    apollo_ui.howto(
                                        f"<span class=msym>crisis_alert</span> <b>集中</b>: 上位出願人 '{top_apps[0][0]}' がノイズの{top1_share:.0%}を占める → <b>戦略的ニッチ</b>",
                                        title="出願人分布の自動解釈")
                                else:
                                    apollo_ui.howto(
                                        "<span class=msym>language</span> <b>分散</b>: 多数の出願人にノイズが分布 → <b>標準化前段階</b>",
                                        title="出願人分布の自動解釈")

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
                                              df_noise.get(abstract_col, pd.Series([''] * len(df_noise), index=df_noise.index)).fillna(''))
                                noise_kws = noise_texts.apply(lambda x: utils.extract_keywords(x, stopwords=stopwords))

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
                                st.caption(":material/menu_book: ノイズ分析データは記録セッションに蓄積されます")
                        except Exception:
                            pass

            # --- クラスタ動態マップ ---
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

    # --- ドリルダウン（旧 PROBE） ---
    with tab_drill:
        # --- おすすめ実行バー（メインマップと同じ型: クラスタを選んで1ボタン） ---
        drilldown_options = [('(選択してください)', 'NONE')]
        if "saturnv_labels_map" in st.session_state:
            drilldown_options += [(f"{label} ({count}件)", cid) for cid, label in st.session_state.saturnv_labels_map.items() if cid != -1 for count in [st.session_state.df_main['cluster'].value_counts().get(cid, 0)]]

        with st.container(border=True):
            _db_l, _db_r = st.columns([2.6, 1], vertical_alignment="center")
            with _db_l:
                selected_drilldown_target_drill = st.selectbox(
                    "深掘りするクラスタを選んでください:", options=drilldown_options,
                    format_func=lambda x: x[0], key="drill_target_select",
                    help="俯瞰図マップで見つかった技術テーマ（クラスタ）の中身を、さらに細かいサブクラスタに分けて詳しく見ます。")
                st.caption(":material/smart_toy: おすすめ設定でそのまま描けます — サブクラスタの粒度は自動調整。期間・出願人の絞り込みなどは下の「詳細設定」からできます。")
            with _db_r:
                drill_run_clicked = st.button(
                    ":material/map: 選択クラスタの俯瞰図を描く", type="primary", key="drill_run_button",
                    use_container_width=True,
                    disabled=(selected_drilldown_target_drill[1] == "NONE"))
        drilldown_target_id = selected_drilldown_target_drill[1]

        if drilldown_target_id == "NONE":
            df_subset_filter = pd.DataFrame(columns=df_main.columns)
            if not st.session_state.saturnv_cluster_done:
                st.info("先に「俯瞰図マップ」タブで描画すると、ここでクラスタを深掘りできます。")
        else:
            df_subset_filter = df_main[df_main['cluster'] == drilldown_target_id].copy()

        def on_drill_interval_change():
            if "drill_date_filter_w" in st.session_state: del st.session_state.drill_date_filter_w

        # --- 詳細設定（フィルタ・粒度・ラベルはここに格納。通常は触らなくてよい） ---
        with st.expander("詳細設定（期間・出願人フィルタ、サブクラスタの粒度・ラベル）", expanded=False):
            st.markdown("**フィルタ（期間・出願人）**")
            if drilldown_target_id != "NONE":
                col1, col2 = st.columns(2)
                with col1:
                    if 'year' in df_subset_filter.columns and df_subset_filter['year'].notna().any():
                        drill_bin_interval_w_val = st.selectbox("期間の粒度:", [5, 3, 2, 1], index=0, key="drill_interval_w", on_change=on_drill_interval_change, help="表示期間を区切る年数の幅です。大きくすると5年単位などの大まかな期間に、小さくすると1年単位など細かい期間に分かれます（下の「表示期間」の選択肢が変わります）。")
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
            else:
                st.caption("上でクラスタを選ぶと、期間・出願人のフィルタが表示されます。")

            st.markdown("**サブクラスタの粒度・ラベル**")
            # :material/smart_toy: 自動最適化（メインと同じ HDBSCAN 2パラメータ掃引をドリルダウンにも・既定ON）
            drill_auto_hdbscan = st.checkbox(
                ":material/smart_toy: 自動最適化（最小クラスタサイズ・最小サンプル数を掃引してサブクラスタ数を適正化）",
                value=True, key="drill_auto_hdbscan",
                help="ドリルダウン対象の件数に合わせて HDBSCAN の2パラメータを自動で掃引し、サブクラスタ数が少なすぎ(2,3)も多すぎもしない設定を自動選択します（手動調整が不要になります）。ONの間は下の手動値は無視されます。")
            drill_target_k_w = None
            if drill_auto_hdbscan:
                # メインマップと同じく、対象の件数に応じて目標サブクラスタ数を初期化する（suggest_target_k）。
                # 対象クラスタを変えるたびに初期値が件数適応されるよう、key に対象IDを含める。
                _n_sub = len(df_subset_filter) if drilldown_target_id != "NONE" else 0
                drill_target_k_w = st.number_input(
                    "目標サブクラスタ数（目安）", min_value=2, max_value=80,
                    value=utils.suggest_target_k(_n_sub), key=f"drill_target_k_w_{drilldown_target_id}",
                    help="この数に近づくよう2パラメータを掃引します（対象の件数から自動初期化）。"
                         "結果のサブクラスタ数や粒度に満足できない場合は、この値を増減して再度「選択クラスタの俯瞰図を描く」を押してください。"
                         "実際の値は密度構造に依存するため、目標ちょうどにならないこともあります（品質 DBCV を優先して選びます）。")
                utils.render_dbcv_help()
                # 前回の自動決定を常時再表示（メインマップと同じ挙動）。対象が一致する結果のみ表示する。
                _dar = st.session_state.get('saturnv_drill_auto_result')
                if _dar and _dar.get('target_id') == drilldown_target_id:
                    _rv = _dar.get('validity')
                    _rv_txt = f"・地図の品質（DBCV）={_rv:.2f}" if isinstance(_rv, (int, float)) else ""
                    st.info(
                        f":material/smart_toy: 前回の自動決定: 最小クラスタサイズ=**{_dar['mcs']}** / 最小サンプル数=**{_dar['ms']}** "
                        f"→ サブクラスタ **{_dar['k']}**・ノイズ {_dar['noise'] * 100:.1f}%（目標≈{_dar['target_k']}{_rv_txt}）。"
                        f"　数が合わない/粒度が好みでない場合は上の「目標サブクラスタ数」を変えて再描画してください。")
            col1, col2, col3 = st.columns(3)
            with col1: drill_min_cluster_size_w = st.number_input('最小クラスタサイズ:', min_value=2, value=5, key="drill_min_cluster_size_w", disabled=drill_auto_hdbscan, help="ドリルダウン（選択クラスタの内部再分割）で、1つのサブクラスタとして認める最小の特許件数です（クラスタの粒度設定）。小さくすると細かい技術テーマまで分かれてクラスタ数が増えますが、どこにも属さないノイズ（外れ値）も増えます。大きくすると少数の大まかなクラスタにまとまり安定しますが、細部は埋もれます。対象件数が少ないため小さめの値が向きます。")
            with col2: drill_min_samples_w = st.number_input('最小サンプル数:', min_value=1, value=5, key="drill_min_samples_w", disabled=drill_auto_hdbscan, help="ドリルダウンで、サブクラスタの「核」と認める密度の厳しさです。大きいほど判定が厳しくなり、ノイズ（外れ値）が増えてクラスタは密な中心部だけになります。小さいほど緩くなり、多くの点がクラスタに取り込まれます。通常は最小クラスタサイズ以下に設定します。")
            with col3: drill_label_top_n_w = st.number_input('ラベル単語数:', min_value=1, value=3, key="drill_label_top_n_w", help="各サブクラスタの自動命名に使う特徴語の数です。クラスタを特徴づける語を上位から何語ラベルに並べるかを決めます。多いほど内容を詳しく表せますが冗長になり、少ないほど簡潔になります。")
            drill_show_labels_chk = st.checkbox('マップにラベルを表示する', value=True, key="drill_show_labels_chk", help="各サブクラスタの自動命名ラベルをマップ上に重ねて表示します。オフにすると点の分布だけが見えるため、ラベルが重なって読みにくいときに使います。")

        # terra_drill_intent_rerun / ctx: ドリルダウン版「しっくりこない」ボタンの自動再実行フラグと制約
        _drill_intent_rerun = st.session_state.pop('terra_drill_intent_rerun', False)
        _drill_intent_ctx = st.session_state.pop('terra_drill_intent_ctx', None)
        if not _drill_intent_rerun:
            _drill_intent_ctx = None
        if drill_run_clicked or _drill_intent_rerun:
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
                        if drill_app_values and "ALL" not in drill_app_values:
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

                            if drill_auto_hdbscan:
                                _dpb = st.progress(0.0, text="サブクラスタの粒度を自動調整中（パラメータ掃引）...")
                                # 意図ボタン経由なら「今の粒度と同じ解の再選択」を範囲制約で禁じる
                                _dsweep_kwargs = {}
                                if _drill_intent_ctx and _drill_intent_ctx.get('k_min'):
                                    _dsweep_kwargs['k_min'] = int(_drill_intent_ctx['k_min'])
                                if _drill_intent_ctx and _drill_intent_ctx.get('k_max'):
                                    _dsweep_kwargs['k_max'] = int(_drill_intent_ctx['k_max'])
                                _dsweep = utils.sweep_hdbscan_params(
                                    embedding_drill, target_k=int(drill_target_k_w),
                                    progress_callback=lambda f: _dpb.progress(min(f, 1.0), text="サブクラスタの粒度を自動調整中（パラメータ掃引）..."),
                                    **_dsweep_kwargs)
                                _dpb.empty()
                                df_subset['drill_cluster'] = _dsweep['labels']
                                # 自動決定の結果を保持し、次回以降も「前回の自動決定」として再表示する（メインマップと同じ）。
                                st.session_state['saturnv_drill_auto_result'] = {
                                    'mcs': _dsweep['min_cluster_size'], 'ms': _dsweep['min_samples'],
                                    'k': _dsweep['n_clusters'], 'noise': _dsweep['noise_ratio'],
                                    'target_k': _dsweep['target_k'], 'validity': _dsweep.get('validity'),
                                    'target_id': drilldown_target_id}
                                _drv = _dsweep.get('validity')
                                _drv_txt = f"・地図の品質（DBCV）={_drv:.2f}" if isinstance(_drv, (int, float)) else ""
                                st.caption(
                                    f":material/smart_toy: 自動決定: 最小クラスタサイズ={_dsweep['min_cluster_size']} / 最小サンプル数={_dsweep['min_samples']} "
                                    f"→ サブクラスタ {_dsweep['n_clusters']}・ノイズ {_dsweep['noise_ratio'] * 100:.0f}%（目標≈{_dsweep['target_k']}{_drv_txt}）")
                            else:
                                clusterer_drill = hdbscan.HDBSCAN(min_cluster_size=int(drill_min_cluster_size_w), min_samples=int(drill_min_samples_w), metric='euclidean', cluster_selection_method='eom')
                                df_subset['drill_cluster'] = clusterer_drill.fit_predict(embedding_drill)

                            # utils.safe_auto_label で c-TF-IDF ラベリング（ドリルダウン、Janome 例外耐性付き）
                            drill_texts = utils.combine_text_columns(
                                df_subset, col_map, keys=('title', 'abstract'))
                            drill_labels_map = utils.safe_auto_label(
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
                            # 意図ボタン経由の場合: 変化の有無の通知を預ける（カード描画時にインライン表示）
                            if _drill_intent_ctx:
                                _dnk = int(len(set(df_subset['drill_cluster'].unique()) - {-1}))
                                _dnz = float((df_subset['drill_cluster'] == -1).mean())
                                _dim = _drill_intent_ctx.get('mode')
                                _dpk = _drill_intent_ctx.get('prev_k')
                                _dpn = _drill_intent_ctx.get('prev_noise')
                                _dfb = None
                                if _dim == 'finer':
                                    if _dpk and _dnk > _dpk:
                                        _dfb = {'msg': f"サブクラスタ数が {_dpk} → {_dnk} に増えました", 'changed': True}
                                    elif _dpk:
                                        _dfb = {'msg': f"これ以上細かく分かれませんでした（サブクラスタ数 {_dnk} のまま）。"
                                                       "データの密度構造の限界です", 'changed': False}
                                elif _dim == 'coarser':
                                    if _dpk and _dnk < _dpk:
                                        _dfb = {'msg': f"サブクラスタ数が {_dpk} → {_dnk} にまとまりました", 'changed': True}
                                    elif _dpk:
                                        _dfb = {'msg': f"これ以上大きなくくりにできませんでした（サブクラスタ数 {_dnk} のまま）",
                                                'changed': False}
                                elif _dim == 'less_noise':
                                    if _dpn is not None and _dnz < _dpn - 1e-9:
                                        _dfb = {'msg': f"未分類が {_dpn * 100:.0f}% → {_dnz * 100:.0f}% に減りました",
                                                'changed': True}
                                    elif _dpn is not None:
                                        _dfb = {'msg': f"未分類は減りませんでした（{_dnz * 100:.0f}%）。"
                                                       "もう一度押すと、さらに条件を緩めて試します", 'changed': False}
                                if _dfb:
                                    st.session_state['terra_drill_intent_feedback'] = _dfb
                            st.success("完了しました。")
                            st.rerun()

                    except Exception as e:
                        st.error(f"エラー: {e}")

        if "df_drilldown_result" in st.session_state:
            df_drill = st.session_state.df_drilldown_result.copy()
            drill_labels_map = st.session_state.drill_labels_map



            st.subheader("ドリルダウンマップ")

            # --- UIレイアウト ---
            drill_map_mode = st.radio("表示モード:", ["クラスタ領域 (Clusters)", "密度マップ (Density)", "散布図 (Scatter)"], horizontal=True, key="drill_map_mode_radio", help="俯瞰図の描き方を切り替えます。クラスタ領域＝各クラスタを色付きの領域で表示、密度マップ＝点の混み具合をヒートマップで表示、散布図＝個々の特許を点で表示。全体像は領域、混雑度は密度、個別確認は散布図が見やすいです。")

            d_c1, d_c2, d_c3 = st.columns(3)
            with d_c1:
                st.markdown("**密度マップ設定**")
                drill_mesh_size = st.number_input("メッシュサイズ (Grid)", value=40, min_value=10, max_value=200, step=5, key="drill_mesh_size", help="密度マップ（ヒートマップ）を描くときの格子の細かさです。大きいほど細かい格子になり局所的な濃淡が見えますが、点がまばらだと粗く見えます。小さいほど滑らかで大まかな分布になります。")
            with d_c2:
                st.markdown("**フィルタ**")
                drill_remove_noise_chk = st.checkbox("ノイズを除く (Exclude Noise)", value=False, key="drill_remove_noise", help="どのサブクラスタにも属さないノイズ（外れ値）の特許をマップから隠します。オンにすると主要なサブクラスタ構造がすっきり見えますが、外れ値の特許は見えなくなります。")
            with d_c3:
                st.empty()

            if drill_remove_noise_chk:
                df_drill_plot = df_drill[df_drill['drill_cluster'] != -1]
            else:
                df_drill_plot = df_drill

            fig_drill = go.Figure()

            # サブクラスタ→固定色（点・勢力圏・ラベル共通。メインマップと同じ俯瞰図パレット）
            _land_cmap_d = utils.landscape_color_map(df_drill_plot['drill_cluster'])
            _is_density_d = (drill_map_mode == "密度マップ (Density)")

            if _is_density_d:
                # クラスタ別ソフト密度（技術の地形図スタイル・全サブクラスタ共通メッシュ・端は余白ビンで見切れ防止）
                _dbx, _dby = utils.landscape_density_bins(df_drill_plot['drill_x'], df_drill_plot['drill_y'], drill_mesh_size)
                for _cid, _grp in df_drill_plot[df_drill_plot['drill_cluster'] != -1].groupby('drill_cluster'):
                    utils.add_landscape_density(
                        fig_drill, _grp['drill_x'], _grp['drill_y'],
                        _land_cmap_d.get(_cid, '#8C93A6'), xbins=_dbx, ybins=_dby)

            if drill_map_mode == "クラスタ領域 (Clusters)":
                # 平滑化した勢力圏（点と同じ固定色）
                for _cid, _grp in df_drill_plot[df_drill_plot['drill_cluster'] != -1].groupby('drill_cluster'):
                    utils.add_landscape_region(
                        fig_drill, _grp[['drill_x', 'drill_y']].values, _land_cmap_d.get(_cid, '#8C93A6'))

            # 点は常時白フチ。密度モードは地形が主役なので点を小さく薄く（メインマップと同じ規則）。
            marker_line_d = dict(width=0.6 if _is_density_d else 0.8, color='white')
            _pt_size_d = 4 if _is_density_d else utils.LANDSCAPE_POINT_SIZE
            _pt_opacity_d = 0.65 if _is_density_d else utils.LANDSCAPE_POINT_OPACITY

            # Split Data into Signal (Valid) and Noise
            df_drill_valid = df_drill_plot[df_drill_plot['drill_cluster'] != -1]
            df_drill_noise = df_drill_plot[df_drill_plot['drill_cluster'] == -1]

            # 1. Plot Noise (背景に沈める) - Always plot if present and not filtered out
            # 凸包（go.Scatter=SVG）と座標系を揃えるため、プロット点も go.Scatter を使う。
            # go.Scattergl（WebGL）だと凸包領域とプロット点がズレる（メイン俯瞰図は両方 Scatter で
            # ズレない）。サブクラスタは点数が少なく Scatter でも性能問題はない。
            if not df_drill_noise.empty:
                fig_drill.add_trace(go.Scatter(
                    x=df_drill_noise['drill_x'], y=df_drill_noise['drill_y'], mode='markers',
                    marker=dict(line=dict(width=0), **utils.LANDSCAPE_NOISE_STYLE),
                    hoverinfo='text', hovertext=df_drill_noise['drill_hover_text'], name='Noise'
                ))

            # 2. Plot Valid Sub-clusters（勢力圏・ラベルと同じ固定色を点にも使う）
            if not df_drill_valid.empty:
                fig_drill.add_trace(go.Scatter(
                    x=df_drill_valid['drill_x'], y=df_drill_valid['drill_y'], mode='markers',
                    marker=dict(
                        color=[_land_cmap_d.get(_c, '#8C93A6') for _c in df_drill_valid['drill_cluster']],
                        size=_pt_size_d,
                        opacity=_pt_opacity_d,
                        line=marker_line_d
                    ),
                    hoverinfo='text', hovertext=df_drill_valid['drill_hover_text'], name='Sub-clusters'
                ))

            if drill_show_labels_chk:
                # ピル型ラベル（点/勢力圏と同色ドット・件数上位サブクラスタは強調・端は見切れ防止）
                _dsizes = df_drill_valid['drill_cluster'].value_counts().to_dict()
                _dtop = utils.landscape_top_clusters(_dsizes)
                _dlab_xr = (float(df_drill_plot['drill_x'].min()), float(df_drill_plot['drill_x'].max()))
                _dlab_yr = (float(df_drill_plot['drill_y'].min()), float(df_drill_plot['drill_y'].max()))
                for cid, grp in df_drill_plot[df_drill_plot['drill_cluster'] != -1].groupby('drill_cluster'):
                    mean_pos = grp[['drill_x', 'drill_y']].mean()
                    utils.add_landscape_label(
                        fig_drill, mean_pos['drill_x'], mean_pos['drill_y'],
                        drill_labels_map.get(cid, ""), _land_cmap_d.get(cid, '#8C93A6'),
                        emphasized=(cid in _dtop), x_range=_dlab_xr, y_range=_dlab_yr)
            # 図内タイトルは置かず（メインマップと同じ規則）、対象はキャプションで示す
            utils.update_fig_layout(fig_drill, "", height=1000, show_legend=False)
            fig_drill.update_layout(margin=dict(l=16, r=16, t=24, b=16))
            st.caption(f"ドリルダウン対象: {st.session_state.drill_base_label}")
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

            # 読み方カード（ドリルダウンマップの直下）
            apollo_ui.howto(
                "<b>選んだクラスタの中だけ</b>を取り出し、座標を計算し直した詳細地図です。"
                "<b>色ごとのかたまり</b>がサブクラスタ（より細かい技術テーマ）、<b>灰色の点</b>はどのサブクラスタにも入らなかった特許です。"
                "座標は俯瞰図マップとは別に計算しているため、<b>点の位置は親マップとは対応しません</b>。"
            )

            # --- しっくりこないときの調整（意図ベース・メインマップと同じ操作感） ---
            with st.container(border=True):
                try:
                    _dlab_now = df_drill['drill_cluster']
                    _dk_now = int(len(set(_dlab_now.unique()) - {-1}))
                    _dk_txt = f"（現在 {_dk_now} サブクラスタ・未分類 {float((_dlab_now == -1).mean()) * 100:.0f}%）"
                except Exception:
                    _dk_txt = ""
                st.markdown(
                    f"**この分け方、しっくりきましたか？**{_dk_txt} — "
                    "ボタンを押すと同じ対象・フィルタのまま、その場で分け直します。")
                # 直前の意図ボタン実行の結果通知（インライン表示・1描画限り）
                _dfb = st.session_state.pop('terra_drill_intent_feedback', None)
                if _dfb:
                    if _dfb.get('changed'):
                        st.success(_dfb['msg'], icon=":material/check_circle:")
                    else:
                        st.info(_dfb['msg'], icon=":material/info:")
                _dib1, _dib2, _dib3 = st.columns(3)
                _dib_disabled = (drilldown_target_id == "NONE")
                _dib1.button(":material/search: もっと細かく分ける", key="terra_drill_intent_finer", use_container_width=True,
                             on_click=_terra_drill_adjust_intent, args=('finer',),
                             disabled=_dib_disabled,
                             help="サブクラスタの数を増やして、より細かい技術テーマまで分けます")
                _dib2.button(":material/map: もっと大きなくくりに", key="terra_drill_intent_coarser", use_container_width=True,
                             on_click=_terra_drill_adjust_intent, args=('coarser',),
                             disabled=_dib_disabled,
                             help="サブクラスタの数を減らして、大まかなテーマにまとめます")
                _dib3.button(":material/label: 未分類（灰色）を減らす", key="terra_drill_intent_noise", use_container_width=True,
                             on_click=_terra_drill_adjust_intent, args=('less_noise',),
                             disabled=_dib_disabled,
                             help="どのサブクラスタにも入らなかった特許を、できるだけ近くのまとまりに取り込みます")
                if _dib_disabled:
                    st.caption("上で「深掘りするクラスタ」を選び直すとボタンが使えます。")
                else:
                    st.caption("細かい調整（目標サブクラスタ数の直接指定）は上の「詳細設定」からもできます。")

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

            snap_data['chart_data'] = utils_ai.df_to_markdown(df_snap_safe_d, index=False)


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
            # ドリルダウンはクラスタを切り替えるたびにビューが変わるため、key を対象クラスタ別にする。
            # 静的キーだと1枚撮った後に別クラスタへ切り替えても「保存済み」のまま出てしまう。
            _drill_snap_slug = re.sub(r'\s+', '_', str(st.session_state.get('drill_base_label', 'target')))[:30] or 'target'
            utils.render_snapshot_button(
                title=f"俯瞰図ドリルダウン - {st.session_state.drill_base_label}",
                description=f"Detailed analysis of cluster: {st.session_state.drill_base_label}",
                fig=fig_drill,
                data_summary=snap_data,
                key=f"saturn_drill_snap_{_drill_snap_slug}",
                group="saturn_drill_snap"
            )
            apollo_ui.snapshot_hint()

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
                        _dcnt = drill_counts.get(cid, 0)
                        count = int(_dcnt) if pd.notna(_dcnt) else 0
                        cid_mask = df_drill['drill_cluster'] == cid
                        cx = float(df_drill.loc[cid_mask, 'drill_x'].mean()) if cid_mask.any() else 0
                        cy = float(df_drill.loc[cid_mask, 'drill_y'].mean()) if cid_mask.any() else 0
                        reps_full = drill_reps.get(cid, []) if cid in drill_reps else []
                        reps_raw = [r[:200] if len(r) > 200 else r for r in reps_full]
                        drill_clusters_json.append({
                            "cluster_id": int(cid) if pd.notna(cid) else -1,
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
                    # 親クラスタ別ファイル名で保存（複数クラスタを続けて深掘りしても上書きされない）
                    _probe_tgt = str(st.session_state.get('drill_base_label', '') or 'cluster')
                    _probe_slug = re.sub(r'\s+', '_', _probe_tgt).strip('_')[:40] or 'cluster'
                    capcom.save_data(f"saturnv_drilldown_{_probe_slug}.json", drill_json)
            except Exception as e:
                st.caption(f":material/warning: クラスタ詳細（ドリルダウン） の CAPCOM 保存に失敗しました（要確認）: {e}")

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
            st.subheader("クラスタ・テキスト分析（テキストマイニング）")
            col_tm1, col_tm2 = st.columns(2)
            with col_tm1:
                cooc_top_n = st.slider("共起: 上位単語数", 30, 100, 70, key="cooc_top_n", help="ネットワークに表示するキーワード（ノード）の数です。出現頻度の上位から何語を使うかを決めます。多いほど網羅的ですが密集して読みにくく、少ないほど主要語に絞られ見やすくなります。")
                cooc_threshold = st.slider("共起: Jaccard係数 閾値", 0.01, 0.3, 0.03, 0.01, key="cooc_threshold", help="2つのキーワードを線（エッジ）で結ぶ基準の強さです。共起の強さを0〜1で表すJaccard係数がこの値以上のペアだけを結びます。高くすると強い関係だけが残ってスッキリし、低くすると弱い関係も含め線が増えて密になります。")

            if st.button("テキスト分析を実行", key="run_text_mining"):
                with st.spinner("分析中..."):
                    # 文献ごとに抽出して集約（doc_words は共起ネットワークでも再利用し二度抽出を回避）
                    words, doc_words = [], []
                    for _, row in df_drill.iterrows():
                        dt = ""
                        if col_map['title'] and pd.notna(row[col_map['title']]): dt += str(row[col_map['title']]) + " "
                        if col_map['abstract'] and pd.notna(row[col_map['abstract']]): dt += str(row[col_map['abstract']]) + " "
                        dw = extract_compound_nouns(dt, stopwords)
                        words.extend(dw)
                        doc_words.append(dw)

                    if not words: st.warning("有効なキーワードなし")
                    else:
                        st.markdown("##### 1. ワードクラウド")
                        generate_wordcloud_and_list(words, f"クラスタ: {st.session_state.drill_base_label}", 30, FONT_PATH, capcom_key="saturnv_drill")

                        st.markdown("##### 2. 共起ネットワーク")
                        word_freq = Counter(words)
                        top_words = [w for w, c in word_freq.most_common(cooc_top_n)]
                        pair_counts = Counter()
                        for dw_list in doc_words:
                            dw = {w for w in set(dw_list) if w in top_words}
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
                                rep_lines_net.append(f"{i+1}. 【{r['title']}】[{r.get('number','N/A')}] ({r['applicant']}) - {r['abstract'][:80]}...")

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

    # --- 統計（特許マップ） ---
    with tab_stats:
        st.subheader("特許マップ（統計分析）")
        if st.session_state.saturnv_cluster_done:
            cluster_counts_stats = st.session_state.df_main['cluster_label'].value_counts()
            cluster_options_stats = [(f"(全クラスタ) ({len(st.session_state.df_main)}件)", "ALL")] + [
                (f"{st.session_state.saturnv_labels_map.get(cid)} ({cluster_counts_stats.get(st.session_state.saturnv_labels_map.get(cid), 0)}件)", cid)
                for cid in sorted(st.session_state.df_main['cluster'].unique())
            ]
            stats_cluster_filter_w = st.multiselect("集計対象クラスタ:", cluster_options_stats, default=[cluster_options_stats[0]], format_func=lambda x: x[0], key="stats_cluster_filter", help="統計集計の対象にするクラスタを選びます。特定の技術テーマだけに絞って件数推移や出願人ランキングを見たいときに使います。")

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

                s_year = st.number_input('開始年:', min_value=1900, max_value=2100, key="stats_start_year", step=1, help="集計・表示の対象とする年の範囲（始まり）です。")
                e_year = st.number_input('終了年:', min_value=1900, max_value=2100, key="stats_end_year", step=1, help="集計・表示の対象とする年の範囲（終わり）です。")
            with c2:
                n_apps = st.number_input('表示人数:', min_value=1, value=15, key="stats_num_assignees", help="出願人ランキング等で上位何件まで表示するかです（表示数のみ変わり、分析結果は変わりません）。")

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
        else:
            st.info("先に「俯瞰図マップ」タブで描画すると、クラスタ別の統計マップ（出願推移・出願人ランキング等）を集計できます。")

    # --- エクスポート ---
    with tab_export:
        st.subheader("データエクスポート")
        if (not st.session_state.saturnv_cluster_done) and ("df_drilldown_result" not in st.session_state):
            st.info("先に「俯瞰図マップ」タブで描画すると、クラスタ付きの結果データをダウンロードできます。")
        if st.session_state.saturnv_cluster_done:
            cols_drop = ['hover_text', 'parsed_date', 'drill_cluster', 'drill_cluster_label', 'drill_hover_text', 'drill_x', 'drill_y', 'temp_date_bin']
            csv = df_main.drop(columns=cols_drop, errors='ignore').to_csv(encoding='utf-8-sig', index=False).encode('utf-8-sig')
            st.download_button("俯瞰図マップ全データ (CSV)", csv, "APOLLO_SaturnV_Main.csv", "text/csv")

            # パラメータのエクスポート
            param_content = f"APOLLO Saturn V（俯瞰図分析）分析パラメータ\n"
            param_content += f"-----------------------------------\n"
            param_content += f"最小クラスタサイズ: {min_cluster_size_w}\n"
            param_content += f"最小サンプル数: {min_samples_w}\n"
            param_content += f"クラスタラベル単語数: {label_top_n_w}\n"
            st.download_button("パラメータ設定 (TXT)", param_content, "APOLLO_SaturnV_Params.txt", "text/plain")

        if "df_drilldown_result" in st.session_state:
            cols_drop_d = ['hover_text', 'parsed_date', 'date_bin', 'drill_hover_text', 'drill_date_bin', 'temp_date_bin']
            csv_d = st.session_state.df_drilldown_result.drop(columns=cols_drop_d, errors='ignore').to_csv(encoding='utf-8-sig', index=False).encode('utf-8-sig')
            st.download_button("ドリルダウン結果 (CSV)", csv_d, "APOLLO_SaturnV_Drill.csv", "text/csv")
