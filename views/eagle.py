# ==================================================================
# APOLLO — EAGLE（探索的クラスタリング）
#
# 投げ縄で俯瞰図上の任意の領域を手動選択し、独自のクラスタを構築する。
# 自動クラスタリングの結果に納得できないとき、分析者自身の判断で
# 境界を引き直すための操縦桿にあたる。
# ==================================================================
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
from sklearn.feature_extraction.text import TfidfVectorizer
import io
import capcom
import utils
import utils_ai
import utils_spatial
import apollo_ui
import flight_recorder
import patiroha
from umap import UMAP
import hdbscan
from wordcloud import WordCloud
from janome.tokenizer import Tokenizer
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

warnings.filterwarnings('ignore')

# 俯瞰図ファミリー共通スタイル（utils.py 0c）を EAGLE に当てるうえでの意図的な差分。
# EAGLE の未割り当て点（eagle_cluster == -1）は「まだクラスタに入れていない作業対象」で、
# 俯瞰図分析のノイズ（HDBSCAN の外れ値＝背景に沈めるもの）とは意味が違う。
# utils.LANDSCAPE_NOISE_STYLE をそのまま当てると投げ縄で掴む対象が見えなくなるため、
# EAGLE だけは未割り当て点に視認できる濃さを残す。
# 濃さは実描画で決めた（淡いスレート #AEB6C4 / 不透明度 0.5 では地形も点も読めなかった）。
EAGLE_TERRAIN_COLOR = "#8C93A6"                                      # 未割り当て点の地形（密度）
EAGLE_UNASSIGNED_STYLE = dict(color="#7C8798", size=5, opacity=0.7)  # 未割り当て点

# 等高線の輪郭線。俯瞰図ファミリーの既定は塗りだけだが、EAGLE は点が密なので
# 塗りが点に隠れて「等高線図」に見えなくなる。線を出して地形の段差を読めるようにする。
EAGLE_CONTOUR_LINE_WIDTH = 0.6
EAGLE_CONTOUR_LINE_COLOR = "rgba(60,70,90,0.35)"
# この件数を超えたら「密な図」とみなして点を小さく薄くする（地形を透かすため）
EAGLE_DENSE_VIEW_THRESHOLD = 3000

# フォント設定
FONT_PATH = utils.get_japanese_font_path()
if FONT_PATH:
    try:
        prop = fm.FontProperties(fname=FONT_PATH)
        plt.rcParams['font.family'] = prop.get_name()
    except Exception:
        pass

# ==================================================================
# --- テキスト処理設定 ---
# ==================================================================
@st.cache_resource
def load_tokenizer_eagle():
    return Tokenizer()

t = load_tokenizer_eagle()


def _get_stopwords():
    """ストップワード（ユーザー定義があれば優先）。

    モジュールは1回しか読み込まれないため、セッション状態に依存する値を
    モジュールレベルで束縛すると初回の値で凍結される。都度呼び出しで解決する。
    """
    sw = st.session_state.get("stopwords")
    return sw if sw else patiroha.get_stopwords()

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
                if capcom.is_active():
                    wc_data = {
                        "metadata": {"module": "EAGLE", "title": title, "top_n": top_n},
                        "word_frequencies": {w: c for w, c in word_freq.most_common(100)}
                    }
                    capcom.save_data(f"{capcom_key}_wordcloud.json", wc_data)
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                    buf.seek(0)
                    capcom.save_snapshot_image(f"{capcom_key}_wordcloud", buf.read())
            except Exception as e:
                st.caption(f":material/warning: WARN ワードクラウド の CAPCOM 保存に失敗しました（要確認）: {e}")

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
    # ベクトル化: pandas の文字列演算で一括処理する（数千〜万行でも高速）
    parts = pd.Series([""] * len(df), index=df.index)

    title_c = col_map.get('title')
    if title_c and title_c in df.columns:
        seg = "<b>名称:</b> " + df[title_c].astype(str).str[:50] + "...<br>"
        parts = parts + seg.where(df[title_c].notna(), "")

    num_c = col_map.get('app_num')
    if num_c and num_c in df.columns:
        seg = "<b>番号:</b> " + df[num_c].astype(str) + "<br>"
        parts = parts + seg.where(df[num_c].notna(), "")

    app_c = col_map.get('applicant')
    if app_c and app_c in df.columns:
        seg = "<b>出願人:</b> " + df[app_c].astype(str).str[:50] + "...<br>"
        parts = parts + seg.where(df[app_c].notna(), "")

    if 'characteristic_words' in df.columns:
        parts = parts + ("<b>特徴語:</b> " + df['characteristic_words'].astype(str) + "<br>")

    return parts.tolist()


def render():
    apollo_ui.page_header(
        "探索的クラスタリング", "EAGLE",
        "俯瞰図上の任意の領域を投げ縄で選択し、分析者の判断でクラスタを定義します",
    )
    # スナップショットのモジュール記録キー（CAPCOM スキーマ互換のため値も旧名のまま）
    st.session_state['current_page'] = 'EAGLE'

    # --- データガード ---
    apollo_ui.require_data()

    stopwords = _get_stopwords()
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
    if ('hover_text' not in st.session_state.df_eagle.columns
            or st.session_state.df_eagle.empty
            or '特徴語' not in st.session_state.df_eagle['hover_text'].iloc[0]):
        st.session_state.df_eagle['hover_text'] = update_hover_text_eagle(st.session_state.df_eagle, col_map)

    # ヘルパー: 投げ縄/クリックの選択結果 → 特許のインデックス列
    def selected_patent_indices(selection):
        """選択イベントから特許の行インデックスだけを取り出す。

        勢力圏（fill='toself' の線トレース）や地形（等高線）には customdata が無いため、
        選択に混じると .loc が None を引いて KeyError になる。customdata を持つ点のみ拾う。
        """
        out = []
        try:
            points = (selection or {}).get("selection", {}).get("points") or []
        except Exception:
            return out
        for _p in points:
            _cd = _p.get("customdata")
            if isinstance(_cd, (list, tuple)):
                if not _cd:
                    continue
                _cd = _cd[0]
            if _cd is None:
                continue
            out.append(_cd)
        return out

    # ヘルパー: ラベル生成（単一クラスタ用、投げ縄選択時に使用）
    def generate_label_for_cluster(df_sub, tfidf_mat, feat_names, top_n=3):
        """単一クラスタのラベルを生成する。patiroha.auto_labelのc-TF-IDFと同等。

        要約カラムは任意（README で「推奨」・J-PlatPat の要約なし形式も受け入れる）なので、
        col_map に無い列・データに無い列は黙って飛ばす。以前は df_sub[None] で KeyError に
        なり、**クラスタは作られたのにラベルだけ付かない**状態（マップに数字だけが出て、
        ラベル編集にも行が出ない）を生んでいた。
        """
        if df_sub.empty:
            return "Empty"
        _text_cols = [col_map.get(k) for k in ('title', 'abstract')]
        _text_cols = [c for c in _text_cols if c and c in df_sub.columns]
        if not _text_cols:
            return "(テキスト列なし)"
        texts = df_sub[_text_cols[0]].fillna('').astype(str)
        for _tc in _text_cols[1:]:
            texts = texts + ' ' + df_sub[_tc].fillna('').astype(str)
        # 全件を同一クラスタ(0)として扱い、c-TF-IDFでラベル生成
        dummy_labels = np.zeros(len(df_sub), dtype=int)
        label_map = utils.safe_auto_label(texts, dummy_labels, method='c-tfidf', top_n=top_n)
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

    # ラベルが欠けているクラスタを描画前に補う（自己修復）。
    # 欠けると ①マップのラベルが数字だけになる ②ラベル編集は labels_map を元に行を作るため
    # 手で直すこともできない、という二重の詰みになる。欠ける経路は上の (a)(b) で塞いだが、
    # 既に壊れたセッションや、クラスタ削除と再作成の組み合わせでも起こり得るため、
    # 原因側の対処とは別に毎回突き合わせる。
    _label_ids_now = [int(c) for c in st.session_state.df_eagle['eagle_cluster'].dropna().unique()]
    _missing_labels = sorted({c for c in _label_ids_now
                              if c != -1 and c not in st.session_state.eagle_labels_map})
    if _missing_labels:
        for _mid in _missing_labels:
            _sub = st.session_state.df_eagle[st.session_state.df_eagle['eagle_cluster'] == _mid]
            try:
                _lbl = generate_label_for_cluster(_sub, tfidf_matrix, feature_names)
            except Exception:
                _lbl = "(ラベル未設定)"
            st.session_state.eagle_labels_map[_mid] = f"[{_mid}] {_lbl}"
        st.caption(
            f":material/build: ラベルが未設定だったクラスタ {len(_missing_labels)} 件"
            f"（ID: {', '.join(str(i) for i in _missing_labels)}）に特徴語から名前を付けました。"
            "下の「クラスタ・ラベル編集」で変更できます。")

    # --- 共通設定 ---
    col_common, _ = st.columns([1, 2])
    with col_common:
        resolution = st.number_input("メッシュサイズ (Grid)", min_value=10, max_value=200, value=30, step=5, key="eagle_resolution_common", help="密度マップ（ヒートマップ）を描くときの格子の細かさです。大きいほど細かい格子になり局所的な濃淡が見えますが、点がまばらだと粗く見えます。小さいほど滑らかで大まかな分布になります。")

    st.markdown("---")

    # --- フィルタリングとデータレイヤリング (Saturn Vアーキテクチャ) ---
    st.subheader("フィルタリング設定")

    # 1. Universe (全体)
    df_universe = st.session_state.df_eagle.copy()

    # 密度メッシュは全体集合 df_universe の範囲で1度だけ決める（フィルタでは変えない）。
    # 期間や出願人で絞っても格子と濃さの基準が動かないので、絞り込みどうしを比べられる。
    # メッシュ範囲に余白ビンを持たせるのは、データ範囲ちょうどで切ると端の等高線が
    # 四角く途切れて見えるため（utils.landscape_density_bins）。
    # 絶対密度スケール用の全体ZMax。描画と同じメッシュで数えたビン件数から決める。
    # ⚠️ 最大ビンを基準にすると、極端に密なクラスタが1つあるだけで他が全部薄くなる
    #    （実データ7,789件で最大166件・非ゼロビンの中央値6件＝大半が最大値の4%で描かれていた）。
    #    非ゼロビンの95パーセンタイルを基準にし、それより密な場所は飽和させる。
    # 失敗時はメッシュ未指定（自動ビン）に落とし、密度自体は描けるようにする。
    try:
        _eagle_bx, _eagle_by = utils.landscape_density_bins(
            df_universe['umap_x'], df_universe['umap_y'], resolution)
        _H, _, _ = np.histogram2d(df_universe['umap_x'], df_universe['umap_y'],
                                  bins=[utils.landscape_bin_edges(_eagle_bx),
                                        utils.landscape_bin_edges(_eagle_by)])
        _H_nz = _H[_H > 0]
        eagle_global_zmax = float(np.percentile(_H_nz, 95)) if _H_nz.size else None
    except Exception:
        _eagle_bx = _eagle_by = None
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
        edit_mode = st.radio("モード:", ["編集中 (Edit)", "閲覧中 (FIX)"], horizontal=True, key="eagle_edit_mode", help="マップの操作モードを切り替えます。編集中＝投げ縄（Lasso）で範囲を選び、新規クラスタを作成・削除できます。閲覧中（FIX）＝クラスタ構成をロックし、点をクリックして特許の詳細を確認できます。クラスタを作るときは編集中、結果を見るときは閲覧中にします。")

    is_editing = (edit_mode == "編集中 (Edit)")

    if is_editing:
        st.markdown("グラフ上の「Lasso Select」等で範囲を選択し、新規クラスタを作成してください。<br>不要なクラスタは下部から削除できます。", unsafe_allow_html=True)
    else:
        st.markdown("クラスタリングはロックされています。修正する場合は「編集中」に切り替えてください。", unsafe_allow_html=True)

    # コントロール (ラベル & 密度固定)
    col_ctrl1, col_ctrl2 = st.columns([1, 2])
    with col_ctrl1:
        show_labels_chk = st.checkbox("マップにラベルを表示する", value=True, key="eagle_main_show_labels", help="俯瞰図に各クラスタの名前ラベルを重ねて表示するかどうかです。オンにすると技術テーマが一目で分かりますが、クラスタが多いと重なって読みにくくなります。点の分布だけを見たいときはオフにします。")
    with col_ctrl2:
        fix_density_chk = st.checkbox("密度マップを固定 (全体基準)", value=True, key="eagle_fix_density", help="密度マップ（ヒートマップ）の濃淡の基準を全データ共通に固定するかどうかです。オンにすると期間や出願人で絞り込んでも色の濃さが同じ尺度になり、期間どうしの混雑度を正しく比較できます。オフにすると表示中のデータだけで色を割り当てるため、絞り込んだ範囲内の相対的な濃淡が見やすくなります。")

    # 現在のクラスタを表示
    # fig_lasso をフィルタ・表示条件・クラスタ構成が変わったときだけ再構築し、
    # それ以外のリラン（投げ縄選択・他ウィジェット操作等）では session_state のキャッシュを再利用する。
    _eagle_fig_key = (
        date_filter_val, tuple(selected_apps), is_editing, fix_density_chk, show_labels_chk,
        int(resolution),
        tuple(sorted(st.session_state.eagle_labels_map.items())),
        tuple(st.session_state.df_eagle['eagle_cluster'].value_counts().sort_index().items()),
    )
    if (st.session_state.get('eagle_main_fig_key') == _eagle_fig_key
            and 'eagle_main_fig' in st.session_state):
        fig_lasso = st.session_state['eagle_main_fig']
    else:
        fig_lasso = go.Figure()

        # クラスタ→固定色の対応（地形・勢力圏・点・ラベルで共通）。フィルタに依らず
        # 全体集合 df_universe 基準で割り当てるので、期間や出願人で絞っても色が変わらない。
        _land_cmap = utils.landscape_color_map(df_universe['eagle_cluster'])

        # 1. 地形（密度背景）— どこが混んでいるかを見て投げ縄を掛けるための下地。
        #    クラスタに入れた点はそのクラスタ色、未割り当ての点は中立のスレートで描く。
        #    俯瞰図分析（Saturn V）はクラスタ別着色だけだが、EAGLE は最初クラスタが1つも
        #    無いため、未割り当て分を地形に残さないと背景が真っ白になり掛ける場所が見えない。
        if not df_trend.empty:
            _zmax = eagle_global_zmax if fix_density_chk else None
            for _cid, _grp in df_trend.groupby('eagle_cluster'):
                utils.add_landscape_density(
                    fig_lasso, _grp['umap_x'], _grp['umap_y'],
                    EAGLE_TERRAIN_COLOR if _cid == -1 else _land_cmap.get(_cid, '#8C93A6'),
                    zmax=_zmax, xbins=_eagle_bx, ybins=_eagle_by,
                    line_width=EAGLE_CONTOUR_LINE_WIDTH, line_color=EAGLE_CONTOUR_LINE_COLOR)

        # 2. ゴーストポイント (除外データ)
        if not df_ghost.empty:
            fig_lasso.add_trace(go.Scatter(
                x=df_ghost['umap_x'], y=df_ghost['umap_y'], mode='markers',
                marker=dict(color='#dddddd', size=3, opacity=0.4, line=dict(width=0)),
                name='その他 (Ghost)',
                hoverinfo='skip'
            ))

        # 3. クラスタ領域（平滑化した勢力圏・点とラベルと同じ固定色）。
        #    df_universe 基準で描き、期間フィルタで勢力圏が伸縮しないようにする。
        for _cid, _grp in df_universe[df_universe['eagle_cluster'] != -1].groupby('eagle_cluster'):
            utils.add_landscape_region(
                fig_lasso, _grp[['umap_x', 'umap_y']].values, _land_cmap.get(_cid, '#8C93A6'))

        # 4. フォーカスポイント (クラスタリング対象)
        uniq = sorted(df_focus['eagle_cluster'].unique())

        is_applicant_filtered = "ALL" not in selected_apps

        # 編集中は点に濃い枠を付けて「掴める」ことを示す。閲覧中(FIX)は俯瞰図ファミリー標準の
        # 白フチにする（点が重なっても粒が読める）。
        marker_border = dict(width=1, color='#333333') if is_editing else dict(width=0.8, color='white')

        # 件数が多いと点が地形を覆い隠し、等高線が見えなくなる（実データ7,789件で発覚。
        # 濃さや色の問題ではなく、点の面積で潰れていた）。密な図では点を小さく薄くして
        # 下の地形を透かす。件数が少ない図では従来どおり粒をしっかり見せる。
        _dense_view = len(df_focus) > EAGLE_DENSE_VIEW_THRESHOLD
        _pt_size = 5 if _dense_view else utils.LANDSCAPE_POINT_SIZE
        _pt_opacity = 0.85 if _dense_view else utils.LANDSCAPE_POINT_OPACITY
        _un_style = dict(EAGLE_UNASSIGNED_STYLE)
        if _dense_view:
            _un_style.update(size=4, opacity=0.45)

        if is_applicant_filtered:
            # 出願人着色モード（企業識別色も俯瞰図パレットで統一）
            for i, app_name in enumerate(selected_apps):
                # この出願人でフィルタ
                mask = df_focus[col_map['applicant']].fillna('').str.contains(re.escape(app_name))
                d_app = df_focus[mask]

                if not d_app.empty:
                    # 動的ホバーテキスト構築（内部クラスタIDをラベルにマッピング）
                    dynamic_hover = d_app['hover_text'] + d_app['eagle_cluster'].apply(
                        lambda x: f"<b>クラスタ:</b> {st.session_state.eagle_labels_map.get(x, str(x))}" if x != -1 else "")

                    fig_lasso.add_trace(go.Scatter(
                        x=d_app['umap_x'], y=d_app['umap_y'], mode='markers',
                        marker=dict(color=utils.landscape_color(i), size=max(_pt_size, 5),
                                    opacity=_pt_opacity, line=marker_border),
                        name=app_name,
                        customdata=d_app.index,
                        hoverinfo='text',
                        hovertext=dynamic_hover,
                        showlegend=True
                    ))
        else:
            # クラスタ着色モード
            for c in uniq:
                d = df_focus[df_focus['eagle_cluster'] == c]
                if d.empty: continue
                name = st.session_state.eagle_labels_map.get(c, str(c))
                if c == -1:
                    # 未割り当ては投げ縄で掴む対象。Saturn V のノイズのようには沈めない（冒頭の定数参照）
                    marker = dict(line=dict(width=0), **_un_style)
                else:
                    marker = dict(color=_land_cmap.get(c, '#8C93A6'), size=_pt_size,
                                  opacity=_pt_opacity, line=marker_border)

                # クラスタモードの場合、d内の全点はクラスタc(name)に属する
                dynamic_hover_c = d['hover_text'] + (f"<b>クラスタ:</b> {name}" if c != -1 else "")

                fig_lasso.add_trace(go.Scatter(
                    x=d['umap_x'], y=d['umap_y'], mode='markers',
                    marker=marker,
                    name=name,
                    customdata=d.index,
                    hoverinfo='text',
                    hovertext=dynamic_hover_c,
                    showlegend=False
                ))

        # 5. ラベル（ピル型・地形と勢力圏と点と同色のドット付き・件数上位クラスタは強調）
        if show_labels_chk:
            _valid_lab = df_focus[df_focus['eagle_cluster'] != -1]
            if not _valid_lab.empty:
                _top_cids = utils.landscape_top_clusters(
                    _valid_lab['eagle_cluster'].value_counts().to_dict())
                # 端のクラスタのラベルが図の外に見切れないよう、範囲を渡してアンカーを内側に倒す
                _lab_xr = (float(_valid_lab['umap_x'].min()), float(_valid_lab['umap_x'].max()))
                _lab_yr = (float(_valid_lab['umap_y'].min()), float(_valid_lab['umap_y'].max()))
                for _cid, _grp in _valid_lab.groupby('eagle_cluster'):
                    utils.add_landscape_label(
                        fig_lasso, _grp['umap_x'].mean(), _grp['umap_y'].mean(),
                        st.session_state.eagle_labels_map.get(_cid, str(_cid)),
                        _land_cmap.get(_cid, '#8C93A6'), emphasized=(_cid in _top_cids),
                        x_range=_lab_xr, y_range=_lab_yr)

        update_fig_eagle(fig_lasso, "Current Clusters", show_legend=False)

        # 表示範囲を明示固定し、かつフィルタやクラスタ構成に依存せず一定にする。
        # 必ず全宇宙 df_universe の【全点】から範囲を算出する（ノイズ/クラスタで絞らない）。
        # ※ 有効クラスタ(eagle_cluster != -1)だけで算出すると、EAGLE は投げ縄で
        #   手動クラスタを作るため、1つ目のクラスタを作った瞬間に範囲がそのクラスタの外接矩形へ
        #   縮小し「最初のクラスタにズームインする」ことになる。全点基準なら枠が動かず、
        #   「全体表示に戻す」のオートスケール（全データ収容）とも一致する。
        _bounds_df = df_universe
        if not _bounds_df.empty and 'umap_x' in _bounds_df.columns and 'umap_y' in _bounds_df.columns:
            _ex_min, _ex_max = _bounds_df['umap_x'].min(), _bounds_df['umap_x'].max()
            _ey_min, _ey_max = _bounds_df['umap_y'].min(), _bounds_df['umap_y'].max()
            _ex_pad = (_ex_max - _ex_min) * 0.02 if _ex_max > _ex_min else 1.0
            _ey_pad = (_ey_max - _ey_min) * 0.02 if _ey_max > _ey_min else 1.0
            fig_lasso.update_layout(
                xaxis=dict(range=[_ex_min - _ex_pad, _ex_max + _ex_pad], autorange=False, constrain="domain"),
                yaxis=dict(range=[_ey_min - _ey_pad, _ey_max + _ey_pad], autorange=False,
                           scaleanchor="x", scaleratio=1, constrain="domain"),
            )
            # 「全体表示に戻す」で固定レンジを再適用できるよう、算出した範囲を保持する。
            st.session_state['_eagle_xy_range'] = (
                [_ex_min - _ex_pad, _ex_max + _ex_pad],
                [_ey_min - _ey_pad, _ey_max + _ey_pad],
            )

        st.session_state['eagle_main_fig'] = fig_lasso
        st.session_state['eagle_main_fig_key'] = _eagle_fig_key
    # インタラクティブロジック
    if is_editing:
        # 操作UI（選択数・新規クラスタ作成・削除）をマップの「上」に表示するためのプレースホルダ。
        # 中身はチャートを描画して選択（戻り値）を得たあとで埋めるので、ボタンが上にあっても
        # 直近の選択を確実に使える。
        top_ui = st.container()
        st.markdown("---")

        # --- マップ（投げ縄選択）---
        # 選択はチャートの戻り値から取得する（最も確実な方法）。disable_selection_fade で
        # 非選択点のフェード（選択部だけ浮き上がって見える現象）を無効化する。
        # チャートの key を可変にし、「全体表示に戻す」ボタンや新規クラスタ作成時に key を変えることで、
        # 新規コンポーネントとして全体表示で再生成する（＝選択ズームのリセット）。
        _map_key_suffix = int(st.session_state.get('_eagle_map_key_suffix', 0))
        fig_lasso.update_layout(dragmode='lasso', clickmode='event+select')
        # 「全体表示に戻す」ボタンの挙動 = Plotly のオートスケールと同じ。ボタン押下時のみ
        # 軸を autorange に切り替え、全データが収まる範囲へ自動調整する（手動オートスケール相当）。
        # それ以外のリランでは、フレームを安定させる固定レンジを再適用する。
        # さらに uirevision を suffix に連動させる: 値が変わると Plotly はユーザーの手動ズーム/
        # パン/選択を破棄して新しい軸設定（autorange）を適用するため、ズーム中でも確実にリセットされる。
        # 値が変わらない通常リラン（投げ縄選択等）ではズームを保持する。
        if st.session_state.pop('_eagle_autoscale', False):
            fig_lasso.update_layout(
                xaxis=dict(autorange=True, constrain="domain"),
                yaxis=dict(autorange=True, scaleanchor="x", scaleratio=1, constrain="domain"),
            )
        else:
            _xy_range = st.session_state.get('_eagle_xy_range')
            if _xy_range:
                fig_lasso.update_layout(
                    xaxis=dict(range=_xy_range[0], autorange=False, constrain="domain"),
                    yaxis=dict(range=_xy_range[1], autorange=False,
                               scaleanchor="x", scaleratio=1, constrain="domain"),
                )
        fig_lasso.update_layout(uirevision=f"eagle_edit_{_map_key_suffix}")
        utils.disable_selection_fade(fig_lasso)
        selection = st.plotly_chart(fig_lasso, use_container_width=True, on_select="rerun", key=f"eagle_edit_map_{_map_key_suffix}", config={
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

        # 選択をチャートの戻り値から取得
        selected_indices = selected_patent_indices(selection)

        # 上のプレースホルダに操作UIを描画
        with top_ui:
            col_info, col_reset = st.columns([3, 1])
            with col_info:
                st.write(f"選択中: {len(selected_indices)} 件")
            with col_reset:
                # 押すとオートスケール（全データが収まる範囲へ自動調整）を作動させる。
                # autoscale フラグ + suffix（key/uirevision）変更で、手動ズーム状態からも確実にリセット。
                if st.button(":material/autorenew: 全体表示に戻す", key="eagle_reset_view", use_container_width=True):
                    st.session_state['_eagle_autoscale'] = True
                    st.session_state['_eagle_map_key_suffix'] = _map_key_suffix + 1
                    st.rerun()

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
                    st.write("")
                    if st.button("選択範囲を新規クラスタにする"):
                        st.session_state.df_eagle.loc[selected_indices, 'eagle_cluster'] = new_id
                        sub_df = st.session_state.df_eagle.loc[selected_indices]
                        # ラベル生成が失敗しても、直前のクラスタ割り当ては済んでいる。
                        # ここで例外を通すと「点はあるのにラベルが無いクラスタ」が残るので、
                        # 必ず何らかの名前を入れる（あとからラベル編集で直せる）。
                        try:
                            lbl = generate_label_for_cluster(sub_df, tfidf_matrix, feature_names)
                        except Exception:
                            lbl = "(ラベル未設定)"
                        st.session_state.eagle_labels_map[new_id] = f"[{new_id}] {lbl}"
                        # CAPCOM: patents.csvにeagle_cluster列を更新
                        try:
                            if capcom.is_active():
                                capcom.save_patents_csv()
                        except Exception as e:
                            st.caption(f":material/warning: WARN クラスタ列付き特許データ の CAPCOM 保存に失敗しました（要確認）: {e}")
                        # 作成後は key を変えて選択をクリア＋全体表示に戻す
                        st.session_state['_eagle_map_key_suffix'] = _map_key_suffix + 1
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
                    st.write("")
                    if st.button("削除実行"):
                        # Reset to -1
                        st.session_state.df_eagle.loc[st.session_state.df_eagle['eagle_cluster'] == del_target_id, 'eagle_cluster'] = -1
                        if del_target_id in st.session_state.eagle_labels_map:
                            del st.session_state.eagle_labels_map[del_target_id]
                        # CAPCOM: patents.csvにeagle_cluster列を更新
                        try:
                            if capcom.is_active():
                                capcom.save_patents_csv()
                        except Exception as e:
                            st.caption(f":material/warning: WARN クラスタ列付き特許データ の CAPCOM 保存に失敗しました（要確認）: {e}")
                        st.success(f"ID {del_target_id} を削除しました")
                        st.rerun()

    else:
        # 固定モード
        fig_lasso.update_layout(dragmode='pan') # 選択ロック
        @st.fragment
        def _eagle_click_main():
            utils.disable_selection_fade(fig_lasso)
            selection_eagle = st.plotly_chart(fig_lasso, use_container_width=True,
                on_select="rerun", selection_mode="points", key="eagle_main_map", config={
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
            # 閲覧(FIX)モードでは点クリック → 特許詳細ポップアップ（編集モードは投げ縄選択を維持）
            utils.handle_map_click(selection_eagle, "eagle_main", title="クリックした特許")
        _eagle_click_main()

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

        # :material/image: 整理版ランドスケープ（スライド/レポート用・上位クラスタのみ）
        if st.checkbox(":material/image: 整理版ランドスケープ（スライド/レポート用・上位クラスタのみ）を表示", key="eagle_curated_show"):
            st.caption("全クラスタを密にラベルすると重なるため、件数上位クラスタだけを大きく示したスライド向けの俯瞰図です。")
            _ecl_topn = st.slider("ラベル表示するクラスタ数（件数上位）", 3, 15, 8, key="eagle_curated_topn", help="整理版（スライド/レポート用）の俯瞰図で、件数が多い上位何クラスタにだけ大きなラベルを付けるかです。全クラスタに付けると重なって読めないため、主要クラスタだけを強調します。")
            _ecl_style_lbl = st.radio("表示モード:", ["クラスタ領域 (Clusters)", "密度マップ (Density)", "散布図 (Scatter)"],
                                      horizontal=True, key="eagle_curated_style", help="俯瞰図の描き方を切り替えます。クラスタ領域＝各クラスタを色付きの領域で表示、密度マップ＝点の混み具合をヒートマップで表示、散布図＝個々の特許を点で表示。全体像は領域、混雑度は密度、個別確認は散布図が見やすいです。")
            _ecl_style = {"クラスタ領域 (Clusters)": "hull", "密度マップ (Density)": "density", "散布図 (Scatter)": "points"}.get(_ecl_style_lbl, "hull")
            try:
                _ecl_df = df_focus.copy()
                _ecl_df['_curated_label'] = _ecl_df['eagle_cluster'].map(st.session_state.get('eagle_labels_map', {}))
                _ecl_fig = utils.build_curated_landscape(
                    _ecl_df, cluster_col='eagle_cluster', label_col='_curated_label',
                    x_col='umap_x', y_col='umap_y', top_n=_ecl_topn, region_style=_ecl_style)
                st.plotly_chart(_ecl_fig, use_container_width=True, config={'editable': False})
                _ecl_nall = int(df_focus[df_focus['eagle_cluster'] != -1]['eagle_cluster'].nunique()) if 'eagle_cluster' in df_focus.columns else 0
                # CAPCOMアクティブ時はクリーンPNGをZIPに同梱（スライド/レポート用の整理版を下流へ）
                utils.save_curated_to_capcom(
                    _ecl_fig, snap_id="eagle_curated_landscape",
                    cache_token=f"{_ecl_topn}_{_ecl_style}")
                utils.render_report_png_button(
                    _ecl_fig, key="eagle_curated_report",
                    default_title="技術ランドスケープ：主要クラスタ",
                    default_subtitle=f"出願 {len(df_focus):,}件 / 上位{_ecl_topn}クラスタ（全{_ecl_nall}クラスタ）",
                    default_caption="",
                    label=":material/palette: 整理版をスライド/レポート用PNGに書き出す")
            except Exception as _e:
                st.warning(f"整理版ランドスケープの生成に失敗しました: {_e}")

        # クラスタ動態（前回ラン分を session_state から）とノイズ（萌芽技術）を insight に反映。
        # 動態は本 insight より後段（クラスタ動態マップ）で計算されるため前回ラン分を参照する。
        _eg_noise = (int((df_focus['eagle_cluster'] == -1).sum())
                     if 'eagle_cluster' in df_focus.columns and -1 in df_focus['eagle_cluster'].values else 0)
        _eg_map_extra, _eg_map_inst = utils_ai.build_map_dynamics_noise_addon(
            dynamics_data=st.session_state.get('eagle_dynamics_data'),
            noise_count=_eg_noise, total_count=len(df_focus))
        main_prompt = utils_ai.generate_ai_insight_prompt(
            insight_role, insight_context, snap_data, insight_instruction + _eg_map_inst,
            extra_content=f"\n# 空間配置情報 (Spatial Context)\n{spatial_info}\n{_eg_map_extra}"
        )
        utils_ai.render_ai_insight_button(main_prompt, "eagle_main_insight")

        # CAPCOM data/ JSON出力（EAGLE クラスタ）
        try:
            if capcom.is_active():
                eagle_clusters_json = []
                cluster_counts_eagle = df_focus['eagle_cluster'].value_counts()
                for cid in sorted(df_focus['eagle_cluster'].unique()):
                    if cid == -1:
                        continue
                    label = st.session_state.eagle_labels_map.get(cid, f"Cluster {cid}")
                    _eg_count_raw = cluster_counts_eagle.get(cid, 0)
                    count = int(_eg_count_raw) if pd.notna(_eg_count_raw) else 0
                    cid_mask = df_focus['eagle_cluster'] == cid
                    cx = float(df_focus.loc[cid_mask, 'umap_x'].mean()) if cid_mask.any() else 0
                    cy = float(df_focus.loc[cid_mask, 'umap_y'].mean()) if cid_mask.any() else 0
                    reps_raw = cluster_reps.get(cid, []) if 'cluster_reps' in dir() and cid in cluster_reps else []
                    eagle_clusters_json.append({
                        "cluster_id": int(cid) if pd.notna(cid) else -1,
                        "label": label,
                        "count": count,
                        "centroid": [round(cx, 4), round(cy, 4)],
                        "representative_patents": reps_raw
                    })
                _eg_noise_raw = (df_focus['eagle_cluster'] == -1).sum() if -1 in df_focus['eagle_cluster'].values else 0
                noise_count = int(_eg_noise_raw) if pd.notna(_eg_noise_raw) else 0
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
            st.caption(f":material/warning: WARN EAGLEクラスタ の CAPCOM 保存に失敗しました（要確認）: {e}")

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
                    # メインマップ insight が次回ラン時に参照できるよう session_state に保持
                    st.session_state['eagle_dynamics_data'] = dyn_data
                    try:
                        if capcom.is_active():
                            capcom.save_data('eagle_cluster_dynamics', {'cluster_dynamics': dyn_data})
                    except Exception as e:
                        st.caption(f":material/warning: WARN クラスタ動態 の CAPCOM 保存に失敗しました（要確認）: {e}")

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
                def on_drill_interval_change(): pass
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
        drill_method = st.radio("手法を選択:", ["自動 (HDBSCAN)", "手動 (Lasso)"], horizontal=True, key="eagle_drill_method", help="選択したクラスタの内部をさらに細かく分ける方法を切り替えます。自動 (HDBSCAN)＝密度ベースのアルゴリズムが自動でサブクラスタに分割します。手動 (Lasso)＝再計算した詳細マップ上で投げ縄選択し、自分でサブクラスタを作ります。客観的に分けたいときは自動、意図した区切りで分けたいときは手動が向きます。")

        if drill_method == "自動 (HDBSCAN)":
            c1, c2, c3 = st.columns(3)
            with c1: drill_min_cluster_size_w = st.number_input('最小クラスタサイズ:', min_value=2, value=5, key="eagle_drill_min_cluster_size_w", disabled=st.session_state.get("eagle_drill_auto_hdbscan", False), help="1つのクラスタとして認める最小の特許件数です（クラスタの粒度設定）。小さくすると細かい技術テーマまで分かれてクラスタ数が増えますが、どこにも属さないノイズ（外れ値）も増えます。大きくすると少数の大まかなクラスタにまとまり安定しますが、細部は埋もれます。目安は母集団の約1〜2%（例: 2,000件なら20前後）。推奨10〜50。")
            with c2: drill_min_samples_w = st.number_input('最小サンプル数:', min_value=1, value=5, key="eagle_drill_min_samples_w", disabled=st.session_state.get("eagle_drill_auto_hdbscan", False), help="クラスタの「核」と認める密度の厳しさです。大きいほど判定が厳しくなり、ノイズ（外れ値）が増えてクラスタは密な中心部だけになります。小さいほど緩くなり、多くの点がクラスタに取り込まれます。通常は最小クラスタサイズ以下に設定します。推奨5〜20。")
            with c3: drill_label_top_n_w = st.number_input('ラベル単語数:', min_value=1, value=3, key="eagle_drill_label_top_n_w", help="各クラスタの自動命名に使う特徴語の数です。クラスタを特徴づける語を上位から何語ラベルに並べるかを決めます。多いほど内容を詳しく表せますが冗長になり、少ないほど簡潔になります。")
            # :material/smart_toy: 自動最適化（メインと同じ HDBSCAN 2パラメータ掃引をドリルダウンにも）
            drill_auto_hdbscan = st.checkbox(
                ":material/smart_toy: 自動最適化（最小クラスタサイズ・最小サンプル数を掃引してサブクラスタ数を適正化）",
                key="eagle_drill_auto_hdbscan",
                help="ドリルダウン対象の件数に合わせて HDBSCAN の2パラメータを自動で掃引し、サブクラスタ数を適正化します。ONの間は上の手動値は無視されます。")
            drill_target_k_w = None
            if drill_auto_hdbscan:
                # メインマップと同じく、対象の件数に応じて目標サブクラスタ数を初期化する（suggest_target_k）。
                # 対象クラスタを変えるたびに初期値が件数適応されるよう、key に対象IDを含める。
                _n_sub = len(df_subset_filter)
                drill_target_k_w = st.number_input(
                    "目標サブクラスタ数（目安）", min_value=2, max_value=80,
                    value=utils.suggest_target_k(_n_sub), key=f"eagle_drill_target_k_w_{drilldown_target_id}",
                    help="この数に近づくよう2パラメータを掃引します（対象の件数から自動初期化）。"
                         "結果のサブクラスタ数や粒度に満足できない場合は、この値を増減して再度「選択クラスタで詳細マップ作成」を押してください。"
                         "実際の値は密度構造に依存するため、目標ちょうどにならないこともあります（品質 DBCV を優先して選びます）。")
                utils.render_dbcv_help()
                # 前回の自動決定を常時再表示（メインマップと同じ挙動）。対象が一致する結果のみ表示する。
                _dar = st.session_state.get('eagle_drill_auto_result')
                if _dar and _dar.get('target_id') == drilldown_target_id:
                    _rv = _dar.get('validity')
                    _rv_txt = f"・品質DBCV={_rv:.2f}" if isinstance(_rv, (int, float)) else ""
                    st.info(
                        f":material/smart_toy: 前回の自動決定: 最小クラスタサイズ=**{_dar['mcs']}** / 最小サンプル数=**{_dar['ms']}** "
                        f"→ サブクラスタ **{_dar['k']}**・ノイズ {_dar['noise'] * 100:.1f}%（目標≈{_dar['target_k']}{_rv_txt}）。"
                        f"　数が合わない/粒度が好みでない場合は上の「目標サブクラスタ数」を変えて再描画してください。")
        else:
            drill_min_cluster_size_w, drill_min_samples_w, drill_label_top_n_w = 0, 0, 3 # Dummy
            drill_auto_hdbscan = False
            drill_target_k_w = None

        drill_show_labels_chk = st.checkbox('マップにラベルを表示する', value=True, key="eagle_drill_show_labels_chk", help="詳細マップに各サブクラスタの名前ラベルを重ねて表示するかどうかです。オンにするとサブテーマが一目で分かりますが、数が多いと重なって読みにくくなります。点の分布だけを見たいときはオフにします。")

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
                    if drill_app_values and "ALL" not in drill_app_values:
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
                    
                        if drill_method == "自動 (HDBSCAN)" and drill_auto_hdbscan:
                            _dpb = st.progress(0.0, text="サブクラスタのパラメータを掃引中...")
                            _dsweep = utils.sweep_hdbscan_params(
                                embedding_drill, target_k=int(drill_target_k_w),
                                progress_callback=lambda f: _dpb.progress(min(f, 1.0), text="サブクラスタのパラメータを掃引中..."))
                            _dpb.empty()
                            df_subset['drill_cluster'] = _dsweep['labels']
                            # 自動決定の結果を保持し、次回以降も「前回の自動決定」として再表示する（メインマップと同じ）。
                            st.session_state['eagle_drill_auto_result'] = {
                                'mcs': _dsweep['min_cluster_size'], 'ms': _dsweep['min_samples'],
                                'k': _dsweep['n_clusters'], 'noise': _dsweep['noise_ratio'],
                                'target_k': _dsweep['target_k'], 'validity': _dsweep.get('validity'),
                                'target_id': drilldown_target_id}
                            _drv = _dsweep.get('validity')
                            _drv_txt = f"・品質DBCV={_drv:.2f}" if isinstance(_drv, (int, float)) else ""
                            st.caption(
                                f":material/smart_toy: 自動決定: 最小クラスタサイズ={_dsweep['min_cluster_size']} / 最小サンプル数={_dsweep['min_samples']} "
                                f"→ サブクラスタ {_dsweep['n_clusters']}・ノイズ {_dsweep['noise_ratio'] * 100:.0f}%（目標≈{_dsweep['target_k']}{_drv_txt}）")
                        elif drill_method == "自動 (HDBSCAN)":
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
                            drill_labels_map = utils.safe_auto_label(
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
            
                drill_map_mode = st.radio("表示モード:", ["クラスタ領域 (Clusters)", "密度マップ (Density)", "散布図 (Scatter)"], horizontal=True, key="eagle_drill_map_mode_radio", help="俯瞰図の描き方を切り替えます。クラスタ領域＝各クラスタを色付きの領域で表示、密度マップ＝点の混み具合をヒートマップで表示、散布図＝個々の特許を点で表示。全体像は領域、混雑度は密度、個別確認は散布図が見やすいです。")

                d_c1, d_c2, d_c3 = st.columns(3)
                with d_c1:
                    drill_mesh_size = st.number_input("メッシュサイズ", value=40, min_value=10, max_value=200, step=5, key="eagle_drill_mesh_size", help="密度マップ（ヒートマップ）を描くときの格子の細かさです。大きいほど細かい格子になり局所的な濃淡が見えますが、点がまばらだと粗く見えます。小さいほど滑らかで大まかな分布になります。")
                with d_c2:
                    drill_remove_noise_chk = st.checkbox("ノイズを除く", value=False, key="eagle_drill_remove_noise", help="どのサブクラスタにも属さないノイズ（外れ値）の点を表示から除くかどうかです。オンにすると主要なサブクラスタだけが残り見やすくなりますが、孤立した特許は見えなくなります。萌芽的・例外的な技術も確認したいときはオフにします。")
                with d_c3: pass

                if drill_remove_noise_chk:
                    df_drill_plot = df_drill[df_drill['drill_cluster'] != -1]
                else:
                    df_drill_plot = df_drill

                fig_drill = go.Figure()

                # サブクラスタ→固定色（地形・勢力圏・点・ラベルで共通）
                _land_cmap_d = utils.landscape_color_map(df_drill_plot['drill_cluster'])
                _is_density_d = (drill_map_mode == "密度マップ (Density)")

                if _is_density_d and not df_drill_plot.empty:
                    # サブクラスタごとの地形（端の見切れ防止に共通メッシュ + 余白ビン）。
                    # 手動モードでは全点が未割り当てなので、そのぶんを中立のスレートで描いて
                    # 背景が空にならないようにする。
                    _dbx, _dby = utils.landscape_density_bins(
                        df_drill_plot['drill_x'], df_drill_plot['drill_y'], drill_mesh_size)
                    for _cid, _grp in df_drill_plot.groupby('drill_cluster'):
                        utils.add_landscape_density(
                            fig_drill, _grp['drill_x'], _grp['drill_y'],
                            EAGLE_TERRAIN_COLOR if _cid == -1 else _land_cmap_d.get(_cid, '#8C93A6'),
                            xbins=_dbx, ybins=_dby,
                            line_width=EAGLE_CONTOUR_LINE_WIDTH, line_color=EAGLE_CONTOUR_LINE_COLOR)

                if drill_map_mode == "クラスタ領域 (Clusters)":
                    # 平滑化した勢力圏（生の凸包から差し替え・点とラベルと同じ固定色）
                    for _cid, _grp in df_drill_plot[df_drill_plot['drill_cluster'] != -1].groupby('drill_cluster'):
                        utils.add_landscape_region(
                            fig_drill, _grp[['drill_x', 'drill_y']].values,
                            _land_cmap_d.get(_cid, '#8C93A6'))

                # 点は常時白フチ（重なっても粒が読める）。密度モードは地形が主役なので小さく薄く。
                # 件数が多い図でも点で地形が潰れるため、同じように軽くする（メインマップと同基準）。
                _dense_d = _is_density_d or len(df_drill_plot) > EAGLE_DENSE_VIEW_THRESHOLD
                marker_line_d = dict(width=0.6 if _dense_d else 0.8, color='white')
                _pt_size_d = 4 if _dense_d else utils.LANDSCAPE_POINT_SIZE
                _pt_opacity_d = 0.65 if _dense_d else utils.LANDSCAPE_POINT_OPACITY

                # --- Drill-down Scatter with Manual Selection Support ---
                # 勢力圏（go.Scatter=SVG）と座標系を揃えるため点も Scatter を使う（Scattergl=WebGL
                # だと領域とプロット点がズレる）。サブクラスタは点数が少なく lasso/性能とも問題なし。
                for cid in sorted(df_drill_plot['drill_cluster'].unique()):
                    d_sub = df_drill_plot[df_drill_plot['drill_cluster'] == cid]
                    if d_sub.empty: continue

                    if cid == -1:
                        # 未割り当ては投げ縄で掴む対象なので沈めない（冒頭の定数参照）
                        _un_d = dict(EAGLE_UNASSIGNED_STYLE)
                        if _dense_d:
                            _un_d.update(size=4, opacity=0.45)
                        marker_d = dict(line=dict(width=0), **_un_d)
                    else:
                        marker_d = dict(color=_land_cmap_d.get(cid, '#8C93A6'), size=_pt_size_d,
                                        opacity=_pt_opacity_d, line=marker_line_d)

                    fig_drill.add_trace(go.Scatter(
                        x=d_sub['drill_x'], y=d_sub['drill_y'], mode='markers',
                        marker=marker_d,
                        hoverinfo='text', hovertext=d_sub['drill_hover_text'],
                        name=drill_labels_map.get(cid, str(cid)),
                        customdata=d_sub.index
                    ))

                # ラベル（ピル型・点と勢力圏と同色のドット付き・件数上位サブクラスタは強調）
                if drill_show_labels_chk:
                    _dvalid = df_drill_plot[df_drill_plot['drill_cluster'] != -1]
                    if not _dvalid.empty:
                        _dtop = utils.landscape_top_clusters(
                            _dvalid['drill_cluster'].value_counts().to_dict())
                        _dxr = (float(_dvalid['drill_x'].min()), float(_dvalid['drill_x'].max()))
                        _dyr = (float(_dvalid['drill_y'].min()), float(_dvalid['drill_y'].max()))
                        for cid, grp in _dvalid.groupby('drill_cluster'):
                            utils.add_landscape_label(
                                fig_drill, grp['drill_x'].mean(), grp['drill_y'].mean(),
                                drill_labels_map.get(cid, str(cid)),
                                _land_cmap_d.get(cid, '#8C93A6'), emphasized=(cid in _dtop),
                                x_range=_dxr, y_range=_dyr)
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
                # 対象クラスタ別の key（静的キーだと対象を切り替えても「保存済み」のまま出るため）
                _eagle_drill_slug = re.sub(r'\W+', '_', str(drilldown_target_id))[:30] or 'target'
                utils.render_snapshot_button(
                    title=f"EAGLE: ドリルダウンマップ（{drilldown_target_id}）",
                    description="選択クラスタの詳細分析マップ（サブクラスタリング）。",
                    key=f"eagle_drill_snap_{_eagle_drill_slug}",
                    group="eagle_drill_snap",
                    fig=fig_drill,
                    data_summary=snap_data_d
                )

                drill_prompt = utils_ai.generate_ai_insight_prompt(
                    drill_insight_role, drill_insight_context, snap_data_d, drill_insight_instruction,
                    extra_content=f"\n# 空間配置情報 (Spatial Context)\n{d_spatial_info}"
                )
                utils_ai.render_ai_insight_button(drill_prompt, "eagle_drill_insight")


                # --- Manual Lasso Logic for Drill-down ---
                s_indices_d = selected_patent_indices(selection_drill)
            
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
                            try:
                                lbl = generate_label_for_cluster(sub_df_d, tfidf_matrix, feature_names, top_n=3)
                            except Exception:
                                lbl = "(ラベル未設定)"
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
                    cooc_top_n = st.slider("共起: 上位単語数", 30, 100, 70, key="eagle_cooc_top_n", help="ネットワークに表示するキーワード（ノード）の数です。出現頻度の上位から何語を使うかを決めます。多いほど網羅的ですが密集して読みにくく、少ないほど主要語に絞られ見やすくなります。")
                    cooc_threshold = st.slider("共起: Jaccard係数 閾値", 0.01, 0.3, 0.03, 0.01, key="eagle_cooc_threshold", help="2つのキーワードを線（エッジ）で結ぶ基準の強さです。共起の強さを0〜1で表すJaccard係数がこの値以上のペアだけを結びます。高くすると強い関係だけが残ってスッキリし、低くすると弱い関係も含め線が増えて密になります。")
            
                if st.button("テキスト分析を実行", key="eagle_run_text_mining"):
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
                            generate_wordcloud_and_list(words, f"クラスタ: {st.session_state.eagle_drill_base_label}", 30, FONT_PATH, capcom_key="eagle_drill")
                        
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
                    s_year = st.number_input('開始年:', min_value=1900, max_value=2100, value=auto_min_year, key="eagle_stats_start_year", step=1, help="集計・表示の対象年の範囲です。")
                    e_year = st.number_input('終了年:', min_value=1900, max_value=2100, value=auto_max_year, key="eagle_stats_end_year", step=1, help="集計・表示の対象年の範囲です。")
                with c2:
                    n_apps = st.number_input('表示人数:', min_value=1, value=15, key="eagle_stats_num_assignees", help="ランキングで上位何件まで表示するかです（表示数のみ変わり分析結果は不変）。")
            
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
