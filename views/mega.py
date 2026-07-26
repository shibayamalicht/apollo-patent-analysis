# ==================================================================
# APOLLO — MEGA（動態分析）
# 旧 APOLLO pages/4_:material/trending_up:_MEGA.py の移植版。
# 分析ロジック・session_state キー・CAPCOM 保存スキーマは元のまま、
# UI構造と文言のみ APOLLO 共通シェルに合わせて変換している。
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

# 機械学習・自然言語処理
from umap import UMAP
import hdbscan
from wordcloud import WordCloud
from janome.tokenizer import Tokenizer
import networkx as nx

# 描画用
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import utils
import utils_ai
import patiroha
import apollo_ui
utils.configure_matplotlib_font()

# 警告を非表示
warnings.filterwarnings('ignore')

# ==================================================================
# --- 1. フォント設定（モジュールレベルに維持） ---
# ==================================================================

FONT_PATH = utils.get_japanese_font_path()
if FONT_PATH:
    try:
        prop = fm.FontProperties(fname=FONT_PATH)
        plt.rcParams['font.family'] = prop.get_name()
    except:
        pass


def _tide_drill_adjust_intent(mode):
    """クラスタ動態（ドリルダウン地図）版「しっくりこない」ボタンの共通コールバック
    （俯瞰図分析のドリルダウンと同じ設計）。

    実測サブクラスタ数より細かい/粗い解のみ許容する制約を mega_drill_intent_ctx で
    実行側に渡し、目標はクリックごとに前進させる。結果の通知はカード内にインライン表示。
    """
    achieved_k = None
    prev_noise = None
    try:
        _lab = st.session_state.df_drilldown['cluster_id']
        achieved_k = int(len(set(_lab.unique()) - {-1}))
        prev_noise = float((_lab == -1).mean())
    except Exception:
        pass
    _tsel = st.session_state.get('drill_target')
    _tid = _tsel[1] if isinstance(_tsel, (tuple, list)) and len(_tsel) > 1 else _tsel
    if _tid in (None, '(分析対象を選択)'):
        return
    _k_key = f"mega_drill_target_k_{_tid}"
    cur_target = st.session_state.get(_k_key) or achieved_k or 8

    if mode == 'finer':
        if achieved_k is not None and achieved_k >= 80:
            st.session_state['mega_drill_intent_feedback'] = {
                'msg': "すでに上限（80サブクラスタ）です。これ以上は細かくできません", 'changed': False}
            return
        base_k = max(int(cur_target), int(achieved_k or 0))
        new_k = min(80, max(base_k + 2, round(base_k * 1.4)))
        st.session_state['mega_drill_auto_hdbscan'] = True
        st.session_state[_k_key] = int(new_k)
        _k_floor = (int(achieved_k) + 1) if achieved_k else None
        st.session_state['mega_drill_intent_ctx'] = {
            'mode': mode, 'prev_k': achieved_k, 'prev_noise': prev_noise,
            'k_min': _k_floor, 'k_max': max(int(new_k) * 2, (_k_floor or 2) + 1)}
    elif mode == 'coarser':
        if achieved_k is not None and achieved_k <= 2:
            st.session_state['mega_drill_intent_feedback'] = {
                'msg': "すでに最小（2サブクラスタ）です。これ以上は大きくまとめられません", 'changed': False}
            return
        base_k = min(int(cur_target), int(achieved_k or 10 ** 9))
        new_k = max(2, min(base_k - 2, round(base_k * 0.65)))
        st.session_state['mega_drill_auto_hdbscan'] = True
        st.session_state[_k_key] = int(new_k)
        _k_ceil = max(2, int(achieved_k) - 1) if achieved_k else None
        st.session_state['mega_drill_intent_ctx'] = {
            'mode': mode, 'prev_k': achieved_k, 'prev_noise': prev_noise,
            'k_min': 2, 'k_max': _k_ceil}
    elif mode == 'less_noise':
        # 直前が自動決定ならその値を土台に、核の判定（最小サンプル数）だけ緩める
        _dar = st.session_state.get('mega_drill_auto_result') or {}
        if _dar.get('target_id') == _tid:
            mcs = int(_dar.get('mcs') or st.session_state.get('drill_min_cluster_size', 5))
            ms = int(_dar.get('ms') or st.session_state.get('drill_min_samples', 5))
        else:
            mcs = int(st.session_state.get('drill_min_cluster_size', 5))
            ms = int(st.session_state.get('drill_min_samples', 5))
        new_ms = max(1, int(round(ms * 0.5)))
        if new_ms == ms:
            _nz_txt = f"（現在 {prev_noise * 100:.0f}%）" if prev_noise is not None else ""
            st.session_state['mega_drill_intent_feedback'] = {
                'msg': f"未分類の判定はすでに最も緩い設定です{_nz_txt}。これ以上は自動では減らせません",
                'changed': False}
            return
        st.session_state['mega_drill_auto_hdbscan'] = False
        st.session_state['drill_min_cluster_size'] = mcs
        st.session_state['drill_min_samples'] = new_ms
        st.session_state['mega_drill_intent_ctx'] = {
            'mode': mode, 'prev_k': achieved_k, 'prev_noise': prev_noise}
    st.session_state['mega_drill_intent_rerun'] = True


# ==================================================================
# --- 2. テキスト処理・ストップワード ---
# ==================================================================
@st.cache_resource
def load_tokenizer_mega():
    return Tokenizer()

t = load_tokenizer_mega()

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
    if not words:
        st.subheader(title)
        st.warning("キーワードが見つからなかったため、表示をスキップしました。")
        return None

    word_freq = Counter(words)

    try:
        wc_array = utils.compute_wordcloud_array(tuple(sorted(word_freq.items())), font_path)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc_array, interpolation='bilinear')
        ax.set_title(title, fontsize=20)
        ax.axis('off')
        st.pyplot(fig)

        st.markdown(f"**上位キーワード (Top {top_n})**")
        list_data = { "キーワード": [], "出現頻度": [] }
        for word, freq in word_freq.most_common(top_n):
            list_data["キーワード"].append(word)
            list_data["出現頻度"].append(freq)
        st.dataframe(pd.DataFrame(list_data), height=200)

        # CAPCOM: ワードクラウドデータ保存
        if capcom_key:
            try:
                import capcom
                if capcom.is_active():
                    import io
                    wc_data = {
                        "metadata": {"module": "MEGA", "title": title, "top_n": top_n},
                        "word_frequencies": {w: c for w, c in word_freq.most_common(100)}
                    }
                    capcom.save_data(f"{capcom_key}_wordcloud.json", wc_data)
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                    buf.seek(0)
                    capcom.save_snapshot_image(f"{capcom_key}_wordcloud", buf.read())
            except Exception as e:
                # 保存失敗を握りつぶさず可視化する。
                st.caption(f":material/warning: ワードクラウドの記録セッション保存に失敗しました（要確認）: {e}")

        # スナップショット保存ボタン
        if capcom_key:
            utils.render_snapshot_button(
                title=f"ワードクラウド: {title}",
                description=f"TF-IDFワードクラウド（上位{top_n}語）",
                key=f"{capcom_key}_wordcloud",
                fig=fig,
                data_summary={
                    "module": "MEGA",
                    "type": "wordcloud",
                    "title": title,
                    "top_words": [{"word": w, "freq": c} for w, c in word_freq.most_common(top_n)]
                }
            )
            apollo_ui.snapshot_hint()

    except Exception as e:
        st.error(f"ワードクラウドの描画に失敗しました: {e}")
        if font_path is None:
            st.warning("日本語フォントが見つかりませんでした。")


# ==================================================================
# --- 3. ヘルパー関数 (動態分析ロジック・元 MEGA のまま) ---
# ==================================================================

@st.cache_data
def _get_top_words_filtered(dense_vector, feature_names, stopwords_list, top_n=5):
    """TF-IDFベクトルから上位語を抽出（ストップワード除外）"""
    indices = np.argsort(dense_vector)[::-1]
    top_words = []
    for i in indices:
        word = feature_names[i]
        if word not in stopwords_list and not re.fullmatch(r'[\d０-９]+', word) and len(word) > 1:
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
# --- 4. 画面描画 ---
# ==================================================================

def render():
    apollo_ui.page_header("動態分析", "MEGA", "成長率（CAGR）× 活動量の4象限による技術テーマの動態分析")

    # スナップショットのモジュール判定に使う互換キー（値も 'MEGA' のまま変更不可）
    st.session_state['current_page'] = 'MEGA'

    # --- データガード & セッションデータ読込 ---
    apollo_ui.require_data()
    try:
        df_main = st.session_state.df_main
        col_map = st.session_state.col_map
        sbert_embeddings = st.session_state.sbert_embeddings
        tfidf_matrix = st.session_state.tfidf_matrix
        feature_names = st.session_state.feature_names
    except Exception as e:
        st.error(f"セッションデータの読み込みに失敗しました: {e}")
        st.stop()

    # ストップワード（ユーザー定義があれば優先。元ページ同様に毎回評価する）
    if "stopwords" in st.session_state and st.session_state["stopwords"]:
        stopwords = st.session_state["stopwords"]
    else:
        stopwords = patiroha.get_stopwords()

    # --- 分析対象の選択（全体 / 俯瞰図クラスタ / ルール分類カテゴリ） ---
    # 以降の df_main はここで選んだサブセット。index は元のまま保持されるため、
    # sbert_embeddings / tfidf_matrix への .index 参照はそのまま正しく機能する。
    df_main, tide_scope = apollo_ui.analysis_target_selector(df_main, "tide")
    _tide_scoped = not tide_scope.startswith("全体")

    tab_b, tab_c, tab_d = st.tabs([
        "動態マップ",
        "クラスタ動態",
        "エクスポート"
    ])

    # --- A. 動態マップ（旧 PULSE: 4象限の動態分析） ---
    with tab_b:
        # おすすめ実行バー（分析軸はここで選ぶ — 詳細設定に埋めない）
        with st.container(border=True):
            col_go1, col_go2 = st.columns([2.4, 1], vertical_alignment="center")
            with col_go1:
                axis_options = [('出願人', 'applicant_main'),
                                (f'分類コード {utils.classification_code_label(col_map)} (メイングループ)', 'ipc_main_group')]
                if st.session_state.col_map.get('fterm'): axis_options.append(('Fターム (テーマコード)', 'fterm_main'))
                analysis_axis = st.radio(
                    "何の勢いを見ますか？（分析軸）:", options=axis_options,
                    format_func=lambda x: x[0], horizontal=True, key="mega_analysis_axis",
                    help="4象限にプロットする単位です。出願人=プレイヤー（企業）比較、IPC/Fターム=技術分野の比較になります。")
                st.caption("活動量=直近5年・最小10件で描画します。期間や件数フィルタは下の「詳細設定」で変更できます。")
            with col_go2:
                run_map = st.button("動態分析マップを描画", type="primary",
                                    key="mega_run_map", use_container_width=True)

        # 詳細設定（分析軸以外のパラメータを維持）
        with st.expander("詳細設定（集計期間・フィルタ）"):
            col1, col2 = st.columns(2)
            with col1:
                yaxis_slider = st.slider("Y軸 (現在) の集計年数:", min_value=1, max_value=10, value=5, key="mega_yaxis", help="4象限の「現在の活動量（Y軸）」を直近何年分の出願件数で測るかです。短くすると足元の勢いを、長くすると中期的な活動量を反映します。")
                cagr_end_year = st.number_input("X軸 (過去の勢い) 計算の最終年:", value=datetime.datetime.now().year - 1, key="mega_cagr_year", help="「過去の勢い（X軸）」の成長率を計算する基準となる最終年です。この年までのデータで勢いを算出します。")
            with col2:
                trajectory_past = st.number_input("軌跡 (過去への遡り年数):", min_value=1, value=5, key="mega_trajectory", help="注目企業が4象限のどこからどこへ移動したか、その軌跡を何年前から辿って描くかです。長くするほど長期の移動経路が見えます。")
                min_patents = st.number_input("最小フィルタ件数 (描画対象):", min_value=1, value=10, key="mega_min_patents", help="4象限に描画する最小の出願件数です。これ未満の小規模な出願人を除外し、主要プレイヤーに絞ってノイズを減らします。")

        # ハイライトと軌跡（一度描画すると候補が件数順に並ぶ）
        highlight_options = st.session_state.get("mega_highlight_options", [])
        highlight_targets = st.multiselect("注目対象 (軌跡を表示):", options=highlight_options, format_func=lambda x: x[0],
                                           help="象限移動の軌跡を強調表示する出願人を選びます。選んだ企業の戦略変化（成長/衰退への移動）を追えます。マップを一度描画すると候補が現れます。")

        if run_map:
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
                    st.session_state.mega_axis_col = axis_col
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

        # --- ラベル編集 & 描画 ---
        base_color_map = {'リーダー (Leaders)': '#28a745', '新興・高ポテンシャル (Emerging)': '#ffc107', '成熟・既存勢力 (Established)': '#007bff', '衰退・ニッチ (Declining/Niche)': '#6c757d', 'N/A': '#ced4da'}

        if "df_momentum_result" in st.session_state:
            df_to_plot = st.session_state.df_momentum_result.copy()
            st.session_state.mega_group_map_custom = {}
            with st.expander("象限ラベルの編集（図中の呼び名を変えられます）"):
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
            # axis_col も session_state から復元する。axis_col は「描画ボタン押下時」のみ
            # 定義されるローカル変数で、ボタンを押さない再描画では未定義になり、後続の CAPCOM 保存
            # （軸別ファイル名 mega_momentum_<軸>.json）で NameError になっていた。
            axis_col = st.session_state.get('mega_axis_col', '')
            cagr_start = int(st.session_state.get('cagr_start_year_min', 2000))
            cagr_end = int(st.session_state.get('cagr_end_year_val', datetime.datetime.now().year))
            xaxis_title_label = f"過去の勢い (CAGR, {cagr_start}-{cagr_end}年内の活動期間)"

            fig = go.Figure()

            # 軌跡描画
            if highlight_targets:
                highlight_values = [t[1] for t in highlight_targets]
                palette = utils.APOLLO_COLORS
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

                    # reset_index で軸の値（出願人名/IPC 等）が mega_axis_label 名の列として復活する。
                    # クリック時にこの値を取得できるよう custom_data に追加（customdata[1]）。
                    _axis_label_col = st.session_state.mega_axis_label
                    fig = px.scatter(
                        df_filt.reset_index(),
                        x='X_Present',
                        y='Y_Present_Plot',
                        size='Bubble_Present',
                        size_max=60,
                        color='Group_Custom',
                        color_discrete_map=current_color_map,
                        hover_name=_axis_label_col,
                        log_y=True,
                        custom_data=['Group_Custom', _axis_label_col]
                    )
                    fig.update_traces(hovertemplate=_get_hover_template_mode2())

            fig.add_vline(x=st.session_state.mega_x_threshold, line_width=1, line_dash="dash", line_color="gray")
            # 注: plotly は対数軸でも add_hline/add_vline の座標を自動で log10 変換する。
            # 生値（mega_y_threshold）のまま渡すのが正しい（log10 を渡すと二重変換でズレる）。
            fig.add_hline(y=st.session_state.mega_y_threshold, line_width=1, line_dash="dash", line_color="gray")

            _tide_title = "動態分析マップ" + (f"　対象: {tide_scope}" if _tide_scoped else "")
            utils.update_fig_layout(fig, _tide_title, height=800, show_axes=True, show_legend=False)
            fig.update_layout(
                xaxis_title=f"← 勢い減速 | {xaxis_title_label} | 勢い加速 → (十字線: {st.session_state.mega_x_threshold:.1%})",
                yaxis_title="← 活動鈍化 | 現在の活動量 | 活動活発 →",
                xaxis_tickformat='.0%',
                yaxis_type="log",
                xaxis=dict(showgrid=True, zeroline=True, showline=True),
                yaxis=dict(showgrid=True, zeroline=False, showline=True)
            )

            utils.disable_selection_fade(fig)
            selection_mega = st.plotly_chart(
                fig, use_container_width=True, key="mega_map",
                on_select="rerun", selection_mode="points",
                config={'editable': False}
            )
            st.session_state.df_momentum_export = df_to_plot.copy()

            # --- バブルクリック → その軸値（出願人/IPC/Fターム）に該当する特許群をポップアップ ---
            def _mega_resolve_click(point):
                # 通常モードのバブルのみ customdata を持つ（軌跡モードの go.Scatter は反応しない）
                cd = point.get("customdata")
                if not cd or len(cd) < 2:
                    return None, None
                axis_value = cd[1]  # 軸の値（出願人名 / IPCメイングループ / Fターム）
                if axis_value is None or str(axis_value).strip() == "":
                    return None, None
                axis_value = str(axis_value).strip()

                # 分析軸に応じて df_main のフィルタ列を決定（不明な場合は出願人をデフォルト）
                _label = st.session_state.get("mega_axis_label", "出願人")
                if ("IPC" in _label) or ("FI" in _label):
                    filt_col = "ipc_main_group"
                elif "Fターム" in _label:
                    filt_col = "fterm_main"
                else:
                    filt_col = "applicant_main"
                if filt_col not in df_main.columns:
                    filt_col = "applicant_main"

                # 各軸列はリスト（要素は strip 済み）。クリックした値を含む特許を抽出。
                def _contains(lst):
                    if isinstance(lst, list):
                        return any(str(v).strip() == axis_value for v in lst)
                    return False

                indices = df_main[df_main[filt_col].apply(_contains)].index.tolist()
                dlg_title = f"{_label}: {axis_value}（{len(indices)}件）"
                return indices, dlg_title

            utils.handle_chart_click_to_records(selection_mega, "mega_map", _mega_resolve_click)

            # --- 読み方カード（軌跡モード時は軌跡の読み方も添える） ---
            if highlight_targets:
                apollo_ui.howto(
                    "注目対象ごとに年別の位置をつないだ<b>軌跡</b>を表示しています。"
                    "<b>色が濃い点ほど現在に近く</b>、薄い点は過去の位置です（点の上の 'YY は西暦の下2桁）。"
                    "<b>右上へ向かう軌跡は投資強化</b>、<b>左下へ向かう軌跡は撤退・停滞</b>のサインとして読めます。"
                    "十字線（象限の区切り）をまたいだ年の前後に、その対象の戦略転換が起きた可能性があります。",
                    title="軌跡の見方",
                )
            _n_recent = int(st.session_state.get('mega_y_axis_years', 5))
            apollo_ui.howto(
                f"<b>横軸=過去の勢い（CAGR: 年平均成長率）、縦軸=現在の活動量（直近{_n_recent}年の出願件数・対数目盛）</b>、"
                "バブルの大きさは総出願件数です。平均値の十字線で4つの象限に分かれます — "
                "<b>右上: リーダー</b>（勢い・活動量とも平均以上）、<b>右下: 新興・高ポテンシャル</b>（勢いは高いが規模はまだ小さい）、"
                "<b>左上: 成熟・既存勢力</b>（規模は大きいが勢いは鈍化）、<b>左下: 衰退・ニッチ</b>（勢い・活動量とも平均以下）。"
                "まず<b>右下（新興）に将来のリーダー候補が潜んでいないか</b>を、バブルの大きさ（これまでの蓄積）とあわせて確認してください。"
                "誤読に注意 — <b>CAGRは出発点の件数が小さいほど大きく出やすい</b>ため右下の高成長は規模の小ささの裏返しでもあること、"
                "十字線は<b>この母集団内の平均で引いた相対的な区切り</b>で絶対評価ではないこと、"
                "縦軸は対数のため上下の差は見た目以上に大きいこと、直近年は未公開の出願が反映されず実際より少なく見えることがあります。"
                "バブルをクリックすると該当する特許の一覧を確認できます。",
                title="4象限マップの見方",
            )

            # --- Snapshot: 4象限マップ ---
            _pulse_summary = f"動態分析マップ: {axis_label}軸\n"
            _group_counts = df_to_plot['Group_Auto'].value_counts().to_dict()
            for _g, _c in _group_counts.items():
                _pulse_summary += f"  {_g}: {_c}件\n"
            _pulse_snap_data = {
                'module': 'MEGA',
                'type': 'pulse_4quadrant',
                'chart_data': _pulse_summary + "\n" + utils_ai.df_to_markdown(df_to_plot[['X_Present', 'Y_Present', 'Bubble_Present', 'Group_Auto']].head(50), index=True, index_label='対象'),
            }
            _tide_scope_note = f"分析対象={tide_scope}。" if _tide_scoped else ""
            utils.render_snapshot_button(
                title=f"動態分析マップ（{axis_label}軸）",
                description=f"{_tide_scope_note}CAGR×活動量の4象限で{axis_label}を分類した動態分析マップ。",
                key="mega_pulse_snap",
                fig=fig,
                data_summary=_pulse_snap_data
            )
            apollo_ui.snapshot_hint()

            # --- AI Insight: ポートフォリオ戦略分析 ---
            _meta_pulse = utils_ai.build_common_metadata(df_main=df_main, col_map=col_map)
            _meta_pulse['分析軸'] = axis_label
            _meta_pulse['CAGR計算期間'] = f"{int(st.session_state.get('cagr_start_year_min', 2000))}年～{int(st.session_state.get('cagr_end_year_val', datetime.datetime.now().year))}年"
            _meta_pulse['活動量算出期間'] = f"直近{st.session_state.get('mega_y_axis_years', 5)}年"
            _meta_pulse['最小特許件数フィルタ'] = int(min_patents)
            _meta_pulse['CAGR閾値'] = f"{st.session_state.get('mega_x_threshold', 0):.1%}"
            _meta_pulse['活動量閾値'] = f"{st.session_state.get('mega_y_threshold', 0):.1f}"
            _meta_pulse['象限別件数'] = {str(k): int(v) for k, v in _group_counts.items()}
            _meta_pulse['分析対象数'] = len(df_to_plot)

            _pulse_data_str = utils_ai.df_to_markdown(df_to_plot[['X_Present', 'Y_Present', 'Bubble_Present', 'Group_Auto']].sort_values('Bubble_Present', ascending=False).head(30), index=True, index_label='対象')
            _pulse_prompt = utils_ai.generate_ai_insight_prompt(
                role="特許戦略・ポートフォリオ分析の専門家として、CAGR×活動量の4象限動態分析マップを戦略的に分析してください。",
                context="""\
散布図による動態分析マップ（PULSE）を表示しています。
- X軸: CAGR（成長率）— 過去の出願勢い
- Y軸: 現在の活動量（直近N年の出願件数）— 対数スケール
- バブルサイズ: 総出願件数
- 4象限: リーダー(右上)/新興(右下)/成熟(左上)/衰退(左下)
- 十字線: 平均値で区切り""",
                data_summary=_pulse_data_str,
                instructions="""\
以下の観点で分析してください:
1. **象限分布**: 各象限のプレイヤー分布パターンと市場構造の評価
2. **リーダー分析**: リーダー象限の特徴と今後の展望
3. **新興勢力**: 高成長率プレイヤーの特定と将来性評価
4. **成熟・衰退**: 既存勢力の動向と衰退リスクの評価
5. **戦略提言**: ポートフォリオ全体のバランスと推奨アクション

各主張には必ず具体的な数値（CAGR値、活動量、件数等）を1つ以上含めてください。""",
                metadata=_meta_pulse,
                constraints="CAGRは単純な比率であり、出願戦略の変更や規制変更等の外部要因も考慮すること。",
                output_format="Markdown形式。見出し付きの構造化された分析レポート。"
            )
            utils_ai.render_ai_insight_button(_pulse_prompt, "mega_pulse_insight")

            # CAPCOM data/ JSON出力（MEGA モメンタム）
            try:
                import capcom
                if capcom.is_active():
                    entities = []
                    for _, row in df_to_plot.iterrows():
                        entities.append({
                            "name": str(row.name),
                            "cagr": round(float(row['X_Present']), 4) if pd.notna(row.get('X_Present')) else 0,
                            # NaN を int() に渡すと ValueError になるため NaN 安全化する。
                            "activity": int(row['Y_Present']) if pd.notna(row.get('Y_Present')) else 0,
                            "total": int(row['Bubble_Present']) if pd.notna(row.get('Bubble_Present')) else 0,
                            "quadrant": str(row.get('Group_Auto', ''))
                        })
                    mega_json = {
                        "metadata": {
                            "module": "MEGA",
                            "mode": "PULSE",
                            "axis": axis_label,
                            "total_entities": len(entities)
                        },
                        "entities": entities,
                        "quadrant_summary": {str(k): int(v) for k, v in _group_counts.items()}
                    }
                    # 軸別ファイル名で保存（出願人/IPC/Fタームを続けて実行しても上書きされない）
                    _axis_suffix = {'applicant_main': 'applicant', 'ipc_main_group': 'ipc',
                                    'fterm_main': 'fterm'}.get(axis_col, 'other')
                    capcom.save_data(f"mega_momentum_{_axis_suffix}.json", mega_json)
                    capcom.save_patents_csv()
            except Exception as e:
                # 失敗を可視化して原因を追えるようにする（保存失敗時も他モジュールの処理は継続）。
                st.caption(f":material/warning: 動態分析データの記録セッション保存に失敗しました（要確認）: {e}")

    # --- B. クラスタ動態（旧 TELESCOPE: 対象を絞ったドリルダウン地図） ---
    with tab_c:
        st.caption("動態マップで気になった出願人・分類を1つ選び、その特許群を内容の近さでクラスタ地図にします。期間で絞り込むと注力テーマの移り変わりを追えます。")

        st.subheader("分析対象の選択")
        drilldown_options = st.session_state.get("mega_drilldown_options", [('(分析対象を選択)', '(分析対象を選択)')])
        drilldown_target = st.selectbox("ドリルダウン対象:", options=drilldown_options, format_func=lambda x: x[0], key="drill_target",
                                        help="「動態マップ」タブでマップを一度描画すると、候補が件数の多い順に並びます。")[1]

        # 対象を選んで1ボタン（俯瞰図分析のドリルダウンと同じ型。粒度などは詳細設定に格納）
        _tide_run_clicked = st.button(
            ":material/map: 選択対象の俯瞰図を描く", type="primary", key="drill_run_map",
            help="選んだ出願人・分類の特許群だけを取り出し、内容の近さでサブクラスタに分けた俯瞰図を描きます。サブクラスタの粒度は自動調整されます。")

        with st.expander("詳細設定（サブクラスタの粒度・ラベル語数）", expanded=False):
            # :material/smart_toy: 自動最適化（既定ON。クラスタの粒度を決める2パラメータを自動掃引）
            drill_auto_hdbscan = st.checkbox(
                ":material/smart_toy: 自動最適化（クラスタの粒度を自動調整してサブクラスタ数を適正化）",
                value=True,
                key="mega_drill_auto_hdbscan",
                help="ドリルダウン対象の件数に合わせて HDBSCAN の2パラメータ（最小クラスタサイズ・最小サンプル数）を自動で掃引し、サブクラスタ数を適正化します。ONの間は下の手動値は無視されます。")
            drill_target_k = None
            if drill_auto_hdbscan:
                # メインマップと同じく、対象の件数に応じて目標サブクラスタ数を初期化する（suggest_target_k）。
                # 件数は選択肢ラベルの「(N件)」から取得し、対象を変えるたびに初期値が件数適応されるよう key に対象名を含める。
                _n_sub = 0
                for _lbl, _val in drilldown_options:
                    if _val == drilldown_target:
                        _m = re.search(r'\((\d[\d,]*)\s*件\)', _lbl)
                        if _m:
                            _n_sub = int(_m.group(1).replace(',', ''))
                        break
                drill_target_k = st.number_input(
                    "目標サブクラスタ数（目安）", min_value=2, max_value=80,
                    value=utils.suggest_target_k(_n_sub), key=f"mega_drill_target_k_{drilldown_target}",
                    help="この数に近づくよう2パラメータを掃引します（対象の件数から自動初期化）。"
                         "結果のサブクラスタ数や粒度に満足できない場合は、この値を増減して再度「:material/map: 選択対象の俯瞰図を描く」を押してください。"
                         "実際の値は密度構造に依存するため、目標ちょうどにならないこともあります（品質 DBCV を優先して選びます）。")
                utils.render_dbcv_help()
                # 前回の自動決定を常時再表示（メインマップと同じ挙動）。対象が一致する結果のみ表示する。
                _dar = st.session_state.get('mega_drill_auto_result')
                if _dar and _dar.get('target_id') == drilldown_target:
                    _rv = _dar.get('validity')
                    _rv_txt = f"・品質（DBCV）={_rv:.2f}" if isinstance(_rv, (int, float)) else ""
                    st.info(
                        f":material/smart_toy: 前回の自動決定: 最小クラスタサイズ=**{_dar['mcs']}** / 最小サンプル数=**{_dar['ms']}** "
                        f"→ サブクラスタ **{_dar['k']}**・ノイズ {_dar['noise'] * 100:.1f}%（目標≈{_dar['target_k']}{_rv_txt}）。"
                        f"　数が合わない/粒度が好みでない場合は上の「目標サブクラスタ数」を変えて再描画してください。")
            col1, col2, col3 = st.columns(3)
            with col1: drill_min_cluster_size = st.number_input('最小クラスタサイズ:', min_value=2, value=5, key="drill_min_cluster_size", disabled=drill_auto_hdbscan, help="1つのクラスタとして認める最小の特許件数です（クラスタの粒度設定）。小さくすると細かい技術テーマまで分かれてクラスタ数が増えますが、どこにも属さないノイズ（外れ値）も増えます。大きくすると少数の大まかなクラスタにまとまり安定しますが、細部は埋もれます。目安は母集団の約1〜2%。推奨10〜50。")
            with col2: drill_min_samples = st.number_input('最小サンプル数:', min_value=1, value=5, key="drill_min_samples", disabled=drill_auto_hdbscan, help="クラスタの「核」と認める密度の厳しさです。大きいほど判定が厳しくなりノイズ（外れ値）が増え、クラスタは密な中心部だけになります。小さいほど緩くなり多くの点が取り込まれます。通常は最小クラスタサイズ以下に設定します。推奨5〜20。")
            with col3: drill_label_top_n = st.number_input('ラベル単語数:', min_value=1, value=3, key="drill_label_top_n", help="各クラスタの自動命名に使う特徴語の数です。多いほど内容を詳しく表せますが冗長になり、少ないほど簡潔になります。")

        # mega_drill_intent_rerun / ctx: 意図ボタンの自動再実行フラグと制約（俯瞰図分析と同じ設計）
        _tide_intent_rerun = st.session_state.pop('mega_drill_intent_rerun', False)
        _tide_intent_ctx = st.session_state.pop('mega_drill_intent_ctx', None)
        if not _tide_intent_rerun:
            _tide_intent_ctx = None
        if _tide_run_clicked or _tide_intent_rerun:
            if drilldown_target == '(分析対象を選択)': st.error("選択してください。")
            else:
                with st.spinner("計算中..."):
                    try:
                        axis_col = ("applicant_main" if st.session_state.mega_axis_label == "出願人"
                                    else "ipc_main_group" if (("IPC" in st.session_state.mega_axis_label) or ("FI" in st.session_state.mega_axis_label))
                                    else "fterm_main")
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

                        if drill_auto_hdbscan:
                            _dpb = st.progress(0.0, text="サブクラスタのパラメータを掃引中...")
                            # 意図ボタン経由なら「今の粒度と同じ解の再選択」を範囲制約で禁じる
                            _tsweep_kwargs = {}
                            if _tide_intent_ctx and _tide_intent_ctx.get('k_min'):
                                _tsweep_kwargs['k_min'] = int(_tide_intent_ctx['k_min'])
                            if _tide_intent_ctx and _tide_intent_ctx.get('k_max'):
                                _tsweep_kwargs['k_max'] = int(_tide_intent_ctx['k_max'])
                            _dsweep = utils.sweep_hdbscan_params(
                                df_plot[['x', 'y']].values, target_k=int(drill_target_k),
                                progress_callback=lambda f: _dpb.progress(min(f, 1.0), text="サブクラスタのパラメータを掃引中..."),
                                **_tsweep_kwargs)
                            _dpb.empty()
                            cluster_labels = _dsweep['labels']
                            # 自動決定の結果を保持し、次回以降も「前回の自動決定」として再表示する（メインマップと同じ）。
                            st.session_state['mega_drill_auto_result'] = {
                                'mcs': _dsweep['min_cluster_size'], 'ms': _dsweep['min_samples'],
                                'k': _dsweep['n_clusters'], 'noise': _dsweep['noise_ratio'],
                                'target_k': _dsweep['target_k'], 'validity': _dsweep.get('validity'),
                                'target_id': drilldown_target}
                            _drv = _dsweep.get('validity')
                            _drv_txt = f"・品質（DBCV）={_drv:.2f}" if isinstance(_drv, (int, float)) else ""
                            st.caption(
                                f":material/smart_toy: 自動決定: 最小クラスタサイズ={_dsweep['min_cluster_size']} / 最小サンプル数={_dsweep['min_samples']} "
                                f"→ サブクラスタ {_dsweep['n_clusters']}・ノイズ {_dsweep['noise_ratio'] * 100:.0f}%（目標≈{_dsweep['target_k']}{_drv_txt}）")
                        else:
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

                        # utils.safe_auto_label で c-TF-IDF ラベリング（ドリルダウン、Janome 例外耐性付き）
                        mega_drill_texts = (
                            df_plot[col_map['title']].fillna('') + ' ' +
                            df_plot[col_map.get('abstract', '')].fillna('') if col_map.get('abstract') else df_plot[col_map['title']].fillna('')
                        )
                        label_map = utils.safe_auto_label(
                            mega_drill_texts,
                            df_plot['cluster_id'].values,
                            method='c-tfidf',
                            top_n=int(drill_label_top_n),
                        )

                        df_plot['label'] = df_plot['cluster_id'].map(label_map)
                        st.session_state.df_drilldown = df_plot
                        st.session_state.sbert_sub_cluster_map_auto = label_map
                        st.session_state.drilldown_target_name = drilldown_target
                        # 意図ボタン経由の場合: 変化の有無の通知を預ける（カード描画時にインライン表示）
                        if _tide_intent_ctx:
                            _tnk = int(len(set(df_plot['cluster_id'].unique()) - {-1}))
                            _tnz = float((df_plot['cluster_id'] == -1).mean())
                            _tim = _tide_intent_ctx.get('mode')
                            _tpk = _tide_intent_ctx.get('prev_k')
                            _tpn = _tide_intent_ctx.get('prev_noise')
                            _tfb = None
                            if _tim == 'finer':
                                if _tpk and _tnk > _tpk:
                                    _tfb = {'msg': f"サブクラスタ数が {_tpk} → {_tnk} に増えました", 'changed': True}
                                elif _tpk:
                                    _tfb = {'msg': f"これ以上細かく分かれませんでした（サブクラスタ数 {_tnk} のまま）。"
                                                   "データの密度構造の限界です", 'changed': False}
                            elif _tim == 'coarser':
                                if _tpk and _tnk < _tpk:
                                    _tfb = {'msg': f"サブクラスタ数が {_tpk} → {_tnk} にまとまりました", 'changed': True}
                                elif _tpk:
                                    _tfb = {'msg': f"これ以上大きなくくりにできませんでした（サブクラスタ数 {_tnk} のまま）",
                                            'changed': False}
                            elif _tim == 'less_noise':
                                if _tpn is not None and _tnz < _tpn - 1e-9:
                                    _tfb = {'msg': f"未分類が {_tpn * 100:.0f}% → {_tnz * 100:.0f}% に減りました",
                                            'changed': True}
                                elif _tpn is not None:
                                    _tfb = {'msg': f"未分類は減りませんでした（{_tnz * 100:.0f}%）。"
                                                   "もう一度押すと、さらに条件を緩めて試します", 'changed': False}
                            if _tfb:
                                st.session_state['mega_drill_intent_feedback'] = _tfb
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                        import traceback
                        st.exception(traceback.format_exc())

        # --- 描画 ---
        if "df_drilldown" in st.session_state:
            df_d = st.session_state.df_drilldown.copy()

            map_mode = st.radio("表示モード:", ["クラスタ領域 (Clusters)", "密度マップ (Density)", "散布図 (Scatter)"], horizontal=True, key="mega_map_mode", help="俯瞰図の描き方を切り替えます。クラスタ領域＝色付き領域で表示、密度マップ＝混み具合をヒートマップで表示、散布図＝個々の特許を点で表示。全体像は領域、混雑度は密度、個別確認は散布図が見やすいです。")

            c1, c2, c3 = st.columns(3)
            with c1:
                mesh_size = st.number_input("メッシュサイズ (Grid)", 40, 200, 40, step=5, key="mega_mesh", help="密度マップ（ヒートマップ）の格子の細かさです。大きいほど細かい格子で局所的な濃淡が見えますが、まばらだと粗く見えます。小さいほど滑らかで大まかな分布になります。")
                use_abs = st.checkbox("密度スケール固定", False, key="mega_abs") if map_mode == "密度マップ (Density)" else False
            with c2:
                no_noise = st.checkbox("ノイズを除く", False, key="mega_noise")
            with c3:
                show_label = st.checkbox("ラベルを表示", True, key="mega_label")

            c_t1, c_t2 = st.columns(2)
            with c_t1: interval = st.selectbox("期間の粒度:", [1, 2, 3, 5], index=2, key="mega_interval", help="トレンドを何年ごとに区切って集計するかです。粗く（大きく）すると大まかな傾向、細かく（小さく）すると年単位の動きが見えます。")

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

            # サブクラスタ→固定色（点・勢力圏・ラベル共通。全期間 df_d 基準で割り当てるため
            # 表示期間を切り替えても色が変わらない）
            _land_cmap_t = utils.landscape_color_map(df_d['cluster_id'])
            _is_density_t = (map_mode == "密度マップ (Density)")

            # 密度マップ（クラスタ別ソフト密度 = 技術の地形図スタイル）
            if _is_density_t and not df_in.empty:
                # 全期間 df_d の範囲で共通メッシュを定義（期間を切り替えても格子が動かない・端は余白ビンで見切れ防止）
                _tbx, _tby = utils.landscape_density_bins(df_d['x'], df_d['y'], mesh_size)
                # 「密度スケール固定」ON: 全期間・全サブクラスタでの最大ビン件数を濃さ基準にする
                # （期間を絞っても濃さの基準が変わらず、期間同士を見比べられる）
                _tzmax = None
                if use_abs:
                    try:
                        _xe = utils.landscape_bin_edges(_tbx)
                        _ye = utils.landscape_bin_edges(_tby)
                        _tzmax = max(
                            float(np.histogram2d(_g['x'], _g['y'], bins=[_xe, _ye])[0].max())
                            for _, _g in df_d[df_d['cluster_id'] != -1].groupby('cluster_id'))
                    except Exception:
                        _tzmax = None
                for _cid, _grp in df_in[df_in['cluster_id'] != -1].groupby('cluster_id'):
                    utils.add_landscape_density(
                        fig, _grp['x'], _grp['y'], _land_cmap_t.get(_cid, '#8C93A6'),
                        zmax=_tzmax, xbins=_tbx, ybins=_tby)

            # クラスタ領域（平滑化した勢力圏・点と同じ固定色）
            if map_mode == "クラスタ領域 (Clusters)" and not df_in.empty:
                for _cid, _grp in df_in[df_in['cluster_id'] != -1].groupby('cluster_id'):
                    utils.add_landscape_region(fig, _grp[['x', 'y']].values, _land_cmap_t.get(_cid, '#8C93A6'))

            # Ghost (期間外/背景)
            # 勢力圏（go.Scatter=SVG）と座標系を揃えるため、点も go.Scatter を使う
            # （go.Scattergl だと WebGL と SVG の座標系の差で勢力圏が点群からズレるため）。
            if not df_out.empty:
                fig.add_trace(go.Scatter(x=df_out['x'], y=df_out['y'], mode='markers', marker=dict(color='#cccccc', size=3, opacity=0.5), name='期間外', hoverinfo='skip'))

            # フォーカス (散布図・勢力圏/ラベルと同じ固定色。密度モードは点を小さく薄く)
            m_line = dict(width=0.6 if _is_density_t else 0.8, color='white')
            _pt_size_t = 4 if _is_density_t else utils.LANDSCAPE_POINT_SIZE
            _pt_opacity_t = 0.65 if _is_density_t else utils.LANDSCAPE_POINT_OPACITY
            _df_in_valid = df_in[df_in['cluster_id'] != -1]
            _df_in_noise = df_in[df_in['cluster_id'] == -1]

            if not _df_in_valid.empty:
                fig.add_trace(go.Scatter(
                    x=_df_in_valid['x'], y=_df_in_valid['y'], mode='markers',
                    marker=dict(color=[_land_cmap_t.get(_c, '#8C93A6') for _c in _df_in_valid['cluster_id']],
                                size=_pt_size_t, opacity=_pt_opacity_t, line=m_line),
                    hovertext=_df_in_valid['label'] + "<br>" + _df_in_valid[col_map['title']], name='期間内'
                ))
            # 未分類（ノイズ）は背景に沈める
            if not _df_in_noise.empty:
                fig.add_trace(go.Scatter(
                    x=_df_in_noise['x'], y=_df_in_noise['y'], mode='markers',
                    marker=dict(line=dict(width=0), **utils.LANDSCAPE_NOISE_STYLE),
                    hovertext=_df_in_noise['label'] + "<br>" + _df_in_noise[col_map['title']], name='未分類'
                ))

            # ラベル（ピル型・点/勢力圏と同色ドット・件数上位サブクラスタは強調・端は見切れ防止）
            if show_label:
                _tsizes = df_d[df_d['cluster_id'] != -1]['cluster_id'].value_counts().to_dict()
                _ttop = utils.landscape_top_clusters(_tsizes)
                _tlab_xr = (float(df_d['x'].min()), float(df_d['x'].max()))
                _tlab_yr = (float(df_d['y'].min()), float(df_d['y'].max()))
                for cid, grp in df_in[df_in['cluster_id'] != -1].groupby('cluster_id'):
                    if grp.empty: continue
                    mx, my = grp['x'].mean(), grp['y'].mean()
                    label_txt = st.session_state.sbert_sub_cluster_map_auto.get(cid, str(cid))
                    utils.add_landscape_label(
                        fig, mx, my, label_txt, _land_cmap_t.get(cid, '#8C93A6'),
                        emphasized=(cid in _ttop), x_range=_tlab_xr, y_range=_tlab_yr)

            # 図内タイトルは置かず（俯瞰図分析と同じ規則）、対象と期間はキャプションで示す
            st.caption(f"俯瞰図対象: {st.session_state.drilldown_target_name}{title_s}")
            utils.update_fig_layout(fig, "", height=1000, show_legend=False)
            fig.update_layout(margin=dict(l=16, r=16, t=24, b=16))
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

            # --- 読み方カード ---
            apollo_ui.howto(
                "選んだ対象の特許を内容の近さで並べ替えた地図です。<b>近くにある点は内容が似た特許</b>で、色の違いはサブクラスタ（技術のかたまり）を表します。"
                "<b>表示期間を絞ると期間内の出願だけが色付きになり、灰色の点は期間外</b>です。期間を切り替えながら、どのかたまりが厚くなり・薄くなったか（クラスタの動態）を追ってください。"
                "誤読に注意 — 地図の縦横の座標そのものに意味はなく、<b>読み取れるのは点どうしの近い・遠いだけ</b>です。"
                "クラスタ名は特徴語からの自動命名のため実際の内容とズレることがあり（下の「クラスタ・ラベル編集」で修正できます）、どのかたまりにも属さない点はノイズ（外れ値）として扱われます。",
                title="クラスタ地図の見方",
            )

            # --- しっくりこないときの調整（意図ベース・俯瞰図分析のドリルダウンと同じ型） ---
            with st.container(border=True):
                try:
                    _tlab_now = st.session_state.df_drilldown['cluster_id']
                    _tk_now = int(len(set(_tlab_now.unique()) - {-1}))
                    _tk_txt = f"（現在 {_tk_now} サブクラスタ・未分類 {float((_tlab_now == -1).mean()) * 100:.0f}%）"
                except Exception:
                    _tk_txt = ""
                st.markdown(
                    f"**この分け方、しっくりきましたか？**{_tk_txt} — "
                    "ボタンを押すと同じ対象のまま、その場で分け直します。何回でも試せます。")
                # 直前の意図ボタン実行の結果通知（インライン表示・1描画限り）
                _tfb = st.session_state.pop('mega_drill_intent_feedback', None)
                if _tfb:
                    if _tfb.get('changed'):
                        st.success(_tfb['msg'], icon=":material/check_circle:")
                    else:
                        st.info(_tfb['msg'], icon=":material/info:")
                _tib1, _tib2, _tib3 = st.columns(3)
                _tib_disabled = (drilldown_target == '(分析対象を選択)')
                _tib1.button(":material/search: もっと細かく分ける", key="mega_drill_intent_finer", use_container_width=True,
                             on_click=_tide_drill_adjust_intent, args=('finer',),
                             disabled=_tib_disabled,
                             help="サブクラスタの数を増やして、より細かい技術テーマまで分けます")
                _tib2.button(":material/map: もっと大きなくくりに", key="mega_drill_intent_coarser", use_container_width=True,
                             on_click=_tide_drill_adjust_intent, args=('coarser',),
                             disabled=_tib_disabled,
                             help="サブクラスタの数を減らして、大まかなテーマにまとめます")
                _tib3.button(":material/label: 未分類（灰色）を減らす", key="mega_drill_intent_noise", use_container_width=True,
                             on_click=_tide_drill_adjust_intent, args=('less_noise',),
                             disabled=_tib_disabled,
                             help="どのサブクラスタにも入らなかった特許を、できるだけ近くのまとまりに取り込みます")
                if _tib_disabled:
                    st.caption("上で「ドリルダウン対象」を選び直すとボタンが使えます。")
                else:
                    st.caption("細かい調整（目標サブクラスタ数の直接指定）は上の「詳細設定」からもできます。")

            st.session_state.df_drilldown_export = df_d

            # --- Snapshot: ドリルダウン地図 ---
            _drill_cluster_info = ""
            if 'sbert_sub_cluster_map_auto' in st.session_state:
                for _cid, _lbl in st.session_state.sbert_sub_cluster_map_auto.items():
                    _cnt = len(df_d[df_d['cluster_id'] == _cid]) if 'cluster_id' in df_d.columns else 0
                    _drill_cluster_info += f"  Cluster {_cid} ({_lbl}): {_cnt}件\n"
            _drill_snap_data = {
                'module': 'MEGA',
                'type': 'telescope_drilldown',
                'chart_data': f"ドリルダウン対象: {st.session_state.get('drilldown_target_name', 'N/A')}\n{_drill_cluster_info}",
            }
            # ドリルダウン対象別の key（静的キーだと対象を切り替えても「保存済み」のまま出るため）
            _mega_tel_slug = re.sub(r'\s+', '_', str(st.session_state.get('drilldown_target_name', 'target')))[:30] or 'target'
            utils.render_snapshot_button(
                title=f"クラスタ動態: {st.session_state.get('drilldown_target_name', 'N/A')}",
                description=f"ドリルダウン対象の技術ポートフォリオマップ（UMAP+HDBSCAN）。",
                key=f"mega_telescope_snap_{_mega_tel_slug}",
                group="mega_telescope_snap",
                fig=fig,
                data_summary=_drill_snap_data
            )
            apollo_ui.snapshot_hint()

            # --- AI Insight: 技術クラスタの深掘り分析 ---
            _meta_drill = utils_ai.build_common_metadata(df_main=df_main, col_map=col_map)
            _meta_drill['ドリルダウン対象'] = st.session_state.get('drilldown_target_name', 'N/A')
            _meta_drill['対象の分析軸'] = st.session_state.get('mega_axis_label', 'N/A')
            _meta_drill['クラスタ数'] = len(st.session_state.get('sbert_sub_cluster_map_auto', {}))
            # df_d はこのブロック冒頭（df_drilldown 取得時）で必ず定義済み。
            _meta_drill['分析対象件数'] = len(df_d)

            _drill_prompt = utils_ai.generate_ai_insight_prompt(
                role="技術戦略・特許ランドスケープの専門家として、ドリルダウン分析による技術クラスタの構造を分析してください。",
                context=f"""\
UMAP+HDBSCANによるクラスタリングマップを表示しています。
特定の出願人または分類コード（{utils.classification_code_label(col_map)}）の技術ポートフォリオを細分化し、各クラスタがどのような技術テーマに対応するかを可視化しています。""",
                data_summary=_drill_cluster_info,
                instructions="""\
以下の観点で分析してください:
1. **技術ポートフォリオ構造**: 主要クラスタの特徴と技術テーマの多様性
2. **注力領域**: 最大クラスタの技術的意味と事業との関連
3. **新規領域**: 小規模クラスタに潜む新技術・新規事業の可能性
4. **クラスタ間関係**: 技術テーマの関連性・統合可能性
5. **戦略提言**: ポートフォリオの強み・弱み・今後の方向性

各主張には必ず具体的な数値を1つ以上含めてください。""",
                metadata=_meta_drill,
                constraints="クラスタラベルはTF-IDF自動生成の場合があり、実際の技術内容と完全には一致しない可能性がある。",
                output_format="Markdown形式。見出し付きの構造化された分析レポート。"
            )
            utils_ai.render_ai_insight_button(_drill_prompt, "mega_telescope_insight")

            # CAPCOM data/ JSON出力（MEGA TELESCOPE ドリルダウン）
            try:
                import capcom
                if capcom.is_active() and 'cluster_id' in df_d.columns:
                    drill_clusters_json = []
                    # 代表特許（重心に近い3件）をクラスタIDごとに一括取得（埋め込みは関数内で session_state から取得）
                    _reps_by_cid = utils.get_cluster_representatives(df_d, cluster_col='cluster_id', n_reps=3)
                    for _cid in sorted(df_d['cluster_id'].unique()):
                        _grp = df_d[df_d['cluster_id'] == _cid]
                        _label = st.session_state.sbert_sub_cluster_map_auto.get(_cid, str(_cid))
                        drill_clusters_json.append({
                            "cluster_id": int(_cid) if pd.notna(_cid) else -1,
                            "label": _label,
                            "count": len(_grp),
                            "representatives": _reps_by_cid.get(_cid, [])
                        })
                    mega_drill_json = {
                        "metadata": {
                            "module": "MEGA",
                            "mode": "TELESCOPE",
                            "drilldown_target": st.session_state.get('drilldown_target_name', 'N/A'),
                            "n_clusters": len(drill_clusters_json)
                        },
                        "clusters": drill_clusters_json
                    }
                    # 対象別ファイル名で保存（複数対象を続けてドリルダウンしても上書きされない）
                    _drill_tgt = str(st.session_state.get('drilldown_target_name', 'N_A'))
                    _drill_slug = re.sub(r'\s+', '_', _drill_tgt).strip('_')[:40] or 'target'
                    capcom.save_data(f"mega_drilldown_{_drill_slug}.json", mega_drill_json)
                    capcom.save_patents_csv()
            except Exception as e:
                # 保存失敗を握りつぶさず可視化する。
                st.caption(f":material/warning: ドリルダウンデータの記録セッション保存に失敗しました（要確認）: {e}")

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
                cooc_top_n = st.slider("共起: 上位単語数", 30, 100, 70, key="mega_cooc_top_n", help="ネットワークに表示するキーワード（ノード）の数です。出現頻度の上位から何語を使うかを決めます。多いほど網羅的ですが密集して読みにくく、少ないほど主要語に絞られ見やすくなります。")
                cooc_threshold = st.slider("共起: Jaccard係数 閾値", 0.01, 0.3, 0.03, 0.01, key="mega_cooc_threshold", help="2つのキーワードを線（エッジ）で結ぶ基準の強さです。共起の強さを0〜1で表すJaccard係数がこの値以上のペアだけを結びます。高くすると強い関係だけが残ってスッキリし、低くすると弱い関係も含め線が増えて密になります。")

            if st.button("テキスト分析を実行", key="mega_run_text_mining"):
                with st.spinner("分析中..."):
                    # 文献ごとに抽出して集約（doc_words は共起ネットワークでも再利用し二度抽出を回避）
                    words, doc_words = [], []
                    for _, row in df_in.iterrows():
                        dt = ""
                        if col_map['title'] and pd.notna(row[col_map['title']]): dt += str(row[col_map['title']]) + " "
                        if col_map.get('abstract') and col_map['abstract'] in row and pd.notna(row[col_map['abstract']]):
                            dt += str(row[col_map['abstract']]) + " "
                        dw = extract_compound_nouns(dt, stopwords)
                        words.extend(dw)
                        doc_words.append(dw)

                    if not words: st.warning("有効なキーワードなし")
                    else:
                        st.markdown("##### 1. ワードクラウド")
                        generate_wordcloud_and_list(words, f"対象: {st.session_state.drilldown_target_name}{title_s}", 30, FONT_PATH, capcom_key="mega_drill")

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

    # --- C. エクスポート ---
    with tab_d:
        st.subheader("データエクスポート")
        if ("df_momentum_export" not in st.session_state) and ("df_drilldown_export" not in st.session_state):
            st.info("エクスポートできるデータがまだありません。「動態マップ」タブでマップを描画すると、ここからCSVをダウンロードできます。", icon=":material/upload:")
        if "df_momentum_export" in st.session_state:
            st.download_button("動態分析データ (CSV)", convert_df_to_csv(st.session_state.df_momentum_export), "MEGA_PULSE.csv", "text/csv")
        if "df_drilldown_export" in st.session_state:
            st.download_button("ドリルダウンデータ (CSV)", convert_df_to_csv(st.session_state.df_drilldown_export), "MEGA_TELESCOPE.csv", "text/csv")
