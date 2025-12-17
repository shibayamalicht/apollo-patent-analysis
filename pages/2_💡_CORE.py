import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import warnings
import unicodedata
import re
import json
import traceback

from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

# ==================================================================
# --- 1. ページ設定 ---
# ==================================================================
st.set_page_config(
    page_title="APOLLO | CORE", 
    page_icon="💡", 
    layout="wide"
)

pio.templates.default = "plotly_white"
warnings.filterwarnings('ignore')

# ==================================================================
# --- 2. デザインテーマ設定 & 共通CSS ---
# ==================================================================
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

def get_theme_config(theme_name):
    themes = {
        "APOLLO Standard": { "bg_color": "#ffffff", "text_color": "#333333", "plotly_template": "plotly_white", "color_sequence": px.colors.qualitative.G10, "css": """[data-testid="stHeader"] { background-color: #ffffff; } h1, h2, h3 { color: #003366; }""" },
        "Modern Presentation": { "bg_color": "#fdfdfd", "text_color": "#2c3e50", "plotly_template": "plotly_white", "color_sequence": ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#8ab17d"], "css": """[data-testid="stSidebar"] { background-color: #eaeaea; } [data-testid="stHeader"] { background-color: #fdfdfd; } h1, h2, h3 { color: #264653; font-family: "Georgia", serif; } .stButton>button { background-color: #264653; color: white; border-radius: 0px; }""" }
    }
    return themes.get(theme_name, themes["APOLLO Standard"])

# ==================================================================
# --- 3. ヘルパー関数 & リソースロード ---
# ==================================================================
@st.cache_resource
def load_tokenizer_core(): return Tokenizer()
t = load_tokenizer_core()

STOP_WORDS = {
    "する","ある","なる","ため","こと","よう","もの","これ","それ","あれ","ここ","そこ","どれ","どの","この","その","当該","該","および","及び","または","また","例えば","例えばは","において","により","に対して","に関して","について","として","としては","場合","一方","他方","さらに","そして","ただし","なお","等","など","等々","いわゆる","所謂","同様","同時","前記","本","同","各","各種","所定","所望","一例","他","一部","一つ","複数","少なくとも","少なくとも一つ","上記","下記","前述","後述","既述","関する","基づく","用いる","使用","利用","有する","含む","備える","設ける","すなわち","従って","しかしながら","次に","特に","具体的に","詳細に","いずれ","うち","それぞれ","とき","かかる","かような","かかる場合","本件","本願","本出願","本明細書",
    "できる", "いる", "提供", "明細書", 
    "本発明","発明","実施例","実施形態","変形例","請求","請求項","図","図面","符号","符号の説明","図面の簡単な説明","発明の詳細な説明","技術分野","背景技術","従来技術","発明が解決しようとする課題","課題","解決手段","効果","要約","発明の効果","目的","手段", "実施の形態","実施の態様","態様","変形","修正例","図示","図示例","図示しない","参照","参照符号","段落","詳細説明","要旨","一実施形態","他の実施形態","一実施例","別の側面","付記","適用例","用語の定義","開示","本開示","開示内容","記載","記述","掲載","言及","内容","詳細","説明","表記","表現","箇条書き","以下の","以上の","全ての","任意の","特定の",
    "上部","下部","内部","外部","内側","外側","表面","裏面","側面","上面","下面","端面","先端","基端","後端","一端","他端","中心","中央","周縁","周辺","近傍","方向","位置","空間","領域","範囲","間隔","距離","形状","形態","状態","種類","層","膜","部","部材","部位","部品","機構","装置","容器","組成","材料","用途","適用","適用例","片側","両側","左側","右側","前方","後方","上流","下流","隣接","近接","離間","間置","介在","重畳","概ね","略","略中央","固定側","可動側","伸長","収縮","係合","嵌合","取付","連結部","支持体","支持部","ガイド部",
    "データ","情報","信号","出力","入力","制御","演算","取得","送信","受信","表示","通知","設定","変更","更新","保存","削除","追加","実行","開始","終了","継続","停止","判定","判断","決定","選択","特定","抽出","検出","検知","測定","計測","移動","回転","変位","変形","固定","配置","生成","付与","供給","適用","照合","比較","算出","解析","同定","初期化","読出","書込","登録","記録","配信","連携","切替","起動","復帰","監視","通知処理","取得処理","演算処理",
    "良好","容易","簡便","適切","有利","有用","有効","効果的","高い","低い","大きい","小さい","新規","改良","改善","抑制","向上","低減","削減","増加","減少","可能","好適","好ましい","望ましい","優れる","優れた","高性能","高効率","低コスト","コスト","簡易","安定","安定性","耐久","耐久性","信頼性","簡素","簡略","単純","最適","最適化","汎用","汎用性","実現","達成","確保","維持","防止","回避","促進","不要","必要","高精度","省電力","省資源","高信頼","低負荷","高純度","高密度","高感度","迅速","円滑","簡略化","低価格","実効的","可能化","有効化","非必須","適合","互換",
    "出願","出願人","出願番号","出願日","出願書","出願公開","公開","公開番号","公開公報","公報","公報番号","特許","特許番号","特許文献","非特許文献","引用","引用文献","先行技術","審査","審査官","拒絶","意見書","補正書","優先","優先日","分割出願","継続出願","国内移行","国際出願","国際公開","PCT","登録","公開日","審査請求","拒絶理由","補正","訂正","無効審判","異議","取消","取下げ","事件番号","代理人","弁理士","係属","経過",
    "第","第一","第二","第三","第1","第２","第３","第１","第２","第３","一","二","三","四","五","六","七","八","九","零","数","複合","多数","少数","図1","図2","図3","図4","図5","図6","図7","図8","図9","表1","表2","表3","式1","式2","式3",
    "%","％","wt%","vol%","質量%","重量%","容量%","mol","mol%","mol/L","M","mm","cm","m","nm","μm","μ","rpm","Pa","kPa","MPa","GPa","N","W","V","A","mA","Hz","kHz","MHz","GHz","℃","°C","K","mL","L","g","kg","mg","wt","vol","h","hr","hrs","min","s","sec","ppm","ppb","bar","Ω","ohm","J","kJ","Wh","kWh",
    "株式会社","有限会社","合資会社","合名会社","合同会社","Inc","Inc.","Ltd","Ltd.","Co","Co.","Corp","Corp.","LLC", "GmbH","AG","BV","B.V.","S.A.","S.p.A.","（株）","㈱","（有）",
    "溶液","溶媒","触媒","反応","生成物","原料","成分","含有","含有量","配合","混合","混合物","濃度","温度","時間","割合","比率","基","官能基","化合物","組成物","樹脂","ポリマー","モノマー","基板","基材","フィルム","シート","粒子","粉末","比較例","参考例","試験","試料","評価","条件","実験","実験例","反応条件","反応時間","反応温度",
    "処理装置","端末","ユニット","モジュール","回路","素子","電源","電圧","電流","信号線","配線","端子","端部","接続", "接続部","演算部","記憶部","記憶装置","記録媒体","ユーザ","利用者","クライアント","サーバ","画面","UI","GUI","インターフェース","データベース","DB","ネットワーク","通信","要求","応答","リクエスト","レスポンス","パラメータ","引数","属性","プロパティ","フラグ","ID","ファイル","データ構造","テーブル","レコード",
    "軸","シャフト","ギア","モータ","エンジン","アクチュエータ","センサ","バルブ","ポンプ","筐体","ハウジング","フレーム","シャーシ","駆動","伝達","支持","連結"
}

@st.cache_data
def _core_text_preprocessor(text):
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFKC', text).lower()
    text = re.sub(r'[（(][^）)]{1,80}[）)]', ' ', text)
    text = re.sub(r'(?:図|Fig|FIG|fig)[. 　]*\d+', ' ', text)
    text = re.sub(r'[!！?"“”#$%＆&\'()（）*＋+,\-．.\:：;；<=>?？@\[\]［］\\^_`{|}~〜〜／/]', ' ', text)
    return text

@st.cache_data
def advanced_tokenize_core(text):
    text = _core_text_preprocessor(text)
    if not text: return ""
    tokens = list(t.tokenize(text))
    processed_tokens = []
    i = 0
    while i < len(tokens):
        token1 = tokens[i]
        base1 = token1.base_form if token1.base_form != '*' else token1.surface
        if base1 in STOP_WORDS: i += 1; continue
        pos1 = token1.part_of_speech.split(',')
        if len(base1) < 2 and pos1[0] != '名詞': i += 1; continue
        if pos1[0] == '名詞' and (len(pos1) > 1 and pos1[1] == '数'): i += 1; continue
        if (i + 1) < len(tokens):
            token2 = tokens[i+1]
            base2 = token2.base_form if token2.base_form != '*' else token2.surface
            pos2 = token2.part_of_speech.split(',')
            if pos1[0] == '名詞' and pos2[0] == '名詞' and base2 not in STOP_WORDS and (len(pos2) > 1 and pos2[1] != '数'):
                processed_tokens.append(base1 + base2); i += 2; continue
        if pos1[0] == '名詞' or (pos1[0] in ['動詞', '形容詞'] and len(pos1)>1 and pos1[1] == '自立'):
            processed_tokens.append(base1)
        i += 1
    return " ".join(processed_tokens)

# --- CORE 検索エンジン ---
def build_regex_pattern(keyword): return re.escape(keyword)
def build_near_regex(a, b, n): return r'(?:{}.{{0,{}}}?{}|{}.{{0,{}}}?{})'.format(a, n, b, b, n, a)
def build_adj_regex(a, b, n): return r'{}.{{0,{}}}?{}'.format(a, n, b)
def build_or_regex(a, b): return r'(?:{}|{})'.format(a, b)

def split_by_operator(text, operator):
    parts = []; balance = 0; current_chunk_start = 0
    for i, char in enumerate(text):
        if char == '(': balance += 1
        elif char == ')': balance -= 1
        elif char == operator and balance == 0:
            parts.append(text[current_chunk_start:i].strip()); current_chunk_start = i + 1
    parts.append(text[current_chunk_start:].strip())
    return parts

@st.cache_data
def parse_core_rule(rule_str):
    tokens = re.findall(r'\(|\)|' r'\bnear\d+\b|' r'\badj\d+\b|' r'[\+]|' r'[^()\s\+]+', rule_str, re.IGNORECASE)
    tokens = [t.strip() for t in tokens if t and t.strip()]
    output_queue, op_stack = [], []; op_precedence = {}
    for op in tokens:
        if op.lower() == '+': op_precedence[op] = 1
        elif op.lower().startswith(('near', 'adj')): op_precedence[op] = 3
    for token in tokens:
        if token == '(': op_stack.append(token)
        elif token == ')':
            while op_stack and op_stack[-1] != '(': output_queue.append(op_stack.pop())
            if op_stack: op_stack.pop()
        elif token.lower() in op_precedence:
            while (op_stack and op_stack[-1] != '(' and op_precedence.get(op_stack[-1].lower(), 0) >= op_precedence[token.lower()]):
                output_queue.append(op_stack.pop())
            op_stack.append(token)
        else: output_queue.append(token)
    while op_stack: output_queue.append(op_stack.pop())
    
    regex_stack = []
    for token in output_queue:
        if token.lower() not in op_precedence and token not in '()':
            norm = unicodedata.normalize('NFKC', token).lower()
            regex_stack.append(build_regex_pattern(norm))
        else:
            if len(regex_stack) < 2: raise ValueError(f"Invalid rule: {rule_str}")
            b, a = regex_stack.pop(), regex_stack.pop()
            tl = token.lower()
            if tl == '+': regex_stack.append(build_or_regex(a, b))
            elif tl.startswith('near'): regex_stack.append(build_near_regex(a, b, int(re.findall(r'\d+', tl)[0])))
            elif tl.startswith('adj'): regex_stack.append(build_adj_regex(a, b, int(re.findall(r'\d+', tl)[0])))
    if len(regex_stack) != 1: raise ValueError(f"Invalid rule: {rule_str}")
    return re.compile(regex_stack[0], re.IGNORECASE | re.DOTALL)

@st.cache_data
def prepare_axis_data_core(df, axis_col_name, delimiter):
    df_processed = df.copy()
    if axis_col_name not in df_processed.columns: return pd.DataFrame()
    df_processed[axis_col_name] = df_processed[axis_col_name].fillna('N/A')
    if axis_col_name == 'year':
        df_processed[axis_col_name] = df_processed[axis_col_name].apply(lambda x: str(int(x)) if pd.notna(x) else 'N/A')
    if delimiter:
        df_processed[axis_col_name] = df_processed[axis_col_name].astype(str).str.split(delimiter)
        df_processed = df_processed.explode(axis_col_name)
    df_processed[axis_col_name] = df_processed[axis_col_name].astype(str).str.strip().replace('', 'N/A')
    return df_processed

@st.cache_data
def convert_df_to_csv_core(df): return df.to_csv(encoding='utf-8-sig').encode('utf-8-sig')

# ==================================================================
# --- 4. アプリケーション初期化 & UI構成 ---
# ==================================================================
with st.sidebar:
    st.title("APOLLO") 
    st.markdown("Advanced Patent & Overall Landscape-analytics Logic Orbiter")
    st.markdown("**v.3**")
    st.markdown("---")
    st.subheader("Home"); st.page_link("Home.py", label="Mission Control", icon="🛰️")
    st.subheader("Modules")
    st.page_link("pages/1_🌍_ATLAS.py", label="ATLAS", icon="🌍")
    st.page_link("pages/2_💡_CORE.py", label="CORE", icon="💡")
    st.page_link("pages/3_🚀_Saturn_V.py", label="Saturn V", icon="🚀")
    st.page_link("pages/4_📈_MEGA.py", label="MEGA", icon="📈")
    st.page_link("pages/5_🧭_Explorer.py", label="Explorer", icon="🧭")
    st.page_link("pages/6_🔗_CREW.py", label="CREW", icon="🔗")
    st.markdown("---")
    st.caption("ナビゲーション:")
    st.caption("1. Mission Control でデータをアップロードし、前処理を実行します。")
    st.caption("2. 上のリストから分析モジュールを選択します。")
    st.markdown("---")
    st.caption("© 2025 しばやま")

st.title("💡 CORE")
st.markdown("Contextual Operator & Rule Engine: **論理式ベースの特許分類ツール**です。")

col_theme, _ = st.columns([1, 3])
with col_theme:
    selected_theme = st.selectbox("表示テーマ:", ["APOLLO Standard", "Modern Presentation"], key="core_theme_selector")
theme_config = get_theme_config(selected_theme)
st.markdown(f"<style>{theme_config['css']}</style>", unsafe_allow_html=True)

if not st.session_state.get("preprocess_done", False):
    st.error("分析データがありません。"); st.stop()
else:
    df_main = st.session_state.df_main
    col_map = st.session_state.col_map

if "core_classification_rules" not in st.session_state: st.session_state.core_classification_rules = {}
if "core_df_classified" not in st.session_state: st.session_state.core_df_classified = None
if "core_current_axis" not in st.session_state: st.session_state.core_current_axis = ""
if "core_reanalyze_result" not in st.session_state: st.session_state.core_reanalyze_result = ""

# ==================================================================
# --- 5. CORE アプリケーション ---
# ==================================================================
current_phase = st.radio("フェーズ選択:", ["フェーズ 1: AIアシスタント (KMeans)", "フェーズ 2: 分類ルール定義", "フェーズ 3: 分類実行", "フェーズ 4: 特許マップ作成"], horizontal=True, key="core_phase_selector")
st.markdown("---")

# --- フェーズ 1: AIアシスタント ---
if current_phase.startswith("フェーズ 1"):
    st.subheader("フェーズ 1: AIによる分類サジェスト (オプション)")
    col_map_options = [v for k, v in col_map.items() if k in ['title', 'abstract', 'claim']]
    target_column = st.selectbox("分析対象カラム:", options=col_map_options, key="core_target_col")
    
    col1, col2 = st.columns(2)
    with col1: ai_k_w = st.number_input("トピック数 (K)", min_value=2, value=8, key="core_k")
    with col2: ai_n_w = st.number_input("サンプル数 (N)", min_value=1, value=5, key="core_n")
    
    use_mece = st.checkbox("MECEモード (自動決定)", value=True, key="core_use_mece")
    
    if not use_mece:
        st.markdown("<b>生成する分類の数 (手動設定):</b>", unsafe_allow_html=True)
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1: ai_cat_count_tech = st.number_input("技術分類:", min_value=1, value=6, key="core_cat_tech")
        with col_c2: ai_cat_count_prob = st.number_input("課題分類:", min_value=1, value=6, key="core_cat_prob")
        with col_c3: ai_cat_count_sol = st.number_input("解決手段分類:", min_value=1, value=6, key="core_cat_sol")

    if st.button("AIアシスタント用プロンプトを生成", key="core_run_ai"):
        with st.spinner("分析中..."):
            try:
                texts_raw = df_main[target_column].astype(str).fillna('')
                tokenized_texts = texts_raw.apply(advanced_tokenize_core)
                vec = TfidfVectorizer(min_df=1, max_df=0.9, token_pattern=r"(?u)\b\w+\b")
                tfidf = vec.fit_transform(tokenized_texts)
                km = KMeans(n_clusters=int(ai_k_w), random_state=42, n_init=10).fit(tfidf)
                
                sampled_docs = []
                for i in range(int(ai_k_w)):
                    c_idx = np.where(km.labels_ == i)[0]
                    if len(c_idx) == 0: continue
                    dists = euclidean_distances(tfidf[c_idx], km.cluster_centers_[i].reshape(1,-1))
                    top_idx = c_idx[dists.flatten().argsort()[:int(ai_n_w)]]
                    sampled_docs.append(f"\n--- Cluster {i} ---\n" + "\n".join([f"・{_core_text_preprocessor(texts_raw.iloc[idx])}" for idx in top_idx]))
                
                if use_mece:
                    instruction_text = (
                        "この特許母集団全体を網羅的に分類するための、**「技術分類」「課題分類」「解決手段分類」**の3つの分類軸について、**分類定義**（分類名、定義、CORE論理式のセット）を設計してください。\n"
                        "\n# 重要: MECE (Mutually Exclusive, Collectively Exhaustive) の原則\n"
                        "- 生成する各分類軸内のカテゴリは、相互に排他的（ダブりがない）であり、かつ全体として網羅的（モレがない）であるように設計してください。\n"
                        "- 各軸のカテゴリ数は、MECEを満たすのに最適だとあなたが判断する数（目安として5〜10個程度）にしてください。"
                    )
                else:
                    instruction_text = "\n".join([
                        "この特許母集団全体を網羅的に分類するための、以下の3つの分類軸について、指定された個数で**分類定義**を設計してください。",
                        f"- **技術分類**: {ai_cat_count_tech}個",
                        f"- **課題分類**: {ai_cat_count_prob}個",
                        f"- **解決手段分類**: {ai_cat_count_sol}個"
                    ])

                sampled_docs_str = "".join(sampled_docs)

                prompt = f"""
あなたは優秀な特許情報ストラテジストです。
以下の「代表文献サンプル」は、ある特許母集団（{len(df_main)}件）をK-Means法で{ai_k_w}個のクラスタに分類し、各クラスタから代表的な文献の「{target_column}」を{ai_n_w}件ずつ抽出したものです。

# 依頼内容
{instruction_text}

以下の形式の **JSONデータのみ** を出力してください。解説は不要です。
JSONをコピーしてシステムにそのままインポートします。

# JSONフォーマット (厳守)
{{
  "技術分類": [
    {{
      "name": "カテゴリ名 (例: CO2分離膜)",
      "definition": "カテゴリの定義...",
      "rule": "CORE論理式 (例: (CO2 + 二酸化炭素) * (膜 + メンブレン))"
    }},
    ...
  ],
  "課題分類": [ ... ],
  "解決手段分類": [ ... ]
}}

# CORE論理式文法 (厳守)
- `A + B` (OR): A または B
- `A * B` (AND): A かつ B (順序問わず)
- `A nearN B` (近傍): AとBが**N文字**以内で出現 (順序問わず)。Nは10〜40程度を推奨。
- `A adjN B` (順序指定近傍): AがBの**N文字**以内にA→Bの順で出現。Nは1〜10程度を推奨。
- **重要:** キーワードはスペースを含まない単一語（例: `二酸化炭素`）にしてください。スペースを含むフレーズは `adj1` で表現してください。

# 最重要ルール (キーワード拡張と表記ゆれ)
- サンプルに存在するキーワードをそのまま使うだけでは不十分です。
- AIの知識を活用し、そのキーワードの**類義語、関連語、上位/下位概念、特許特有の表現、表記ゆれ（カタカナ、ひらがな、漢字）**を、あなたの知識ベースから網羅的に想起してください。
- **特許用語の網羅:** （例: 「保持」→「担持」「固着」「係止」など、特許で使われる言い換えを網羅）
- **概念の階層化:** 上位概念（例: 「車両」）と下位概念（例: 「自動車」「二輪車」）の両方を含め、取りこぼしを防ぎます。
- **カタカナ:** キーワードにカタカナを使用する場合は、**必ず全角（例: `ポリマー`）**を使用し、**半角（例: `ﾎﾟﾘﾏｰ`）は絶対に使用しないでください**。

# 代表文献サンプル
{sampled_docs_str}
"""
                st.success("プロンプトを生成しました。右上のコピーボタンでコピーしてください。")
                st.code(prompt, language='markdown')
            except Exception as e: st.error(f"エラー: {e}")

# --- フェーズ 2: 分類ルール定義 ---
elif current_phase.startswith("フェーズ 2"):
    st.subheader("フェーズ 2: 分類ルール定義")
    
    tab_manual, tab_json = st.tabs(["手動追加・修正", "JSON一括インポート"])
    existing = list(st.session_state.core_classification_rules.keys())
    
    with tab_manual:
        is_edit_mode = "core_edit_target" in st.session_state and st.session_state.core_edit_target is not None
        
        mode = st.radio("軸の指定:", ["新規作成", "既存に追加"], horizontal=True, index=1 if is_edit_mode else 0)
        
        if mode == "既存に追加" and existing:
            default_idx = 0
            if is_edit_mode:
                try: default_idx = existing.index(st.session_state.core_edit_target["axis"])
                except: pass
            elif st.session_state.core_current_axis in existing:
                try: default_idx = existing.index(st.session_state.core_current_axis)
                except: pass
            axis = st.selectbox("追加/修正先の軸:", existing, index=default_idx)
        else:
            axis = st.text_input("新規軸名:", value=st.session_state.core_edit_target["axis"] if is_edit_mode else "", placeholder="例: 課題分類")
            
        c_name = st.text_input("分類名:", value=st.session_state.core_edit_target["cat"] if is_edit_mode else "", placeholder="例: 耐久性向上")
        c_def = st.text_area("定義:", value=st.session_state.core_edit_target["def"] if is_edit_mode else "", height=68)
        c_rule = st.text_input("論理式:", value=st.session_state.core_edit_target["rule"] if is_edit_mode else "", placeholder="(耐久性 + 寿命) * 向上")
        
        btn_label = "ルールを更新" if is_edit_mode else "ルールを追加"
        
        if st.button(btn_label, key="add_manual"):
            if all([axis, c_name, c_rule]):
                try:
                    parse_core_rule(c_rule)
                    if axis not in st.session_state.core_classification_rules:
                        st.session_state.core_classification_rules[axis] = {}
                    st.session_state.core_classification_rules[axis][c_name] = {'rule': c_rule, 'definition': c_def}
                    st.session_state.core_current_axis = axis
                    if is_edit_mode: del st.session_state.core_edit_target
                    st.success(f"{btn_label}しました: {c_name}")
                    st.rerun()
                except Exception as e: st.error(f"文法エラー: {e}")
        
        if is_edit_mode:
            if st.button("編集をキャンセル"):
                del st.session_state.core_edit_target
                st.rerun()
    
    with tab_json:
        st.markdown("AIが生成したJSONをここに貼り付けてください。既存のルールは維持され、新しい軸が追加されます。")
        json_input = st.text_area("JSON入力:", height=300)
        if st.button("JSONを一括インポート"):
            try:
                cleaned_json = re.sub(r'^```json\s*|\s*```$', '', json_input.strip(), flags=re.MULTILINE)
                data = json.loads(cleaned_json)
                count = 0
                for axis_name, categories in data.items():
                    if axis_name not in st.session_state.core_classification_rules:
                        st.session_state.core_classification_rules[axis_name] = {}
                    for cat in categories:
                        name = cat.get('name'); rule = cat.get('rule'); defn = cat.get('definition', '')
                        if name and rule:
                            st.session_state.core_classification_rules[axis_name][name] = {'rule': rule, 'definition': defn}
                            count += 1
                st.success(f"{count} 個のルールをインポートしました！")
                st.rerun()
            except Exception as e: st.error(f"JSONパースエラー: {e}")

    st.markdown("---")
    st.subheader("現在のルール一覧")
    
    if st.button("全ルールを削除", type="primary"):
        st.session_state.core_classification_rules = {}
        st.rerun()
        
    for ax, cats in st.session_state.core_classification_rules.items():
        with st.expander(f"軸: {ax} ({len(cats)}件)"):
            for cn, cd in cats.items():
                r = cd['rule'] if isinstance(cd, dict) else cd[0]
                d = cd.get('definition', '') if isinstance(cd, dict) else ""
                
                c1, c2, c3 = st.columns([1, 4, 1])
                with c1:
                    if st.button("編集", key=f"edit_{ax}_{cn}"):
                        st.session_state.core_edit_target = {"axis": ax, "cat": cn, "rule": r, "def": d}
                        st.rerun()
                with c2:
                    st.text(f"【{cn}】 {r}")
                with c3:
                    if st.button("削除", key=f"del_{ax}_{cn}"):
                        del st.session_state.core_classification_rules[ax][cn]
                        if not st.session_state.core_classification_rules[ax]:
                            del st.session_state.core_classification_rules[ax]
                        st.rerun()

# --- フェーズ 3: 分類実行 ---
elif current_phase.startswith("フェーズ 3"):
    st.subheader("フェーズ 3: 分類実行")
    
    st.info("※ 探索範囲は自動的に「発明の名称 + 要約 + 請求項」の結合テキストとなります。")
    
    if st.button("すべての分類を実行", type="primary"):
        if not st.session_state.core_classification_rules:
            st.error("ルールがありません。")
        else:
            with st.spinner("実行中..."):
                try:
                    df_res = df_main.copy()
                    
                    search_cols = []
                    if col_map.get('title') in df_res.columns: search_cols.append(df_res[col_map['title']].fillna(''))
                    if col_map.get('abstract') in df_res.columns: search_cols.append(df_res[col_map['abstract']].fillna(''))
                    if col_map.get('claim') in df_res.columns: search_cols.append(df_res[col_map['claim']].fillna(''))
                    
                    combined_text = search_cols[0]
                    for s in search_cols[1:]:
                        combined_text = combined_text + " " + s
                    
                    rules = st.session_state.core_classification_rules
                    compiled_rules = {}
                    for ax, cats in rules.items():
                        compiled_rules[ax] = []
                        for cn, cd in cats.items():
                            r_str = cd['rule'] if isinstance(cd, dict) else cd[0]
                            or_parts = split_by_operator(r_str, '+')
                            comp_or = []
                            for op in or_parts:
                                and_parts = split_by_operator(op, '*')
                                comp_and = [parse_core_rule(ap.strip()) for ap in and_parts]
                                comp_or.append(comp_and)
                            compiled_rules[ax].append((cn, comp_or))
                    
                    def apply_rules(text, ax_rules):
                        text = _core_text_preprocessor(str(text))
                        hits = []
                        for c_name, c_logic in ax_rules:
                            match_or = False
                            for and_block in c_logic:
                                match_and = True
                                for regex in and_block:
                                    if not regex.search(text): match_and = False; break
                                if match_and: match_or = True; break
                            if match_or: hits.append(c_name)
                        return ";".join(hits) if hits else "その他"

                    bar = st.progress(0)
                    for i, ax in enumerate(rules.keys()):
                        df_res[ax] = combined_text.apply(lambda x: apply_rules(x, compiled_rules[ax]))
                        bar.progress((i+1)/len(rules))
                    
                    st.session_state.core_df_classified = df_res
                    st.success("完了！")
                    
                    st.subheader("分類結果サマリー")
                    cols = st.columns(len(rules))
                    for i, ax in enumerate(rules.keys()):
                        with cols[i]:
                            st.markdown(f"**{ax}**")
                            counts = df_res[ax].str.split(';').explode().value_counts()
                            st.dataframe(counts)
                    
                    csv_core = convert_df_to_csv_core(df_res)
                    st.download_button("分類結果CSVをダウンロード", csv_core, "CORE_classified.csv", "text/csv")
                    
                except Exception as e: st.error(f"エラー: {e}")

    st.markdown("---")
    st.subheader("🔍 未分類データの再分析 (『その他』を減らす)")
    if st.session_state.core_df_classified is not None:
        rules = st.session_state.core_classification_rules
        if rules:
            col_re1, col_re2 = st.columns(2)
            with col_re1: reanalyze_axis = st.selectbox("再分析する軸を選択:", list(rules.keys()), key="core_reanalyze_axis")
            
            col_k, col_n = st.columns(2)
            with col_k: re_k = st.number_input("抽出トピック数 (K)", value=5, key="re_k")
            with col_n: re_n = st.number_input("1トピックあたりのサンプル数 (N)", value=3, key="re_n")
            
            re_mece = st.checkbox("MECEモード (自動)", value=True, key="re_mece")
            re_cnt = 3 if re_mece else st.number_input("追加するカテゴリ数", value=3, key="re_cnt")

            if st.button("『その他』を分析して新ルールを提案", key="core_btn_reanalyze"):
                try:
                    df_c = st.session_state.core_df_classified
                    others_df = df_c[df_c[reanalyze_axis] == 'その他']
                    if others_df.empty:
                        st.info("『その他』はありません。")
                    else:
                        with st.spinner(f"『その他』({len(others_df)}件) を分析中..."):
                            search_cols = []
                            if col_map.get('title') in others_df.columns: search_cols.append(others_df[col_map['title']].fillna(''))
                            if col_map.get('abstract') in others_df.columns: search_cols.append(others_df[col_map['abstract']].fillna(''))
                            if col_map.get('claim') in others_df.columns: search_cols.append(others_df[col_map['claim']].fillna(''))
                            texts = search_cols[0]
                            for s in search_cols[1:]: texts = texts + " " + s
                            
                            toks = texts.apply(advanced_tokenize_core)
                            vec = TfidfVectorizer(min_df=1, max_df=0.9, token_pattern=r"(?u)\b\w+\b")
                            tfidf = vec.fit_transform(toks)
                            
                            actual_k = min(int(re_k), len(others_df))
                            if actual_k < 2: actual_k = 1
                            km = KMeans(n_clusters=actual_k, random_state=42).fit(tfidf)
                            
                            s_docs = []
                            for i in range(actual_k):
                                c_idx = np.where(km.labels_ == i)[0]
                                if len(c_idx) == 0: continue
                                dists = euclidean_distances(tfidf[c_idx], km.cluster_centers_[i].reshape(1,-1))
                                top_idx = c_idx[dists.flatten().argsort()[:int(re_n)]]
                                s_docs.append(f"\n--- その他グループ {i} ---\n" + "\n".join([f"・{_core_text_preprocessor(texts.iloc[idx])}" for idx in top_idx]))
                            
                            s_docs_str = "".join(s_docs)
                            exist_rules = [f"- {cat}: {d['rule']}" for cat, d in rules[reanalyze_axis].items()]
                            exist_rules_str = "\n".join(exist_rules)
                            
                            instruction_part = "MECEを意識し、カテゴリ数は自動で最適化してください。" if re_mece else f"**{re_cnt}個** の新しいカテゴリを追加してください。"
                            
                            p_re = f"""
あなたは特許情報ストラテジストです。
現在、分類軸「{reanalyze_axis}」を作成中ですが、以下の「既存の分類」に当てはまらない特許が「その他」として残っています。

# 既存の分類リスト
{exist_rules_str}

# 依頼内容
以下の「未分類特許のサンプル」を分析し、**既存の分類とは概念的に重複しない、新しい分類カテゴリ**を提案してください。
出力は **JSON形式のみ** としてください。
{instruction_part}

# JSONフォーマット
{{
  "{reanalyze_axis}": [
    {{
      "name": "新カテゴリ名",
      "definition": "...",
      "rule": "論理式"
    }}, ...
  ]
}}

# 未分類特許のサンプル
{s_docs_str}
"""
                            st.session_state.core_reanalyze_result = p_re
                except Exception as e: st.error(f"エラー: {e}")
        
        if st.session_state.core_reanalyze_result:
            st.success("再分析プロンプトを生成しました。"); st.code(st.session_state.core_reanalyze_result, language='markdown')

# --- フェーズ 4: 特許マップ ---
elif current_phase.startswith("フェーズ 4"):
    st.subheader("フェーズ 4: 特許マップ作成")
    
    if st.session_state.core_df_classified is None:
        st.warning("先に分類を実行してください。")
    else:
        df_c = st.session_state.core_df_classified
        axes = list(st.session_state.core_classification_rules.keys())
        meta_axes = []
        if 'year' in df_c.columns: meta_axes.append('出願年')
        if col_map.get('applicant') in df_c.columns: meta_axes.append('出願人')
        all_axes = axes + meta_axes
        
        c1, c2, c3 = st.columns(3)
        with c1: x_ax = st.selectbox("X軸", all_axes, index=0)
        with c2: y_ax = st.selectbox("Y軸", all_axes, index=min(1, len(all_axes)-1))
        with c3: chart_type = st.radio("グラフタイプ", ["ヒートマップ", "バブルチャート"])
        
        col_f1, col_f2 = st.columns(2)
        with col_f1: exclude_other = st.checkbox("「その他」を除外する", value=True)
        
        if st.button("描画"):
            def get_col_data(ax_name):
                if ax_name == '出願年': return df_c['year'].fillna(0).astype(int).astype(str), None
                if ax_name == '出願人': return df_c[col_map['applicant']].fillna('Unknown'), ';' 
                if ax_name in axes: return df_c[ax_name], ';'
                return None, None

            x_data, x_sep = get_col_data(x_ax); y_data, y_sep = get_col_data(y_ax)
            temp_df = pd.DataFrame({'X': x_data, 'Y': y_data})
            if x_sep: temp_df['X'] = temp_df['X'].astype(str).str.split(x_sep); temp_df = temp_df.explode('X')
            if y_sep: temp_df['Y'] = temp_df['Y'].astype(str).str.split(y_sep); temp_df = temp_df.explode('Y')
            
            temp_df = temp_df.replace({'nan': np.nan, 'None': np.nan}).dropna()
            if exclude_other:
                temp_df = temp_df[(temp_df['X'] != 'その他') & (temp_df['Y'] != 'その他')]
            
            if temp_df.empty: st.warning("データなし")
            else:
                ct = pd.crosstab(temp_df['Y'], temp_df['X'])
                
                if x_ax == '出願年': x_ord = sorted(ct.columns, key=lambda x: int(x) if x.isdigit() else x)
                else: x_ord = ct.sum(axis=0).sort_values(ascending=False).index.tolist()
                
                if y_ax == '出願年': y_ord = sorted(ct.index, key=lambda x: int(x) if x.isdigit() else x)
                else: y_ord = ct.sum(axis=1).sort_values(ascending=False).index.tolist()
                
                ct = ct.reindex(index=y_ord, columns=x_ord).fillna(0)
                
                if chart_type == "ヒートマップ":
                    fig = px.imshow(
                        ct, 
                        labels=dict(x=x_ax, y=y_ax, color="件数"),
                        x=ct.columns,
                        y=ct.index,
                        aspect="auto",
                        color_continuous_scale='YlGnBu',
                        text_auto=True
                    )
                    
                    fig.update_layout(
                        title=f"{x_ax} × {y_ax}",
                        height=max(600, len(ct)*40),
                        yaxis=dict(title=y_ax),
                        xaxis=dict(title=x_ax, side='bottom')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                else: # バブルチャート
                    ct_long = ct.reset_index().melt(id_vars='Y', var_name='X', value_name='Count')
                    ct_long = ct_long[ct_long['Count'] > 0]
                    atlas_colors = theme_config["color_sequence"]
                    
                    fig = px.scatter(
                        ct_long, x='X', y='Y', size='Count', color='Y',
                        size_max=60, color_discrete_sequence=atlas_colors,
                        category_orders={'X': x_ord, 'Y': y_ord} 
                    )
                    fig.update_yaxes(categoryorder='array', categoryarray=y_ord, autorange='reversed', title=y_ax, type='category')
                    fig.update_xaxes(categoryorder='array', categoryarray=x_ord, title=x_ax, side='bottom', type='category')
                    
                    fig.update_layout(title=f"{x_ax} × {y_ax}", height=max(600, len(ct)*40), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
        
        # CSVダウンロード
        st.markdown("---")
        csv_core = convert_df_to_csv_core(df_c)
        st.download_button("分類結果付き全データCSVをダウンロード", csv_core, "CORE_classified_full.csv", "text/csv")