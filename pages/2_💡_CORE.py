import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import warnings
import unicodedata
import re
import traceback

from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

# 設定
pio.templates.default = "plotly_white"
warnings.filterwarnings('ignore')

# ==================================================================
# --- 1. ヘルパー関数 & リソースロード ---
# ==================================================================

@st.cache_resource
def load_tokenizer_core():
    """Janome Tokenizerをロードおよびキャッシュ"""
    return Tokenizer()

t = load_tokenizer_core()

# 分析除外用ストップワード
STOP_WORDS = {
    "する","ある","なる","ため","こと","よう","もの","これ","それ","あれ","ここ","そこ","どれ","どの","この","その","当該","該","および","及び","または","また","例えば","例えばは","において","により","に対して","に関して","について","として","としては","場合","一方","他方","さらに","そして","ただし","なお","等","など","等々","いわゆる","所謂","同様","同時","前記","本","同","各","各種","所定","所望","一例","他","一部","一つ","複数","少なくとも","少なくとも一つ","上記","下記","前述","後述","既述","関する","基づく","用いる","使用","利用","有する","含む","備える","設ける","すなわち","従って","しかしながら","次に","特に","具体的に","詳細に","いずれ","うち","それぞれ","とき","かかる","かような","かかる場合","本件","本願","本出願","本明細書",
    "できる", "いる", "提供", "明細書", 
    "本発明","発明","実施例","実施形態","変形例","請求","請求項","図","図面","符号","符号の説明","図面の簡単な説明","発明の詳細な説明","技術分野","背景技術","従来技術","発明が解決しようとする課題","課題","解決手段","効果","要約","発明の効果","目的","手段", "実施の形態","実施の態様","態様","変形","修正例","図示","図示例","図示しない","参照","参照符号","段落","詳細説明","要旨","一実施形態","他の実施形態","一実施例","別の側面","付記","適用例","用語の定義","開示","本開示","開示内容",
    "出願","出願人","出願番号","出願日","出願書","出願公開","公開","公開番号","公開公報","公報","公報番号","特許","特許番号","特許文献","非特許文献","引用","引用文献","先行技術","審査","審査官","拒絶","意見書","補正書","優先","優先日","分割出願","継続出願","国内移行","国際出願","国際公開","PCT","登録","公開日","審査請求","拒絶理由","補正","訂正","無効審判","異議","取消","取下げ","事件番号","代理人","弁理士","係属","経過",
    "第","第一","第二","第三","第1","第２","第３","第１","第２","第３","一","二","三","四","五","六","七","八","九","零","数","複合","多数","少数","図1","図2","図3","図4","図5","図6","図7","図8","図9","表1","表2","表3","式1","式2","式3",
    "%","％","wt%","vol%","質量%","重量%","容量%","mol","mol%","mol/L","M","mm","cm","m","nm","μm","μ","rpm","Pa","kPa","MPa","GPa","N","W","V","A","mA","Hz","kHz","MHz","GHz","℃","°C","K","mL","L","g","kg","mg","wt","vol","h","hr","hrs","min","s","sec","ppm","ppb","bar","Ω","ohm","J","kJ","Wh","kWh",
    "株式会社","有限会社","合資会社","合名会社","合同会社","Inc","Inc.","Ltd","Ltd.","Co","Co.","Corp","Corp.","LLC", "GmbH","AG","BV","B.V.","S.A.","S.p.A.","（株）","㈱","（有）"
}

@st.cache_data
def _core_text_preprocessor(text):
    """テキストの前処理（正規化、記号除去）"""
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFKC', text).lower()
    text = re.sub(r'[（(][^）)]{1,80}[）)]', ' ', text)
    text = re.sub(r'(?:図|Fig|FIG|fig)[. 　]*\d+', ' ', text)
    text = re.sub(r'[!！?"“”#$%＆&\'()（）*＋+,\-．.\:：;；<=>?？@\[\]［］\\^_`{|}~〜〜／/]', ' ', text)
    return text

@st.cache_data
def advanced_tokenize_core(text):
    """形態素解析と複合名詞抽出"""
    text = _core_text_preprocessor(text)
    if not text: return ""

    tokens = list(t.tokenize(text))
    processed_tokens = []
    i = 0
    while i < len(tokens):
        token1 = tokens[i]
        base1 = token1.base_form if token1.base_form != '*' else token1.surface
        if base1 in STOP_WORDS:
            i += 1
            continue
        
        part_of_speech = token1.part_of_speech.split(',')
        pos_major = part_of_speech[0]
        pos_minor = part_of_speech[1] if len(part_of_speech) > 1 else ''
        
        if len(base1) < 2 and pos_major != '名詞':
            i += 1
            continue
        if pos_major == '名詞' and pos_minor == '数':
            i += 1
            continue
            
        # 複合名詞結合
        if (i + 1) < len(tokens):
            token2 = tokens[i+1]
            base2 = token2.base_form if token2.base_form != '*' else token2.surface
            part_of_speech_2 = token2.part_of_speech.split(',')
            pos_major_2 = part_of_speech_2[0]
            pos_minor_2 = part_of_speech_2[1] if len(part_of_speech_2) > 1 else ''
            
            if pos_major == '名詞' and pos_major_2 == '名詞' and \
               base2 not in STOP_WORDS and pos_minor_2 != '数':
                compound_word = base1 + base2
                processed_tokens.append(compound_word)
                i += 2
                continue
        
        if pos_major == '名詞':
            processed_tokens.append(base1)
        elif pos_major == '動詞' and pos_minor == '自立':
            processed_tokens.append(base1)
        elif pos_major == '形容詞' and pos_minor == '自立':
            processed_tokens.append(base1)
        i += 1
    return " ".join(processed_tokens)

# --- CORE 検索エンジン用関数 ---

def build_regex_pattern(keyword):
    return re.escape(keyword)

def build_near_regex(a, b, n):
    a_b = r'{}.{{0,{}}}?{}'.format(a, n, b)
    b_a = r'{}.{{0,{}}}?{}'.format(b, n, a)
    return r'(?:{}|{})'.format(a_b, b_a)

def build_adj_regex(a, b, n):
    return r'{}.{{0,{}}}?{}'.format(a, n, b)

def build_or_regex(a, b):
    return r'(?:{}|{})'.format(a, b)

def split_by_operator(text, operator):
    """括弧の外側にある演算子でのみ分割する"""
    parts = []
    balance = 0
    current_chunk_start = 0
    for i, char in enumerate(text):
        if char == '(':
            balance += 1
        elif char == ')':
            balance -= 1
        elif char == operator and balance == 0:
            parts.append(text[current_chunk_start:i].strip())
            current_chunk_start = i + 1
    parts.append(text[current_chunk_start:].strip())
    return parts

@st.cache_data
def parse_core_rule(rule_str):
    """CORE論理式をパースして正規表現オブジェクトを生成する"""
    tokens = re.findall(r'\(|\)|' r'\bnear\d+\b|' r'\badj\d+\b|' r'[\+]|' r'[^()\s\+]+', rule_str, re.IGNORECASE)
    tokens = [t.strip() for t in tokens if t and t.strip()]
    output_queue, op_stack = [], []
    op_precedence = {}
    
    for op in tokens:
        op_lower = op.lower()
        if op_lower == '+': op_precedence[op] = 1
        elif op_lower.startswith('near'): op_precedence[op] = 3
        elif op_lower.startswith('adj'): op_precedence[op] = 3
        
    for token in tokens:
        token_lower = token.lower()
        if token == '(':
            op_stack.append(token)
        elif token == ')':
            while op_stack and op_stack[-1] != '(':
                output_queue.append(op_stack.pop())
            if not op_stack: raise ValueError(f"文法エラー: 括弧の対応が取れません (「{rule_str}」)")
            op_stack.pop() 
        elif token_lower in op_precedence:
            while (op_stack and op_stack[-1] != '(' and op_precedence.get(op_stack[-1].lower(), 0) >= op_precedence[token_lower]):
                output_queue.append(op_stack.pop())
            op_stack.append(token)
        else:
            output_queue.append(token)
            
    while op_stack:
        op = op_stack.pop()
        if op == '(': raise ValueError(f"文法エラー: 括弧の対応が取れません (「{rule_str}」)")
        output_queue.append(op)
        
    regex_stack = []
    for token in output_queue:
        token_lower = token.lower()
        if token_lower not in op_precedence and token not in '()':
            normalized_token = unicodedata.normalize('NFKC', token).lower()
            if not normalized_token:
                raise ValueError(f"文法エラー: 空のキーワードが含まれています (「{rule_str}」)")
            regex_stack.append(build_regex_pattern(normalized_token))
        else:
            if len(regex_stack) < 2: raise ValueError(f"文法エラー: 演算子 '{token}' が不正です (「{rule_str}」)")
            b, a = regex_stack.pop(), regex_stack.pop()
            if token_lower == '+':
                regex_stack.append(build_or_regex(a, b))
            elif token_lower.startswith('near'):
                n = int(re.findall(r'(\d+)', token_lower)[0])
                regex_stack.append(build_near_regex(a, b, n))
            elif token_lower.startswith('adj'):
                n = int(re.findall(r'(\d+)', token_lower)[0])
                regex_stack.append(build_adj_regex(a, b, n))
                
    if len(regex_stack) != 1: raise ValueError(f"文法エラー: 最終式が不正です (「{rule_str}」)")
    return re.compile(regex_stack[0], re.IGNORECASE | re.DOTALL) 

@st.cache_data
def prepare_axis_data_core(df, axis_col_name, delimiter):
    """ヒートマップ描画用の軸データを前処理する"""
    df_processed = df.copy()
    if axis_col_name not in df_processed.columns:
        st.error(f"エラー: カラム '{axis_col_name}' がデータに存在しません。")
        return pd.DataFrame() 
    
    df_processed[axis_col_name] = df_processed[axis_col_name].fillna('N/A')
    
    if axis_col_name == 'year':
        df_processed[axis_col_name] = df_processed[axis_col_name].apply(
            lambda x: str(int(x)) if pd.notna(x) else 'N/A'
        )
        
    if delimiter:
        df_processed[axis_col_name] = df_processed[axis_col_name].astype(str).str.split(delimiter)
        df_processed = df_processed.explode(axis_col_name)
    
    df_processed[axis_col_name] = df_processed[axis_col_name].astype(str).str.strip()
    df_processed[axis_col_name] = df_processed[axis_col_name].replace('', 'N/A')
    return df_processed

@st.cache_data
def convert_df_to_csv_core(df):
    return df.to_csv(encoding='utf-8-sig').encode('utf-8-sig')


# ==================================================================
# --- 2. アプリケーション初期化 & UI構成 ---
# ==================================================================

st.set_page_config(
    page_title="APOLLO | CORE", 
    page_icon="💡", 
    layout="wide"
)

# 共通CSSの適用
st.markdown("""
<style>
    html, body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
    [data-testid="stSidebar"] h1 { color: #003366; font-weight: 900 !important; font-size: 2.5rem !important; }
    h1 { color: #003366; font-weight: 700; }
    h2, h3 { color: #333333; font-weight: 500; border-bottom: 2px solid #f0f0f0; padding-bottom: 5px; }
    [data-testid="stSidebarNav"] { display: none !important; }
    [data-testid="stSidebar"] .block-container { padding-top: 2rem; padding-bottom: 1rem; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stButton>button { font-weight: 600; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #f0f2f6; border-radius: 8px 8px 0 0; padding: 10px 15px; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-bottom: 2px solid #003366; }
</style>
""", unsafe_allow_html=True)

# --- サイドバー ---
with st.sidebar:
    st.title("APOLLO") 
    st.markdown("Advanced Patent & Overall Landscape-analytics Logic Orbiter")
    st.markdown("---")
    st.subheader("Home")
    st.page_link("Home.py", label="Mission Control", icon="🛰️")
    st.subheader("Modules")
    st.page_link("pages/1_🌍_ATLAS.py", label="ATLAS", icon="🌍")
    st.page_link("pages/2_💡_CORE.py", label="CORE", icon="💡")
    st.page_link("pages/3_🚀_Saturn_V.py", label="Saturn V", icon="🚀")
    st.page_link("pages/4_📈_MEGA.py", label="MEGA", icon="📈")
    st.page_link("pages/5_🧭_Explorer.py", label="Explorer", icon="🧭")
    st.markdown("---")
    st.caption("ナビゲーション:\n1. Mission Control でデータをアップロードし、前処理を実行します。\n2. 上のリストから分析モジュールを選択します。")
    st.markdown("---")
    st.caption("© 2025 しばやま")

# --- メインエリア ---
st.title("💡 CORE")
st.markdown("Contextual Operator & Rule Engine: **論理式ベースの特許分類ツール**です。")

# データロードチェック
if not st.session_state.get("preprocess_done", False):
    st.error("分析データがありません。")
    st.warning("先に「Mission Control」（メインページ）でファイルをアップロードし、「分析エンジン起動」を実行してください。")
    st.stop()
else:
    df_main = st.session_state.df_main
    col_map = st.session_state.col_map

# セッション状態の初期化
if "core_classification_rules" not in st.session_state:
    st.session_state.core_classification_rules = {}
if "core_df_classified" not in st.session_state:
    st.session_state.core_df_classified = None
if "core_current_axis" not in st.session_state:
    st.session_state.core_current_axis = ""
if "core_reanalyze_result" not in st.session_state:
    st.session_state.core_reanalyze_result = ""


# ==================================================================
# --- 3. CORE アプリケーション ---
# ==================================================================

phase_options = [
    "フェーズ 1: AIアシスタント (KMeans)",
    "フェーズ 2: 分類ルール定義",
    "フェーズ 3: 分類実行",
    "フェーズ 4: 特許マップ作成"
]

current_phase = st.radio(
    "フェーズ選択:", 
    phase_options, 
    horizontal=True, 
    key="core_phase_selector"
)

st.markdown("---")

# --- フェーズ 1: AIアシスタント ---
if current_phase == phase_options[0]:
    st.subheader("フェーズ 1: AIによる分類サジェスト (オプション)")
    st.markdown("K-Meansクラスタリングに基づき、分類ルール作成のためのAIプロンプトを生成します。")
    
    col_map_options = [v for k, v in col_map.items() if k in ['title', 'abstract', 'claim']]
    target_column = st.selectbox("分析対象カラム:", options=col_map_options, key="core_target_col")
    
    col1, col2 = st.columns(2)
    with col1:
        ai_k_w = st.number_input("トピック数 (K: クラスタリング用):", min_value=2, value=8, key="core_k")
        ai_n_w = st.number_input("各トピックの代表文献数 (N):", min_value=1, value=5, key="core_n")
    
    use_mece = st.checkbox("MECEモード (AIが最適な分類数を自動決定)", value=True, key="core_use_mece")

    if not use_mece:
        st.markdown("<b>生成する分類の数 (手動設定):</b>", unsafe_allow_html=True)
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            ai_cat_count_tech = st.number_input("技術分類:", min_value=1, value=6, key="core_cat_tech")
        with col_c2:
            ai_cat_count_prob = st.number_input("課題分類:", min_value=1, value=6, key="core_cat_prob")
        with col_c3:
            ai_cat_count_sol = st.number_input("解決手段分類:", min_value=1, value=6, key="core_cat_sol")

    if st.button("AIアシスタント用プロンプトを生成", key="core_run_ai"):
        if not target_column or target_column not in df_main.columns:
            st.error("エラー: 分析対象カラムを正しく選択してください。")
        else:
            try:
                k = int(ai_k_w)
                n = int(ai_n_w)
                
                with st.spinner(f"K-Means (K={k}) とサンプリング (N={n}) を実行中..."):
                    texts_raw = df_main[target_column].astype(str).fillna('')
                    tokenized_texts = texts_raw.apply(advanced_tokenize_core)
                    
                    vectorizer = TfidfVectorizer(min_df=1, max_df=0.9, token_pattern=r"(?u)\b\w+\b")
                    tfidf_matrix = vectorizer.fit_transform(tokenized_texts)
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(tfidf_matrix)
                    centroids = kmeans.cluster_centers_
                    
                    sampled_abstracts = []
                    for cluster_id in range(k):
                        cluster_indices = np.where(clusters == cluster_id)[0]
                        if len(cluster_indices) == 0: continue
                        centroid = centroids[cluster_id]
                        distances = euclidean_distances(tfidf_matrix[cluster_indices], centroid.reshape(1, -1))
                        closest_indices_in_cluster = distances.flatten().argsort()[:n]
                        original_indices = cluster_indices[closest_indices_in_cluster]
                        sampled_abstracts.append(f"\n--- (AIによる推定) クラスタ {cluster_id} の代表文献 ---")
                        for original_index in original_indices:
                            abstract_original = texts_raw.iloc[original_index]
                            abstract_processed = _core_text_preprocessor(abstract_original)
                            sampled_abstracts.append(f"・ {abstract_processed}")
                    
                    if use_mece:
                        instruction_text = """
                        f"この特許母集団全体を網羅的に分類するための、**「技術分類」「課題分類」「解決手段分類」**の3つの分類軸について、**分類定義**（分類名、定義、CORE論理式のセット）を設計してください。",
                        \n# 重要: MECE (Mutually Exclusive, Collectively Exhaustive) の原則
                        - 生成する各分類軸内のカテゴリは、相互に排他的（ダブりがない）であり、かつ全体として網羅的（モレがない）であるように設計してください。
                        - 各軸のカテゴリ数は、MECEを満たすのに最適だとあなたが判断する数（目安として5〜10個程度）にしてください。
                        """
                    else:
                        instruction_text = [
                            f"この特許母集団全体を網羅的に分類するための、以下の3つの分類軸について、指定された個数で**分類定義**（分類名、定義、CORE論理式のセット）を設計してください。",
                            f"- **技術分類**: {ai_cat_count_tech}個",
                            f"- **課題分類**: {ai_cat_count_prob}個",
                            f"- **解決手段分類**: {ai_cat_count_sol}個"
                        ]
                        instruction_text = "\n".join(instruction_text)

                    prompt_parts = [
                        "あなたは優秀な特許情報ストラテジストです。",
                        "\n# 依頼内容",
                        f"以下の「代表文献サンプル」は、ある特許母集団（{len(df_main)}件）をK-Means法で{k}個のクラスタに分類し、各クラスタから代表的な文献の「{target_column}」を{n}件ずつ抽出したものです。",
                        instruction_text,
                        "\n# あなた（AI）の思考プロセス",
                        "1. **熟読:** まず、`# 代表文献サンプル` を**すべて**熟読し、この技術分野の全体像（どのような技術トピックがあり、どのような課題が議論されているか）を把握します。",
                        "2. **分類:** 次に、各文献の文脈から、技術の「目的（課題）」と「手段（解決策）」と「核となる技術要素」を心の中で分類します。",
                        "3. **キーワード選定:** 各分類軸（技術・課題・解決手段）にふさわしいキーワードを選定します。",
                        "4. **キーワード拡張:** 「**最重要ルール**」に基づき、選定したキーワードの**類義語、関連語、上位/下位概念、特許特有の表現、表記ゆれ（カタカナ、ひらがな、漢字）**を、あなたの知識ベースから網羅的に想起します。",
                        "   - **特許用語の網羅:** （例: 「保持」→「担持」「固着」「係止」など、特許で使われる言い換えを網羅）",
                        "   - **概念の階層化:** 上位概念（例: 「車両」）と下位概念（例: 「自動車」「二輪車」）の両方を含め、取りこぼしを防ぎます。",
                        "5. **論理式構築:** これらのキーワード群を「**CORE論理式文法**」を駆使して組み合わせ、**「その他」に分類される特許を極限まで減らせるような、網羅性の高い**論理式を構築します。",
                        f"6. **出力:** 最後に、「### 良い出力例」のフォーマットに厳密に従って、分類軸を生成します。",
                        "\n# CORE論理式文法 (厳守)",
                        "- `A + B` (OR): A または B",
                        "- `A * B` (AND): A かつ B (順序問わず)",
                        "- `A nearN B` (近傍): AとBが**N文字**以内で出現 (順序問わず)。Nは10〜40程度を推奨。",
                        "- `A adjN B` (順序指定近傍): AがBの**N文字**以内にA→Bの順で出現。Nは1〜10程度を推奨。",
                        "- **重要:** キーワードはスペースを含まない単一語（例: `二酸化炭素`）にしてください。スペースを含むフレーズ（例: `AI agent`）は、`AI adj1 agent` のように演算子で表現してください。",
                        "\n# 最重要ルール (キーワード拡張と表記ゆれ)",
                        "- サンプルに存在するキーワードをそのまま使うだけでは不十分です。",
                        "- AIの知識を活用し、そのキーワードの**類義語、関連語、上位/下位概念**を想起してください。",
                        "- **特に、カタカナ（例: `ポリマー`）、ひらがな（例: `ばね`）、漢字（例: `樹脂`）**といった**日本語の**表記ゆれを `+` 演算子で網羅してください。",
                        "- **注意:** 論理式に英語（英単語）は含めず、日本語（漢字、カタカナ、ひらがな）のみを使用してください。",
                        "- **カタカナ:** キーワードにカタカナを使用する場合は、**必ず全角（例: `ポリマー`）**を使用し、**半角（例: `ﾎﾟﾘﾏｰ`）は絶対に使用しないでください**。",
                        "\n### 良い出力例",
                        "```",
                        "## 技術分類",
                        "1.  **CO2分離膜**",
                        "    * **定義:** CO2を分離・回収するための膜（中空糸膜、高分子膜など）に関連する技術。",
                        "    * **論理式:** (CO2 + 二酸化炭素 + 炭酸ガス) * (膜 + 分離膜 + フィルター + 中空糸 + メンブレン)",
                        "2.  **アミン吸収液**",
                        "    * **定義:** アミン化合物（MEA, MDEA等）を用いた化学吸収液によるCO2回収技術。",
                        "    * **論理式:** (アミン + 吸収液 + 吸収剤) + (MEA + MDEA + モノエタノールアミン)",
                        "\n## 課題分類",
                        "1.  **耐久性の向上**",
                        "    * **定義:** 膜や吸収液の劣化を抑制し、長期間安定して使用可能にすること。",
                        "    * **論理式:** (耐久性 + 信頼性 + 劣化 + 寿命 + 安定性 + 耐熱性 + 耐薬品性) * (向上 + 改善 + 抑制 + 高める + 防止 + 維持)",
                        "2.  **コストの削減**",
                        "    * **定義:** 製造コストや運用コストを低減し、経済性を高めること。",
                        "    * **論理式:** (コスト + 製造費用 + 安価 + 低廉 + 経済性 + 省エネルギー + 省電力) * (削減 + 低減 + 安く + 効率化)",
                        "\n## 解決手段分類",
                        "1.  **多孔質担体の利用**",
                        "    * **定義:** ゼオライト、MOF、活性炭などの多孔質な担体に機能性材料を担持させる手法。",
                        "    * **論理式:** (多孔質 + ポーラス + 担体 + 細孔 + ハニカム) + (ゼオライト + MOF + 活性炭 + 金属有機構造体)",
                        "2.  **新規アミンの添加**",
                        "    * **定義:** 既存のアミン吸収液に、性能向上のための新規アミン化合物を添加する手法。",
                        "    * **論理式:** (アミン + 溶剤) adj10 (新規 + 添加 + 混合 + 開発 + 配合)",
                        "```",
                        "\n# 代表文献サンプル",
                        "\n".join(sampled_abstracts)
                    ]
                    final_prompt = "\n".join(prompt_parts)
                    
                    st.success("プロンプトの生成が完了しました。")
                    st.text_area("以下のプロンプトとサンプルデータをコピーし、AIアシスタントに貼り付けてください。", final_prompt, height=400)

            except Exception as e:
                st.error(f"分析中にエラーが発生しました: {e}")
                import traceback
                st.exception(traceback.format_exc())

# --- フェーズ 2: 分類ルール定義 ---
elif current_phase == phase_options[1]:
    st.subheader("フェーズ 2: 分類ルール定義")
    st.markdown("AIアシスタントの出力を参考に、分類ルールを定義します。")
    
    existing_axes = list(st.session_state.core_classification_rules.keys())
    
    # 軸の選択・入力
    if existing_axes:
        input_mode = st.radio("軸の指定方法:", ["新規に軸を作成する", "既存の軸に追加する"], horizontal=True, key="core_axis_mode")
        if input_mode == "既存の軸に追加する":
            default_index = 0
            if st.session_state.core_current_axis in existing_axes:
                default_index = existing_axes.index(st.session_state.core_current_axis)
            axis_name_text = st.selectbox("追加先の軸を選択:", existing_axes, index=default_index, key="core_axis_select")
        else:
            axis_name_text = st.text_input("新しい分類軸の名前:", placeholder="例: 課題、解決手段、技術要素など", key="core_axis_name")
    else:
        axis_name_text = st.text_input("分類軸の名前:", placeholder="例: 課題、解決手段、技術要素など", key="core_axis_name")

    category_name_text = st.text_input("分類名:", placeholder="例: 耐久性、コストダウンなど", key="core_category_name")
    category_def_text = st.text_area("定義:", placeholder="この分類の定義やAIプロンプト用の説明を入力...", key="core_category_def")
    
    st.markdown("""
    <b>論理式文法 (N = 文字数)</b>
    <ul>
        <li><b><code>A + B</code></b> (OR): A または B</li>
        <li><b><code>A * B</code></b> (AND): A かつ B (順序問わず)</li>
        <li><b><code>A nearN B</code></b> (近傍): AとBが<b>N文字</b>以内で出現 (順序問わず)</li>
        <li><b><code>A adjN B</code></b> (順序指定近傍): AがBの<b>N文字</b>以内にA→Bの順で出現</li>
        <li><b><code>( )</code></b> (括弧): 演算の優先順位を指定</li>
    </ul>
    """, unsafe_allow_html=True) 

    keywords_text = st.text_input("論理式:", placeholder="例: (樹脂 + ポリマー) * (高強度 near50 耐久性)", key="core_keywords")
    
    # ボタンアクション
    col1, col2 = st.columns(2)
    with col1:
        if st.button("この分類ルールを追加", key="core_add_rule"):
            axis_name = axis_name_text
            category_name = category_name_text
            category_def = category_def_text
            rule_str = keywords_text
            
            if not all([axis_name, category_name, rule_str]):
                st.warning("「分類軸の名前」「分類名」「論理式」は必須です。")
            else:
                try:
                    or_clauses_str = split_by_operator(rule_str, '+')
                    compiled_or_clauses = []
                    for or_part_str in or_clauses_str:
                        and_clauses_str = split_by_operator(or_part_str, '*')
                        compiled_and_clauses = []
                        for and_part_str in and_clauses_str:
                            sub_rule = and_part_str.strip()
                            if not sub_rule: raise ValueError("空のルールがあります")
                            compiled_and_clauses.append(parse_core_rule(sub_rule))
                        compiled_or_clauses.append(compiled_and_clauses)
                    
                    if axis_name not in st.session_state.core_classification_rules:
                        st.session_state.core_classification_rules[axis_name] = {}
                    
                    st.session_state.core_classification_rules[axis_name][category_name] = {
                        'rule': rule_str,
                        'compiled': compiled_or_clauses,
                        'definition': category_def
                    }
                    
                    st.session_state.core_current_axis = axis_name
                    st.success(f"ルールを追加しました: {category_name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"文法エラー: {e}")

    with col2:
        if st.button("この分類軸の定義を完了 (次の軸へ)", key="core_finish_axis"):
            axis_name = st.session_state.core_current_axis
            if not axis_name or axis_name not in st.session_state.core_classification_rules:
                st.warning(f"軸 '{axis_name}' にルールがありません。")
            else:
                st.success(f"軸 '{axis_name}' の定義を完了。")
                st.session_state.core_current_axis = ""
                st.rerun()

    st.markdown("---")
    st.subheader("ルールの管理・修正")
    
    if st.button("全ルールをリセット", type="secondary", key="core_reset_all"):
        st.session_state.core_classification_rules = {}
        st.session_state.core_current_axis = ""
        st.success("リセットしました。")
        st.rerun()

    if st.session_state.core_classification_rules:
        rule_items = []
        for axis, categories in st.session_state.core_classification_rules.items():
            for cat, data in categories.items():
                if isinstance(data, tuple): # 旧データ形式対応
                    rule_str = data[0]
                    def_str = ""
                else:
                    rule_str = data['rule']
                    def_str = data.get('definition', "")
                
                rule_items.append({
                    "label": f"[{axis}] {cat}: {rule_str[:20]}...",
                    "axis": axis, "cat": cat, "rule": rule_str, "def": def_str
                })
        
        if rule_items:
            selected_rule_label = st.selectbox("編集/削除するルール:", [r["label"] for r in rule_items], key="core_select_rule_edit")
            target_rule = next((r for r in rule_items if r["label"] == selected_rule_label), None)
            
            col_edit, col_del = st.columns(2)
            
            def load_rule_to_edit(axis, cat, rule, definition):
                if existing_axes:
                    st.session_state.core_axis_mode = "既存の軸に追加する"
                    st.session_state.core_axis_select = axis
                else:
                    st.session_state.core_axis_name = axis
                st.session_state.core_current_axis = axis
                st.session_state.core_category_name = cat
                st.session_state.core_keywords = rule
                st.session_state.core_category_def = definition

            with col_edit:
                if target_rule:
                    st.button("編集 (入力欄にセット)", key="core_btn_edit", on_click=load_rule_to_edit, args=(target_rule["axis"], target_rule["cat"], target_rule["rule"], target_rule["def"]))
            with col_del:
                if st.button("削除", key="core_btn_delete"):
                    if target_rule:
                        del st.session_state.core_classification_rules[target_rule["axis"]][target_rule["cat"]]
                        if not st.session_state.core_classification_rules[target_rule["axis"]]:
                            del st.session_state.core_classification_rules[target_rule["axis"]]
                        st.success("削除しました。")
                        st.rerun()

    st.markdown("---")
    st.subheader("定義済みルールログ")
    if st.session_state.core_classification_rules:
        for axis, rules in st.session_state.core_classification_rules.items():
            st.markdown(f"**軸: {axis}**")
            for category, data in rules.items():
                if isinstance(data, tuple): r_str = data[0]
                else: r_str = data['rule']
                st.code(f"  - {category}: {r_str}", language="text")

# --- フェーズ 3: 分類実行 ---
elif current_phase == phase_options[2]:
    st.subheader("フェーズ 3: 分類実行")
    
    # ターゲットカラムの取得とフォールバック
    target_column = st.session_state.get("core_target_col")
    if not target_column and col_map.get('abstract'):
        target_column = col_map['abstract']
    
    if st.button("すべての分類を実行", type="primary", key="core_run_classification"):
        if not st.session_state.core_classification_rules:
            st.error("エラー: ルールが定義されていません。")
        elif not target_column or target_column not in df_main.columns:
            st.error("エラー: 分析対象カラムを選択してください。")
        else:
            with st.spinner("分類処理を実行中..."):
                try:
                    df_classified = df_main.copy()
                    status_area = st.empty()
                    progress_bar = st.progress(0, "分類処理中...")
                    rules = st.session_state.core_classification_rules
                    total_axes = len(rules)
                    
                    for i, (axis_name, ruleset) in enumerate(rules.items()):
                        status_area.write(f"軸 '{axis_name}' の分類処理中... ({i+1}/{total_axes})")
                        df_classified[axis_name] = ''
                        target_texts = df_classified[target_column].astype(str).fillna("")
                        
                        def apply_rules_for_axis(search_text):
                            search_text_processed = _core_text_preprocessor(search_text)
                            found_categories = []
                            for category, data in ruleset.items():
                                if isinstance(data, tuple): compiled_or = data[1]
                                else: compiled_or = data['compiled']
                                
                                is_or_match = False
                                for compiled_and in compiled_or:
                                    is_and_match = True
                                    for sub_regex in compiled_and:
                                        if not sub_regex.search(search_text_processed): 
                                            is_and_match = False; break 
                                    if is_and_match: is_or_match = True; break
                                if is_or_match: found_categories.append(category)
                            return ";".join(found_categories) if found_categories else 'その他'
                                
                        df_classified[axis_name] = target_texts.apply(apply_rules_for_axis)
                        progress_bar.progress((i + 1) / total_axes)

                    st.session_state.core_df_classified = df_classified.copy()
                    st.session_state.core_reanalyze_result = ""
                    status_area.empty(); progress_bar.empty()
                    st.success("完了しました。")
                    
                    # 結果表示
                    st.subheader("分類結果サマリー")
                    for axis_name in rules.keys():
                        st.markdown(f"**軸: {axis_name}**")
                        for cat in rules[axis_name].keys():
                            count = df_classified[axis_name].str.contains(re.escape(cat), na=False).sum()
                            st.write(f"- {cat}: {count}件")
                        st.write(f"- その他: {(df_classified[axis_name] == 'その他').sum()}件")

                    csv_core = convert_df_to_csv_core(df_classified)
                    st.download_button("分類結果CSVをダウンロード", csv_core, "CORE_classified.csv", "text/csv")
                    
                except Exception as e:
                    st.error(f"エラー: {e}")
                    import traceback
                    st.exception(traceback.format_exc())

    # --- 再分析セクション ---
    st.markdown("---")
    st.subheader("🔍 未分類データの再分析 (『その他』を減らす)")
    
    if st.session_state.core_df_classified is not None:
        rules = st.session_state.core_classification_rules
        if rules:
            col_re1, col_re2 = st.columns(2)
            with col_re1:
                reanalyze_axis = st.selectbox("再分析する軸を選択:", list(rules.keys()), key="core_reanalyze_axis")
            
            with st.expander("詳細設定 (AIパラメータ)", expanded=False):
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    re_k = st.number_input("トピック数 (K)", min_value=1, value=5, key="core_re_k")
                with col_p2:
                    re_n = st.number_input("サンプル数 (N)", min_value=1, value=3, key="core_re_n")
                
                re_use_mece = st.checkbox("MECEモード (自動決定)", value=True, key="core_re_mece")
                
                re_cat_count = 3 
                if not re_use_mece:
                    re_cat_count = st.number_input("追加する分類数", min_value=1, value=3, key="core_re_count")

            def run_reanalysis():
                try:
                    df_classified = st.session_state.core_df_classified
                    target_col = st.session_state.get("core_target_col", col_map.get('abstract'))
                    axis = st.session_state.core_reanalyze_axis
                    k_in = st.session_state.core_re_k
                    n_in = st.session_state.core_re_n
                    use_mece_in = st.session_state.core_re_mece
                    cat_count_in = st.session_state.core_re_count if not use_mece_in else None
                    
                    others_df = df_classified[df_classified[axis] == 'その他']
                    
                    if others_df.empty:
                        st.session_state.core_reanalyze_result = "『その他』のデータはありません。"
                        return

                    with st.spinner(f"『その他』({len(others_df)}件) を分析中..."):
                        texts_others = others_df[target_col].astype(str).fillna('')
                        tokenized_others = texts_others.apply(advanced_tokenize_core)
                        vec_others = TfidfVectorizer(min_df=1, max_df=0.9, token_pattern=r"(?u)\b\w+\b")
                        tfidf_others = vec_others.fit_transform(tokenized_others)
                        
                        actual_k = min(int(k_in), len(others_df))
                        if actual_k < 2: actual_k = 1
                        
                        kmeans_others = KMeans(n_clusters=actual_k, random_state=42, n_init=10)
                        clusters_others = kmeans_others.fit_predict(tfidf_others)
                        centroids_others = kmeans_others.cluster_centers_
                        
                        sampled_others_text = []
                        for cid in range(actual_k):
                            c_indices = np.where(clusters_others == cid)[0]
                            if len(c_indices) == 0: continue
                            centroid = centroids_others[cid]
                            dists = euclidean_distances(tfidf_others[c_indices], centroid.reshape(1, -1))
                            actual_n = int(n_in)
                            closest_idx = dists.flatten().argsort()[:actual_n]
                            orig_indices = c_indices[closest_idx]
                            sampled_others_text.append(f"\n--- その他グループ {cid} の代表文献 ---")
                            for o_idx in orig_indices:
                                raw_txt = texts_others.iloc[o_idx]
                                sampled_others_text.append(f"・ {_core_text_preprocessor(raw_txt)}")
                        
                        existing_rules_str = []
                        for cat, data in rules[axis].items():
                            if isinstance(data, tuple): r_s, d_s = data[0], ""
                            else: r_s, d_s = data['rule'], data.get('definition', "")
                            existing_rules_str.append(f"- {cat}: {d_s} (論理式: {r_s})")
                        
                        if use_mece_in:
                            instruction_part = "MECE（漏れなくダブりなく）を意識し、既存の分類の隙間を埋めるような定義にしてください。カテゴリ数はあなたが最適と考える数にしてください。"
                        else:
                            instruction_part = f"既存の分類の隙間を埋めるような定義で、**{cat_count_in}個** の新しい分類カテゴリを追加してください。"

                        context_intro = f"以下の「未分類（その他）特許のサンプル」は、『その他』グループをK-Means法で{actual_k}個のクラスタに分類し、各クラスタから代表的な文献の「{target_col}」を{actual_n}件ずつ抽出したものです。"

                        reanalyze_prompt = [
                            "あなたは優秀な特許情報ストラテジストです。",
                            f"現在、分類軸「{axis}」を作成中ですが、以下の「既存の分類」に当てはまらない特許が「その他」として残っています。",
                            "\n# 既存の分類リスト（これらとは重複しないようにしてください）",
                            "\n".join(existing_rules_str),
                            "\n# 依頼内容",
                            context_intro,
                            "これを分析し、**既存の分類とは概念的に重複しない、新しい分類カテゴリ（分類名、定義、論理式）**を提案してください。",
                            instruction_part,
                            "\n# あなた（AI）の思考プロセス",
                            "1. **熟読:** 「未分類特許のサンプル」を読み、既存分類で拾いきれなかった技術概念を特定します。",
                            "2. **定義:** 既存の分類と被らないように、新しい概念の定義を明確にします。",
                            "3. **キーワード拡張:** 「**最重要ルール**」に基づき、類義語、関連語、上位/下位概念、特許特有の表現を網羅します。",
                            "4. **論理式構築:** 「**CORE論理式文法**」を駆使して、ノイズを避けつつ網羅性の高い論理式を構築します。",
                            "\n# CORE論理式文法 (厳守)",
                            "- `A + B` (OR): A または B",
                            "- `A * B` (AND): A かつ B (順序問わず)",
                            "- `A nearN B` (近傍): AとBが**N文字**以内で出現 (順序問わず)。Nは10〜40程度を推奨。",
                            "- `A adjN B` (順序指定近傍): AがBの**N文字**以内にA→Bの順で出現。Nは1〜10程度を推奨。",
                            "- **重要:** キーワードはスペースを含まない単一語（例: `二酸化炭素`）にしてください。",
                            "\n# 最重要ルール (キーワード拡張と表記ゆれ)",
                            "- AIの知識を活用し、そのキーワードの**類義語、関連語、上位/下位概念**を想起してください。",
                            "- **特許用語の網羅:** （例: 「保持」→「担持」「固着」「係止」など）",
                            "- **概念の階層化:** 上位概念（例: 「車両」）と下位概念（例: 「自動車」「二輪車」）の両方を含めること。",
                            "- **カタカナ:** キーワードにカタカナを使用する場合は、**必ず全角**を使用してください。",
                            "\n# 未分類（その他）特許のサンプル",
                            "\n".join(sampled_others_text),
                            "\n### 出力フォーマット",
                            "```",
                            f"## {axis} (追加提案)",
                            "1. **[分類名]**",
                            "   * **定義:** [定義]",
                            "   * **論理式:** (キーワードA + キーワードB) * (キーワードC + キーワードD)",
                            "```"
                        ]
                        st.session_state.core_reanalyze_result = "\n".join(reanalyze_prompt)

                except Exception as e:
                    st.session_state.core_reanalyze_result = f"エラーが発生しました: {e}"

            with col_re2:
                st.button("『その他』を分析して新ルールを提案", key="core_btn_reanalyze", on_click=run_reanalysis)
        
        if st.session_state.core_reanalyze_result:
            if "エラー" in st.session_state.core_reanalyze_result:
                st.error(st.session_state.core_reanalyze_result)
            else:
                st.success("再分析プロンプトを生成しました。")
                st.text_area("以下のプロンプトをAIに貼り付けてください。", st.session_state.core_reanalyze_result, height=400)


# --- フェーズ 4: 特許マップ作成 ---
elif current_phase == phase_options[3]:
    st.subheader("フェーズ 4: 特許マップ作成 (ヒートマップ)")
    
    if st.session_state.core_df_classified is None:
        st.info("先に「フェーズ 3: 分類実行」タブで分類を実行してください。")
    else:
        df_graph = st.session_state.core_df_classified
        
        st.subheader("マップ設定")
        
        core_axes = list(st.session_state.core_classification_rules.keys())
        app_py_axes = []
        
        if 'year' in df_graph.columns:
            app_py_axes.append("出願年")
        
        if col_map.get('applicant') and col_map['applicant'] in df_graph.columns:
            app_py_axes.append("出願人")
        
        all_axis_options = core_axes + app_py_axes
        
        if len(all_axis_options) < 2:
            st.error("エラー: グラフ化できる軸（分類軸、出願年、出願人のうち2つ以上）がありません。")
            st.stop()
            
        col1, col2 = st.columns(2)
        with col1:
            x_axis_name = st.selectbox("X軸:", all_axis_options, key="core_x_axis", index=min(0, len(all_axis_options)-1))
            x_top_n = st.number_input("X軸 表示件数 (Top N):", min_value=1, value=20, key="core_x_top_n", help="「出願年」を軸にした場合は、この設定は無視されます。")
            x_exclude_other_w = st.checkbox("X軸から「その他」を除外", value=False, key="core_x_exclude_other")
            
        with col2:
            y_axis_name = st.selectbox("Y軸:", all_axis_options, key="core_y_axis", index=min(1, len(all_axis_options)-1))
            y_top_n = st.number_input("Y軸 表示件数 (Top N):", min_value=1, value=20, key="core_y_top_n", help="「出願年」を軸にした場合は、この設定は無視されます。")
            y_exclude_other_w = st.checkbox("Y軸から「その他」を除外", value=False, key="core_y_exclude_other")
        
        delimiter_w = st.text_input("区切り文字 (分類軸・出願人用):", value=';', key="core_delimiter")

        x_is_year = (st.session_state.core_x_axis == "出願年")
        y_is_year = (st.session_state.core_y_axis == "出願年")
        
        if x_is_year or y_is_year:
            st.markdown("---")
            st.subheader("期間フィルタ設定")
            st.info("X軸またはY軸に「出願年」が選択されたため、以下の期間でデータを絞り込みます。")
            
            def callback_autoset_core_year():
                if 'year' in df_graph.columns and df_graph['year'].notna().any():
                    valid_years = df_graph['year'].dropna().astype(int)
                    st.session_state.core_start_year = int(valid_years.min())
                    st.session_state.core_end_year = int(valid_years.max())
                else:
                    st.session_state.core_start_year = 2010
                    st.session_state.core_end_year = 2024
            
            if 'core_start_year' not in st.session_state:
                callback_autoset_core_year()

            d_col1, d_col2, d_col3 = st.columns([1, 1, 2])
            with d_col1:
                st.number_input("開始年:", key="core_start_year", step=1, format="%d")
            with d_col2:
                st.number_input("終了年:", key="core_end_year", step=1, format="%d")
            with d_col3:
                st.button("（全期間を自動設定）", on_click=callback_autoset_core_year, key="core_autoset_year")
        
        st.markdown("---")
        st.subheader("マップの実行と表示")
        
        if st.button("5. 特許マップを作成", type="primary", key="core_run_graph"):
            x_axis_key = st.session_state.core_x_axis
            y_axis_key = st.session_state.core_y_axis
            
            if x_axis_key == y_axis_key:
                st.error("エラー: ヒートマップではX軸とY軸に異なるカラムを選択してください。")
            else:
                with st.spinner("ヒートマップを作成中..."):
                    try:
                        delimiter = delimiter_w.strip()
                        
                        df_filtered = df_graph.copy()
                        if x_is_year or y_is_year:
                            start_year_val = int(st.session_state.core_start_year)
                            end_year_val = int(st.session_state.core_end_year)
                            df_filtered = df_filtered[
                                (df_filtered['year'].notna()) &
                                (df_filtered['year'] >= start_year_val) & 
                                (df_filtered['year'] <= end_year_val)
                            ]

                        if x_axis_key == "出願年":
                            x_col_name = 'year'
                            x_delimiter = None
                        elif x_axis_key == "出願人":
                            x_col_name = col_map['applicant']
                            x_delimiter = delimiter
                        else:
                            x_col_name = x_axis_key 
                            x_delimiter = delimiter

                        if y_axis_key == "出願年":
                            y_col_name = 'year'
                            y_delimiter = None
                        elif y_axis_key == "出願人":
                            y_col_name = col_map['applicant']
                            y_delimiter = delimiter
                        else:
                            y_col_name = y_axis_key
                            y_delimiter = delimiter

                        df_plot_x = prepare_axis_data_core(df_filtered, x_col_name, x_delimiter)
                        if df_plot_x.empty: st.stop()
                        
                        df_plot_xy = prepare_axis_data_core(df_plot_x, y_col_name, y_delimiter)
                        if df_plot_xy.empty: st.stop()
                        
                        if st.session_state.core_x_exclude_other:
                            df_plot_xy = df_plot_xy[df_plot_xy[x_col_name] != 'その他']
                        
                        if st.session_state.core_y_exclude_other:
                            df_plot_xy = df_plot_xy[df_plot_xy[y_col_name] != 'その他']
                        
                        x_top_n_val = int(st.session_state.core_x_top_n)
                        y_top_n_val = int(st.session_state.core_y_top_n)

                        if x_axis_key != "出願年":
                            x_top_labels = df_plot_xy[
                                (df_plot_xy[x_col_name] != 'N/A') & 
                                (df_plot_xy[x_col_name] != 'その他')
                            ][x_col_name].value_counts().head(x_top_n_val).index.tolist()
                            
                            x_allowed_labels = x_top_labels + ['N/A']
                            if not st.session_state.core_x_exclude_other:
                                x_allowed_labels.append('その他')
                                
                            df_plot_xy = df_plot_xy[df_plot_xy[x_col_name].isin(x_allowed_labels)]
                        
                        if y_axis_key != "出願年":
                            y_top_labels = df_plot_xy[
                                (df_plot_xy[y_col_name] != 'N/A') & 
                                (df_plot_xy[y_col_name] != 'その他')
                            ][y_col_name].value_counts().head(y_top_n_val).index.tolist()
                            
                            y_allowed_labels = y_top_labels + ['N/A']
                            if not st.session_state.core_y_exclude_other:
                                y_allowed_labels.append('その他')

                            df_plot_xy = df_plot_xy[df_plot_xy[y_col_name].isin(y_allowed_labels)]

                        df_plot_final = df_plot_xy[
                            (df_plot_xy[x_col_name] != 'N/A') & 
                            (df_plot_xy[y_col_name] != 'N/A')
                        ]
                        
                        if df_plot_final.empty:
                            st.warning("該当するデータがありません。（Top N フィルタや N/A 除外の結果かもしれません）")
                        else:
                            matrix = pd.crosstab(df_plot_final[y_col_name], df_plot_final[x_col_name])
                            
                            # 軸を文字列型に統一し、ソート
                            matrix.index = matrix.index.astype(str)
                            matrix.columns = matrix.columns.astype(str)

                            if x_axis_key == "出願年":
                                x_category_order = sorted(matrix.columns, key=lambda x: int(x) if x.isdigit() else x)
                            else:
                                x_category_order = matrix.sum(axis=0).sort_values(ascending=False).index.tolist()
                            
                            if y_axis_key == "出願年":
                                y_category_order = sorted(matrix.index, key=lambda x: int(x) if x.isdigit() else x)
                            else:
                                y_category_order = matrix.sum(axis=1).sort_values(ascending=False).index.tolist()
                            
                            # 0埋めしてNaNを回避
                            matrix = matrix.reindex(index=y_category_order, columns=x_category_order).fillna(0)
                            z = matrix.values
                            z_max = z.max() if z.size > 0 else 1
                            
                            x_labels = matrix.columns.tolist()
                            y_labels = matrix.index.tolist()
                            x_indices = np.arange(len(x_labels))
                            y_indices = np.arange(len(y_labels))

                            annotations = []
                            for i, row_idx in enumerate(y_indices):
                                for j, col_idx in enumerate(x_indices):
                                    val = z[i][j]
                                    text_val = str(int(val))
                                    color = "white" if val > z_max * 0.6 else "black"
                                    annotations.append(dict(
                                        x=col_idx, y=row_idx, text=text_val,
                                        xref="x", yref="y", showarrow=False,
                                        font=dict(color=color, size=14)
                                    ))
                            
                            h = max(600, len(matrix.index) * 100 + 200)
                            
                            fig = go.Figure(data=go.Heatmap(
                                z=z, x=x_indices, y=y_indices,
                                colorscale='YlGnBu', showscale=True
                            ))
                            fig.update_layout(
                                title=f"'{y_axis_key}' × '{x_axis_key}' ヒートマップ",
                                height=h, annotations=annotations,
                                xaxis=dict(
                                    title=x_axis_key, 
                                    tickmode='array', 
                                    tickvals=x_indices, 
                                    ticktext=x_labels, 
                                    side='bottom', 
                                    tickangle=-90
                                ),
                                yaxis=dict(
                                    title=y_axis_key, 
                                    tickmode='array', 
                                    tickvals=y_indices, 
                                    ticktext=y_labels, 
                                    autorange='reversed'
                                ),
                                margin=dict(l=150, b=150)
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"グラフ作成中にエラーが発生しました: {e}")
                        import traceback
                        st.exception(traceback.format_exc())

# --- 共通サイドバーフッター ---
st.sidebar.markdown("---") 
st.sidebar.caption("ナビゲーション:")
st.sidebar.caption("1. Mission Control でデータをアップロードし、前処理を実行します。")
st.sidebar.caption("2. 左のリストから分析モジュールを選択します。")
st.sidebar.markdown("---")
st.sidebar.caption("© 2025 しばやま")