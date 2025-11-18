%%writefile pages/2_💡_CORE.py
# ==================================================================
# --- 1. ライブラリのインポート ---
# ==================================================================
import streamlit as st
import pandas as pd
import numpy as np
import io
import datetime
import warnings
import unicodedata
import re

# Janome / Sklearn
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

# Plotly
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "plotly_white"

# 警告を非表示
warnings.filterwarnings('ignore')

# ==================================================================
# --- 2. CORE専用ヘルパー関数 ---
# ==================================================================

# COREは独自のTokenizerとStopWordsを持つ
@st.cache_resource
def load_tokenizer_core():
    print("... CORE: Janome Tokenizerをロード中 ...")
    return Tokenizer()

t = load_tokenizer_core()

# CORE専用のストップワード
stop_words = {
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
    """KMeans(フェーズ1)と分類実行(フェーズ3)で共通のテキスト前処理"""
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFKC', text).lower()
    text = re.sub(r'[（(][^）)]{1,80}[）)]', ' ', text) # 括弧内を除去
    text = re.sub(r'(?:図|Fig|FIG|fig)[. 　]*\d+', ' ', text) # 図番を除去
    text = re.sub(r'[!！?"“”#$%＆&\'()（）*＋+,\-．.\:：;；<=>?？@\[\]［］\\^_`{|}~〜〜／/]', ' ', text) # 記号を除去
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
        if base1 in stop_words:
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
        if (i + 1) < len(tokens):
            token2 = tokens[i+1]
            base2 = token2.base_form if token2.base_form != '*' else token2.surface
            part_of_speech_2 = token2.part_of_speech.split(',')
            pos_major_2 = part_of_speech_2[0]
            pos_minor_2 = part_of_speech_2[1] if len(part_of_speech_2) > 1 else ''
            if pos_major == '名詞' and pos_major_2 == '名詞' and \
               base2 not in stop_words and pos_minor_2 != '数':
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

# CORE 検索エンジン
def build_regex_pattern(keyword):
    return re.escape(keyword)
def build_near_regex(a, b, n):
    a_b = r'{}.{{0,{}}}?{}'.format(a, n, b); b_a = r'{}.{{0,{}}}?{}'.format(b, n, a); return r'(?:{}|{})'.format(a_b, b_a)
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
    # このパーサーは `+`, `near`, `adj`, `()` のみを処理
    # `*` は上位の `split_by_operator` で処理される
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
        if token == '(': op_stack.append(token)
        elif token == ')':
            while op_stack and op_stack[-1] != '(': output_queue.append(op_stack.pop())
            if not op_stack: raise ValueError(f"文法エラー: 括弧の対応が取れません (「{rule_str}」)")
            op_stack.pop() 
        elif token_lower in op_precedence:
            while (op_stack and op_stack[-1] != '(' and op_precedence.get(op_stack[-1].lower(), 0) >= op_precedence[token_lower]):
                output_queue.append(op_stack.pop())
            op_stack.append(token)
        else: output_queue.append(token)
    while op_stack:
        op = op_stack.pop();
        if op == '(': raise ValueError(f"文法エラー: 括弧の対応が取れません (「{rule_str}」)");
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
            if token_lower == '+': regex_stack.append(build_or_regex(a, b))
            elif token_lower.startswith('near'):
                n = int(re.findall(r'(\d+)', token_lower)[0]); regex_stack.append(build_near_regex(a, b, n))
            elif token_lower.startswith('adj'):
                n = int(re.findall(r'(\d+)', token_lower)[0]); regex_stack.append(build_adj_regex(a, b, n))
    if len(regex_stack) != 1: raise ValueError(f"文法エラー: 最終式が不正です (「{rule_str}」)")
    return re.compile(regex_stack[0], re.IGNORECASE | re.DOTALL) 

@st.cache_data
def prepare_axis_data_core(df, axis_col_name, delimiter):
    """ヒートマップ専用のデータ準備関数"""
    df_processed = df.copy()
    if axis_col_name not in df_processed.columns:
        st.error(f"エラー: カラム '{axis_col_name}' がデータに存在しません。")
        return pd.DataFrame() # 空のDFを返す
    
    df_processed[axis_col_name] = df_processed[axis_col_name].fillna('N/A')
    
    # 'year' カラムの場合 (floatをint文字列に)
    if axis_col_name == 'year':
        df_processed[axis_col_name] = df_processed[axis_col_name].apply(
            lambda x: str(int(x)) if pd.notna(x) else 'N/A'
        )
    
    # '出願人' または '分類軸' の場合 (デリミタで分割)
    if delimiter:
        df_processed[axis_col_name] = df_processed[axis_col_name].astype(str).str.split(delimiter)
        df_processed = df_processed.explode(axis_col_name)
    
    # 共通のクリーニング
    df_processed[axis_col_name] = df_processed[axis_col_name].astype(str).str.strip()
    df_processed[axis_col_name] = df_processed[axis_col_name].replace('', 'N/A')
    return df_processed


# ==================================================================
# --- 3. Streamlit UI ---
# ==================================================================
st.set_page_config(
    page_title="APOLLO | CORE", 
    page_icon="💡", 
    layout="wide"
)

st.title("💡 CORE")
st.markdown("Contextual Operator & Rule Engine: **論理式ベースの特許分類ツール**です。")

# ==================================================================
# --- 4. セッション状態の確認と初期化 ---
# ==================================================================
if not st.session_state.get("preprocess_done", False):
    st.error("分析データがありません。")
    st.warning("先に「Mission Control」（メインページ）でファイルをアップロードし、「分析エンジン起動」を実行してください。")
    st.stop()
else:
    df_main = st.session_state.df_main
    col_map = st.session_state.col_map

if "core_classification_rules" not in st.session_state:
    st.session_state.core_classification_rules = {}
if "core_df_classified" not in st.session_state:
    st.session_state.core_df_classified = None
if "core_current_axis" not in st.session_state:
    st.session_state.core_current_axis = ""

# ==================================================================
# --- 5. CORE アプリケーション ---
# ==================================================================

tab_ai, tab_rule, tab_run, tab_graph = st.tabs([
    "フェーズ 1: AIアシスタント (KMeans)",
    "フェーズ 2: 分類ルール定義",
    "フェーズ 3: 分類実行",
    "フェーズ 4: 特許マップ作成"
])

# --- フェーズ 1: AIアシスタント ---
with tab_ai:
    st.subheader("フェーズ 1: AIによる分類サジェスト (オプション)")
    st.markdown("K-Meansクラスタリングに基づき、分類ルール作成のためのAIプロンプトを生成します。")
    
    col_map_options = [v for k, v in col_map.items() if k in ['title', 'abstract', 'claim']]
    target_column = st.selectbox(
        "分析対象カラム:",
        options=col_map_options,
        key="core_target_col"
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        ai_k_w = st.number_input("トピック数 (K):", min_value=2, value=8, key="core_k")
    with col2:
        ai_n_w = st.number_input("各トピックの代表文献数 (N):", min_value=1, value=5, key="core_n")
    with col3:
        ai_cat_count_w = st.number_input("AIが生成する分類名の数:", min_value=1, value=6, key="core_cat_count")
        
    if st.button("AIアシスタント用プロンプトを生成", key="core_run_ai"):
        if not target_column or target_column not in df_main.columns:
            st.error("エラー: 分析対象カラムを正しく選択してください。")
        else:
            try:
                k = int(ai_k_w)
                n = int(ai_n_w)
                cat_count = int(ai_cat_count_w)
                
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
                    
                    prompt_parts = [
                        "あなたは優秀な特許情報ストラテジストです。",
                        "\n# 依頼内容",
                        f"以下の「代表文献サンプル」は、ある特許母集団（{len(df_main)}件）をK-Means法で{k}個のクラスタに分類し、各クラスタから代表的な文献の「{target_column}」を{n}件ずつ抽出したものです。",
                        f"この特許母集団全体を網羅的に分類するための、**「技術分類」「課題分類」「解決手段分類」**の3つの分類軸について、**分類定義**（分類名、定義、CORE論理式のセット）をそれぞれ**{cat_count}個**ずつ設計してください。",
                        "\n# あなた（AI）の思考プロセス",
                        "1. **熟読:** まず、`# 代表文献サンプル` を**すべて**熟読し、この技術分野の全体像（どのような技術トピックがあり、どのような課題が議論されているか）を把握します。",
                        "2. **分類:** 次に、各文献の文脈から、技術の「目的（課題）」と「手段（解決策）」と「核となる技術要素」を心の中で分類します。",
                        "3. **キーワード選定:** 各分類軸（技術・課題・解決手段）にふさわしいキーワードを選定します。",
                        "4. **キーワード拡張:** 「**最重要ルール**」に基づき、選定したキーワードの**類義語、関連語、表記ゆれ（カタカナ、ひらがな、漢字）**を、あなたの知識ベースから網羅的に想起します。",
                        "5. **論理式構築:** これらのキーワード群を「**CORE論理式文法**」を駆使して組み合わせ、**ノイズに強く、かつ網羅的（モレが少ない）**な論理式を構築します。",
                        f"6. **出力:** 最後に、「### 良い出力例」のフォーマットに厳密に従って、3つの分類軸を（それぞれ{cat_count}個ずつ）生成します。",
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
                        "    * **論理式:** (CO2 + 二酸化炭素 + 炭酸ガス) * (膜 + 分離膜 + フィルター + 中空糸)",
                        "2.  **アミン吸収液**",
                        "    * **定義:** アミン化合物（MEA, MDEA等）を用いた化学吸収液によるCO2回収技術。",
                        "    * **論理式:** (アミン + 吸収液) + (MEA + MDEA + モノエタノールアミン)",
                        "\n## 課題分類",
                        "1.  **耐久性の向上**",
                        "    * **定義:** 膜や吸収液の劣化を抑制し、長期間安定して使用可能にすること。",
                        "    * **論理式:** (耐久性 +信頼性 + 劣化 + 寿命 + 安定性) * (向上 + 改善 + 抑制 + 高める)",
                        "2.  **コストの削減**",
                        "    * **定義:** 製造コストや運用コストを低減し、経済性を高めること。",
                        "    * **論理式:** (コスト + 製造費用 + 安価 + 低廉 + 経済性) * (削減 + 低減 + 安く)",
                        "\n## 解決手段分類",
                        "1.  **多孔質担体の利用**",
                        "    * **定義:** ゼオライト、MOF、活性炭などの多孔質な担体に機能性材料を担持させる手法。",
                        "    * **論理式:** (多孔質 + ポーラス + 担体 + 細孔) + (ゼオライト + MOF + 活性炭)",
                        "2.  **新規アミンの添加**",
                        "    * **定義:** 既存のアミン吸収液に、性能向上のための新規アミン化合物を添加する手法。",
                        "    * **論理式:** (アミン + 溶剤) adj10 (新規 + 添加 + 混合 + 開発)",
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
with tab_rule:
    st.subheader("フェーズ 2: 分類ルール定義")
    st.markdown("AIアシスタントの出力を参考に、分類ルールを定義します。")
    
    axis_name_text = st.text_input("分類軸の名前:", placeholder="例: 課題、解決手段、技術要素など", key="core_axis_name")
    category_name_text = st.text_input("分類名:", placeholder="例: 耐久性、コストダウンなど", key="core_category_name")

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
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("この分類ルールを追加", key="core_add_rule"):
            axis_name = axis_name_text
            category_name = category_name_text
            rule_str = keywords_text
            
            if not all([axis_name, category_name, rule_str]):
                st.warning("「分類軸の名前」「分類名」「論理式」のすべてを入力してください。")
            else:
                try:
                    or_clauses_str = split_by_operator(rule_str, '+')
                    compiled_or_clauses = []
                    
                    for or_part_str in or_clauses_str:
                        and_clauses_str = split_by_operator(or_part_str, '*')
                        compiled_and_clauses = []
                        
                        for and_part_str in and_clauses_str:
                            sub_rule = and_part_str.strip()
                            if not sub_rule:
                                raise ValueError("'*' または '+' 演算子の間に空のルールがあります。")
                            compiled_and_clauses.append(parse_core_rule(sub_rule))
                        
                        compiled_or_clauses.append(compiled_and_clauses)
                    
                    if axis_name not in st.session_state.core_classification_rules:
                        st.session_state.core_classification_rules[axis_name] = {}
                    
                    st.session_state.core_classification_rules[axis_name][category_name] = (rule_str, compiled_or_clauses)
                    st.success(f"[軸: {axis_name}] カテゴリ '{category_name}' に論理式 '{rule_str}' を登録しました。")
                    st.session_state.core_current_axis = axis_name
                except Exception as e:
                    st.error(f"文法エラー: {e}")

    with col2:
        if st.button("この分類軸の定義を完了 (次の軸へ)", key="core_finish_axis"):
            axis_name = st.session_state.core_current_axis
            if not axis_name or axis_name not in st.session_state.core_classification_rules:
                st.warning(f"軸 '{axis_name}' にルールが1つも登録されていません。")
            else:
                st.success(f"分類軸 '{axis_name}' の定義を完了しました。")
                st.session_state.core_current_axis = "" # クリア

    st.markdown("---")
    st.subheader("定義済みルールログ")
    if not st.session_state.core_classification_rules:
        st.info("まだルールが定義されていません。")
    else:
        for axis, rules in st.session_state.core_classification_rules.items():
            st.markdown(f"**軸: {axis}**")
            for category, (rule_str, _) in rules.items():
                st.code(f"  - {category}: {rule_str}", language="text")

# --- フェーズ 3: 分類実行 ---
with tab_run:
    st.subheader("フェーズ 3: 分類実行")
    st.markdown("定義したすべての分類ルールを実行し、特許リストに分類を付与します。")
    
    if st.button("すべての分類を実行", type="primary", key="core_run_classification"):
        if not st.session_state.core_classification_rules:
            st.error("エラー: 「フェーズ 2」で分類ルールを1つ以上定義してください。")
        elif not target_column or target_column not in df_main.columns:
            st.error("エラー: 「フェーズ 1」で分析対象カラムを正しく選択してください。")
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
                            
                            for category, (rule_str, compiled_or_clauses) in ruleset.items():
                                
                                is_or_match = False
                                for compiled_and_clauses in compiled_or_clauses:
                                    
                                    is_and_match = True
                                    for sub_regex in compiled_and_clauses:
                                        if not sub_regex.search(search_text_processed): 
                                            is_and_match = False
                                            break 
                                    
                                    if is_and_match:
                                        is_or_match = True
                                        break
                                
                                if is_or_match:
                                    found_categories.append(category)

                            if found_categories:
                                return ";".join(found_categories)
                            else:
                                return 'その他'
                                
                        df_classified[axis_name] = target_texts.apply(apply_rules_for_axis)
                        progress_bar.progress((i + 1) / total_axes)

                    st.session_state.core_df_classified = df_classified.copy()
                    
                    status_area.empty()
                    progress_bar.empty()
                    st.success("すべての分類付与が完了しました。")
                    
                    st.subheader("分類結果サマリー")
                    total_docs = len(df_classified)
                    st.write(f"総公報数: {total_docs}件")
                    summary_text = []
                    for axis_name in rules.keys():
                        summary_text.append(f"\n--- 軸: [{axis_name}] ---")
                        for category_name in rules[axis_name].keys():
                            count = df_classified[axis_name].str.contains(re.escape(category_name), na=False, regex=True).sum()
                            summary_text.append(f"  {category_name}: {count}件")
                        other_count = (df_classified[axis_name] == 'その他').sum()
                        summary_text.append(f"  その他: {other_count}件")
                    st.code("\n".join(summary_text), language="text")
                    
                    st.subheader("処理結果のプレビュー")
                    preview_cols = list(rules.keys()) + [target_column]
                    st.dataframe(df_classified[preview_cols].head())
                    
                    @st.cache_data
                    def convert_df_to_csv_core(df):
                        return df.to_csv(encoding='utf-8-sig').encode('utf-8-sig')
                    
                    csv_core = convert_df_to_csv_core(df_classified)
                    st.download_button(
                        label="分類結果 (CORE_classified_output.csv) をダウンロード",
                        data=csv_core,
                        file_name="CORE_classified_output.csv",
                        mime="text/csv",
                    )
                    
                except Exception as e:
                    st.error(f"分類実行中にエラー: {e}")
                    import traceback
                    st.exception(traceback.format_exc())

# --- フェーズ 4: 特許マップ作成 ---
with tab_graph:
    st.subheader("フェーズ 4: 特許マップ作成 (ヒートマップ)")
    
    st.markdown("「フェーズ 3」で分類付与されたデータを対象に、2軸のヒートマップを作成します。")
    st.markdown("---")
    
    if st.session_state.core_df_classified is None:
        st.info("先に「フェーズ 3: 分類実行」タブで分類を実行してください。")
    else:
        df_graph = st.session_state.core_df_classified
        
        st.subheader("マップ設定")
        
        # 1. 軸の選択肢を準備
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
            
        # 2. UIウィジェットを定義
        col1, col2 = st.columns(2)
        with col1:
            x_axis_name = st.selectbox(
                "X軸:", 
                all_axis_options, 
                key="core_x_axis",
                index = min(0, len(all_axis_options)-1) 
            )
            x_top_n = st.number_input(
                "X軸 表示件数 (Top N):", 
                min_value=1, 
                value=20, 
                key="core_x_top_n",
                help="「出願年」を軸にした場合は、この設定は無視されます。"
            )
            x_exclude_other_w = st.checkbox("X軸から「その他」を除外", value=False, key="core_x_exclude_other")
            
        with col2:
            y_axis_name = st.selectbox(
                "Y軸:", 
                all_axis_options, 
                key="core_y_axis",
                index= min(1, len(all_axis_options)-1) 
            )
            y_top_n = st.number_input(
                "Y軸 表示件数 (Top N):", 
                min_value=1, 
                value=20, 
                key="core_y_top_n",
                help="「出願年」を軸にした場合は、この設定は無視されます。"
            )
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
                st.number_input("開始年:", key="core_start_year", step=1)
            with d_col2:
                st.number_input("終了年:", key="core_end_year", step=1)
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
                            
                            if x_axis_key == "出願年":
                                x_category_order = sorted(matrix.columns.astype(int))
                            else:
                                x_category_order = matrix.sum(axis=0).sort_values(ascending=False).index.tolist()
                            
                            if y_axis_key == "出願年":
                                y_category_order = sorted(matrix.index.astype(int))
                            else:
                                y_category_order = matrix.sum(axis=1).sort_values(ascending=False).index.tolist()

                            cell_size_px = 35 
                            x_label_padding = 150 
                            y_label_padding = 200 
                            
                            fig_height = max(400, len(matrix.index) * cell_size_px + x_label_padding)
                            fig_width = max(600, len(matrix.columns) * cell_size_px + y_label_padding)

                            fig = px.imshow(matrix, 
                                            text_auto=True, 
                                            title=f"'{y_axis_key}' × '{x_axis_key}' ヒートマップ",
                                            aspect=None,
                                            color_continuous_scale='YlGnBu',
                                            height=fig_height, 
                                            width=fig_width   
                                           )
                            fig.update_layout(
                                xaxis_title=x_axis_key,
                                yaxis_title=y_axis_key,
                                xaxis_tickangle=-90,
                                xaxis={'categoryarray': x_category_order},
                                yaxis={'categoryarray': y_category_order, 'autorange': 'reversed'}
                            )
                            
                            st.plotly_chart(fig, use_container_width=False)

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