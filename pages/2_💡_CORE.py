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
import utils
import utils_ai
import patiroha

# ==================================================================
# --- 1. ページ設定 ---
# ==================================================================
st.set_page_config(
    page_title="APOLLO v8 | CORE", 
    page_icon="💡", 
    layout="wide"
)

st.session_state['current_page'] = 'CORE'

pio.templates.default = "plotly_white"
warnings.filterwarnings('ignore')

# ==================================================================
# --- 2. デザインテーマ設定 & 共通CSS ---
# ==================================================================




# ==================================================================
# --- 3. ヘルパー関数 & リソースロード ---
# ==================================================================
@st.cache_resource
def load_tokenizer_core(): return Tokenizer()
t = load_tokenizer_core()

if "stopwords" in st.session_state and st.session_state["stopwords"]:
    STOP_WORDS = st.session_state["stopwords"]
else:
    STOP_WORDS = patiroha.get_stopwords()

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
    # 防御層: 超長文で Janome の IndexError を避ける
    if len(text) > 8000:
        text = text[:8000]
    try:
        tokens = list(t.tokenize(text))
    except Exception:
        return ""
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

# --- CORE 検索エンジン (Recursive Descent Parser) ---
class LogicNode:
    def evaluate(self, text): raise NotImplementedError

class AndNode(LogicNode):
    def __init__(self, children): self.children = children
    def evaluate(self, text): return all(c.evaluate(text) for c in self.children)

class OrNode(LogicNode):
    def __init__(self, children): self.children = children
    def evaluate(self, text): return any(c.evaluate(text) for c in self.children)

class RegexNode(LogicNode):
    def __init__(self, pattern): 
        try: self.pattern = re.compile(pattern, re.IGNORECASE | re.DOTALL)
        except: self.pattern = None
    def evaluate(self, text): return bool(self.pattern.search(text)) if self.pattern else False

class CoreLogicParser:
    def __init__(self):
        self.tokens = []
        self.pos = 0

    def tokenize(self, rule_str):
        # Tokenize: (, ), +, *, nearN, adjN, or literals
        raw_tokens = re.findall(r'\(|\)|' r'\bnear\d+\b|' r'\badj\d+\b|' r'[\+\*]' r'|' r'[^\(\)\+\*\s]+', rule_str, re.IGNORECASE)
        self.tokens = [t.strip() for t in raw_tokens if t.strip()]
        self.pos = 0

    def parse(self, rule_str):
        self.tokenize(rule_str)
        if not self.tokens: return None
        node = self.expression()
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token at end: {self.tokens[self.pos]}")
        return node

    def expression(self):
        # Expression -> Term { + Term }  (OR)
        nodes = [self.term()]
        while self.pos < len(self.tokens) and self.tokens[self.pos] == '+':
            self.pos += 1
            nodes.append(self.term())
        return OrNode(nodes) if len(nodes) > 1 else nodes[0]

    def term(self):
        # Term -> Factor { * Factor } (AND)
        nodes = [self.factor()]
        while self.pos < len(self.tokens) and self.tokens[self.pos] == '*':
            self.pos += 1
            nodes.append(self.factor())
        return AndNode(nodes) if len(nodes) > 1 else nodes[0]

    def factor(self):
        # Factor -> Atom { (nearN|adjN) Atom }

        # To support (A+B) near C, we need to compile sub-parts to regex strings if possible.
        # Limitation: near/adj can only apply to "Regex-compatible" nodes (Leaf or OR of Leafs). NO ANDs allowed inside near/adj.
        
        left = self.atom()
        
        while self.pos < len(self.tokens) and re.match(r'^(near|adj)\d+$', self.tokens[self.pos], re.IGNORECASE):
            op = self.tokens[self.pos].lower()
            self.pos += 1
            right = self.atom()
            
            # recursive constraint check: Left and Right must be convertible to Regex String
            l_rex = self.to_regex_string(left)
            r_rex = self.to_regex_string(right)
            n = int(re.findall(r'\d+', op)[0])
            
            if op.startswith('near'): pattern = r'(?:{}.{{0,{}}}?{}|{}.{{0,{}}}?{})'.format(l_rex, n, r_rex, r_rex, n, l_rex)
            else: pattern = r'{}.{{0,{}}}?{}'.format(l_rex, n, r_rex) # adj
            
            left = RegexNode(pattern)
            
        return left

    def atom(self):
        if self.pos < len(self.tokens) and self.tokens[self.pos] == '(':
            self.pos += 1
            node = self.expression()
            if self.pos < len(self.tokens) and self.tokens[self.pos] == ')':
                self.pos += 1
                return node
            else:
                raise ValueError("Missing closing parenthesis")
        elif self.pos < len(self.tokens):
            t = self.tokens[self.pos]
            self.pos += 1
            # Literal
            norm = unicodedata.normalize('NFKC', t).lower()
            return RegexNode(re.escape(norm))
        else:
            raise ValueError("Unexpected end of rule")

    def to_regex_string(self, node):
        # Helper to convert a Node back to regex string if it contains only OR/Literal
        if isinstance(node, RegexNode): 
            # Pattern inside RegexNode is already compiled or string? 
            # In our class, it's compiled. We need the source string. 
            # Implementation trick: Store source pattern in RegexNode
            if hasattr(node, 'pattern') and node.pattern: return node.pattern.pattern
            # If it was constructed blindly? 
            # Let's modifying RegexNode to store source.
            return "" 
        if isinstance(node, OrNode):
            parts = [self.to_regex_string(c) for c in node.children]
            return r'(?:' + '|'.join(parts) + r')'
        if isinstance(node, AndNode):
             raise ValueError("Cannot use AND (*) inside a NEAR/ADJ condition. Use OR (+) only.")
        return ""

# Patch RegexNode to store source for recursion
class RegexNode(LogicNode):
    def __init__(self, pattern): 
        self.source = pattern
        try: self.pattern = re.compile(pattern, re.IGNORECASE | re.DOTALL)
        except: self.pattern = None
    def evaluate(self, text): return bool(self.pattern.search(text)) if self.pattern else False

@st.cache_resource
def parse_core_rule(rule_str):
    try:
        parser = CoreLogicParser()
        return parser.parse(rule_str)
    except Exception:
        # cache 中はエラー表示を抑制。呼び出し側で None を判定する。
        return None

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
utils.render_sidebar()

st.title("💡 CORE")
st.markdown("AND/OR/NEAR/ADJ論理式で特許を分類し、分類軸間のクロス集計で技術ポートフォリオの構造を明らかにします。")

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
- `()`: 括弧を使って優先順位を制御できます。入れ子も可能です。
    - 例: `(水素 * (吸蔵 + 貯蔵) * (合金 + 材料))`
    - 例: `(A * B) + (C * D)`
- `A nearN B` (近傍): AとBがN文字以内 (順序不問)。
- `A adjN B` (順序指定): A→BがN文字以内。
- **重要:** `near` や `adj` の条件の内部には `*` (AND) を含めることはできません（`+` (OR) は可能）。
    - OK: `(A + B) near10 C`
    - NG: `(A * B) near10 C`

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
                    compiled_rule_nodes = {}
                    for ax, cats in rules.items():
                        compiled_rule_nodes[ax] = []
                        for cn, cd in cats.items():
                            r_str = cd['rule'] if isinstance(cd, dict) else cd[0]
                            # Try parse
                            node = parse_core_rule(r_str)
                            if node:
                                compiled_rule_nodes[ax].append((cn, node))
                            else:
                                st.warning(f"Failed to parse rule for {cn}: {r_str}")

                    def apply_rules(text, ax_nodes):
                        text = _core_text_preprocessor(str(text))
                        hits = []
                        for c_name, node in ax_nodes:
                            if node.evaluate(text):
                                hits.append(c_name)
                        return ";".join(hits) if hits else "その他"

                    bar = st.progress(0)
                    for i, ax in enumerate(rules.keys()):
                        df_res[ax] = combined_text.apply(lambda x: apply_rules(x, compiled_rule_nodes[ax]))
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

                    # CAPCOM: patents.csv更新（CORE分類列を追加）
                    try:
                        import capcom
                        if capcom.is_active():
                            capcom.save_patents_csv()
                    except Exception:
                        pass

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
      "rule": "論理式 (Allowed: `(A * B) + C`, `(A+B) near10 C` etc)"
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
        
        if st.button("描画 (分析実行)", type="primary"):
            st.session_state.core_phase4_run = True

        if st.session_state.get("core_phase4_run"):
            st.markdown("---")
            
            # --- 1. グローバルな軸の順序を決定 (母集団全体で固定) ---
            def get_col_data_global(target_df, ax_name):
                if ax_name == '出願年': return target_df['year'].fillna(0).astype(int).astype(str), None
                if ax_name == '出願人': return target_df[col_map['applicant']].fillna('Unknown'), ';' 
                if ax_name in axes: return target_df[ax_name], ';'
                return None, None

            # 全体データでクロス集計して軸順序を決定
            x_data_g, x_sep_g = get_col_data_global(df_c, x_ax)
            y_data_g, y_sep_g = get_col_data_global(df_c, y_ax)
            temp_df_g = pd.DataFrame({'X': x_data_g, 'Y': y_data_g})
            
            if x_sep_g: temp_df_g['X'] = temp_df_g['X'].astype(str).str.split(x_sep_g); temp_df_g = temp_df_g.explode('X')
            if y_sep_g: temp_df_g['Y'] = temp_df_g['Y'].astype(str).str.split(y_sep_g); temp_df_g = temp_df_g.explode('Y')
            
            temp_df_g = temp_df_g.replace({'nan': np.nan, 'None': np.nan}).dropna()
            if exclude_other:
                temp_df_g = temp_df_g[(temp_df_g['X'] != 'その他') & (temp_df_g['Y'] != 'その他')]
            
            # --- 軸が出願人の場合のフィルタリング (Top N / 手動) ---
            if '出願人' in [x_ax, y_ax]:
                # 全データの出願人頻度を計算
                app_s_all = df_c[col_map['applicant']].fillna('Unknown').astype(str).str.split(';')
                app_counts_all = app_s_all.explode().str.strip().value_counts()
                
                st.markdown("##### 👥 出願人軸の表示設定")
                app_filter_mode = st.radio("表示モード:", ["上位指定 (Top N)", "手動選択 (Manual)"], horizontal=True, key="core_app_axis_mode")
                
                target_apps_set = set()
                if app_filter_mode == "上位指定 (Top N)":
                    top_n_val = st.number_input("表示件数 (上位N社):", min_value=5, max_value=200, value=10, step=5, key="core_app_axis_n")
                    target_apps_set = set(app_counts_all.head(top_n_val).index)
                    st.info(f"上位 {top_n_val} 社を表示します（全 {len(app_counts_all)} 社中）")
                else:
                    target_apps_set = set(st.multiselect("表示する出願人を選択:", app_counts_all.index.tolist(), default=app_counts_all.head(10).index.tolist(), key="core_app_axis_manual"))
                
                # フィルタリング適用
                if x_ax == '出願人':
                    temp_df_g = temp_df_g[temp_df_g['X'].isin(target_apps_set)]
                if y_ax == '出願人':
                    temp_df_g = temp_df_g[temp_df_g['Y'].isin(target_apps_set)]
            
            if temp_df_g.empty:
                st.warning("有効なデータがありません。")
            else:
                ct_g = pd.crosstab(temp_df_g['Y'], temp_df_g['X'])
                
                # Sorting Global
                if x_ax == '出願年': x_ord_global = sorted(ct_g.columns, key=lambda x: int(x) if x.isdigit() else x)
                else: x_ord_global = ct_g.sum(axis=0).sort_values(ascending=False).index.tolist()
                
                if y_ax == '出願年': y_ord_global = sorted(ct_g.index, key=lambda x: int(x) if x.isdigit() else x)
                else: y_ord_global = ct_g.sum(axis=1).sort_values(ascending=False).index.tolist()

                # --- Sorting Adjustment: Force 'Others' to the end ---
                if 'その他' in x_ord_global:
                    x_ord_global.remove('その他')
                    x_ord_global.append('その他')
                
                if 'その他' in y_ord_global:
                    y_ord_global.remove('その他')
                    y_ord_global.append('その他')

                # --- 2. 出願人選択 (プルダウン) ---
                target_applicant_options = ["全体 (Overall)"]
                app_name_map = {} # label -> app_name
                
                if col_map.get('applicant') in df_c.columns:
                    app_s = df_c[col_map['applicant']].fillna('Unknown').astype(str).str.split(';')
                    app_exploded = app_s.explode().str.strip()
                    top_apps = app_exploded.value_counts() # All applicants
                    
                    for app_name, count in top_apps.items():
                        if app_name and app_name != 'nan':
                            label = f"{app_name} ({count})"
                            target_applicant_options.append(label)
                            app_name_map[label] = app_name
                
                selected_app_label = st.selectbox("出願人で絞り込み (Focus Applicant):", target_applicant_options)

                # --- 3. データフィルタリング ---
                if selected_app_label == "全体 (Overall)":
                    df_target = df_c
                else:
                    target_app_name = app_name_map[selected_app_label]
                    mask = df_c[col_map['applicant']].fillna('').astype(str).apply(lambda x: target_app_name in [s.strip() for s in x.split(';')])
                    df_target = df_c[mask]
                
                st.markdown(f"**分析対象: {selected_app_label}**")

                # --- 4. 描画関数 (Global Axisを適用) ---
                def render_core_chart(sub_df, wrapper_key):
                    # Local Data Prep
                    x_d, x_s = get_col_data_global(sub_df, x_ax)
                    y_d, y_s = get_col_data_global(sub_df, y_ax)
                    t_df = pd.DataFrame({'X': x_d, 'Y': y_d})
                    if x_s: t_df['X'] = t_df['X'].astype(str).str.split(x_s); t_df = t_df.explode('X')
                    if y_s: t_df['Y'] = t_df['Y'].astype(str).str.split(y_s); t_df = t_df.explode('Y')
                    
                    t_df = t_df.replace({'nan': np.nan, 'None': np.nan}).dropna()
                    if exclude_other:
                        t_df = t_df[(t_df['X'] != 'その他') & (t_df['Y'] != 'その他')]
                    
                    # Create Crosstab
                    ct_local = pd.crosstab(t_df['Y'], t_df['X'])
                    
                    # Reindex with Global Orders (Forces matrix structure)
                    ct_final = ct_local.reindex(index=y_ord_global, columns=x_ord_global).fillna(0)
                    
                    if chart_type == "ヒートマップ":
                        fig = px.imshow(
                            ct_final,
                            labels=dict(x=x_ax, y=y_ax, color="件数"),
                            x=ct_final.columns,
                            y=ct_final.index,
                            aspect="auto",
                            color_continuous_scale='YlGnBu',
                            text_auto=True
                        )
                        # マス間に薄い白線を入れて可読性を高める
                        fig.update_traces(xgap=2, ygap=2)
                        fig.update_layout(
                            height=max(600, len(ct_final)*40),
                            yaxis=dict(title=y_ax),
                            xaxis=dict(title=x_ax, side='bottom'),
                            plot_bgcolor='white'
                        )
                        st.plotly_chart(fig, use_container_width=True, config={'editable': False}, key=f"core_chart_{wrapper_key}")
                        
                    else: # バブルチャート
                        ct_long = ct_final.reset_index().melt(id_vars='Y', var_name='X', value_name='Count')
                        ct_long = ct_long[ct_long['Count'] > 0] 
                        
                        atlas_colors = utils.APOLLO_COLORS
                        
                        fig = px.scatter(
                            ct_long, x='X', y='Y', size='Count', color='Y',
                            size_max=60, color_discrete_sequence=atlas_colors,
                            category_orders={'X': x_ord_global, 'Y': y_ord_global} 
                        )
                        
                        # Explicitly FORCE Range to show all categories, even empty ones
                        x_range = [-0.5, len(x_ord_global) - 0.5]
                        # Y軸: Plotlyの散布図は通常下から上だが、行列形式（上から下）にするために調整
                        # 反転させるために範囲を明示的に指定
                        
                        fig.update_yaxes(
                            categoryorder='array', 
                            categoryarray=y_ord_global, 
                            title=y_ax, 
                            type='category',
                            range=[len(y_ord_global) - 0.5, -0.5] # 上から下の行列形式
                        )
                        fig.update_xaxes(
                            categoryorder='array', 
                            categoryarray=x_ord_global, 
                            title=x_ax, 
                            side='bottom', 
                            type='category',
                            range=[-0.5, len(x_ord_global) - 0.5] # Ensure full width
                        )
                        st.plotly_chart(fig, use_container_width=True, config={'editable': False}, key=f"core_chart_{wrapper_key}")

                    
                    # --- Snapshot (Dynamic) ---
                    # Generate Matrix CSV for AI
                    # Optimize Matrix Size for Large Datasets (Top 20x20)
                    if ct_final.shape[0] > 20 or ct_final.shape[1] > 20:
                         # Calculate Row/Col sums to find Top items
                         top_rows = ct_final.sum(axis=1).sort_values(ascending=False).head(20).index
                         top_cols = ct_final.sum(axis=0).sort_values(ascending=False).head(20).index
                         # Slicing
                         ct_ai_safe = ct_final.loc[top_rows, top_cols]
                         matrix_csv = ct_ai_safe.to_csv()
                         note = f"(Truncated to Top 20x20 from {ct_final.shape})"
                    else:
                         matrix_csv = ct_final.to_csv()
                         note = ""

                    # Construct Data Summary for VOYAGER
                    # Construct Data Summary for VOYAGER
                    snap_data = {
                        'module': 'CORE',
                        'type': 'matrix',
                        'axes': {'x': x_ax, 'y': y_ax},
                        'matrix_data': matrix_csv,
                        'matrix_context': f"Analysis Target: {selected_app_label}",
                        'stats': {
                            'total_count': int(ct_final.sum().sum()),
                            'max_value': int(ct_final.max().max())
                        },
                        'chart_data': f"Matrix ({y_ax} vs {x_ax}) - Target: {selected_app_label}\n{matrix_csv}" 
                    }

                    utils.render_snapshot_button(
                        title=f"CORE Map ({selected_app_label}): {x_ax} vs {y_ax}",
                        description=f"COREマトリクス分析: {x_ax} (X) × {y_ax} (Y)。対象: {selected_app_label}。",
                        key=f"core_snap_{x_ax}_{y_ax}_{selected_app_label}",
                        fig=fig,
                        data_summary=snap_data
                    )

                    # --- AI Insight: 分類結果の戦略的解釈 ---
                    _meta_core = utils_ai.build_common_metadata(df_main=df_main, df_filtered=df_c, col_map=col_map)
                    _meta_core['分析対象'] = selected_app_label
                    _meta_core['X軸'] = x_ax
                    _meta_core['Y軸'] = y_ax
                    _meta_core['グラフタイプ'] = chart_type
                    _meta_core['マトリクスサイズ'] = f"{ct_final.shape[0]}行 × {ct_final.shape[1]}列"
                    _meta_core['合計件数'] = int(ct_final.sum().sum())
                    # 分類ルール一覧
                    _rules_info = {}
                    for _ax_name, _ax_rules in st.session_state.core_classification_rules.items():
                        _rules_info[_ax_name] = {cat: d.get('rule', '') for cat, d in _ax_rules.items()}
                    _meta_core['分類ルール'] = _rules_info
                    # 各カテゴリのヒット件数
                    for _ax_name in [x_ax, y_ax]:
                        if _ax_name in st.session_state.core_classification_rules:
                            _cat_counts = ct_final.sum(axis=1 if _ax_name == y_ax else 0).to_dict()
                            _meta_core[f'{_ax_name}別件数'] = {str(k): int(v) for k, v in _cat_counts.items()}

                    _core_prompt = utils_ai.generate_ai_insight_prompt(
                        role="特許分類・技術戦略の専門家として、ルールベース分類によるクロス集計結果を戦略的に分析してください。",
                        context=f"""\
{chart_type}によるクロス集計マトリクスを表示しています。
- X軸: {x_ax}
- Y軸: {y_ax}
- 対象: {selected_app_label}
各セルの値は該当する特許の件数を示します。""",
                        data_summary=matrix_csv,
                        instructions="""\
以下の観点で分析してください:
1. **注力領域**: 件数の集中パターンから、注力している技術-課題の組み合わせを特定
2. **空白領域**: 件数がゼロまたは少ないセルの意味（未開拓 or 不要な組み合わせ）
3. **バランス**: 分類間の件数のばらつきと「その他」比率の評価
4. **競合比較示唆**: 出願人別分析の場合、各社の技術ポートフォリオ特性の違い
5. **戦略提言**: データから導かれる研究開発・知財戦略の方向性

各主張には必ず具体的な数値を1つ以上含めてください。""",
                        metadata=_meta_core,
                        constraints="分類ルールの定義（論理式）を考慮し、カテゴリの意味を正確に理解した上で分析すること。",
                        output_format="Markdown形式。見出し付きの構造化された分析レポート。"
                    )
                    utils_ai.render_ai_insight_button(_core_prompt, f"core_matrix_insight_{x_ax}_{y_ax}_{selected_app_label}")


                render_core_chart(df_target, "main_display")

                # CAPCOM data/ JSON出力（CORE分類結果）
                try:
                    import capcom
                    if capcom.is_active():
                        # ルール情報の構造化
                        rules_json = {}
                        for ax_name, ax_cats in st.session_state.core_classification_rules.items():
                            rules_json[ax_name] = {}
                            for cat_name, cat_info in ax_cats.items():
                                rules_json[ax_name][cat_name] = {
                                    "rule": cat_info.get('rule', ''),
                                    "definition": cat_info.get('definition', ''),
                                    "count": int((df_c[ax_name] == cat_name).sum()) if ax_name in df_c.columns else 0
                                }
                            # 「その他」カテゴリの件数
                            if ax_name in df_c.columns:
                                other_count = int((df_c[ax_name] == 'その他').sum())
                                if other_count > 0:
                                    rules_json[ax_name]['その他'] = {"rule": "(未分類)", "definition": "いずれのルールにもマッチしなかった特許", "count": other_count}

                        core_json = {
                            "metadata": {
                                "module": "CORE",
                                "total_patents": len(df_c),
                                "axes": list(rules_json.keys())
                            },
                            "rules": rules_json
                        }
                        capcom.save_data("core_classification.json", core_json)
                except Exception as e:
                    pass

        # CSVダウンロード
        st.markdown("---")
        csv_core = convert_df_to_csv_core(df_c)
        st.download_button("分類結果付き全データCSVをダウンロード", csv_core, "CORE_classified_full.csv", "text/csv")