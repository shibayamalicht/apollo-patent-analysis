# ==================================================================
# openalex_query.py -- APOLLO v10.0.0 コマンドライン検索式エンジン
# 特許DB風のコマンドライン検索構文（TI=/AB=/TA=/TX=/FT= + AND/OR/NOT +
# nearN/adjN + ワイルドカード）を解釈する自己完結モジュール。
#   - 2フェーズ方式: ①AST → OpenAlex 候補クエリ構築、②near/adj は取得後にローカル厳密照合
#   - Streamlit / requests には依存しない（純 Python + re）。単体テスト可能。
# ==================================================================

import re
import unicodedata

# TI/AB/TA はローカルで厳密照合できる（タイトル・要旨が手元にある）
STRICT_FIELDS = {"TI", "AB", "TA"}
# TX/FT は OpenAlex 索引による候補取得専用（手元に全文が無いため内部照合しない）
CANDIDATE_ONLY_FIELDS = {"TX", "FT"}

# 候補取得 scope → OpenAlex 検索フィールド
SCOPE_TO_FILTER_KEY = {
    "title": "title.search",
    "abstract": "abstract.search",
    "title_and_abstract": "title_and_abstract.search",
    "fulltext": "fulltext.search",
    # "all" は filter ではなく top-level の search= を使う
}
_SCOPE_DEFAULT_FIELD = {
    "title": "TI", "abstract": "AB", "title_and_abstract": "TA",
    "all": "TX", "fulltext": "FT",
}


class QueryError(ValueError):
    """検索式の構文エラー"""
    pass


# ------------------------------------------------------------------
# 正規化
# ------------------------------------------------------------------
def normalize_input(text: str) -> str:
    """検索式の入力正規化（NFKC・全角記号→半角・スマートクォート統一）"""
    s = unicodedata.normalize("NFKC", str(text or ""))
    s = s.replace("“", '"').replace("”", '"')   # “ ”
    s = s.replace("‘", "'").replace("’", "'")   # ‘ ’
    s = s.replace("＝", "=").replace("（", "(").replace("）", ")")
    return s.strip()


def search_normalize(text: str) -> str:
    """照合用テキスト正規化（NFKC・小文字化・ハイフン/中黒/スラッシュ→空白・アポストロフィ除去）"""
    s = unicodedata.normalize("NFKC", str(text or "")).lower()
    s = re.sub(r"[‐‑‒–—−ーｰ・·/\\]+", " ", s)
    s = re.sub(r"[’'`´]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def search_tokens(text: str, keep_wildcards: bool = False):
    """正規化テキストから語トークンを抽出（keep_wildcards で * ? を温存）"""
    norm = search_normalize(text)
    if keep_wildcards:
        return re.findall(r"(?:[^\W_]|[*?])+", norm)
    return re.findall(r"[^\W_]+", norm)


def tokenize_document(text: str) -> dict:
    """タイトル/要旨を語トークン列（位置付き）に変換する"""
    norm = search_normalize(text)
    tokens = [{"text": m.group(0), "start": m.start(), "end": m.end()}
              for m in re.finditer(r"[^\W_]+", norm)]
    return {"text": norm, "tokens": tokens}


def pattern_from_token(token: str):
    """ワイルドカード付きトークンをアンカー付き正規表現に変換（* → .*, ? → .）"""
    esc = re.sub(r"[.+^${}()|\[\]\\]", lambda m: "\\" + m.group(0), token)
    esc = esc.replace("*", ".*").replace("?", ".")
    return re.compile(f"^{esc}$")


# ------------------------------------------------------------------
# トークナイザ（検索式 → トークン列）
# ------------------------------------------------------------------
def tokenize(query: str):
    q = normalize_input(query)
    tokens = []
    i, n = 0, len(q)
    while i < n:
        ch = q[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "(":
            tokens.append({"type": "LPAREN", "value": ch}); i += 1; continue
        if ch == ")":
            tokens.append({"type": "RPAREN", "value": ch}); i += 1; continue
        if ch == "=":
            tokens.append({"type": "EQ", "value": ch}); i += 1; continue
        if ch == '"':
            j = i + 1
            val = ""
            while j < n:
                if q[j] == '"':
                    break
                if q[j] == "\\" and j + 1 < n:
                    val += q[j + 1]; j += 2; continue
                val += q[j]; j += 1
            if j >= n or q[j] != '"':
                raise QueryError("引用符が閉じていません。")
            tokens.append({"type": "PHRASE", "value": val})
            i = j + 1
            continue
        j = i
        val = ""
        while j < n and not q[j].isspace() and q[j] not in "()=":
            val += q[j]; j += 1
        if val:
            tokens.append({"type": "WORD", "value": val})
        i = j
    return tokens


_NEAR_RE = re.compile(r"^(near|adj)\d+$", re.IGNORECASE)
_NEAR_PARSE_RE = re.compile(r"^(near|adj)(\d+)$", re.IGNORECASE)
_FIELDS = {"TI", "AB", "TA", "TX", "FT"}


# ------------------------------------------------------------------
# パーサ（トークン列 → AST）。AST ノードは dict。
#   {type: AND/OR, left, right} / {type: NOT, child} / {type: FIELD, field, child}
#   {type: PROX, op, n, left, right} / {type: TERM, value} / {type: PHRASE, value}
# ------------------------------------------------------------------
def parse(query: str) -> dict:
    tokens = tokenize(query)
    pos = {"i": 0}

    def peek(k=0):
        idx = pos["i"] + k
        return tokens[idx] if 0 <= idx < len(tokens) else None

    def consume():
        t = tokens[pos["i"]]
        pos["i"] += 1
        return t

    def is_word(t, word):
        return bool(t and t["type"] == "WORD" and t["value"].upper() == word)

    def is_near(t):
        return bool(t and t["type"] == "WORD" and _NEAR_RE.match(t["value"]))

    def is_field(t):
        return bool(t and t["type"] == "WORD" and t["value"].upper() in _FIELDS)

    def is_primary_start(t):
        if not t:
            return False
        if t["type"] in ("LPAREN", "PHRASE"):
            return True
        if t["type"] != "WORD":
            return False
        if t["value"].upper() in ("AND", "OR"):
            return False
        if is_near(t):
            return False
        return True

    def parse_expression():
        return parse_or()

    def parse_or():
        node = parse_and()
        while is_word(peek(), "OR"):
            consume()
            node = {"type": "OR", "left": node, "right": parse_and()}
        return node

    def parse_and():
        node = parse_not()
        while True:
            if is_word(peek(), "AND"):
                consume()
                node = {"type": "AND", "left": node, "right": parse_not()}
            elif is_primary_start(peek()) and not is_word(peek(), "OR") and not is_word(peek(), "AND"):
                node = {"type": "AND", "left": node, "right": parse_not(), "implicit": True}
            else:
                break
        return node

    def parse_not():
        if is_word(peek(), "NOT"):
            consume()
            return {"type": "NOT", "child": parse_not()}
        return parse_proximity()

    def parse_proximity():
        node = parse_primary()
        while is_near(peek()):
            op_token = consume()["value"]
            m = _NEAR_PARSE_RE.match(op_token)
            op = m.group(1).lower()
            nval = int(m.group(2))
            node = {"type": "PROX", "op": op, "n": nval, "left": node, "right": parse_primary()}
        return node

    def parse_primary():
        t = peek()
        if not t:
            raise QueryError("式が途中で終わっています。")
        if is_field(t) and peek(1) and peek(1)["type"] == "EQ":
            field = consume()["value"].upper()
            consume()  # EQ
            if peek() and peek()["type"] == "LPAREN":
                consume()
                child = parse_expression()
                if not peek() or peek()["type"] != "RPAREN":
                    raise QueryError(f"{field}= の閉じ括弧がありません。")
                consume()
            else:
                child = parse_primary()
            return {"type": "FIELD", "field": field, "child": child}
        if t["type"] == "LPAREN":
            consume()
            node = parse_expression()
            if not peek() or peek()["type"] != "RPAREN":
                raise QueryError("閉じ括弧がありません。")
            consume()
            return node
        if t["type"] == "PHRASE":
            consume()
            return {"type": "PHRASE", "value": t["value"]}
        if t["type"] == "WORD":
            v = consume()["value"]
            if v.upper() in ("AND", "OR", "NOT") or _NEAR_RE.match(v):
                raise QueryError(f"演算子 {v} の位置が不正です。")
            return {"type": "TERM", "value": v}
        raise QueryError(f"予期しないトークン: {t['value']}")

    if not tokens:
        return {"type": "EMPTY"}
    ast = parse_expression()
    if pos["i"] < len(tokens):
        raise QueryError(f"解釈できない残りの入力があります: {tokens[pos['i']]['value']}")
    _validate_wildcards(ast)
    return ast


def _validate_wildcards(node):
    if not node:
        return
    t = node.get("type")
    if t in ("TERM", "PHRASE"):
        if t == "PHRASE" and re.search(r"[*?]", node["value"]):
            raise QueryError("引用符フレーズ内のワイルドカードは未対応です。")
    for key in ("left", "right", "child"):
        if node.get(key):
            _validate_wildcards(node[key])


# ------------------------------------------------------------------
# AST 解析ヘルパー
# ------------------------------------------------------------------
def _walk(node, fn):
    if not node:
        return
    fn(node)
    for key in ("left", "right", "child"):
        if node.get(key):
            _walk(node[key], fn)


def has_explicit_field(node) -> bool:
    found = [False]
    def f(n):
        if n.get("type") == "FIELD":
            found[0] = True
    _walk(node, f)
    return found[0]


def has_unfielded_expression(node, inside_field=False) -> bool:
    if not node:
        return False
    t = node.get("type")
    if t == "FIELD":
        return has_unfielded_expression(node["child"], True)
    if t in ("TERM", "PHRASE"):
        return not inside_field
    return (has_unfielded_expression(node.get("left"), inside_field) or
            has_unfielded_expression(node.get("right"), inside_field) or
            has_unfielded_expression(node.get("child"), inside_field))


def contains_prox(node) -> bool:
    found = [False]
    _walk(node, lambda n: found.__setitem__(0, found[0] or n.get("type") == "PROX"))
    return found[0]


def contains_not(node) -> bool:
    found = [False]
    _walk(node, lambda n: found.__setitem__(0, found[0] or n.get("type") == "NOT"))
    return found[0]


def contains_wildcard(node) -> bool:
    found = [False]
    def f(n):
        if n.get("type") in ("TERM", "PHRASE") and re.search(r"[*?]", n.get("value", "")):
            found[0] = True
    _walk(node, f)
    return found[0]


def collect_positive_fields(node, current_field="TX", out=None, in_not=False):
    """NOT 配下を除く、各語が属するフィールドの集合"""
    if out is None:
        out = set()
    if not node:
        return out
    t = node.get("type")
    if t == "FIELD":
        return collect_positive_fields(node["child"], node["field"], out, in_not)
    if t == "NOT":
        return collect_positive_fields(node["child"], current_field, out, True)
    if t in ("TERM", "PHRASE", "PROX"):
        if not in_not:
            out.add(current_field)
        if t == "PROX":
            collect_positive_fields(node.get("left"), current_field, out, in_not)
            collect_positive_fields(node.get("right"), current_field, out, in_not)
        return out
    collect_positive_fields(node.get("left"), current_field, out, in_not)
    collect_positive_fields(node.get("right"), current_field, out, in_not)
    return out


def candidate_scope(node, default_field="TX") -> str:
    """OpenAlex 候補取得 scope を決定する（all/title/abstract/title_and_abstract/fulltext）"""
    fields = collect_positive_fields(node, default_field)
    if not fields:
        return "all"
    if all(f in ("TI", "AB", "TA") for f in fields):
        if len(fields) == 1:
            f = next(iter(fields))
            return {"TI": "title", "AB": "abstract", "TA": "title_and_abstract"}[f]
        return "title_and_abstract"
    if len(fields) == 1:
        f = next(iter(fields))
        return {"TX": "all", "FT": "fulltext"}.get(f, "all")
    return "all"


def requires_internal_filter(node) -> bool:
    """near/adj を含む（→ 取得後にローカル厳密照合が必要）か"""
    return contains_prox(node) or contains_not(node) or contains_wildcard(node)


# ------------------------------------------------------------------
# 候補クエリ構築（AST → OpenAlex 検索文字列。NOT は落とす）
# ------------------------------------------------------------------
def _format_candidate_term(node) -> str:
    v = str(node.get("value", "")).strip()
    if node.get("type") == "PHRASE" or re.search(r"\s", v):
        return '"' + v.replace('"', '\\"') + '"'
    return v


def _candidate_alternatives(node, limit=33):
    """OR を代替案に展開し、AND/PROX を結合した語リスト群を返す。(items, overflow)"""
    def pack(items, overflow=False):
        return {"items": items, "overflow": overflow}

    def concat_unique(arrays):
        seen, out = set(), []
        for arr in arrays:
            for v in arr:
                v = v.strip()
                if v and v not in seen:
                    seen.add(v); out.append(v)
        return out

    def combine(a, b):
        out = []
        overflow = a["overflow"] or b["overflow"]
        for la in a["items"]:
            for rb in b["items"]:
                out.append(concat_unique([la, rb]))
                if len(out) > limit:
                    return pack(out, True)
        return pack(out, overflow)

    if not node or node.get("type") == "EMPTY":
        return pack([[]])
    t = node["type"]
    if t in ("TERM", "PHRASE"):
        # ワイルドカード(* ?)を含む語は OpenAlex の .search フィルタに送ると HTTP 400 になり、
        # かつ索引が前方一致非対応のため候補取得にも使えない。候補からは除外し（NOT と同様に空扱い）、
        # 取得後のローカル厳密照合（paper_matches）に判定を委ねる。
        if t == "TERM" and re.search(r"[*?]", str(node.get("value", ""))):
            return pack([[]])
        return pack([[_format_candidate_term(node)]])
    if t == "FIELD":
        return _candidate_alternatives(node["child"], limit)
    if t == "NOT":
        return pack([[]])
    if t == "OR":
        l = _candidate_alternatives(node["left"], limit)
        r = _candidate_alternatives(node["right"], limit)
        items = l["items"] + r["items"]
        return pack(items[:limit + 1], l["overflow"] or r["overflow"] or len(items) > limit)
    if t in ("AND", "PROX"):
        return combine(_candidate_alternatives(node["left"], limit),
                       _candidate_alternatives(node["right"], limit))
    return pack([[]])


def _compact_candidate_string(node) -> str:
    if not node or node.get("type") == "EMPTY":
        return ""
    t = node["type"]
    if t in ("TERM", "PHRASE"):
        # ワイルドカード語は候補取得に使えない（上記 _candidate_alternatives 参照）ため空扱い。
        if t == "TERM" and re.search(r"[*?]", str(node.get("value", ""))):
            return ""
        return _format_candidate_term(node)
    if t == "FIELD":
        return _compact_candidate_string(node["child"])
    if t == "NOT":
        return ""
    if t == "OR":
        l = _compact_candidate_string(node["left"]); r = _compact_candidate_string(node["right"])
        return f"({l} OR {r})" if l and r else (l or r)
    if t in ("AND", "PROX"):
        l = _compact_candidate_string(node["left"]); r = _compact_candidate_string(node["right"])
        return f"({l} AND {r})" if l and r else (l or r)
    return ""


def build_candidate_queries(node, max_queries=32):
    """OpenAlex に投げる候補クエリ文字列のリストを返す"""
    max_q = max(1, int(max_queries or 32))
    alts = _candidate_alternatives(node, max_q + 1)
    if not alts["overflow"] and len(alts["items"]) <= max_q:
        queries = []
        for parts in alts["items"]:
            uniq = [p.strip() for p in dict.fromkeys(parts) if p.strip()]
            if uniq:
                queries.append(" AND ".join(uniq))
        # 重複除去（順序保持）
        return list(dict.fromkeys(queries)) or [""]
    compact = _compact_candidate_string(node)
    return [compact] if compact else [""]


# ------------------------------------------------------------------
# ローカル照合（near/adj / NOT / ワイルドカードの厳密検証）
# ------------------------------------------------------------------
def make_docs(title: str, abstract: str) -> dict:
    ta = " / ".join([t for t in [title or "", abstract or ""] if t])
    return {
        "TI": {"field": "TI", **tokenize_document(title or "")},
        "AB": {"field": "AB", **tokenize_document(abstract or "")},
        "TA": {"field": "TA", **tokenize_document(ta)},
    }


def _term_patterns(node):
    is_phrase = node.get("type") == "PHRASE"
    toks = search_tokens(node["value"], keep_wildcards=not is_phrase)
    return [pattern_from_token(t) for t in toks]


def find_term_spans(node, doc):
    patterns = _term_patterns(node)
    toks = doc["tokens"]
    if not patterns or not toks:
        return []
    spans = []
    last = len(toks) - len(patterns)
    for i in range(0, last + 1):
        ok = True
        for j, pat in enumerate(patterns):
            if not pat.match(toks[i + j]["text"]):
                ok = False
                break
        if ok:
            spans.append({"start": i, "end": i + len(patterns) - 1})
    return spans


def _compare_proximity(l, r, op, n):
    if op == "adj":
        if l["end"] < r["start"]:
            gap = r["start"] - l["end"] - 1
            if gap <= n:
                return {"ok": True, "gap": gap,
                        "start": l["start"], "end": r["end"]}
        return {"ok": False}
    if l["end"] < r["start"]:
        gap = r["start"] - l["end"] - 1
    elif r["end"] < l["start"]:
        gap = l["start"] - r["end"] - 1
    else:
        gap = 0
    if gap <= n:
        return {"ok": True, "gap": gap,
                "start": min(l["start"], r["start"]), "end": max(l["end"], r["end"])}
    return {"ok": False}


def _eval_in_doc(node, doc):
    """単一 doc に対する AST 評価。(matched, spans) を返す。"""
    if not node or node.get("type") == "EMPTY":
        return True, []
    t = node["type"]
    if t in ("TERM", "PHRASE"):
        spans = find_term_spans(node, doc)
        return (len(spans) > 0), spans
    if t == "FIELD":
        return _eval_in_doc(node["child"], doc)
    if t == "NOT":
        m, _ = _eval_in_doc(node["child"], doc)
        return (not m), []
    if t == "OR":
        lm, ls = _eval_in_doc(node["left"], doc)
        if lm:
            return True, ls
        rm, rs = _eval_in_doc(node["right"], doc)
        return (rm, rs if rm else [])
    if t == "AND":
        lm, ls = _eval_in_doc(node["left"], doc)
        rm, rs = _eval_in_doc(node["right"], doc)
        if not (lm and rm):
            return False, []
        return True, (ls if ls else rs)
    if t == "PROX":
        lm, ls = _eval_in_doc(node["left"], doc)
        rm, rs = _eval_in_doc(node["right"], doc)
        if not (lm and rm):
            return False, []
        best = None
        for a in ls:
            for b in rs:
                cmp = _compare_proximity(a, b, node["op"], node["n"])
                if cmp["ok"]:
                    span = {"start": cmp["start"], "end": cmp["end"]}
                    if best is None or (span["end"] - span["start"]) < (best["end"] - best["start"]):
                        best = span
        if best is None:
            return False, []
        return True, [best]
    return False, []


def _eval_in_field(node, docs, field):
    if field in CANDIDATE_ONLY_FIELDS:
        return True  # TX/FT は候補専用（内部照合しない）
    doc = docs.get(field) or docs["TA"]
    m, _ = _eval_in_doc(node, doc)
    return m


def eval_ast(node, docs, default_field="TX") -> bool:
    """論文（docs = make_docs(title, abstract)）が AST 全体に一致するか"""
    if not node or node.get("type") == "EMPTY":
        return True
    t = node["type"]
    if t == "FIELD":
        return _eval_in_field(node["child"], docs, node["field"])
    if t == "AND":
        return eval_ast(node["left"], docs, default_field) and eval_ast(node["right"], docs, default_field)
    if t == "OR":
        return eval_ast(node["left"], docs, default_field) or eval_ast(node["right"], docs, default_field)
    if t == "NOT":
        return not eval_ast(node["child"], docs, default_field)
    # bare TERM/PHRASE/PROX → デフォルトフィールドで評価
    return _eval_in_field(node, docs, default_field)


def paper_matches(node, title, abstract, default_field="TX") -> bool:
    return eval_ast(node, make_docs(title, abstract), default_field)


# ------------------------------------------------------------------
# 複数行コマンド（#NN 行 + T= 結合論理式）の展開
# ------------------------------------------------------------------
_CMD_LINE_RE = re.compile(r"^\s*#0*(\d+)\s*(.+)$")
_REF_RE = re.compile(r"#0*(\d+)")


def expand_command_query(text: str) -> str:
    """複数行コマンド入力を 1 本の検索式へ展開する。

    形式:
        #01  TI=(solid adj3 electrolyte)
        #02  AB=((degradation AND capacity) near8 mechanism)
        T=(#01 AND #02) AND NOT #02
    - `#1` と `#01` は同一視。
    - 1 行のみで T 式が無ければ自動的にその行を採用。
    - `T=` 行が無く複数行ある場合はエラー。
    戻り値: 展開済みの単一検索式。
    """
    raw = normalize_input(text)
    if not raw:
        return ""
    rows = {}          # index(int) -> expr(str)
    order = []
    logic = ""
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _CMD_LINE_RE.match(line)
        if m:
            idx = int(m.group(1))
            rows[idx] = m.group(2).strip()
            order.append(idx)
            continue
        # T= 結合論理式行
        tline = re.sub(r"^T\s*=\s*", "", line, flags=re.IGNORECASE).strip()
        if tline:
            logic = tline

    if not rows:
        # #NN が無ければ全体を 1 本の式として扱う（単一式モード）
        return raw

    if not logic:
        if len(rows) == 1:
            logic = f"#{order[0]:02d}"
        else:
            raise QueryError("複数のコマンド行がある場合は結合論理式 T を入力してください。例: T=(#01 AND #02)")

    if re.search(r"#(?!\d)", logic):
        raise QueryError("結合論理式の行参照は #01 または #1 の形式で指定してください。")

    used = set()

    def _sub(mo):
        idx = int(mo.group(1))
        if idx not in rows:
            raise QueryError(f"結合論理式が存在しない行 #{idx:02d} を参照しています。")
        used.add(idx)
        return f"({rows[idx]})"

    expanded = _REF_RE.sub(_sub, logic)
    if not used:
        raise QueryError("結合論理式には #01 などの行参照を 1 つ以上含めてください。")
    return expanded


# ------------------------------------------------------------------
# 上位 API: 検索式をコンパイルして実行に必要な情報をまとめて返す
# ------------------------------------------------------------------
def compile_command_query(text: str, require_field: bool = True) -> dict:
    """コマンドライン検索式をコンパイルする。

    Returns dict:
        ast, expanded (str), scope (str), default_field (str),
        candidate_queries (list[str]), needs_local_filter (bool)
    例外: QueryError（構文・フィールド指定エラー）
    """
    expanded = expand_command_query(text)
    if not expanded:
        raise QueryError("検索式を入力してください。")
    ast = parse(expanded)
    if ast.get("type") == "EMPTY":
        raise QueryError("検索式が空です。")
    if require_field:
        if not has_explicit_field(ast):
            raise QueryError("コマンドラインモードでは TI=/AB=/TA=/TX=/FT= のいずれかを明示してください。")
        if has_unfielded_expression(ast):
            raise QueryError("フィールド未指定の語・式があります。各部分に TI=/AB=/TA= 等の範囲を明示してください。")
    scope = candidate_scope(ast)
    candidate_queries = build_candidate_queries(ast)
    # ワイルドカード語のみ／NOT のみの式は OpenAlex から候補を取得できない（具体語が無い）。
    # この場合は静かに 0 件で返さず、原因の分かるエラーにする。
    if not any(q.strip() for q in candidate_queries):
        raise QueryError(
            "候補取得に使える具体的な検索語がありません。"
            "少なくとも 1 つの通常語を含めてください"
            "（ワイルドカードのみ・NOT のみの検索式は実行できません。"
            "例: TI=(battery AND electrol*)）。"
        )
    return {
        "ast": ast,
        "expanded": expanded,
        "scope": scope,
        "default_field": _SCOPE_DEFAULT_FIELD.get(scope, "TX"),
        "candidate_queries": candidate_queries,
        "needs_local_filter": requires_internal_filter(ast),
    }
