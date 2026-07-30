# ==================================================================
# APOLLO フライトレコーダー — 分析条件の自動記録
#
# 分析者が「この結果をどう説明するか」に答えられるようにするための記録装置。
# 母集団の来歴と、各モジュールの実行パラメータ・乱数シード・実行時刻を
# セッション中に貯め、画面カード・ZIP・レポート付録の3か所へ出す。
#
# 記録は全パラメータを対象とするが、レポートへの出し方は3層に分ける。
# 層を分けずに丸ごと AI へ渡すと、本文に内部パラメータ値が漏れて
# 用語ルール（terminology.md）に抵触するため、層は JSON に明示して運ぶ。
#
#   層1 LINEAGE  母集団の来歴          → レポート本文で使える
#   層2 JUDGMENT 判断に効くパラメータ  → 「結論の前提と見直しのサイン」の材料。
#                                        数値そのものではなく条件文に変換して書く
#   層3 REPRO    再現用の全パラメータ  → 付録の表に載せるだけ。本文への記載は禁止
# ==================================================================
import datetime

import streamlit as st

SCHEMA_VERSION = 1
APP_VERSION = "10.0.1"

LAYER_LINEAGE = 1
LAYER_JUDGMENT = 2
LAYER_REPRO = 3

LAYER_POLICY = {
    "1": "母集団の来歴。レポート本文（母集団の定義・信頼性注記）で使用してよい。",
    "2": "判断に効くパラメータ。『結論の前提と見直しのサイン』の材料として、"
         "数値そのものではなく「粒度を上げれば分割されうる」等の条件文に変換して書く。",
    "3": "再現用パラメータ。付録の表に載せるだけ。本文への数値記載は禁止。",
}

_KEY = "flight_recorder"

# 画面・付録での表示順（未知のモジュールは末尾に回す）
_MODULE_ORDER = ["Mission Control", "ATLAS", "CORE", "Saturn V", "MEGA",
                 "Explorer", "CREW", "EAGLE", "NEBULA", "CAPCOM"]


def _now():
    return datetime.datetime.now().replace(microsecond=0).isoformat()


def _store():
    """記録の実体を返す（無ければ初期化する）"""
    rec = st.session_state.get(_KEY)
    if not isinstance(rec, dict):
        rec = {
            "schema_version": SCHEMA_VERSION,
            "app_version": APP_VERSION,
            "started_at": _now(),
            "lineage": {"layer": LAYER_LINEAGE},
            "runs": [],
        }
        st.session_state[_KEY] = rec
    return rec


def reset():
    """新しいデータを読み込んだときに記録をまっさらにする"""
    st.session_state[_KEY] = None
    _store()


# ------------------------------------------------------------------
# 層1: 母集団の来歴
# ------------------------------------------------------------------
def set_lineage(**fields):
    """母集団の来歴を記録する（既存の値は上書き・None の値は無視）"""
    lin = _store()["lineage"]
    for k, v in fields.items():
        if v is not None:
            lin[k] = v
    lin["updated_at"] = _now()


def get_lineage():
    return dict(_store()["lineage"])


# ------------------------------------------------------------------
# 層2・層3: モジュールの実行記録
# ------------------------------------------------------------------
def record(module, action, judgment=None, repro=None):
    """モジュールの実行を1件記録する。

    module   内部モジュール名（'Saturn V' 等・スナップショットの module 値と揃える）
    action   何を実行したか（'俯瞰図の生成' 等・画面と付録にそのまま出る）
    judgment 層2 — 結論の妥当性に効くパラメータと結果（粒度・クラスタ数・未分類率・品質指標）
    repro    層3 — 再現のためのパラメータ（近傍数・最小距離・乱数シード・モデル名・所要秒数）

    同一モジュール・同一アクションの再実行は最新の1件で置き換える（履歴ではなく
    「いまの結果がどの条件で出たか」を残すのが目的のため）。
    """
    rec = _store()
    entry = {
        "module": module,
        "action": action,
        "at": _now(),
        "judgment": dict(judgment or {}),
        "repro": dict(repro or {}),
    }
    runs = rec["runs"]
    for i, r in enumerate(runs):
        if r.get("module") == module and r.get("action") == action:
            runs[i] = entry
            break
    else:
        runs.append(entry)


def get_runs(module=None):
    runs = _store()["runs"]
    if module:
        return [r for r in runs if r.get("module") == module]
    return list(runs)


def _sorted_runs():
    def key(r):
        m = r.get("module", "")
        return (_MODULE_ORDER.index(m) if m in _MODULE_ORDER else len(_MODULE_ORDER), r.get("at", ""))
    return sorted(_store()["runs"], key=key)


# ------------------------------------------------------------------
# 書き出し
# ------------------------------------------------------------------
def to_json():
    """ZIP 同梱用の辞書を返す（層の意味と取り扱いルールを同梱する）"""
    rec = _store()
    return {
        "schema_version": rec.get("schema_version", SCHEMA_VERSION),
        "app_version": rec.get("app_version", APP_VERSION),
        "started_at": rec.get("started_at"),
        "exported_at": _now(),
        "layer_policy": LAYER_POLICY,
        "lineage": rec.get("lineage", {}),
        "runs": [
            {
                "module": r.get("module"),
                "action": r.get("action"),
                "at": r.get("at"),
                "judgment": {"layer": LAYER_JUDGMENT, **r.get("judgment", {})},
                "repro": {"layer": LAYER_REPRO, **r.get("repro", {})},
            }
            for r in _sorted_runs()
        ],
    }


# 画面・付録で使う来歴の表示名（キー → ラベル）。定義順に表示する。
_LINEAGE_LABELS = [
    ("source_files", "読み込みファイル"),
    ("rows_loaded", "読込件数"),
    ("rows_merged", "結合後件数"),
    ("rows_deduped", "重複除去"),
    ("rows_final", "母集団件数"),
    ("stopword_count", "除外語（ストップワード）"),
    ("abstract_available", "要約の有無"),
    ("classification_code_label", "分類コード"),
    ("database_name", "データベース"),
    ("coverage_years", "収録年"),
    ("query_logic", "母集団論理式"),
]


def _fmt(value):
    if isinstance(value, bool):
        return "あり" if value else "なし"
    if isinstance(value, float):
        return f"{value:,.3f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, (list, tuple)):
        if not value:
            return "—"
        s = "・".join(str(v) for v in value)
        return s if len(s) <= 60 else f"{s[:60]}…（{len(value)}件）"
    s = str(value)
    return s if len(s) <= 80 else s[:80] + "…"


def summary_rows(module=None):
    """(ラベル, 値) の一覧を返す。module 指定時はそのモジュールの最新実行のみ。"""
    rows = []
    lin = get_lineage()
    for key, label in _LINEAGE_LABELS:
        if key in lin and lin[key] not in (None, "", []):
            rows.append((label, _fmt(lin[key])))
    for r in (get_runs(module) if module else _sorted_runs()):
        head = r.get("action") if module else f"{r.get('module')} — {r.get('action')}"
        rows.append((head, r.get("at", "").replace("T", " ")))
        for k, v in r.get("judgment", {}).items():
            rows.append((f"　{k}", _fmt(v)))
        if module:  # 単一モジュール表示のときは再現パラメータも畳まずに出す
            for k, v in r.get("repro", {}).items():
                rows.append((f"　{k}", _fmt(v)))
    return rows


def render_card(module=None, title="飛行記録"):
    """画面に置く記録カード。module 指定でそのモジュールの条件だけを出す。"""
    rows = summary_rows(module)
    if not rows:
        return
    body = "".join(
        f'<tr><td class="k">{k}</td><td class="v">{v}</td></tr>' for k, v in rows)
    st.markdown(
        f'<div class="ap-rec"><div class="hd"><span class="pill">REC</span>'
        f'<span class="ttl">{title}</span></div><table>{body}</table></div>',
        unsafe_allow_html=True,
    )
