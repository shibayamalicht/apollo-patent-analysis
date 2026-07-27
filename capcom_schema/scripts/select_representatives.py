#!/usr/bin/env python3
# 代表特許の決定的選定スクリプト — reports/representative_patents.json を生成する
# 使い方: cd <session_dir> && python3 capcom_schema/scripts/select_representatives.py
#
# 狙い: 「どの特許を代表として引くか」をモデルの自由選択から決まった手順に移し、
# つまみ食い（結論に都合の良い特許の選択）の余地を消す。
# ⚠️ 正直な限界: これは「どの特許を選んだか」の恣意性だけを潰す。引用した特許の
# 周りの文章での解釈のつまみ食い（同じ特許を都合よく読む）は消さない。
#
# 選定規則（すべて決定的・タイブレークまで固定）:
#  - Saturn V : saturnv_clusters.json の representative_patents（patiroha が SBERT 全次元
#               空間の重心近傍で算出済み＝決定的）を第一ソースにする。番号が patents.csv に
#               照合できないクラスタは UMAP 重心近傍の上位N件にフォールバック
#  - MEGA     : mega_pulse_group の各象限内で 出願年昇順 → 公開番号昇順 の上位N件
#  - Explorer : 共起ネットワークの中心性上位キーワードごとに、そのキーワードを
#               発明名称/要約に含む特許のうち 出願年昇順 → 公開番号昇順 の先頭1件
#  - CREW     : crew_network.json があれば、媒介中心性上位ノード名で inventor_main /
#               applicant_main を絞り、出願年昇順 → 公開番号昇順 の先頭1件
#  - ATLAS    : 出願年を四分位で4期に区切り、各期の先頭N件（年昇順→番号昇順＝各時代の
#               「幕開けの特許」）。ライフサイクル各段階の象徴特許はここから引く
#  - CORE     : 技術×課題マトリクスの件数上位5セル（「その他」を含むセルは除外）の
#               各セル内で 出願年昇順 → 公開番号昇順 の上位N件
#
# ミクロ分析A はこの JSON に載った番号だけを引用する（analysis/deep_dive_guide.md）。
# phase_d_gate.sh Check 35 が、ミクロ分析節に JSON 外の番号が引かれていないかを検査する。

import glob
import json
import os
import re
import sys

N_PER_DIM = 3        # 各次元（クラスタ/象限）あたりの件数
N_HUBS = 5           # Explorer/CREW の上位ノード数

def die(msg):
    print("❌ select_representatives: %s" % msg)
    sys.exit(1)

try:
    import pandas as pd
except Exception:
    die("pandas が必要です (pip install pandas)")

if not os.path.isfile("data/patents.csv"):
    die("data/patents.csv がありません（セッションルートで実行してください）")

df = pd.read_csv("data/patents.csv", encoding="utf-8-sig")
title_col = next((c for c in df.columns if "発明名称" in c), None)
if title_col is None:
    die("発明名称カラムが見つかりません")

def norm_num(s):
    """特許番号らしき文字列を 'YYYY-NNNNNN' に正規化。取れなければ None"""
    if not isinstance(s, str):
        s = str(s) if s is not None and s == s else ""
    m = re.search(r"([0-9]{4})\s*[-−]\s*([0-9]{5,7})", s)
    return "%s-%s" % (m.group(1), m.group(2)) if m else None

def record(row):
    """patents.csv の1行を引用用レコードに整形（公開番号優先・無ければ出願番号）"""
    pub = norm_num(row.get("公開番号"))
    app = norm_num(row.get("出願番号"))
    return {
        "pub_number": pub,                       # 公開番号（特開◯◯として引用）
        "app_number": app,                       # 出願番号（未公開時「出願番号◯◯」として引用）
        "cite_as": ("特開%s" % pub) if pub else ("出願番号 %s" % app if app else "(番号なし)"),
        "title": str(row.get(title_col, "")).strip(),
        "applicant": str(row.get("applicant_main", "")).strip(),
        "year": int(row["year"]) if row.get("year") == row.get("year") else None,
    }

def sort_key(row):
    """決定的な並び: 出願年昇順 → 公開番号昇順 → 出願番号昇順"""
    y = row.get("year")
    y = int(y) if y == y else 9999
    return (y, str(row.get("公開番号", "")), str(row.get("出願番号", "")))

out = {"_meta": {
    "generated_by": "capcom_schema/scripts/select_representatives.py",
    "n_per_dimension": N_PER_DIM,
    "rules": {
        "saturnv": "saturnv_clusters.json の representative_patents（SBERT重心近傍・決定的）を第一ソース。照合不能時は UMAP 重心近傍の上位N件",
        "mega": "各象限内で出願年昇順→公開番号昇順の上位N件",
        "explorer": "中心性上位キーワードを発明名称/要約に含む特許の最先（年昇順→番号昇順）1件",
        "crew": "媒介中心性上位ノードの関与特許の最先（年昇順→番号昇順）1件",
        "atlas": "出願年の四分位4期それぞれの先頭N件（年昇順→番号昇順＝各時代の幕開けの特許）",
        "core": "技術×課題の件数上位5セル（その他セル除外）の各セル内で年昇順→番号昇順の上位N件",
        "saturnv_drill": "各ドリルダウンJSONのサブクラスタごとに第一代表を1件（図1枚あたり最大6件）",
    },
}}

# --- 番号 → patents.csv 行 の索引（公開番号・出願番号の両方で引けるように） ---
idx = {}
for _, row in df.iterrows():
    for col in ("公開番号", "出願番号"):
        k = norm_num(row.get(col))
        if k and k not in idx:
            idx[k] = row

# --- Saturn V: クラスタ代表 ---
if os.path.isfile("data/saturnv_clusters.json"):
    sat = json.load(open("data/saturnv_clusters.json", encoding="utf-8"))
    sv = {}
    for c in sat.get("clusters", []):
        cid = c.get("cluster_id")
        if cid is None or int(cid) < 0:
            continue
        key = "cluster_%s" % cid
        recs = []
        # 第一ソース: JSON の representative_patents（文字列から番号を抽出し CSV に照合）
        for rp in (c.get("representative_patents") or [])[:N_PER_DIM]:
            k = norm_num(rp if isinstance(rp, str) else json.dumps(rp, ensure_ascii=False))
            if k is not None and k in idx:
                recs.append(record(idx[k]))
        # フォールバック: UMAP 重心近傍（不足分を補う）
        if len(recs) < N_PER_DIM and {"cluster", "umap_x", "umap_y"} <= set(df.columns):
            sub = df[df["cluster"] == int(cid)].copy()
            cen = c.get("centroid")
            if cen and len(sub) > 0:
                sub["_d"] = (sub["umap_x"] - cen[0]) ** 2 + (sub["umap_y"] - cen[1]) ** 2
                have = {r["pub_number"] for r in recs} | {r["app_number"] for r in recs}
                for _, row in sub.sort_values(["_d", "公開番号"]).iterrows():
                    r = record(row)
                    if r["pub_number"] in have or r["app_number"] in have:
                        continue
                    recs.append(r)
                    if len(recs) >= N_PER_DIM:
                        break
        if recs:
            sv[key] = recs
    if sv:
        out["saturnv"] = sv

# --- Saturn V ドリルダウン: 各サブクラスタの代表（data/saturnv_drilldown_*.json） ---
#     ドリルダウン図（saturn_drill_snap_*）を論じる段落の代表特許はここから引く
drill = {}
for f in sorted(glob.glob("data/saturnv_drilldown_*.json")):
    try:
        dd = json.load(open(f, encoding="utf-8"))
    except Exception:
        continue
    name = os.path.basename(f)[len("saturnv_drilldown_"):-len(".json")]
    recs = []
    for c in dd.get("clusters", []):
        cid = c.get("cluster_id")
        if cid is None or int(cid) < 0:
            continue
        # 各サブクラスタの第一代表を1件（サブクラスタ数が多くても図1枚あたり最大6件）
        for rp in (c.get("representative_patents") or []):
            k = norm_num(rp if isinstance(rp, str) else json.dumps(rp, ensure_ascii=False))
            if k is not None and k in idx:
                r = record(idx[k])
                r["sub_cluster"] = "%s: %s" % (cid, str(c.get("label", ""))[:30])
                recs.append(r)
                break
        if len(recs) >= 6:
            break
    if recs:
        drill["drill_%s" % name] = recs
if drill:
    out["saturnv_drill"] = drill

# --- MEGA: 象限代表 ---
if "mega_pulse_group" in df.columns:
    mg = {}
    for grp, sub in df.dropna(subset=["mega_pulse_group"]).groupby("mega_pulse_group"):
        rows = sorted((row for _, row in sub.iterrows()), key=sort_key)
        mg[str(grp)] = [record(r) for r in rows[:N_PER_DIM]]
    if mg:
        out["mega"] = mg

# --- Explorer: 中心性上位キーワードに対応する特許 ---
if os.path.isfile("data/explorer_global_network.json"):
    net = json.load(open("data/explorer_global_network.json", encoding="utf-8"))
    nodes = net.get("nodes", [])
    ckey = next((k for k in ("degree_centrality", "centrality", "degree") if nodes and k in nodes[0]), None)
    if ckey:
        hubs = sorted(nodes, key=lambda n: (-n.get(ckey, 0), n.get("keyword", "")))[:N_HUBS]
        ex = {}
        text = (df[title_col].fillna("") + " " + df.get("要約", pd.Series([""] * len(df))).fillna(""))
        for n in hubs:
            kw = n.get("keyword", "")
            if not kw:
                continue
            sub = df[text.str.contains(re.escape(kw), na=False)]
            rows = sorted((row for _, row in sub.iterrows()), key=sort_key)
            if rows:
                ex["hub_%s" % kw] = [record(rows[0])]
        if ex:
            out["explorer"] = ex

# --- ATLAS: 時期区分（出願年の四分位4期）ごとの時代代表 ---
if "year" in df.columns:
    ys = df["year"].dropna()
    if len(ys) > 0:
        qs = [int(ys.quantile(q)) for q in (0, 0.25, 0.5, 0.75, 1)]
        at = {}
        for i in range(4):
            lo, hi = qs[i], qs[i + 1]
            # 期の境界は「次の期の開始年未満」（最終期のみ両端含む）で重複させない
            sub = df[(df["year"] >= lo) & (df["year"] <= hi)] if i == 3 else df[(df["year"] >= lo) & (df["year"] < qs[i + 1])]
            rows = sorted((row for _, row in sub.iterrows()), key=sort_key)
            if rows:
                at["period_%d-%d" % (lo, hi)] = [record(r) for r in rows[:N_PER_DIM]]
        if at:
            out["atlas"] = at

# --- CORE: 技術×課題マトリクスの重点セル（件数上位5セル）代表 ---
TECH, ISSUE = "core_技術分類", "core_課題分類"
if TECH in df.columns and ISSUE in df.columns:
    sub = df.dropna(subset=[TECH, ISSUE]).copy()
    sub = sub[(sub[TECH].astype(str).str.strip() != "") & (sub[ISSUE].astype(str).str.strip() != "")]
    sub = sub[~(sub[TECH].astype(str).str.contains("その他") | sub[ISSUE].astype(str).str.contains("その他"))]
    if len(sub) > 0:
        top_cells = sub.groupby([TECH, ISSUE]).size().sort_values(ascending=False).head(5)
        co = {}
        for (t, i), _cnt in top_cells.items():
            cell = sub[(sub[TECH] == t) & (sub[ISSUE] == i)]
            rows = sorted((row for _, row in cell.iterrows()), key=sort_key)
            if rows:
                co["cell_%s×%s" % (t, i)] = [record(r) for r in rows[:N_PER_DIM]]
        if co:
            out["core"] = co

# --- CREW: 媒介中心性上位ノードの関与特許 ---
if os.path.isfile("data/crew_network.json"):
    crew = json.load(open("data/crew_network.json", encoding="utf-8"))
    nodes = (crew.get("top_by_metric") or {}).get("媒介中心性") or crew.get("top_nodes") or []
    bkey = next((k for k in ("媒介中心性", "betweenness", "betweenness_centrality") if nodes and k in nodes[0]), None)
    if bkey:
        tops = sorted(nodes, key=lambda n: (-n.get(bkey, 0), str(n.get("ノード", n.get("name", n.get("id", ""))))))[:N_HUBS]
        cw = {}
        for n in tops:
            name = str(n.get("ノード", n.get("name", n.get("id", "")))).strip()
            if not name:
                continue
            mask = pd.Series([False] * len(df))
            for col in ("inventor_main", "applicant_main"):
                if col in df.columns:
                    mask = mask | df[col].fillna("").astype(str).str.contains(re.escape(name), na=False)
            rows = sorted((row for _, row in df[mask].iterrows()), key=sort_key)
            if rows:
                cw["node_%s" % name] = [record(rows[0])]
        if cw:
            out["crew"] = cw

os.makedirs("reports", exist_ok=True)
with open("reports/representative_patents.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)

# サマリ表示
total = 0
for mod, dims in out.items():
    if mod == "_meta":
        continue
    n = sum(len(v) for v in dims.values())
    total += n
    print("✅ %-8s: %d次元 %d件" % (mod, len(dims), n))
print("✅ reports/representative_patents.json を生成（合計 %d件）。ミクロ分析A はこの番号だけを引用する" % total)
if total < 15:
    print("⚠️  合計が15件未満。deep_dive_guide.md の規則に従い、不足分は重心近傍/中心性の順で N を増やして再実行するか、gate の警告に従う")
