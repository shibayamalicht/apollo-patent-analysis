---
title: APOLLO v8
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.41.1
app_file: Home.py
pinned: false
short_description: Patent Analysis × Multi-Agent CAPCOM — End-to-End from Data to Strategic Report
license: mit
---

# 🚀 APOLLO v8.0.0

**特許情報分析 × マルチエージェント CAPCOM — 母集団設計から戦略レポートまで、全部おまかせ。**

**Patent Analysis × Multi-Agent CAPCOM — From population design to strategic reports, fully automated.**

> "Houston, we have ~~a problem~~ a report — and this time, across three agents with one voice." — APOLLO v8

---

## これは何？ / What is this?

**APOLLO v8** は、APOLLO v7 をベースに**マルチエージェント CAPCOM**（Claude Code / Codex CLI / Antigravity IDE）と**母集団設計の文書化機能**を統合した版です。10モジュールで特許データを多角的に分析し、**CAPCOM** が結果を選択した AI エージェントに橋渡しし、**品質ゲート + 用語統一ルール付きの戦略レポート**を執筆します。

**APOLLO v8** is the successor to APOLLO v7, adding a **multi-agent CAPCOM** (Claude Code / Codex CLI / Antigravity IDE) and **population-design documentation**. It analyzes patent data through 10 specialized modules, **CAPCOM** bridges the results to the user-selected AI agent, and the agent writes **strategic reports with built-in quality gates and unified terminology**.

```
CSV/Excel  →  APOLLO v8(分析・可視化)  →  CAPCOM(In-Memory + 母集団メタ情報 + ツール選択)  →  ZIP DL(パッチ同梱済)  →  Claude Code / Codex / Antigravity(レポート執筆)
              Analysis & Viz              In-Memory + Population Meta + Tool Selection       ZIP (Pre-patched)        Report Writing (Multi-Agent)
                                                                                                                             ↓
                                                                                                                      Typst PDF 完成 🎉
```

v7 から v8 への主な進化:

**入力・取得の強化**
- 🗂️ **母集団設計の文書化**（論理式・設計意図・収録年・データベース名を任意入力 → 分析・付録に反映）
- 📅 **OpenALEX 年別取得モード**（年ごとに最大上限まで取得、10,000 件/クエリ制限を回避）
- 🎓 **OpenALEX 高品質フィルタ**（要約ありのみ / 英語のみ — タイトル側も言語判定）
- 🔍 **OpenALEX 検索プレビュー拡張**（要約列 + 取得率 🟢/🟡/🔴 + 分析対象カラム明示）
- 🔍 **OpenALEX 論文種別選択**（article / review / book-chapter など 10 種の複数選択 + CSV DL）

**母集団設計の読解（4 層誤読防止）**:
- 🔬 **query_logic 構造化読解**（7 DB 構文リファレンス + 意図↔論理整合性検査 + データ逆読み）
- 👥 **母集団 5 タイプ分類**（業界全体 / 技術領域 / 競合限定 / 単一企業 / 特定テーマ）
- 🎯 **スコープ限定ルール**（本母集団 vs 業界全体の誤読防止）
- 🧭 **設計意図の一貫性**（サブクエスチョン化 + 問い/答え形式禁止 + 「分析過程で確認された追加的事項」章）
- 📋 **`_phase_a_decisions.json`**（Phase A の決定を構造化 JSON として永続化）

**レポート品質と用語統一**
- 📝 **レポート用語統一**（内部識別子の露出を禁止、全エージェントで同一呼称を保証）
- 🛡️ **J-PlatPat 等の具体名の自動補完を禁止**（ユーザー未指定なら汎用表記で統一）
- 📋 **Phase D 品質ゲート 13 項目**（定量 + 用語 + スコープ + 母集団タイプ + 設計意図 + NEBULA 戦略の自動検証）

**CAPCOM マルチエージェント対応**
- 🤝 **マルチエージェント CAPCOM**（Claude Code / Codex / Antigravity を複数選択可、パッチ自動同梱）
- 🌐 **NEBULA 3 モード対応**（通常実行 / Web 調査で補完 / 省略 — 特許情報のみの分析も成立）

**UX と耐障害性**
- 📝 **大規模ラベル編集対応**（30 クラスタ超で `st.data_editor` に自動切替、数百クラスタでも安定動作）
- 🤖 **AI ラベルサジェスト**（TSV 推奨 + JSON/Markdown/平文 5 形式自動判別 + 部分応答の追記マージ）
- 🛡️ **Janome 例外防御**（特許テキストの異常入力で分析が止まらない）

Evolution highlights from APOLLO v7:

**Input & Retrieval**
- 🗂️ **Population-design documentation** (query logic, design intent, coverage years, DB name → auto-embedded into analysis & appendix)
- 📅 **OpenALEX year-by-year retrieval** (bypass 10k/query limit for wide-range bulk acquisition)
- 🎓 **OpenALEX quality filters** (abstract-only / English-only with title secondary check)
- 🔍 **OpenALEX preview enhancements** (abstract column + acquisition rate indicator 🟢/🟡/🔴)
- 🔍 **OpenALEX publication-type multi-select** (10 types) + CSV download

**Population-design reading (4-layer misreading prevention)**
- 🔬 **Structured query_logic reading** (7-DB syntax reference + intent↔logic divergence check + data reverse-reading)
- 👥 **5 population types** (industry-wide / tech-domain / competitor-limited / single-company / specific-theme)
- 🎯 **Scope-limiting rule** (prevents confusion between "within the population" and "industry-wide")
- 🧭 **Design-intent consistency** (sub-question decomposition + no Q/A format + "additional observations" chapter)
- 📋 **`_phase_a_decisions.json`** (persists Phase-A decisions as structured JSON)

**Report quality & terminology**
- 📝 **Unified report terminology** (internal identifiers banned from output, consistent naming across all agents)
- 🛡️ **No auto-injection of specific DB names** (generic terms used when user leaves it blank)
- 📋 **13-check Phase-D quality gate** (quantitative + terminology + scope + population type + design intent + NEBULA strategy)

**CAPCOM multi-agent**
- 🤝 **Multi-agent CAPCOM** (Claude Code / Codex / Antigravity selectable, patches pre-bundled in ZIP)
- 🌐 **NEBULA 3-mode handling** (execute / web-compensation / omit — patent-only analysis works too)

**UX & fault tolerance**
- 📝 **Large-scale label editor** (auto-switch to `st.data_editor` beyond 30 clusters, stable at hundreds of clusters)
- 🤖 **AI label suggestion** (TSV-preferred + 5-format auto-detect (JSON/Markdown/plain) + partial-merge mode)
- 🛡️ **Janome exception guards** (catch IndexError in compound-noun extraction so analysis won't stop)

v7 から引き継がれる主要機能（継続）:
- 🌱 萌芽技術の自動発見（ノイズ分析・クラスタ動態マップ・多様性3指標）
- 🌌 学術・ニュース・政策の統合環境分析（OpenALEX API + Hype Cycle）
- 📡 4フェーズ + 品質ゲートの構造化レポート生成
- ☁️ Hugging Face Spaces / Streamlit Cloud で動く（In-Memory 化）
- 🧪 コアライブラリ patiroha（pytest 84件で品質保証）

---

## 🚀 クイックスタート / Quick Start

### A. Hugging Face Spaces(推奨・環境構築ゼロ)/ Hugging Face Spaces (recommended, zero setup)

```
1. Hugging Face Spaces で APOLLO v8 を開く
   Open APOLLO v8 on Hugging Face Spaces

2. CSV/Excel の特許データをアップロード
   Upload patent CSV/Excel data

3. 各モジュールで分析 → CAPCOM で母集団メタ情報を入力 + 使用ツールを選択 → ZIP ダウンロード
   Analyze → input population meta & select agents in CAPCOM → Download ZIP

4. ZIP をローカル展開 → 選択したツール(Claude Code / Codex / Antigravity)でレポート生成
   Extract ZIP locally → Generate report in the selected agent
```

> ⚠️ **セッションはブラウザを閉じると消失します**。必ず CAPCOM ページから ZIP をダウンロードしてください。
>
> ⚠️ **Session data is lost when the browser closes.** Always download the ZIP from the CAPCOM page before leaving.

### B. ローカル実行 / Local execution

#### 1. Python 環境（必須 / required）

```bash
# Python 3.10 以上を推奨（3.12 で動作確認済）
python -m venv .venv
source .venv/bin/activate         # macOS / Linux
# .venv\Scripts\activate          # Windows PowerShell

pip install -r requirements.txt
streamlit run Home.py
# http://localhost:8501 でコーヒー片手にどうぞ ☕
# Open http://localhost:8501 — grab a coffee while you're at it ☕
```

#### 2. Typst（CAPCOM レポートの PDF 化に必要 / required for PDF generation）

CAPCOM でダウンロードした ZIP を AI エージェント（Claude Code / Codex / Antigravity）で開いてレポート生成する際、最終成果物の `report.typ` / `report_executive.typ` を PDF に変換するために **Typst** が必要です。APOLLO 本体（Streamlit）の分析・可視化のみ使う場合は不要です。

Typst is required when compiling the generated `report.typ` / `report_executive.typ` into PDF via AI agents (Claude Code / Codex / Antigravity). **Not required for APOLLO's Streamlit analysis/visualization itself.**

```bash
# macOS (Homebrew)
brew install typst

# Windows (winget / Scoop)
winget install --id Typst.Typst
# または: scoop install typst

# Linux (Snap / Cargo / バイナリ)
snap install typst
# または: cargo install --git https://github.com/typst/typst --locked typst-cli
# または: 公式リリース https://github.com/typst/typst/releases

# インストール確認
typst --version
```

**レポート PDF 化コマンド** / Compile commands:

```bash
# 本編 / Main report
typst compile --root ".." reports/report.typ reports/report.pdf

# 別冊（経営層向け要約版、別冊生成を選択した場合）/ Executive summary edition (if generated)
typst compile --root ".." reports/report_executive.typ reports/report_executive.pdf
```

> 💡 AI エージェントに「レポートを書いて」と依頼すると `reports/report.typ` 等が生成されます。その後、上記コマンドで PDF 化してください。エージェントが自動で実行してくれる場合もあります。
>
> 💡 When you ask the AI agent to write a report, `reports/report.typ` is generated. Then run the command above to compile PDF. Some agents do this automatically.

### 基本ワークフロー / Basic Workflow

1. **Home.py** で特許データをアップロード → 前処理(SBERT + TF-IDF + メタデータ正規化)
   Upload patent data in Home.py → Preprocess (SBERT + TF-IDF + metadata normalization)
2. **CAPCOMセッション開始** → 以降の分析結果は自動的に In-Memory ストアに蓄積
   Start a CAPCOM session → Analysis results are auto-saved to the in-memory store
3. 各モジュール(ATLAS/Saturn V/MEGA/Explorer/CREW/EAGLE/NEBULA など)で分析・可視化
   Analyze & visualize across 10 modules
4. 気になるチャートを **VOYAGER** or 各モジュールの **📸 Snapshot** で収集
   Collect key charts as snapshots
5. **CAPCOM** で Mission Objective + 母集団メタ情報(任意4項目) + 使用ツール(複数可)を設定 → **ZIP ダウンロード**
   Set Mission Objective, population meta (4 optional fields), and agent selection in CAPCOM → Download ZIP
6. **選択したエージェント** で ZIP を展開 → 4フェーズで戦略レポート生成(品質ゲート + 用語統一検証付き)
   Extract ZIP in the selected agent → Generate reports through 4 phases with auto quality & terminology gates

---

## 🧩 10 の分析モジュール / 10 Analysis Modules

APOLLO v8 は 10 モジュールで特許データを多角的に分析します。

APOLLO v8 analyzes patent data across 10 specialized modules.

| # | モジュール / Module | 概要 / Description |
|---|----------|---------|
| 1 | 🌍 ATLAS | 基本統計 + 多様性指標(HHI + Entropy + Gini) — Basic stats + 3 diversity indices |
| 2 | 💡 CORE | AND/OR/NEAR/ADJ 論理式分類 + クロス集計マトリクス + ヒートマップ可読性（マス間白線）— Rule-based classification + cross-tab + improved heatmap readability |
| 3 | 🚀 Saturn V | AIランドスケープ + ノイズ分析 + クラスタ動態マップ — AI landscape + noise analysis + cluster dynamics map |
| 4 | 📈 MEGA | PULSE 4象限動態分析 + クラスタ動態マップ — PULSE quadrant analysis + cluster dynamics |
| 5 | 🧭 Explorer | 共起ネットワーク + 急上昇キーワード + トルネード競合比較 — Co-occurrence + trending keywords + tornado comparison |
| 6 | 🔗 CREW | 発明者・出願人ネットワーク + 媒介中心性 + コミュニティ検出 — Inventor/applicant networks + betweenness + community detection |
| 7 | 🦅 EAGLE | 投げ縄ツールで手動クラスタ + クラスタ動態マップ — Lasso-based manual clusters + cluster dynamics |
| 8 | 📝 VOYAGER | スナップショット収集 + Mission Objective 設定 + Markdown レポート骨格生成 — Snapshot collection + Mission Objective + Markdown skeleton |
| 9 | 🌌 **NEBULA** | OpenALEX API 統合 + Hype Cycle + **学術ランドスケープ**（Saturn V デザイン統一 + クラスタラベル CSV DL）+ **論文種別 10 種の複数選択** + **検索結果 CSV ダウンロード** + **年別取得モード / 要約ありフィルタ / 英語のみフィルタ** + **大規模ラベル編集対応** — OpenALEX + Hype Cycle + academic landscape + 10-type multi-select + CSV + year-by-year mode + abstract-only / English-only filters + large-scale label editor |
| 10 | 📡 CAPCOM | In-Memory セッション管理 + 独立 Mission Objective + 母集団メタ情報 4 項目（任意）+ マルチエージェント選択（Claude Code / Codex / Antigravity）+ パッチ自動同梱 — In-memory session + independent Mission Objective + population meta (4 optional fields) + multi-agent selection + auto-bundled patches |

---

## 📡 CAPCOM — マルチエージェントへの橋渡し / Bridge to multi-agent workflow

**CAPCOM** (Capsule Communicator) は APOLLO と AI レポート執筆エージェントを繋ぐ通信モジュール。**Claude Code / Codex CLI / Antigravity IDE の複数選択**に対応し、選択したエージェント用の資材が ZIP に**展開済みで自動同梱**されるため、ユーザーは ZIP を展開するだけで対応エージェントでそのまま使えます。

**CAPCOM** bridges APOLLO and AI report-writing agents. You can **select multiple agents** (Claude Code / Codex CLI / Antigravity IDE) and the corresponding assets are **auto-bundled into the ZIP in pre-applied form**, so users just extract and run.

### 🗂️ 母集団メタ情報（全項目任意） / Population Meta (all optional)

CAPCOM ページで以下4項目を任意入力できます。入力された内容はレポート本文・付録・分析注記に反映されます。

In the CAPCOM page, you can optionally input the following 4 fields. They are reflected in the report body, appendix, and analysis notes.

| フィールド / Field | レポートでの扱い / Usage in report |
|---|---|
| 🎯 **母集団論理式の設計意図** / Design intent of the population query | Phase A で**ユーザーと対話確認**した上で（`AskUserQuestion` STOP-GATE）、エージェントが咀嚼して「本分析の前提」章の「分析の視座」サブセクションに自然な日本語として書き下し、さらに **Phase B 以降の全 deep_dive・クロス分析・結論章で「分析の視座」として内在化**（ベタ貼り禁止）/ Agent **confirms its understanding with the user via dialogue** in Phase A (`AskUserQuestion` STOP-GATE), then digests it into a natural-language paragraph in the Premise chapter and **internalizes it as the analytical lens** throughout Phase B onwards (no verbatim paste) |
| 🔎 **母集団論理式** / Population query logic | **付録 D に `#raw` ブロックで全文掲載**（検索式は DB 検索のコマンド文字列なので、そのまま原文で掲載して構わない）/ Embedded verbatim as `#raw` block in Appendix D (query strings are DB command syntax — safe to paste as-is) |
| 📅 **収録年情報** / Coverage years | 付録 A の対象期間欄 + 時系列分析の解釈 / Appendix A period field + time-series interpretation |
| 🗄️ **使用した特許データベース名** / Patent DB name | 付録 A + カバレッジ制約注記 / Appendix A + coverage caveat |

**設計思想**: `database_name` が未指定なら **執筆者（エージェント）は具体名（J-PlatPat 等）を勝手に補えません**。代わりに「提供された特許データセット」と汎用表記されます。これは執筆者の勝手な補完を構造的に防止する仕組みです。

**Design principle**: When `database_name` is blank, the agent **cannot fabricate a specific DB name** (like "J-PlatPat"). Instead, generic wording ("the provided patent dataset") is used. This structurally prevents fabrication.

### 🤝 マルチエージェント選択 / Multi-agent Selection

| エージェント / Agent | 配布状態 / Status | 同梱資材 / Bundled Assets |
|---|---|---|
| **Claude Code**（Anthropic） | ✅ デフォルト / Default | `capcom_schema/`、`.claude/skills/`、`CLAUDE.md` |
| **Codex CLI**（OpenAI） | ✅ 選択可 / Selectable | 上記 + `AGENTS.md` + `.codex/skills/apollo-capcom/` + `exec_mode_addendum.md` |
| **Antigravity IDE**（Google） | ✅ 選択可 / Selectable | 上記 + `GEMINI.md` + `.agent/skills/` + `.agent/workflows/` + `artifacts_templates/` |

**仕組み**: CAPCOM ページで選択したエージェント分の資材が **ZIP 直下に展開済みで同梱** されます。ユーザーは ZIP を展開するだけで、対応エージェントでそのまま使えます。

**How it works**: Assets for the selected agents are **pre-bundled at the ZIP root**. Users just extract and run.

### セッション構造(ZIP 展開後)/ Session Structure (after ZIP extraction)

```
session_YYYYMMDD_HHMMSS_<uuid>/
├── data/                    # 全分析データ / All analysis data
│   ├── patents.csv
│   ├── atlas_statistics.json / saturnv_clusters.json / mega_momentum.json
│   ├── explorer_global_network.json / nebula_hype_cycle.json / nebula_academic_clusters.json ...
├── voyager/                 # 戦略ストーリー / Strategic narrative
│   ├── mission.json         # Mission Objective + Evidence 一覧
│   ├── evidence/            # モジュール横断 Evidence 群
│   └── context.json         # population_meta（4 項目）+ capcom_tools を含む
├── snapshots/ prompts/ reports/ metadata.json    # スナップショット画像・AI プロンプト・レポート出力先
│   └── reports/_phase_a_decisions.json  # Phase A の決定を構造化 JSON で永続化（Phase D gate 自動検証の情報源）
├── capcom_schema/           # 分析スキーマ・テンプレート・品質ゲート
│   ├── SKILL.md             # 4 フェーズ手順 + 絶対遵守ゲートルール
│   ├── analysis/
│   │   ├── terminology.md   # 用語統一ルール（最優先・内部識別子の露出禁止・スコープ限定・サブクエスチョン化）
│   │   ├── query_logic_reading.md     # 7 DB 構文リファレンス + 意図整合性検査
│   │   ├── population_type_metrics.md # 母集団 5 タイプ分類 + 指標解釈
│   │   ├── common_framework.md / data_notes.md / deep_dive_guide.md
│   │   ├── cross_module.md / report_structure.md / quality_checklist.md ...
│   ├── references/ exemplars/ templates/
│   └── scripts/             # phase_c_gate.sh / phase_d_gate.sh（13 チェック: 定量・用語・スコープ・母集団タイプ・設計意図・NEBULA 戦略）
├── .claude/skills/          # Claude Code スキル
├── CLAUDE.md                # プロジェクト設計思想
│
└── ── 以下は選択ツール分だけ自動同梱 ── Tool-specific assets (auto-bundled) ──
    ├── AGENTS.md            # Codex & Antigravity 共通ルール
    ├── GEMINI.md            # Antigravity 最優先ルール
    ├── exec_mode_addendum.md # Codex 非対話モード注意書き
    ├── review_policy_recommendation.md # Antigravity Review Policy 推奨設定
    ├── .codex/skills/apollo-capcom/    # Codex 用スキル
    ├── .agent/skills/apollo-capcom/    # Antigravity 用スキル
    ├── .agent/workflows/               # Antigravity 用 Phase 別ワークフロー
    └── artifacts_templates/            # Antigravity 用 task.md / implementation_plan.md 雛形
```

### 選択したエージェントでの使い方 / How to use the selected agent

**Claude Code**
```
ZIP を展開 → claude 起動
→ 「capcom_schema/SKILL.md を読んでレポートを書いて」
```

**Codex CLI**
```
ZIP を展開 → codex 起動（対話モード必須、codex exec 不可）
→ チャットで $apollo-capcom または /skills から選択
```

**Antigravity IDE**
```
ZIP を展開 → Antigravity IDE でフォルダを開く
→ Review Policy を "Request Review" に設定
→ チャットで「apollo-capcom スキルでレポート生成」と依頼
```

### レポート生成モード / Report Generation Mode — 4フェーズ + 13 品質ゲート + 用語統一

| Phase | タスク | 絶対遵守ゲート |
|-------|-------|--------------|
| **A** | ミッション理解 + データ精読 + `terminology.md` 読了 + `population_meta` 確認 | 用語統一ルール + 母集団メタ 4 項目 + 別冊確認 STOP-GATE / **STOP-GATE A（query_logic 構造化読解）** + **query_intent 3 点整理** + **サブクエスチョン化 STOP-GATE** + **STOP-GATE B（意図↔論理整合性）** + **STOP-GATE C（データ逆読み + 母集団タイプ判定）** + **STOP-GATE D（NEBULA 戦略判定）** + `_phase_a_decisions.json` 永続化 |
| **B** | エビデンス精読 + クロス分析 + Web 調査 | 13 種クロスパターンから 3 つ以上選定 + Web 調査テーマのユーザー確認（NEBULA 補完モード時は **4 カテゴリ必須カバー**: 市場規模・政策・学術動向・主要企業動向） |
| **C** | モジュール別 Deep Dive（7 モジュール） | `bash capcom_schema/scripts/phase_c_gate.sh` で行数自動検証 / **Step 0 は `nebula_strategy` で分岐**（execute / web_compensation / omit） |
| **D** | 統合レポート + 品質検証 + 用語統一検証 + 別冊生成（フラグ ON 時） | `bash capcom_schema/scripts/phase_d_gate.sh` で **13 Check** を自動実施（定量 + 用語 + スコープ + 母集団タイプ + 設計意図 + NEBULA 戦略） |

**Phase D gate の 13 Check（`phase_d_gate.sh`）**:
- **Check 1-9** — 行数 / 代表特許 / 4 層モデル / クロス分析分量 / snapshot / Web 出所 / 用語統一 / J-PlatPat 不正補完
- **Check 10** — スコープ限定ルール（「本母集団」vs「業界全体」の誤読防止、無限化語と限定語の比率判定）
- **Check 11** — 母集団タイプ別の不適切表現検出（タイプ B/C/D で「市場集中」「業界シェア」等を禁止、タイプ C で出願人 HHI の言及を禁止）
- **Check 12** — 設計意図の一貫性（意図参照語カウント / 問い/答え形式の禁止 / サブクエスチョンキーワードの結論章カバレッジ）
- **Check 13** — NEBULA 戦略検証（3 モード別: execute なら NEBULA 章の存在、web_compensation なら 4 カテゴリ + `#footnote` 4 件以上、omit なら NEBULA 章なし + 特許のみ対象の注記）

これらの bash スクリプトは 3 エージェント共通で同じ客観判定を提供します。**主観的な「実質 OK」判断で量的基準を上書きできません**。

The bash scripts provide identical objective pass/fail judgments across all 3 agents. **Subjective "good enough" cannot override quantitative criteria.**

---

## 🌟 主な機能 / Main Features

### 1. 母集団設計の文書化 / Population-design Documentation
- **4 項目の任意入力**: 設計意図・論理式・収録年・DB 名 / 4 optional fields
- **設計意図は Phase A でユーザーと対話確認**: エージェントが「分析目的・母集団の輪郭・視座」を自分の言葉でまとめ、`AskUserQuestion` で提示→確定してから分析を開始 / **Design intent is confirmed via user dialogue in Phase A**: the agent rephrases intent as "objective / population contour / analytical lens" and uses `AskUserQuestion` to get user sign-off before proceeding
- **設計意図はエージェントが咀嚼して内在化**: ベタ貼りは禁止。確定した視座を「本分析の前提」章で自然な日本語として書き下し、Phase B 以降の全 deep_dive・クロス分析・結論章で反映 / Once confirmed, intent is **digested and internalized** (no verbatim paste): rephrased as a natural-language premise and used as the analytical lens across all deep_dives, cross-module analysis, and conclusions
- **論理式・収録年・DB 名は付録にそのまま反映**（論理式は DB 検索のコマンド文字列なので原文のまま）/ Query logic, coverage years, DB name are reflected in the appendix as-is (query logic is kept verbatim since it's a DB command string)
- **執筆者の勝手な補完を禁止**: `database_name` 未指定なら汎用表記固定 / No auto-fabrication of DB names

### 2. マルチエージェント CAPCOM / Multi-agent CAPCOM
- **3エージェント対応**: Claude Code / Codex CLI / Antigravity IDE / 3-agent support
- **複数選択可**: 1つのZIPに複数ツール分の資材を同時同梱 / Multi-select (bundle assets for multiple agents at once)
- **資材自動同梱**: ZIP を展開するだけで対応エージェントで即使える / Auto-bundled agent assets (just extract and run)
- **bash 品質ゲート共通**: 3エージェントで同じ客観判定 / Shared bash quality gates for consistent judgments

### 3. レポート用語統一 / Unified Report Terminology
- **`terminology.md` 新設**: 内部識別子（`spatial_context`, `cluster_dynamics`, `*.json`, `*.md` 等）のレポート本文露出を禁止 / New `terminology.md` banning internal identifiers from report body
- **正式日本語呼称の固定**: 「空間配置分析」「クラスタ動態マップ」「Saturn V TELESCOPE 分析」等 / Canonical Japanese terms fixed
- **Phase D 自動検出**: `phase_d_gate.sh` が違反を客観判定 / Auto-detected by `phase_d_gate.sh`
- **執筆者ごとの表現ブレを抑止** / Eliminates agent-to-agent wording drift

### 4. OpenALEX 拡張 / OpenALEX Enhancements
- **論文種別複数選択**: article / review / book-chapter / preprint / dissertation など10種 / 10 publication types (multi-select)
- **検索結果 CSV ダウンロード**: 取得した論文をそのまま CSV で保存可能 / CSV download of search results
- **未選択＝全種別**: デフォルト挙動を維持 / Default = all types (backward compatible)

### 5. CORE ヒートマップ可読性改善 / CORE Heatmap Readability
- **マス間に白線**: `xgap=2, ygap=2` でセル境界を視認可能に / Cell borders via white gaps
- **バブルチャート側は無変更**: 範囲を最小限に限定 / Bubble chart kept as-is

### 6. 経営層向け要約版（別冊）の同時生成 / Executive Summary Edition
- **Phase A の STOP-GATE で確認**: 「レポート書いて」と依頼すると、エージェントがまず `AskUserQuestion` で「本編に加えて別冊（8-12ページの経営層向け要約版）も生成しますか？」と確認 / Phase A STOP-GATE: when user asks for a report, the agent first confirms via `AskUserQuestion` whether to also generate an 8-12 page executive summary edition
- **刈り取り禁止**: 別冊は本編の段落を短縮したものではなく、**本分析の Mission Objective と `query_intent` から導かれる「今回の意思決定テーマ」に即して**エッセンスを再構成した凝縮版。定型の分類軸を機械的に当てはめるのは不可 / Not a truncation: the executive edition is **re-synthesized around the specific decision theme** derived from this analysis's Mission Objective and `query_intent`, not a shortened copy. A fixed set of categories must not be forced on every report
- **So What 原則**: 各段落は「経営判断に何を意味するか」を必ず含み、手法詳細（SBERT/UMAP 等）は混入させない / Each paragraph must answer "So What for executives"; methodology details are excluded
- **別冊専用ガイド**: `capcom_schema/analysis/executive_summary_guide.md` にページ構成・凝縮技法・品質チェックを記載 / Dedicated guide at `executive_summary_guide.md`
- **成果物**: `reports/report.typ`（本編） + `reports/report_executive.typ`（別冊、新規） / Two output files

### 7. NEBULA 学術ランドスケープを Saturn V デザインに統一 + CSV DL / Academic Landscape Aligned with Saturn V + CSV Export
- **可視化の統一**: カラーパレットを `utils.APOLLO_COLORS`（G10）に、マーカーサイズ・ラベル枠線スタイル・高さ（1200px）・aspect 1:1・密度モードのカラースケールを Saturn V メインマップと揃え、全ランドスケープで同じ視覚言語に / Visual unification: color palette `utils.APOLLO_COLORS`, marker size, label border styling, 1200px height, 1:1 aspect ratio, and density colorscale aligned with Saturn V for a consistent visual language
- **trace 構造の刷新**: クラスタごと個別 trace → **全件 1 trace + colorscale でクラスタ着色** に変更（Saturn V と同じ方式） / Unified trace structure: per-cluster traces → single trace with color-by-cluster (same approach as Saturn V)
- **クラスタラベル付き CSV ダウンロード**: ランドスケープ描画の直下に download ボタンを配置。`unified_title`, `unified_content`, `unified_source`, `year`, `citation_count`, `doi`, `acad_cluster`, `acad_cluster_label`, `acad_umap_x`, `acad_umap_y` を UTF-8 BOM で出力（Excel で直接開ける） / Cluster-labeled CSV download button placed right under the landscape plot; outputs columns above as UTF-8 BOM (opens cleanly in Excel)
- **ファイル名**: `APOLLO_NEBULA_Academic_Landscape.csv`

---

## 🔬 母集団設計の読み込みと誤読防止 / Population-design Reading & Misreading Prevention

**母集団設計の読み込みと誤読防止**を 4 層構造で実装しています。

### 8. スコープ限定ルール / Scope-limiting Rule (Check 10)

特許分析レポートで頻発する **「本母集団の観察を業界全体の傾向として誤読させる」問題** を構造的に防止:

❌ NG（無限化表現）:
- 「業界では A 社が最大手である」→ 母集団は A 社を含む検索式で絞り込まれただけ
- 「市場集中度 HHI = 0.28」→ それは母集団内の集中度で、市場全体ではない
- 「全体として成長している」→ 何の全体か不明

✅ OK（母集団限定修飾）:
- 「本母集団では A 社が最大出願人である」
- 「本分析の特許群では出願人集中度 HHI = 0.28」
- 「本データセットの範囲では〜」

**例外条件**: Web 調査で外部データを `#footnote[...]` で引用した上で書く場合のみ、業界全体への一般化が許容されます。

**自動検出**: `phase_d_gate.sh` **Check 10** が無限化語（「業界では」「市場では」等）と限定語（「本母集団では」等）の出現数を比較。限定語が 5 件未満 or 無限化語が限定語の 0.3 倍を超えると FAIL。

### 9. query_logic 構造化読解と 7 DB 構文リファレンス / Structured Query-logic Reading with 7-DB Syntax Reference

**Phase A に 3 つの新 STOP-GATE（A / B / C）を追加**し、検索式を単にコピペするのではなく、エージェントが **DB 識別 → 構文分解 → 意図推定 → ユーザー確認** の 4 ステップで構造化して読解します。

新規リファレンス `capcom_schema/analysis/query_logic_reading.md` に **7 特許 DB** の精密構文を収録:

| DB | 提供元 | 主要演算子 / 近傍演算子 |
|---|---|---|
| J-PlatPat | INPIT（特許庁系・無償） | `*` `+` `-` / `,{n}C,`（順序固定） `,{n}N,`（順不同） |
| JP-NET | JPDS（日本パテントデータサービス） | `&` `+` `!` / `<NW>` `[NW]` |
| Patentfield | Patentfield 株式会社 | `and` `OR` `not` / `*N{n}` `*ONP{n}` |
| Shareresearch | 日立社会情報サービス | `*` `+` / `adj{n}` `near{n}`（国内文字数・国外単語数） |
| BizCruncher | パテント・リザルト | `*` `&` / `+` / `!` / `adj{n}` `near{n}` |
| PatentSQUARE | パナソニック ソリューションテクノロジー | `*` `+` `#`（スペース自動 OR） / `?キーワード?` |
| PatSnap | Patsnap Pte. Ltd.（シンガポール） | AND/OR/NOT / `$Wn` `$PREn` `$SEN` `$PARA` |

+ 欧米 DB（Espacenet / Google Patents / WIPO PATENTSCOPE / USPTO / PatBase / Derwent / Orbit）の要点。

**STOP-GATE B（意図↔論理整合性）**: 8 項目チェック（技術領域 / 用途 / 対象期間 / 地域 / 出願人絞り込み / 除外条件 / 公報種別 / 分類階層）で乖離を 🔴 Critical / 🟡 Warning / 🔵 Info に分類、改善提案付きでユーザー確認。

**STOP-GATE C（データ逆読み）**: patents.csv から Level 2 項目（上位出願人 / 主要 IPC / 年分布 / HHI / 国分布）を算出、自動偏り警告を提示。

### 10. 母集団 5 タイプ分類 / 5 Population Types (Check 11)

母集団の設計によって **使える指標と使えない指標が根本的に変わる** ため、5 タイプで分類・運用:

| タイプ | 名称 | 代表例 | 判定サイン |
|---|---|---|---|
| **A** | 業界全体 | 「全自動車業界の EV 関連特許」 | 上位 10 社シェア < 40% |
| **A'** | 技術領域 | 「全固体電池」「MRAM」 | 上位 10 社シェア 40-70%、出願人絞り込みなし |
| **B** | 競合限定 | 「トヨタ + ホンダ + 日産 の EV 出願」 | 上位 5 社で > 80% |
| **C** | 単一企業 | 「パナソニックの電池特許」 | 上位 1 社シェア > 90% |
| **D** | 特定製品・技術テーマ | 「全固体電池の正極材料のみ」 | 上位 10 社で > 70% + 複合絞り込み |

**タイプ別の指標解釈**:
- 出願人 HHI: A/A' では市場構造として使用可 / B では「対象社内の非対称性」に限定 / **C では HHI = 1.0 で算出無意味・禁止**
- 「市場は寡占」「業界シェア」「市場構造」等の表現は **タイプ B/C/D で禁止**（誤読を誘発するため）

**自動検出**: `phase_d_gate.sh` **Check 11** が `_phase_a_decisions.json` の `population_type.code` を読み、タイプ B/C/D の場合に禁止表現の混入と、タイプ C での出願人 HHI 言及を検出。

新規リファレンス: `capcom_schema/analysis/population_type_metrics.md`

### 11. 設計意図の一貫性（サブクエスチョン化 + 問い/答え形式禁止）/ Design-intent Consistency (Check 12)

設計意図を **確認するだけ** から **分析全体の軸として一貫して機能させる** 段階へ:

**サブクエスチョン化**: `query_intent` の 3 点整理をさらに「本分析が明らかにすべき具体的観点」3-5 個に分解し、執筆者の内部作業メモとして `_phase_a_decisions.json` の `sub_questions` に保存。

**⚠️ 絶対制約: 問い/答え形式の禁止**:
- ❌ NG: `Q1: 最大成長領域はどこか? A1: タイヤ関連である` / `本分析の問い「...」に対しては、〜` / `SQ1 では〜`
- ✅ OK: `本分析の視座である「自社の注力領域選定」に即して成長動向を精査すると、タイヤ用ゴム複合材料領域が CAGR +254% と本母集団内で群を抜いて伸長している`

**5 章での意図参照義務化**: エグゼクティブサマリー冒頭 / 各 deep_dive 冒頭 / クロス分析結論段 / 戦略的提言 / 仮説検証サマリー

**新規章「分析過程で確認された追加的事項」**: 仮説検証サマリーと戦略的提言の間に配置。Phase A の乖離判定や想定外観察をここにまとめる（タイトルは修辞なし・固定）。

**自動検出（Check 12）**:
- **12a**: 意図参照語（「本分析の視座」「設計意図」等）の本文カウント 5 件以上
- **12b**: 問い/答え形式（Q1 / A1 / SQ1 / 問い 1 / サブクエスチョン等）の出現 0 件
- **12c**: `sub_questions.keywords` が結論章（戦略的提言以降）にすべて登場

### 12. NEBULA 3 モード対応 / NEBULA 3-mode Handling (Check 13)

NEBULA モジュールが未実行でもレポートが成立するよう、**3 モード** で分岐:

| モード | 条件 | レポートでの扱い |
|---|---|---|
| `execute` | NEBULA データあり | 通常の NEBULA 章を実施 |
| `web_compensation` | NEBULA 未実行 + ユーザーが Web 補完を選択 | 「外部環境分析（Web 調査）」章を設置、**4 カテゴリ必須カバー**（市場規模 / 政策・規制 / 学術動向 / 主要企業動向）、各主張に `#footnote[...]` で出所明記 |
| `omit` | NEBULA 未実行 + ユーザーが省略を選択 | NEBULA 章なし。「本分析の範囲と限界」章で「特許情報のみを対象」と注記 |

**Phase A STOP-GATE D** でユーザーが選択、`_phase_a_decisions.json` の `nebula_strategy` に保存。

**自動検出（Check 13）**: モード別に分岐判定。execute は NEBULA 章存在確認、web_compensation は 4 カテゴリカバー + `#footnote` 4 件以上、omit は NEBULA 章なし + 特許のみ対象の注記。

### 13. `_phase_a_decisions.json` の導入 / Phase-A Decisions as Structured JSON

Phase A で確定される全情報（母集団タイプ・サブクエスチョン・検索式構造化読解・乖離判定・データ偏り警告・禁止表現リスト・NEBULA 戦略）を **`reports/_phase_a_decisions.json`** に永続化。

- **Phase C/D 執筆時**: エージェントが読み、タイプ別の禁止表現・サブクエスチョンを意識した執筆
- **Phase D gate**: `phase_d_gate.sh` が JSON から直接参照し、タイプ別の自動検証を実施
- **次回セッション**: 前回の決定を引き継ぎ可能

---

## 🔧 UX・耐障害性の運用支援機能 / UX & Fault-tolerance Features

大量データでも UI が安定し、異常な入力でも分析パイプラインが止まらない運用支援機能。

### 14. OpenALEX 年別取得モード / Year-by-Year Retrieval (bypass 10k limit)

**背景**: OpenALEX API はクエリあたり 10,000 件の上限がある。広い年範囲・大量取得のシナリオでこれを超える論文が取得できない問題があった。

**解決**: UI に「📅 年別取得モード」チェックボックスを追加
- **OFF（デフォルト）**: 既存動作（全期間合算して取得上限まで）
- **ON**: 年ごとに `max_per_year=10,000` で取得、ID ベースで重複除去
- 試算表示: `6 年 × 最大 10,000 件 = 最大 60,000 件（重複除去前）`

**プログレス表示**:
- 単一クエリ: `📅 2023 年 (4/6): 5,234 / 10,000 件 | 累計: 18,764 件`
- 複数クエリ: `🔎 クエリ 2/3 | 📅 2022 年 (3/6): 4,123 / 10,000 件 | 統合累計: 28,543 件`

### 15. OpenALEX 高品質フィルタ / Quality Filters

実運用で発覚した問題:
1. 取得論文の 30% 以上が要約なし（SBERT 埋め込み精度低下）
2. `language:en` 指定でもタイトルが日本語の多言語ジャーナル論文が混入

**解決**: 2 つのフィルタを UI に追加

| フィルタ | 実装 | デフォルト |
|---|---|---|
| **📄 要約ありの論文のみ取得** | `filter=has_abstract:true` | ✅ ON（分析精度確保） |
| **🌐 英語論文のみ取得** | `filter=language:en` + **タイトル側の Unicode 判定**（9 言語検出: CJK / ハングル / キリル / アラビア / タイ等） | ☐ OFF |

**英語フィルタの二重判定**: OpenALEX の `language` は abstract ベース判定のため、タイトル側でも非英語スクリプトを検出して除外する。Unicode 文字範囲で 9 言語系統をカバーし、アクセント文字（é, ü, á 等）は英語扱いで通過させる。

### 16. OpenALEX 検索結果プレビュー拡張 / Preview Enhancements

プレビューには要約列が表示され、ユーザーが「取得データが分析に使える品質か」を即座に判断できるようになっています。

**改善内容**:
- プレビューに要約列を追加（5 列構成: タイトル / 要約（先頭 150 字） / 出版日 / ジャーナル / 被引用数）
- 要約取得率を色分け表示:
  - 🟢 80% 以上: 良好
  - 🟡 50-79%: 注意
  - 🔴 50% 未満: 要確認（分析精度が低下する恐れ）
- 分析対象の全カラムをプレビュー下部に明示

### 17. Janome 例外防御層 / Janome Exception Guards (4 modules)

**背景**: 特許テキスト中の異常文字列（超長文・特殊文字・制御文字等）で Janome が `IndexError: list index out of range` を吐き、分析パイプラインが停止する問題があった。

**対策**: 以下 4 モジュールの `extract_compound_nouns()` / `advanced_tokenize_core()` に統一的な防御層を追加:

- `pages/2_💡_CORE.py`
- `pages/3_🚀_Saturn_V.py`
- `pages/4_📈_MEGA.py`
- `pages/7_🦅_EAGLE.py`

**3 層防御**:
```python
# 1. 入力サニタイズ
if not isinstance(text, str) or not text.strip():
    return []
# 2. 超長文切り詰め（Janome lattice サイズ制約）
if len(text) > 8000:
    text = text[:8000]
# 3. 例外吸収
try:
    tokens = t.tokenize(text)
except Exception:
    return []
```

Janome を経由するテキスト処理は計 12 箇所（utils.py 経由 8 + 直接呼び出し 4）で、すべて防御層を通る。1 レコードの異常で全体が停止しない。

### 18. ラベル編集 UI の大規模対応 / Large-scale Label Editor UI

**背景**: NEBULA の学術ランドスケープなどでクラスタ数が 100 を超えると、`st.text_input` を大量描画した際に Streamlit の WebSocket メッセージ制限を超過し、`Bad message format / Tried to use Session before it was initialized` でクラッシュ。

**解決**: `utils.create_label_editor_ui` に閾値判定を追加。

| クラスタ数 | UI |
|---|---|
| ≤ 30 | 従来の `text_input` 形式（UX 変更なし） |
| **> 30** | **`st.data_editor` によるテーブル編集**（1 widget で全行管理、クラッシュしない） |

`st.data_editor` の利点:
- Excel 風の操作（Tab で次セル、Enter で確定、ドラッグ&ペースト対応）
- ソート・検索機能（列ヘッダークリック、右上の検索アイコン）
- 162 クラスタでも安定動作

### 19. AI ラベルサジェスト / AI Label Suggestion

**設計思想**: LLM に大量クラスタの JSON を強制すると中断・省略・括弧エラーが起きやすい。そこで TSV を推奨形式とし、JSON / Markdown / 平文も自動判別。部分再提案の追記マージも可能にすることで「大量編集」を「部分的な繰り返し」に変える。

**仕様**:

**プロンプト側**: 推奨フォーマットは **TSV**
```
# 出力フォーマット（TSV: クラスタID<TAB>ラベル）
0	全固体電池の固体電解質
1	画像認識による異常検知
```
「部分応答 OK」を明記し、LLM 側の出力負担を軽減。

**パーサ側**: 5 形式を自動判別（TSV / JSON / コードフェンス JSON / Markdown 表 / 平文）。クォート（`"..."` / `'...'` / `「...」` / `『...』`）も自動除去。

**追記マージモード**: 既存マップを保持しつつ、新エントリだけ上書き
```
✅ 取り込み完了: 新規 23 件 / 上書き 5 件 (合計 28 件、未対応 0 件)
```
→ 「クラスタ 5, 12, 47 だけ再提案させる」等の部分編集が可能に。

**data_editor との統合**: AI 提案を session_state に保存し、data_editor の「AI 提案」列に自動反映。4 列構成で提案と編集を並列表示:
```
┌────┬──────────┬──────────────┬──────────────┐
│ ID │ 元ラベル │ AI 提案      │ 編集後ラベル │
└────┴──────────┴──────────────┴──────────────┘
```
- 📥 **AI 提案 → 編集後ラベルへ一括コピー** ボタン
- ↩️ **編集後ラベルを元ラベルへリセット** ボタン

**適用範囲**: `utils.render_ai_label_assistant` と `utils.create_label_editor_ui` は共通関数のため、Saturn V / MEGA / EAGLE / NEBULA の全 6 呼び出し箇所で同時に機能改善。

---

## 🏗️ 技術スタック / Tech Stack

| カテゴリ / Category | ライブラリ / Libraries |
|---------|-----------|
| フレームワーク / Framework | Streamlit 1.41.1 |
| コアライブラリ / Core Library | **patiroha[all]** (pandas, janome, sklearn, SBERT, UMAP, HDBSCAN, NetworkX) |
| テキスト埋め込み / Text Embedding | sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`) |
| 日本語処理 / Japanese NLP | Janome(形態素解析) |
| 可視化 / Visualization | Plotly, Matplotlib, WordCloud |
| 学術 API / Academic API | **OpenALEX API** (v8: 論文種別10種選択・CSV DL) |
| レポート生成 / Report Generation | **Typst**（PDF コンパイル、別途インストールが必要：[typst/typst](https://github.com/typst/typst)） |
| AI エージェント連携 / AI Agent Integration | **Claude Code / Codex CLI / Antigravity IDE**（v8: マルチエージェント対応）|
| 品質ゲート / Quality Gates | **bash スクリプト**（Phase C / Phase D + v8: 用語統一検証） |

---

## 📁 プロジェクト構成 / Project Structure

```
apollo_v8/
├── Home.py                  # Mission Control（データ取込・前処理・CAPCOM セッション開始・OpenALEX 統合）
├── utils.py                 # 共通ユーティリティ（描画・サイドバー・スナップショット・クラスタ動態・ラベル編集・AI サジェスト）
├── utils_ai.py              # AI プロンプト生成 / AI prompt generation
├── utils_spatial.py         # 空間分析（patiroha 委譲）
├── capcom.py                # CAPCOM 通信モジュール（In-Memory セッション + selected_tools パッチ自動同梱 + ZIP エクスポート）
├── openalex.py              # OpenALEX API クライアント（NEBULA 学術論文検索、年別取得・要約フィルタ・言語フィルタ対応）
├── pages/                   # 10 の分析モジュール
│   ├── 1_🌍_ATLAS.py        # 基本統計 + 多様性指標（HHI / Entropy / Gini）
│   ├── 2_💡_CORE.py          # AND/OR/NEAR/ADJ 論理式分類 + クロス集計 + ヒートマップ
│   ├── 3_🚀_Saturn_V.py     # AI ランドスケープ + ノイズ分析 + クラスタ動態マップ
│   ├── 4_📈_MEGA.py          # PULSE 4 象限動態分析 + クラスタ動態マップ
│   ├── 5_🧭_Explorer.py     # 共起ネットワーク + トレンドキーワード
│   ├── 6_🔗_CREW.py          # 発明者・出願人ネットワーク + 媒介中心性
│   ├── 7_🦅_EAGLE.py         # 投げ縄ツールで手動クラスタ + クラスタ動態マップ
│   ├── 8_📝_VOYAGER.py      # スナップショット収集 + Mission Objective + Markdown レポート
│   ├── 9_🌌_NEBULA.py       # OpenALEX + Hype Cycle + 学術ランドスケープ（Saturn V デザイン統一 + CSV DL）
│   └── 10_📡_CAPCOM.py      # 母集団メタ 4 項目入力 + マルチエージェント複数選択 + ZIP エクスポート
├── capcom_schema/           # CAPCOM スキーマ・テンプレート・手順書
│   ├── SKILL.md             # 4 フェーズ手順（Phase A STOP-GATE + Phase D 13 品質ゲート）
│   ├── analysis/            # 分析フレームワーク
│   │   ├── terminology.md              # 用語統一ルール（内部識別子の露出禁止、スコープ限定ルール、サブクエスチョン化）
│   │   ├── executive_summary_guide.md  # 経営層向け要約版（別冊）執筆ガイド
│   │   ├── query_logic_reading.md      # 7 DB 構文リファレンス + 意図整合性検査 + データ逆読み
│   │   ├── population_type_metrics.md  # 母集団 5 タイプ分類 + 指標解釈 + `_phase_a_decisions.json` 仕様
│   │   ├── common_framework.md         # 4 層分析モデル + 母集団タイプ別運用
│   │   ├── data_notes.md               # 特許 / NPL 非対称性 + 全章共通のスコープ明示ルール
│   │   ├── report_structure.md         # report.typ 構造 + 付録 + NEBULA 3 モード分岐 + 「分析過程で確認された追加的事項」章
│   │   ├── quality_checklist.md        # Phase A/C/D の全ゲート項目
│   │   ├── deep_dive_guide.md / cross_module.md / map_reading.md / patent_citation.md / noise_analysis.md
│   ├── references/          # JSON スキーマ定義（各モジュール別）
│   ├── exemplars/           # レポート見本（Typst 5 種）
│   ├── templates/           # report_style.typ / slides_spec.md / apollo_template.pptx
│   └── scripts/             # 品質ゲート
│       ├── phase_c_gate.sh  # Phase C deep_dive の行数検証
│       └── phase_d_gate.sh  # Phase D 統合レポートの 13 チェック（定量・用語・スコープ・母集団タイプ・設計意図・NEBULA 戦略）
├── capcom_schema_patches/   # マルチエージェント用オーバーレイ資材
│   ├── README.md
│   ├── codex/               # Codex CLI 用（AGENTS.md + .codex/skills/ + exec_mode_addendum.md）
│   └── antigravity/         # Antigravity IDE 用（GEMINI.md + .agent/ + artifacts_templates/）
├── .claude/skills/          # Claude Code スキル
│   └── apollo-pptx/         # コンサル品質 PowerPoint 生成
├── requirements.txt
├── CLAUDE.md                # プロジェクト設計思想（用語ルール + マルチエージェント方針）
└── README.md                # ← 本ファイル / this file
```

---

## 🤔 FAQ

**Q: APOLLO v7 から何が変わった?**
**What's new compared to APOLLO v7?**

A: 主な進化は以下の 4 つの柱です:

1. **母集団設計の一貫管理**: 設計意図・検索式・収録年・DB 名の文書化、7 DB 構文読解、5 タイプ分類、スコープ限定ルール、`_phase_a_decisions.json` 永続化。これにより「母集団と業界全体の誤読」「無意味な指標算出（例: 単一企業で HHI）」「設計意図が各章でバラバラ」を構造的に防止
2. **マルチエージェント CAPCOM**: Claude Code / Codex CLI / Antigravity IDE を複数選択可能、パッチ自動同梱、3 ツール共通の bash 品質ゲート
3. **レポート品質ゲート 13 項目**: 用語統一 + スコープ + 母集団タイプ別禁止表現 + 設計意図一貫性 + NEBULA 戦略 + 数値/代表特許/クロス分析などを `phase_d_gate.sh` で客観判定
4. **UX と耐障害性**: OpenALEX 年別取得 / 要約ありフィルタ / 英語のみフィルタ（タイトル二次判定）、ラベル編集の `st.data_editor` 自動切替（大量クラスタ対応）、AI ラベルサジェスト（TSV 推奨 + 5 形式自動判別 + 追記マージ）、Janome 例外防御（異常入力耐性）

The four pillars of APOLLO v8: (1) **Consistent population-design management** — documentation of intent/query/coverage/DB name, 7-DB syntax reading, 5 population types, scope-limiting rule, persistence via `_phase_a_decisions.json`. Structurally prevents misreading ("population vs industry-wide"), meaningless metrics (e.g. HHI on a single-company population), and fragmented design intent. (2) **Multi-agent CAPCOM** — select any of Claude Code / Codex CLI / Antigravity IDE, patches auto-bundled, shared bash quality gates across 3 tools. (3) **13-check report quality gate** — terminology + scope + population-type-specific forbidden expressions + design-intent consistency + NEBULA strategy + quantitative checks, all objectively judged by `phase_d_gate.sh`. (4) **UX & fault tolerance** — OpenALEX year-by-year retrieval / abstract-only / English-only (with title check), label editor auto-switching to `st.data_editor` for large clusters, AI label suggestion (TSV preferred + 5-format auto-detect + partial-merge), Janome exception guards.

**Q: 母集団メタ情報は全部入力しないとダメ?**
**Do I have to fill in all 4 population meta fields?**

A: 全て任意です。何も入力しなくても v7 と同じ挙動で動きます。入力した項目のみがレポートに反映され、未入力項目は省略されます（執筆者が勝手に補完することもありません）。

All 4 fields are optional. If you skip all of them, behavior is identical to v7. Only the fields you fill are embedded in the report; the rest are omitted (and never auto-filled by the agent).

**Q: マルチエージェントで複数選択するとどうなる?**
**What happens when I select multiple agents?**

A: 選択した全エージェント分のパッチ資材が1つの ZIP に同梱されます（`.codex/` と `.agent/` は名前空間が分離されているので衝突しません）。同じ ZIP を Claude Code でも Codex でも Antigravity でも開けます。1回の分析で複数ツールを試したいときに便利です。

Assets for every selected agent are bundled into a single ZIP (`.codex/` and `.agent/` are namespace-isolated, no conflicts). You can open the same ZIP in Claude Code, Codex, or Antigravity. Useful when you want to compare multiple agents on the same analysis.

**Q: Codex CLI / Antigravity で品質はどれくらい保たれる?**
**How is quality maintained on Codex CLI / Antigravity?**

A: 3 ツール共通で `bash capcom_schema/scripts/phase_c_gate.sh` / `phase_d_gate.sh` が動作し、客観的な合否判定を行います。v8 ではさらに `terminology.md` による用語統一ルールが追加され、**どのエージェントで生成しても同じ呼称**が使われるようになりました。

Bash gates (`phase_c_gate.sh` / `phase_d_gate.sh`) run identically on all 3 tools with objective pass/fail criteria. v8 adds `terminology.md`, ensuring **the same canonical terms** regardless of which agent is used.

**Q: J-PlatPat 等のデータベース名はどう扱われる?**
**How are specific DB names (like J-PlatPat) handled?**

A: v7 では SKILL.md 内に「J-PlatPat 等」のハードコードがあり、実際に使っていない DB 名がレポートに混入することがありました。v8 ではユーザーが「使用した特許データベース名」欄に入力した場合のみその名前を使い、未入力なら **「提供された特許データセット」と汎用表記**を強制します。執筆者が勝手に具体名を補うことは `phase_d_gate.sh` の Check 9 で検出・ブロックされます。

v7 had "J-PlatPat 等" hardcoded in SKILL.md, which sometimes leaked into reports for other DBs. v8 uses the user-supplied `database_name` only; otherwise it enforces generic wording ("the provided patent dataset"). Auto-injection of specific names is detected and blocked by `phase_d_gate.sh` Check 9.

**Q: Hugging Face Spaces で使うときの制約は?**
**Any limitations when running on Hugging Face Spaces?**

A: セッションは**ブラウザを閉じると消失**します。必ず CAPCOM ページから ZIP をダウンロードしてください。また、SBERT モデルのロードで初回起動に 1〜2 分かかります(2 回目以降はキャッシュで高速化)。

Sessions are **lost when the browser closes**. Always download the ZIP from the CAPCOM page. Initial boot takes 1-2 minutes due to SBERT model loading (cached after first run).

**Q: APOLLO 単体でも使える? AIエージェントなしでも?**
**Can I use APOLLO without any AI agent?**

A: 分析・可視化・Markdown レポート骨格(VOYAGER)は APOLLO 単体で動きます。ただし**本格的な戦略レポート**が欲しい場合は Claude Code / Codex CLI / Antigravity IDE のいずれかとの連携が必要です。

Analysis, visualization, and the VOYAGER Markdown report skeleton work standalone. However, for **full-scale strategic reports**, you need one of Claude Code / Codex CLI / Antigravity IDE.

**Q: 日本語以外の特許データも使える?**
**Does it work with non-Japanese patent data?**

A: 日本語特許に最適化していますが、英語データでも動作します。ただし形態素解析(Janome)は日本語専用のため、多言語混在データはおすすめしません。

Optimized for Japanese patents but works with English data too. Mixed-language datasets aren't recommended since Janome (morphological analyzer) is Japanese-only.

**Q: CAPCOM の品質ゲート・用語ゲートって何?**
**What are CAPCOM's quality and terminology gates?**

A: エージェントがレポート生成で「効率のために省略しよう」と判断するのを防ぐ仕組みです。Phase C(deep_dive 行数)・Phase D(レポート品質 + v8 用語統一)で bash スクリプトが客観的合否を判定し、不合格なら該当 Phase に戻って補強します。v8 では内部識別子（`spatial_context`, `cluster_dynamics`, `*.json`, `*.md`）のレポート本文露出も自動検出します。

Mechanisms that prevent the agent from skipping steps for efficiency. Bash scripts enforce objective pass/fail on deep_dive length (Phase C) and report quality + terminology (Phase D). Failures trigger mandatory loop-backs. v8 also auto-detects internal-identifier leakage (`spatial_context`, `cluster_dynamics`, `*.json`, `*.md`) into the report body.

**Q: APOLLO SPACE とどう違うの?**
**How does this differ from APOLLO SPACE?**

A: 用途と規模が異なります:
- **APOLLO v8(本ツール)**: 本格分析向け。10 モジュール × Streamlit × マルチエージェント CAPCOM 連携で深い分析とレポート生成
- **APOLLO SPACE**: 入門者・初心者向け。単一 HTML で環境構築ゼロ、Gemini API のみで完結

Different use cases and scales:
- **APOLLO v8 (this tool)**: For serious analysis. 10 modules × Streamlit × multi-agent CAPCOM integration for deep analysis and report generation.
- **APOLLO SPACE**: For beginners. Single HTML, zero setup, powered solely by Gemini API.

**Q: PDF レポートはどう生成する?**
**How do I generate PDF reports?**

A: 以下の 3 ステップです:

1. **Typst をインストール**（初回のみ）: macOS なら `brew install typst`、Windows なら `winget install --id Typst.Typst`、Linux なら `snap install typst` など（詳細は「ローカル実行」セクション参照）
2. **エージェントでレポート生成**: 選択したエージェント（Claude Code / Codex / Antigravity）で ZIP 展開フォルダを開き、`capcom_schema/SKILL.md` を読ませると 4 フェーズで `reports/report.typ` を生成
3. **PDF にコンパイル**: 以下のコマンドで PDF 化
   ```bash
   typst compile --root ".." reports/report.typ reports/report.pdf
   # 別冊も生成した場合
   typst compile --root ".." reports/report_executive.typ reports/report_executive.pdf
   ```

エージェントが PDF コンパイルまで自動で実行してくれる場合もあります。

Three steps: (1) Install Typst once (`brew install typst` on macOS, etc., see "Local execution" section for details). (2) Have the agent (Claude Code / Codex / Antigravity) read `capcom_schema/SKILL.md` and generate `reports/report.typ` via the 4 phases. (3) Compile to PDF: `typst compile --root ".." reports/report.typ reports/report.pdf` (and the executive edition if generated). Some agents run the compile step automatically.

**Q: 経営層向け要約版（別冊）ってどんなもの?**
**What is the executive summary edition?**

A: **8-12 ページの凝縮版**で、本編（60-120 ページ）と同時に生成できます。「レポートを書いて」と依頼した際、エージェントが Phase A で「別冊も生成しますか？」と必ず確認します。別冊は本編の刈り取り版ではなく、**本分析の Mission Objective と設計意図から導かれる「今回の意思決定テーマ」に沿ってエッセンスを再構成**したものです。手法詳細（SBERT/UMAP 等）は省き、「So What」を明確化します。15 分で読了し経営会議に持ち込めるレベルを目指します。

An **8-12 page condensed edition** generated alongside the full report (60-120 pages). When you request a report, the agent always asks in Phase A whether to co-generate the executive edition. It's not a truncation: the executive edition is **re-synthesized around the specific decision theme** derived from this analysis's Mission Objective and design intent. Methodology details (SBERT, UMAP, etc.) are stripped away and "So What" is made explicit. Targets a 15-minute read for an executive meeting.

**Q: 母集団設計の「4 層誤読防止」とは?**
**What is the "4-layer misreading prevention" for population design?**

A: レポートでよく起きる誤読を 4 つの層で構造的に防ぎます:
1. **スコープ限定ルール**（Check 10）: 「業界では」「市場では」等の無限化表現を禁止、「本母集団では」等の限定修飾を必須化
2. **query_logic 構造化読解**（STOP-GATE A/B/C）: 検索式を DB 別構文リファレンス（7 DB 対応）で構造化して読解、意図との整合性を検査、データ側から逆読みで実態確認
3. **母集団 5 タイプ分類**（Check 11）: 業界全体 / 技術領域 / 競合限定 / 単一企業 / 特定テーマ に分類、タイプ別の指標解釈ルールを適用（例: 単一企業では HHI 算出禁止）
4. **設計意図の一貫性**（Check 12）: サブクエスチョンを内部メモとして作成しつつ、レポート本文は問い/答え形式を禁止、5 指定章で意図参照を義務化

加えて **NEBULA 3 モード対応**（Check 13）で特許のみの分析でも品質ゲートが成立、**`_phase_a_decisions.json`** で Phase A の全決定を JSON 永続化。詳細は `capcom_schema/analysis/query_logic_reading.md` / `population_type_metrics.md` / `terminology.md §6` を参照。

Prevents common misreadings through a 4-layer structure: (1) Scope-limiting rule (Check 10), (2) Structured query_logic reading with 7-DB reference (STOP-GATEs A/B/C), (3) 5 population types (Check 11), (4) Design-intent consistency with forbidden Q/A format (Check 12). Plus NEBULA 3-mode handling (Check 13) and `_phase_a_decisions.json` for persisting Phase-A decisions.

**Q: 自社の特許だけで分析する場合、CAPCOM は使えるの?**
**Can I use CAPCOM for single-company self-analysis?**

A: はい。**母集団 5 タイプ分類**で **タイプ C（単一企業）** として扱われ、以下のように自動的に運用が変わります:
- 出願人 HHI は算出されない（HHI=1.0 で無意味なため）
- 「市場集中」「業界寡占」等の表現は禁止（Check 11 で自動検出）
- 代わりに**発明者集中度**や**IPC ポートフォリオ分析**で技術戦略を分析
- 提言も「当社は〜」「自社の〜」の主語に限定

Yes. Single-company analysis is classified as **Type C** in the **5 population types**. The system automatically adapts: Applicant HHI is not computed (HHI=1.0 is meaningless), expressions like "market is oligopolistic" are forbidden (auto-detected by Check 11), inventor concentration and IPC portfolio analysis are used instead, and recommendations are scoped to "our company".

**Q: 「NEBULA を実行しない」特許情報のみの分析はできる?**
**Can I generate a report using only patent data (without NEBULA)?**

A: はい。**NEBULA 3 モード対応**で対応しています。Phase A の STOP-GATE D でエージェントが以下 2 択を提示:
1. **Web 補完モード**: 4 カテゴリ（市場規模 / 政策・規制 / 学術動向 / 主要企業動向）を Web 調査で補完、`#footnote` 引用付き
2. **省略モード**: 外部環境章なし、「本分析の範囲と限界」章で特許情報のみ対象と明記

Check 13 が選択に応じて自動検証を分岐させるため、特許情報のみでも品質ゲートが成立します。

Yes, via **NEBULA 3-mode handling**. At Phase A STOP-GATE D, the agent presents 2 options: (1) Web compensation mode — fill 4 categories via web research with `#footnote` citations, or (2) Omit mode — no external-env chapter, just a "scope and limits" note. Check 13 branches the auto-validation accordingly, so patent-only analysis passes the quality gate.

**Q: OpenALEX で 10,000 件を超える論文を取得したい**
**Can I retrieve more than 10,000 papers from OpenALEX?**

A: はい。**年別取得モード**を使ってください。OpenALEX の検索 UI にある「📅 年別取得モード」チェックボックスを ON にすると、年ごとに最大 10,000 件まで取得し重複除去して統合します。例えば 2020-2026 の 7 年間 × 10,000 件 = 最大 70,000 件（重複除去前）まで取得可能です。年数とページネーション回数に比例して時間がかかるので、広い年範囲の大量取得は数分以上かかります。

Yes, use **year-by-year retrieval mode**. Check "📅 Year-by-year mode" in the OpenALEX UI, and results will be retrieved per year with up to 10,000 per year, merged with ID-based deduplication. Example: 7 years × 10,000 = up to 70,000 papers (before dedup). Takes several minutes for wide year ranges due to API rate limits.

**Q: 取得した論文で要約が空のものがあると分析精度が下がる?**
**Does missing abstracts hurt analysis quality?**

A: はい、**大きく下がります**。NEBULA の学術ランドスケープは `unified_title + unified_content（要約）` を SBERT でベクトル化してクラスタリングするため、要約が空だとタイトルのみでベクトル化されて精度が低下します。対策:
1. **デフォルトで `has_abstract:true` フィルタが ON**（要約がある論文のみ取得）。通常これで十分
2. 検索結果プレビュー上部の取得率 🟢/🟡/🔴 で品質を即時確認できる
3. 多言語混在が問題なら「🌐 英語論文のみ」を ON にする（タイトル側の言語判定も含む二重フィルタ）

Yes, significantly. NEBULA's academic landscape uses `unified_title + unified_content (abstract)` for SBERT embedding; missing abstracts collapse clustering precision. Mitigations: (1) **Abstract-only filter is ON by default**, (2) The preview shows acquisition rate 🟢/🟡/🔴, (3) Toggle English-only if multi-language mixing is an issue (includes title-side language check).

**Q: クラスタ数が 100 超でラベル編集するとクラッシュしない?**
**Can I edit 100+ cluster labels without crashing Streamlit?**

A: **大丈夫です**。ラベル編集 UI は **クラスタ数が 30 を超えると自動的に `st.data_editor`（テーブル編集）形式に切り替わる**設計になっており、Streamlit の WebSocket メッセージ制限を回避します。数百クラスタでも安定動作します。テーブル内で Enter / Tab で連続編集、Excel からのコピー&ペーストも可能です。

The label editor **auto-switches to `st.data_editor` (table form) when cluster count exceeds 30**, avoiding Streamlit's WebSocket message limit. Stable at hundreds of clusters. Supports Enter/Tab for sequential editing and Excel copy-paste.

**Q: AI ラベルサジェストで 100+ クラスタの JSON を出すと LLM が途中で止まる**
**The LLM stops midway when asked for 100+ cluster labels in JSON**

A: **TSV 推奨 + 5 形式自動判別 + 追記マージモード** で対応しています:
1. プロンプトの推奨形式は **TSV (`0\tラベル`)**。LLM の出力が大幅に安定
2. それでも JSON / Markdown 表 / 平文 で応答してくれば **自動判別して取り込み**
3. 「クラスタ 5, 12, 47 だけ再提案して」と LLM に指示して、**部分応答を追記マージ**できる（既存マップは保持）
4. AI 提案は `st.data_editor` の「AI 提案」列に自動反映、一括コピーボタンで「編集後ラベル」へ転記可能

Solved with **TSV-preferred + 5-format auto-detection + partial-merge mode**: (1) Prompt recommends TSV over JSON for stable LLM output. (2) JSON/Markdown/plain text responses are auto-detected. (3) Partial re-suggestions like "regenerate only clusters 5, 12, 47" merge into the existing map. (4) AI suggestions populate the "AI 提案" column in `st.data_editor` with a bulk-copy button.

---

## 📄 ライセンス / License

MIT License

---

## 🔗 関連リポジトリ / Related Repositories

- **APOLLO CAPCOM v1.0** — 公開中の安定版 / Stable public release: [GitHub](https://github.com/shibayamalicht/apollo-patent-analysis-capcom)
- **APOLLO SPACE** — 単一HTML版(入門者向け)/ Single-HTML edition for beginners
- **APOLLO Lite** — 軽量版(PyScript)/ Lightweight PyScript edition
- **KATHERINE** — AI対話型分析設計 / AI conversational analysis designer
- **patiroha** — コアライブラリ / Core library

---

© 2025-2026 しばやま
