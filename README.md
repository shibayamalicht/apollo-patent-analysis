---
title: APOLLO v9
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.41.1
app_file: Home.py
pinned: false
short_description: Patent analysis → Strategic reports via multi-agent CAPCOM
license: apache-2.0
---

# 🚀 APOLLO v9.0.0

**特許情報分析 × マルチエージェント CAPCOM — 母集団設計から戦略レポートまで、全部おまかせ。**

**Patent Analysis × Multi-Agent CAPCOM — From population design to strategic reports, fully automated.**

> "Houston, we have ~~a problem~~ a report — and this time, across three agents with one voice." — APOLLO v9

---

## これは何？ / What is this?

**APOLLO v9** は、APOLLO v8 をベースに**マルチエージェント CAPCOM**（Claude Code / Codex CLI / Antigravity IDE）と**母集団設計の文書化機能**を統合した版です。10モジュールで特許データを多角的に分析し、**CAPCOM** が結果を選択した AI エージェントに橋渡しし、**品質ゲート + 用語統一ルール付きの戦略レポート**を執筆します。

**APOLLO v9** is the successor to APOLLO v8, featuring a **multi-agent CAPCOM** (Claude Code / Codex CLI / Antigravity IDE) and **population-design documentation**. It analyzes patent data through 10 specialized modules, **CAPCOM** bridges the results to the user-selected AI agent, and the agent writes **strategic reports with built-in quality gates and unified terminology**.

```
CSV/Excel  →  APOLLO v9(分析・可視化)  →  CAPCOM(In-Memory + 母集団メタ情報 + ツール選択)  →  ZIP DL(パッチ同梱済)  →  Claude Code / Codex / Antigravity(レポート執筆)
              Analysis & Viz              In-Memory + Population Meta + Tool Selection       ZIP (Pre-patched)        Report Writing (Multi-Agent)
                                                                                                                             ↓
                                                                                                                      Typst PDF 完成 🎉
```

v8 → v9 の主な進化:

**分析モジュールの強化**
- 💡 **CORE 分類設計アシストの刷新**（分類ルール設計を支援するクラスタリングに SBERT 特徴量・HDBSCAN・最適 k 自動分析を追加し Phase 1 UI を再構築、AI インサイトを markdown 構造化。※ルールベース分類とクロス集計マトリクス自体は従来からの中核機能）
- ⚖️ **権利化率分析（量×質）**（出願数 × 権利化率の 4 象限で「件数先行型」と「真の強者」を識別。ATLAS に加え MEGA・CORE・CREW・Saturn V でも量×質をクロス評価）
- ✨ **クラスタ動態マップの AI インサイト**（Saturn V / EAGLE / NEBULA の 4 象限動態を自動解釈し戦略示唆を生成）

**レポート/スライド対応**
- 🎨 **レポート/スライド用ビジュアル**（白背景・大文字・高解像度 PNG 書き出し、件数上位クラスタのみの「整理版ランドスケープ」）
- 📦 **整理版ランドスケープの CAPCOM 同梱**（スライド向けクリーン図を ZIP に自動同梱）

**UX・用語**
- 🆘 **初心者向けヘルプ**（各パラメータ設定に「？」ツールチップを約 150 箇所追加、効果と推奨値を平易に解説）
- 🗺️ **用語「俯瞰図分析」へ統一**（Saturn V の旧称「AI ランドスケープ」を特許レポート向けに変更）
- 🖼️ **モジュールアイコンの写実化**

**基盤**
- ⚡ **Mission Control 高速化**（SBERT の GPU 活用 + キーワード抽出のマルチコア並列化）
- 📚 **CAPCOM exemplar の標準化**（例題ドメイン統一・品質ゲート/用語の整合性向上）
- 🐛 **各種バグ修正**（NLP 系・データ整合/統計系）

**データ取得・モデル・スライド**
- 🧠 **SBERT モデル選択**（Mission Control フェーズ4で ⚡高速 `MiniLM`（384次元・既定）/ 🎯高精度 `multilingual-e5-base`（768次元・多言語E5）を切替。重くて止まる環境は高速を選択。選択モデルは AI インサイト・CAPCOM の `metadata.json` に記録）
- 🔑 **OpenALEX 強化**（API キー必須化（2026-02-13〜・サーバ既定キー非対応で公開時のコスト共有を防止）+ コマンドライン検索式 `TI=/AB=/TA=/TX=/FT=` + `AND/OR/NOT` + 近傍 `nearN/adjN` + ワイルドカード + 年別取得 + **API キーのエラー/ログ漏洩対策**）
- 🖼️ **PPTX 生成スキル強化**（`slides_spec.md` v5.0：コンサル品質・**ポンチ絵 15 種**で全分析スライドに視覚要素を保証・テキストのみスライド ≤10%）
- 🤖 **ドリルダウン自動最適化の統一**（Saturn V / MEGA / EAGLE のドリルダウンの HDBSCAN 自動掃引をメインと統一：目標クラスタ数を対象件数に適応・結果と DBCV の常時再表示・自動 ON 時は手動値をグレーアウト）
- 🎭 **分析の立場（叙述スタンス）**（提言を「誰の立場で書くか」を Phase A で確定：self=自社「当社」/ competitor=競合 / neutral=投資家・アナリスト。立場で呼称・分析の力点・提言の型・サブクエスチョンが変わる。`Check 11s` で立場違反を検出）
- 📊 **PPTX 情報密度の底上げ**（1 スライド本文を 250〜400 字・根拠 4〜6 点へ・図がある面も薄くしない。`Check 16e` 強化で薄い面が 3 割超なら FAIL）

**分析の信頼性 — 結論の検証**
- 🧠 **構造化分析技法の標準搭載**（ホイヤー『インテリジェンス分析の心理学』の ACH（競合仮説分析）・リンチピン分析・ミラーイメージング点検をレポート工程に組込み。結論は対立仮説と決め手を添えた「最も矛盾の少ない解釈」として提示され、`Check 30-34` が監査可能性を機械検査）
- 🎯 **代表特許の決定的選定**（`select_representatives.py` がモジュール別の固定規則で代表特許を選定し `representative_patents.json` を生成。結論に都合の良い特許だけを引く「つまみ食い」を排除、`Check 35` で照合）
- 📖 **レポート読者体験の改善**（マップは 1 枚 1 行・全幅で、隣接する本文が図を名指しして論じる。章末まとめボックス・ワードクラウドの定量分析・3,000 字超の壁テキスト検出 — `Check 23-29 / 37`）
- 📔 **経営層向け別冊に「結論の確からしさ」**（各要点に検討した別解釈と決め手を凝縮、提言末尾に「この提言を見直すべきサイン」ボックス — `Check 18b`）
- ⚡ **スキーマ軽量化（−17%）**（規律を SSoT（正本＋ポインタ）方式に一本化し、必読ガイド群を 175.6k → 145.6k 字に削減。品質ゲートは不変更＝回帰テストで結果完全一致）
- 📋 **品質ゲートを 37 系統へ拡張**（v8 の 13 → v9 で Check 1〜37。実データ母集団のフル生成テスト 2 回で GATE PASSED）

> 各機能の詳細は下記「✨ v9.0.0 の新機能 詳解」セクションを参照。/ See the "What's New in v9.0.0 (In Detail)" section below for full descriptions.

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
- 📋 **Phase D 品質ゲート 37 系統**（定量 + 用語 + スコープ + 母集団タイプ + **分析の立場** + 設計意図 + NEBULA 戦略 + 修辞 + **反復・水増し検出** + **PPTX（情報密度含む）** + **特許番号プレースホルダ検出** + **マップ掲載・章末まとめ** + **構造化分析技法（結論検証）** + **壁テキスト** の自動検証。v9 で 22 → 37 系統に拡張）

**CAPCOM マルチエージェント対応**
- 🤝 **マルチエージェント CAPCOM**（Claude Code / Codex / Antigravity を複数選択可、パッチ自動同梱）
- 🌐 **NEBULA 3 モード対応**（通常実行 / Web 調査で補完 / 省略 — 特許情報のみの分析も成立）

**UX と耐障害性**
- 📝 **大規模ラベル編集対応**（30 クラスタ超で `st.data_editor` に自動切替、数百クラスタでも安定動作）
- 🤖 **AI ラベルサジェスト**（TSV 推奨 + JSON/Markdown/平文 4 形式自動判別 + 部分応答の追記マージ）
- 🛡️ **Janome 例外防御**（特許テキストの異常入力で分析が止まらない）

Evolution highlights from APOLLO v8:

**Analysis modules**
- 💡 **CORE classification-assist rework** (the clustering that helps design rules gained SBERT features, HDBSCAN, and automatic optimal-k analysis; Phase 1 UI rebuilt; AI insight reformatted as structured markdown. The rule-based classification and cross-tab matrix themselves are long-standing core features)
- ⚖️ **Grant-rate analysis (quantity × quality)** (applications × grant-rate quadrant separates "volume-first" from "true leaders"; cross-evaluated in ATLAS, MEGA, CORE, CREW, Saturn V)
- ✨ **AI insight for cluster-dynamics maps** (auto-interprets the 4-quadrant dynamics in Saturn V / EAGLE / NEBULA)

**Report / slide output**
- 🎨 **Report/slide-ready visuals** (white background, large fonts, high-res PNG export; "curated landscape" with top clusters only)
- 📦 **Curated landscape bundled into CAPCOM** (clean slide figures auto-included in the ZIP)

**UX & terminology**
- 🆘 **Beginner help** (~150 "?" tooltips across parameter settings, explaining effects and recommended values in plain language)
- 🗺️ **Terminology unified to "landscape overview"** (Saturn V's former "AI landscape", reworded for patent reports)
- 🖼️ **More realistic module icons**

**Foundation**
- ⚡ **Faster Mission Control** (SBERT GPU acceleration + multi-core keyword extraction)
- 📚 **CAPCOM exemplar standardization** (unified example domain, consistent quality gates / terminology)
- 🐛 **Various bug fixes** (NLP and data-integrity/statistics)

**Retrieval / Model / Slides**
- 🧠 **Selectable SBERT model** (choose in Mission Control Phase 4: ⚡fast `MiniLM` (384d, default) / 🎯high-precision `multilingual-e5-base` (768d, multilingual E5); pick fast if heavy/stalling. The chosen model is recorded in AI insights and the CAPCOM `metadata.json`)
- 🔑 **OpenALEX hardening** (API key now mandatory (since 2026-02-13; no server-default key, to avoid sharing the operator's cost on a public deploy) + command-line query syntax `TI=/AB=/TA=/TX=/FT=` + `AND/OR/NOT` + proximity `nearN/adjN` + wildcards + year-by-year retrieval + **API-key redaction in errors/logs**)
- 🖼️ **Stronger PPTX skill** (`slides_spec.md` v5.0: consulting-grade, **15 schematic ("ponchi-e") slide types** guaranteeing a visual on every analytical slide, text-only slides ≤10%)
- 🤖 **Unified drill-down auto-optimization** (Saturn V / MEGA / EAGLE drill-downs use the same HDBSCAN auto-sweep as the main map: size-adaptive target k, persistent results + DBCV, manual values gray out)
- 🎭 **Narrative stance** (decide in Phase A whose decision the report serves: self / competitor / neutral — the stance shapes naming, analytical angle and recommendation type; enforced by Check 11s)
- 📊 **PPTX information density** (per-slide body raised to 250–400 chars with 4–6 evidence points; Check 16e FAILs when >30% of content slides are thin)

**Analytical reliability — conclusion verification**
- 🧠 **Structured analytic techniques as standard** (Heuer's ACH (competing hypotheses) / linchpin analysis / mirror-imaging built into the report pipeline; conclusions ship as "the interpretation with the fewest contradictions" alongside rival hypotheses and deciding facts; machine-audited by Checks 30–34)
- 🎯 **Deterministic representative-patent selection** (`select_representatives.py` picks representatives by fixed per-module rules into `representative_patents.json`, eliminating cherry-picking; cross-checked by Check 35)
- 📖 **Reader-experience overhaul** (maps argued, not pasted — full-width, one per row, explicitly called out by the adjacent text; chapter-summary boxes; quantified word-cloud analysis; wall-of-text detection — Checks 23–29 / 37)
- 📔 **Executive edition gains "how solid is this"** (each key finding carries the alternative considered + the deciding fact; an observable "signs to revisit" box closes the recommendations — Check 18b)
- ⚡ **Schema slimming (−17%)** (single-source-of-truth discipline; the must-read guides shrank from 175.6k to 145.6k chars with the gates untouched — regression-identical results)
- 📋 **Quality gate grown to 37 checks** (13 in v8 → Checks 1–37 in v9; two full-generation field tests passed the gate)

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
- 📋 **37-check Phase-D quality gate** (quantitative + terminology + scope + population type + design intent + NEBULA strategy + rhetoric + executive-edition depth + repetition/padding detection + process-narration detection + **a PPTX machine-check** + **patent-number placeholder detection** + **map-placement & chapter-summary checks** + **structured-analytic-technique (conclusion verification) checks** + **wall-of-text detection** — grown 22 → 37 in v9)

**CAPCOM multi-agent**
- 🤝 **Multi-agent CAPCOM** (Claude Code / Codex / Antigravity selectable, patches pre-bundled in ZIP)
- 🌐 **NEBULA 3-mode handling** (execute / web-compensation / omit — patent-only analysis works too)

**UX & fault tolerance**
- 📝 **Large-scale label editor** (auto-switch to `st.data_editor` beyond 30 clusters, stable at hundreds of clusters)
- 🤖 **AI label suggestion** (TSV-preferred + 4-format auto-detect (JSON/Markdown/plain) + partial-merge mode)
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
1. Hugging Face Spaces で APOLLO v9 を開く
   Open APOLLO v9 on Hugging Face Spaces

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

#### 3. OS パッケージ（任意・日本語フォント / 高解像度 PNG 書き出し用）/ OS packages (optional)

Hugging Face Spaces では `packages.txt`（CJK フォント `fonts-noto-cjk` 等 + `chromium` / `chromium-driver`）が **apt で自動インストール** されます。ローカルで ①Matplotlib / WordCloud の日本語表示 ②スライド/レポート用の高解像度 PNG 書き出し（kaleido 経由）を使う場合は、相当パッケージを別途導入してください（macOS は日本語フォント同梱が多く不要なことが多い／Linux 例: `sudo apt install fonts-noto-cjk chromium`）。無いと日本語が豆腐になったり PNG 書き出しが失敗します。

On HF Spaces, `packages.txt` (CJK fonts + chromium) is auto-installed via apt. For **local** Japanese text rendering and high-res PNG export (kaleido), install the equivalents (e.g. Linux: `sudo apt install fonts-noto-cjk chromium`); macOS usually already has Japanese fonts.

### 基本ワークフロー / Basic Workflow

1. **Home.py** で特許データをアップロード → 前処理(SBERT + TF-IDF + メタデータ正規化)
   Upload patent data in Home.py → Preprocess (SBERT + TF-IDF + metadata normalization)
2. **CAPCOMセッション開始** → 以降の分析結果は自動的に In-Memory ストアに蓄積
   Start a CAPCOM session → Analysis results are auto-saved to the in-memory store
3. 各モジュール(ATLAS/Saturn V/MEGA/Explorer/CREW/EAGLE/NEBULA など)で分析・可視化
   Analyze & visualize across 10 modules
4. 気になるチャートを **VOYAGER** or 各モジュールの **📸 Snapshot** で収集（同じマップから複数カットも保存可。「📸 このマップで N 枚 保存済み」で枚数を確認）
   Collect key charts as snapshots — multiple shots per map are supported (a live "saved N for this map" count is shown)
5. **CAPCOM** で Mission Objective + 母集団メタ情報(任意4項目) + 使用ツール(複数可)を設定 → **ZIP ダウンロード**
   Set Mission Objective, population meta (4 optional fields), and agent selection in CAPCOM → Download ZIP
6. **選択したエージェント** で ZIP を展開 → 4フェーズで戦略レポート生成(品質ゲート + 用語統一検証付き)
   Extract ZIP in the selected agent → Generate reports through 4 phases with auto quality & terminology gates

---

## 🧩 10 の分析モジュール / 10 Analysis Modules

APOLLO v9 は 10 モジュールで特許データを多角的に分析します。

APOLLO v9 analyzes patent data across 10 specialized modules.

| # | モジュール / Module | 概要 / Description |
|---|----------|---------|
| 1 | 🌍 ATLAS | 基本統計 + 多様性指標(HHI + Entropy + Gini) — Basic stats + 3 diversity indices |
| 2 | 💡 CORE | AND/OR/NEAR/ADJ 論理式分類 + クロス集計マトリクス（ヒートマップ／バブル切替・セルクリックで該当特許へ）— Rule-based classification + cross-tab matrix (heatmap/bubble toggle, click a cell to reach its patents) |
| 3 | 🚀 Saturn V | 俯瞰図分析 + ノイズ分析 + クラスタ動態マップ — Landscape overview + noise analysis + cluster dynamics map |
| 4 | 📈 MEGA | PULSE 4象限動態分析（CAGR×活動量）+ TELESCOPE ドリルダウン — PULSE quadrant analysis (CAGR × activity) + TELESCOPE drill-down |
| 5 | 🧭 Explorer | 共起ネットワーク + 急上昇キーワード + トルネード競合比較 — Co-occurrence + trending keywords + tornado comparison |
| 6 | 🔗 CREW | 発明者・出願人ネットワーク + 媒介中心性 + コミュニティ検出 — Inventor/applicant networks + betweenness + community detection |
| 7 | 🦅 EAGLE | 投げ縄ツールで手動クラスタ + クラスタ動態マップ — Lasso-based manual clusters + cluster dynamics |
| 8 | 📝 VOYAGER | スナップショット収集 + Mission Objective 設定 + Markdown レポート骨格生成 — Snapshot collection + Mission Objective + Markdown skeleton |
| 9 | 🌌 **NEBULA** | OpenALEX API 統合（API キー対応）+ Hype Cycle + **学術ランドスケープ**（Saturn V デザイン統一 + クラスタラベル CSV DL）+ **論文種別 10 種の複数選択** + **検索結果 CSV ダウンロード** + **年別取得モード / 要約ありフィルタ / 英語のみフィルタ** + **コマンドライン検索式（TI=/AB=/TA= + near/adj + ワイルドカード）** + **大規模ラベル編集対応** — OpenALEX (API-key) + Hype Cycle + academic landscape + 10-type multi-select + CSV + year-by-year mode + abstract-only / English-only filters + command-line query syntax + large-scale label editor |
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

> 💡 **本体レポートの品質は Claude Code が最も安定**します。3 ツール共通の品質ゲートが「下限」を保証しますが、分析の深さ・論証の質は機械では測りきれないためです。Codex CLI / Antigravity IDE も利用可能で、その場合は推論「高」（xhigh は避ける）・1 フェーズ 1 セッション・引き継ぎ日誌の運用を推奨します（詳細は FAQ「Codex CLI / Antigravity で品質はどれくらい保たれる?」）。
>
> 💡 **Claude Code is the most consistent for the report body.** The shared quality gates guarantee a floor across all 3 tools, but analytical depth and argument quality can't be fully measured mechanically. Codex CLI / Antigravity IDE also work — use reasoning "high" (avoid xhigh), one-phase-per-session, and the carryover diary (see the FAQ).

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
│   │   ├── structured_techniques.md   # 構造化分析技法（ACH 競合仮説分析・リンチピン・ミラーイメージング＝結論の検証）
│   │   ├── cross_module.md / report_structure.md / quality_checklist.md ...
│   ├── references/ exemplars/ templates/
│   └── scripts/             # phase_c_gate.sh / phase_d_gate.sh（Check 1〜37: 定量・用語・スコープ・母集団タイプ・分析の立場・設計意図・NEBULA 戦略・修辞・反復水増し・PPTX・特許番号プレースホルダ・マップ掲載・章末まとめ・構造化分析技法・壁テキスト）
│                            # + select_representatives.py（代表特許の決定的選定 → reports/representative_patents.json）
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
    ├── .agents/skills/apollo-capcom/   # ↑を現行ツールの自動探索パス（複数形 .agents）へ複製＝dual-emit（スキル発見性確保）
    ├── .agents/workflows/              # 同上（ワークフローの複製）
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

### レポート生成モード / Report Generation Mode — 4フェーズ + 37 品質ゲート + 用語統一

| Phase | タスク | 絶対遵守ゲート |
|-------|-------|--------------|
| **A** | ミッション理解 + データ精読 + `terminology.md` 読了 + `population_meta` 確認 | 用語統一ルール + 母集団メタ 4 項目 + 別冊確認 STOP-GATE / **STOP-GATE A（query_logic 構造化読解）** + **query_intent 3 点整理** + **サブクエスチョン化 STOP-GATE** + **STOP-GATE B（意図↔論理整合性）** + **STOP-GATE C（データ逆読み + 母集団タイプ判定 + 分析の立場確定）** + **STOP-GATE D（NEBULA 戦略判定）** + `_phase_a_decisions.json` 永続化 |
| **B** | エビデンス精読 + クロス分析 + Web 調査 | 13 種クロスパターンから **5 つ以上**選定 + Web 調査テーマのユーザー確認（NEBULA 補完モード時は **4 カテゴリ必須カバー**: 市場規模・政策・学術動向・主要企業動向） |
| **C** | 代表特許の決定的選定（`select_representatives.py` を冒頭で 1 回実行）+ モジュール別 Deep Dive（7 モジュール） | `bash capcom_schema/scripts/phase_c_gate.sh` で内容量（非空白文字数）を自動検証 / **Step 0 は `nebula_strategy` で分岐**（execute / web_compensation / omit） |
| **D** | 統合レポート（結論は ACH・リンチピンで検証）+ 品質検証 + 用語統一検証 + 別冊生成（フラグ ON 時） | `bash capcom_schema/scripts/phase_d_gate.sh` で **Check 1〜37（37 系統）** を自動実施（定量 + 用語 + スコープ + 母集団タイプ + **分析の立場** + 設計意図 + NEBULA 戦略 + 修辞 + **反復・水増し検出** + **PPTX** + **特許番号プレースホルダ** + **マップ掲載・章末まとめ** + **構造化分析技法** + **壁テキスト**） |

**Phase D gate の Check（`phase_d_gate.sh`、計 37 系統 = Check 1〜37 + サブチェック）**:
- **Check 1-9** — **内容量（非空白文字数 45000 字以上・行数は参考）** / 代表特許 / 4 層モデル / **クロス分析分量（120 行以上＝5 パターン）** / snapshot / Web 出所 / 仮説検証 / 用語統一（内部識別子＋**工程ナレーション節「後続分析への接続」等の検出 = Check 8e**）/ J-PlatPat 不正補完
- **Check 10** — スコープ限定ルール（「本母集団」vs「業界全体」の誤読防止、無限化語と限定語の比率判定）
- **Check 11** — 母集団タイプ別の不適切表現検出（タイプ B/C/D で「市場集中」「業界シェア」等を禁止、タイプ C で出願人 HHI の言及を禁止）。**Check 11s** — 分析の立場（`narrative_stance`）の一貫性: 立場が `neutral`（中立/投資家・アナリスト）なのに対象企業を一人称「当社」と呼ぶと **FAIL**、`competitor` は WARN（呼称の取り違え防止。`terminology.md §6-2-B`）
- **Check 12** — 設計意図の一貫性（意図参照語カウント / 問い/答え形式の禁止 / サブクエスチョンキーワードの結論章カバレッジ）
- **Check 13** — NEBULA 戦略検証（3 モード別: execute なら NEBULA 章の存在、web_compensation なら 4 カテゴリ + `#footnote` 4 件以上、omit なら NEBULA 章なし + 特許のみ対象の注記）
- **Check 14** — 4 層モデルのラベル（「（事実）」「（解釈）」「（So what）」「経営的含意:」等）の本文露出検出（FAIL。4 層は思考の枠組みであり地の文に溶かす）
- **Check 15** — 国籍トートロジー（日本出願母集団で「日本が牽引/主導」＝循環論法）の検出（WARN）
- **Check 16** — **PPTX（`presentation.pptx`）の機械チェック 12 項目**（存在時のみ・python-pptx で抽出）。出所にモジュール名 / 内部識別子 / プレースホルダ / フッター "APOLLO CAPCOM" = **FAIL**、**情報量の薄いコンテンツ面（図あり<80字 / 図なし<150字）が 3 割超 = FAIL**（章扉は番号で識別し除外）、図ありスライド比率 / 同一タイプ 3 枚連続 / 事業ファクトの出所ミス / 数値不一致 / 「～」副題 / 過剰修辞 = **WARN**
- **Check 17** — 過剰修辞・気取った比喩（越境者・狼煙・地殻変動・旗手・苗床 等）のレポート本文検出（WARN）
- **Check 18** — 別冊（経営層向け要約版 `report_executive.typ`）の充実度（存在時のみ）。表紙＋数段落だけの薄い別冊（120 行未満）は **FAIL**。`executive_summary_guide.md` の 8-12 ページ基準を満たす
- **Check 19** — **反復・水増し検出**（`report.typ` / `report_executive.typ`）。同一文の反復 / 回転名詞だけ変えた定型文の量産 / 「○○観点 1,2,3…」式の連番定型見出し / 重複文比率 / **文末22字の重複（接続句だけ変えた回避）** を python で機械検出し、閾値超は **FAIL**。さらに **Check 19a** で本文を生成する Python スクリプト（`reports/generate_*.py` が `.typ` を書き出す）を検出し **FAIL**。「最低 N 行」を同一文のコピペループやスクリプト生成で埋める水増し（literal なモデルで起きやすく、ゲートを読んで回避を試みる）を根治する。行数(`wc -l`)は1文1行で水増しできるため、内容量は**非空白文字数**で判定（Check 1 / phase_c_gate）
- **Check 20** — **スナップショット網羅**（WARN）。`snapshots/` にあるPNGのうち `report.typ` で参照されていないマップを検出して警告。ユーザーが撮った全マップを分析・掲載する原則（`map_reading.md`）の機械チェック
- **Check 21** — **特許番号のプレースホルダ／捏造検出**（FAIL）。代表特許番号に「特開2023-XXXXXX」等のプレースホルダや、出願年から推測した捏造番号が残っていると失敗（`patent_citation.md §5`）
- **Check 22** — **見出し・目次の数値強調漏れ予防**（WARN）。見出しの「数値＋単位」や裸の `#outline` が `no-num-hl` で打ち消されず色が浮く状態を検出（`report_style.typ` の U+2060 方式と対）
- **Check 23-29** — **マップ掲載と読者体験**。見出し直後のマップ羅列（23・WARN）/ **Explorer 章の「成長率×中心性の象限図」＝APOLLO に存在しない図の捏造（24・FAIL）** / 章末まとめ `#chapter-summary` の有無・位置（25/26・WARN）/ ワードクラウドの「語（N回）」定量分析の痕跡（27・WARN）/ grid での 2 枚横並べ・縮小掲載（28・WARN）/ 分析本文が隣接しない「貼るだけの図」・図を名指ししない隣接段落・分析が薄いドリルダウン図（29/29b/29c・WARN）を検出（`map_reading.md` / `common_framework.md §4`）
- **Check 30-34** — **構造化分析技法（ホイヤー流の結論検証）**。ACH（競合仮説分析）の `#competing-hypotheses` / `#ach-matrix` がゼロ（30・**FAIL**）/ 弁別材料に出所裏付けなし（30b・WARN）/ リンチピン `#linchpin`（結論の要の前提と崩れる条件）がゼロ（31・**FAIL**）/ 崩れる条件が観測不能な空文（31b・WARN）/ ミラーイメージング点検の欠落（32・WARN）/ 全セル○の非弁別マトリクス＝体裁だけの表（33・WARN）/ 提言が検証結論と接続していない「宙に浮いた提言」（34・WARN）を検査（`structured_techniques.md`。旧スキーマ世代のセッションは自動スキップ＝後方互換）
- **Check 35** — **引用特許の出所**（WARN）。ミクロ分析の特許番号が `select_representatives.py` の決定的選定リスト（`representative_patents.json`）由来かを照合（リスト外引用＝つまみ食いの疑い）
- **Check 36** — **裏付けのない断定**（WARN）。「最大」「唯一」「が原因で」等の断定文で、同文・隣接文に数値・特許番号・`#footnote` の裏付けが無いものを「事実の装い」候補として列挙（誤検出前提の注意誘導＝最終判断は人間）
- **Check 37** — **壁テキスト**（WARN）。表・図・ボックス・箇条書き・小見出しなしで地の文が 3,000 字超連続する節を検出（統合・サマリ節は「表 → 散文」の順で分節）

これらの bash スクリプトは 3 エージェント共通で同じ客観判定を提供します。**主観的な「実質 OK」判断で量的基準を上書きできません**。

The bash scripts provide identical objective pass/fail judgments across all 3 agents. **Subjective "good enough" cannot override quantitative criteria.**

---

## ✨ v9.0.0 の新機能 詳解 / What's New in v9.0.0 (In Detail)

> v8 → v9 で追加・刷新された機能の詳細です。v8 で確立した CAPCOM・母集団設計・品質ゲート（後述の「主な機能」）はすべて v9 に継承されています。
> Detailed look at what v9 adds and renews. Everything established in v8 (CAPCOM, population design, quality gates — see "Main Features" below) carries over into v9.

### V1. CORE 分類設計アシストの刷新 / CORE Classification-assist Rework
- **位置づけ（誤解しやすい点）**: ルールベース分類（AND/OR/NEAR/ADJ 論理式で「技術 / 課題 / 解決手段」に振り分け）と技術×課題クロス集計マトリクス（ヒートマップ／バブル・ホワイトスペース・セルクリックで特許到達）は **CORE の従来からの中核機能**であり v9 で新規追加されたものではない。v9 で刷新したのは、その「**分類ルールの設計を支援するクラスタリング・アシスト**」部分 / The rule-based classification and tech×problem cross-tab matrix are **long-standing core features** (not new in v9); v9 reworks the **clustering assist that helps you design the rules**
- **クラスタリングの高度化**: 分類設計の叩き台を作るクラスタリングに、特徴量 SBERT と手法 HDBSCAN を追加（従来は KMeans のみ）。HDBSCAN は UMAP で低次元化してから適用し、失敗時は KMeans に自動フォールバック / Added SBERT features and HDBSCAN to the assist clustering (previously KMeans only); HDBSCAN runs after UMAP reduction with automatic KMeans fallback
- **最適 k 自動分析 + Phase 1 UI 再構築**: スコア曲線と推奨 k を提示し、スライダーが推奨値に自動追従。特徴量×手法の選択に応じて、AI に分類カテゴリ設計を依頼するプロンプトを排他的に生成 / Optimal-k analysis (score curve + recommended k, slider auto-follows); the AI design prompt reflects the selected feature×method exclusively
- **「その他（未分類）」再分析の高度化**: どのルールにも該当しない特許だけの再クラスタリングにも SBERT / HDBSCAN / k 自動を追加 / The "Other (unclassified)" re-analysis also gained SBERT / HDBSCAN / auto-k
- **AI インサイトの markdown 構造化**: マトリクスを表に、分類ルールを巨大 JSON ではなく軸別の構造化 markdown に整形して読みやすく / AI insight reformatted: matrix as a table, classification rules as structured markdown instead of a giant JSON blob

### V2. 権利化率分析（量×質）/ Grant-rate Analysis (Quantity × Quality)
- **出願数 × 権利化率の 4 象限**: 「量で先行する件数主義型」「量・質を両立する真の強者」「少数精鋭で権利化率が高い要警戒プレイヤー」を識別する / Applications × grant-rate quadrant separates volume-first players, true leaders, and high-grant-rate specialists
- **権利化成功の定義**: 権利継続 ＋ 失効（満了・放棄）＝「一度でも登録された出願」。失効は権利化の失敗ではないためマイナス計上しない。出願数・権利化率それぞれの**中央値（表示中の主体で算出）**で 4 象限に区切る（十字が常に表示点群の中央に来る）/ Grant success = ever-registered (active + lapsed); quadrants split by the median of each axis, computed over the displayed players so the crosshair always centers the shown cloud
- **量×質は「一つの観点」**: 権利化率の 4 象限分析は **ATLAS のネイティブ機能**。他モジュール（MEGA/CORE/CREW 等）にアプリ内で権利状況を重ねる UI はなく（Saturn V は俯瞰図の色分け基準に「ステータス」を選べるのみ）、量×質は **分析者・レポート側で ATLAS の権利化率を併読して読み解く観点** として扱う。**必須の主軸ではなく任意の補助観点**（権利化率を全分析の中心に据えず、「量だけの強さ」と「質を伴う強さ」を区別したい場面で使う）/ The grant-rate quadrant is **native to ATLAS only**; other modules have no in-app rights-status overlay (Saturn V can merely color its landscape by a status toggle), so quantity×quality is a lens the analyst combines by reading ATLAS's grant-rate alongside — **one lens among many, not a mandatory theme**
- **経営言語への翻訳 + 暫定値の注意**: 「量は多いが通っていない」等を投資判断の言葉に翻訳。審査係属中比率の高い新興出願人は暫定値として登録到達率の推移で評価 / Translated into management language; new entrants flagged as provisional and tracked over time

### V3. クラスタ動態マップの AI インサイト / AI Insight for Cluster-dynamics Maps
- **4 象限の自動解釈**: 累積件数（規模）× CAGR（成長）で「成長リーダー / 新興 / 成熟 / ニッチ・衰退」を判定し、注力・育成・刈り取り・撤退の戦略示唆を自動生成 / Auto-interprets the cumulative-count × CAGR quadrants into invest / grow / harvest / exit implications
- **3 モジュール共通**: Saturn V / EAGLE / NEBULA（学術）の動態マップに横展開。スナップショットにも解釈の文脈を同梱し、VOYAGER / CAPCOM へ連携する / Shared across Saturn V / EAGLE / NEBULA; the interpretation context is bundled into snapshots for VOYAGER/CAPCOM

### V4. レポート/スライド用ビジュアル + 整理版の CAPCOM 同梱 / Report/Slide-ready Visuals + Curated Landscape in CAPCOM
- **クリーン高解像度 PNG 書き出し**: 白背景・大きな文字の 16:9 / 4:3 / 正方形 PNG（kaleido, scale=2）。要点・見出し・出典はレポート/スライド生成側（CAPCOM）で付与する運用に統一し、二重表示を防ぐ / Clean high-res PNG export (white bg, large fonts); captions/headings/sources are added downstream by CAPCOM to avoid duplication
- **整理版ランドスケープ**: 件数上位クラスタだけを大きなラベル＋件数比例バブルで示し、全クラスタ密ラベルの重なりを回避。領域（凸包）/ 密度 / 散布の 3 表示モード / "Curated landscape": top-N clusters with large labels and size-proportional bubbles (hull / density / scatter modes)
- **CAPCOM 自動同梱**: 整理版を表示中に CAPCOM がアクティブなら、クリーン PNG を ZIP に自動同梱する（`ls snapshots/` で発見される命名）/ Auto-bundled into the CAPCOM ZIP when active so the slide-ready figure reaches the report agent

### V5. 初心者向けヘルプ（「？」ツールチップ 約 150 箇所）/ Beginner Help (~150 "?" tooltips)
- **全パラメータ設定に解説**: 各スライダー／数値入力／選択に「？」アイコンを付け、「何を決めるか・大きく/小さくした時の効果・推奨値」を平易に解説（最小クラスタサイズ、Jaccard 閾値、CAGR 期間、特徴量 TF-IDF vs SBERT、手法 KMeans vs HDBSCAN 等）/ Every parameter widget gets a "?" tooltip explaining what it controls, the effect of changing it, and recommended values
- **用語を統一**: 共通パラメータは全モジュールで同一文言。Home のカラムマッピングも各列の用途を解説し、初学者のつまずきを軽減 / Consistent wording across modules; Home column-mapping is also explained

### V6. 用語「俯瞰図分析」へ統一 / Terminology Unified to "Landscape Overview"
- **Saturn V の呼称変更**: 特許レポートで違和感のあった「AI ランドスケープ」を「俯瞰図分析」に統一。`terminology.md`（CAPCOM 用語の正式ソース）を起点に exemplar・分析ガイド・UI・ドキュメントへ反映。もう一方の正式呼称「Saturn V TELESCOPE 分析」は併存 / Renamed Saturn V's "AI landscape" to "landscape overview" for patent reports, propagated from `terminology.md` to exemplars, guides, UI and docs

### V7. Mission Control 高速化 / Faster Mission Control
- **SBERT の GPU 活用**: CUDA > MPS > CPU を自動検出（`APOLLO_FORCE_CPU=1` で CPU 強制）。配布先の Apple Silicon / NVIDIA 機で自動的に高速化 / SBERT auto-detects CUDA > MPS > CPU (`APOLLO_FORCE_CPU=1` to force CPU)
- **キーワード抽出のマルチコア並列**: joblib によるプロセス並列（Janome は GIL 律速のためプロセス並列が有効）。小データや少コア環境では逐次フォールバックで逆効果を回避 / Multi-core keyword extraction via joblib, with a sequential fallback for small data / low-core environments

### V8. CAPCOM exemplar の標準化 / CAPCOM Exemplar Standardization
- **例題ドメインの統一**: 全 7 モジュールのお手本レポート（exemplar）を「水素貯蔵」に統一し、題材の混在を解消 / Unified the running example domain to hydrogen storage across all 7 module exemplars
- **量×質と v9 概念の反映**: 権利化率のクロス参照、クラスタ動態・ノイズ（萌芽技術）の解釈、俯瞰図分析の用語を織り込み。3 エージェント（Claude Code / Codex / Antigravity）共通の `capcom_schema/` に集約しているため、どのツールでも同じ品質基準が効く / Grant-rate cross-references and v9 concepts woven in; shared via `capcom_schema/` so all 3 agents get the same standard

### V9. 安定性向上（全体レビューによるバグ修正）/ Stability (Bug Fixes from a Full Review)
- コードベース全体をレビューし、実害のあるバグを修正: クラスタ動態スライダーの年幅が狭いデータでのクラッシュ、EAGLE 投げ縄でのクラスタ作成、MEGA の CAPCOM 代表特許出力、NEBULA 学術代表論文の索引整合、ATLAS の IPC 集計の分裂、OpenALEX の重複除去、**OpenALEX コマンドライン検索式のワイルドカード（`electrol*` 等）が候補クエリに混入して HTTP 400 → 検索全体が失敗していた問題（候補からワイルドカード語を除外しローカル厳密照合に委譲、4xx は即時に分かりやすく失敗）**、ほか / Fixed real bugs found in a full-codebase review (dynamics-slider crash on narrow-year data, EAGLE lasso cluster creation, MEGA CAPCOM representatives, NEBULA academic indexing, ATLAS IPC aggregation, OpenALEX dedup, **OpenALEX command-line wildcards (e.g. `electrol*`) leaking into candidate queries and causing an HTTP 400 that failed the whole search — now excluded from candidate retrieval and enforced via local matching; 4xx now fails fast with a clear message**, and more)

### V10. クラスタ数の自動最適化（DBCV 掃引）/ Cluster-count Auto-optimization (DBCV Sweep)
- **「クラスタ数が多すぎ/少なすぎ」を自動で適正化**: `min_cluster_size` × `min_samples` の 2 パラメータを母集団規模に応じて自動掃引し、**DBCV（密度ベースのクラスタ妥当性指標）を主基準**に最良の組み合わせを選ぶ。目標クラスタ数を指定すると、その近傍を優先して提示する / Auto-sweeps `min_cluster_size` × `min_samples` and picks the best combo by **DBCV** (density-based cluster-validity), nudged toward your target cluster count
- **2 パス探索で高速化**: ①各候補のクラスタ数だけ先に取得 → ②目標範囲に入る候補のみ DBCV を計算、という 2 段構えで無駄な重い計算を省く / Two-pass search (count first, DBCV only for in-range candidates) to avoid wasted heavy computation
- **対象 3 モジュール**: Saturn V / NEBULA（学術）/ CORE に「🤖 自動最適化」チェックボックス＋目標クラスタ数入力を追加（EAGLE は投げ縄＝手動選択のため対象外）/ Added to Saturn V / NEBULA (academic) / CORE as a "🤖 auto-optimize" checkbox + target count (EAGLE is manual-lasso, excluded)
- **DBCV の読み方ガイド**: 「？」popover で目安を解説（おおむね 0.3 前後＝良好 / 0.15 以上＝実用 / 0 以下＝要再検討。データ間で比較する相対指標）/ A "?" popover explains DBCV (≈0.3 good / ≥0.15 usable / ≤0 reconsider; a relative metric)

### V11. 文埋め込みモデルの選択（fast / quality）/ Sentence-embedding Model Selection
- **2 モデルを切替**: Mission Control（フェーズ 4）で **fast = MiniLM（384 次元・軽量）** と **quality = multilingual-e5-base（768 次元・高品質）** を選択できる。既定は fast（配布先の負荷対策）。E5 系で必要な `passage:` 接頭辞は内部で自動付与（patiroha は非改変）/ Pick **fast (MiniLM, 384d)** or **quality (multilingual-e5-base, 768d)** in Mission Control; default fast; the E5 `passage:` prefix is auto-applied
- **分析環境の記録**: 選択したモデル名・次元数は CAPCOM `metadata.json` の `analysis_environment` に記録され、レポート付録「分析条件」で再現性を担保。NEBULA 学術側も同じモデル選択に追従 / Selected model/dim recorded in CAPCOM metadata for reproducibility; NEBULA academic follows the same choice

### V12. MEGA 動態の 3 軸対応（出願人 / IPC / F ターム）/ MEGA 3-axis Dynamics
- **3 軸で 4 象限を実行**: PULSE（CAGR × 活動量）を **出願人・IPC メイングループ・F ターム（列がある場合）** の 3 軸で分析できる。各軸が独立した動態マップを生成 / Run the PULSE quadrants on **applicant / IPC main-group / F-term** axes
- **軸別に保存して取り違え防止**: 軸ごとに `mega_momentum_<軸>.json`（出願人=`applicant`・IPC=`ipc`・Fターム=`fterm`）で保存するため、複数軸を続けて実行しても上書きされない。スナップショットのタイトルにも軸名が入り、レポートで各軸を個別に分析できる / Saved per-axis (`mega_momentum_<axis>.json`) so multiple axes never overwrite; snapshot titles carry the axis name for per-axis analysis

### V13. CAPCOM レポート/スライドの品質強化 / CAPCOM Report & Slide Quality Overhaul
- **レポートを土台にしたスライド生成**: PPTX は完成レポート（`report.typ`）の論証を凝縮して図と文で再構成する設計に統一（evidence の断片を寄せ集めない）。1 スライド＝「主張→根拠（数値）→示唆」の最小ロジック、適量の本文、レポートの章順に沿う物語アーク（`slides_spec.md §0.9`）/ Slides are distilled from the finished report (claim→evidence→implication), not harvested from raw evidence
- **平易・正確な記述（過剰修辞の禁止）**: 文芸的な比喩・扇情的な修辞（越境者・狼煙・地殻変動 等）を排し、事実と数値で語る（レポート/別冊/PPTX 共通・`terminology.md §6.5`）。タイトルの波ダッシュ「～」副題も廃止し、言い切り or 全角ダッシュ「—」に / Plain, precise prose — no purple metaphors; the "～" subtitle style is dropped
- **出所の正確化**: 分析モジュール名を「出所」にしない。特許データ由来はデータセット名、事業・市場ファクトは Web 実出所（付録 C）を明記 / Sources cite the data/Web origin, not the in-house analysis module
- **コンサル品質の作図ヘルパー**: 軸付き 2×2 マトリクス・横向き矢羽根フロー・ドーナツ・Issue Tree（ロジックツリー）を python-pptx 実装で追加（破損要因のコネクタは不使用）。フォントを **Noto Sans JP** に統一し、見出し=Black / サブメッセージ=Medium / 本文=Regular / 出典=Light の**多段ウェイト**を活用。サブメッセージ枠は上下中央寄せ＋高さ安定化、箱の充填（でかい四角に少しの文を禁止）/ Added pictogram helpers (matrix/flow/donut/issue-tree); Noto Sans JP with a multi-weight hierarchy
- **数値・固有名の一貫性**: 同一の数値（件数・%・年）をデッキ全体・本編・別冊で統一 / Numbers/entities kept consistent across deck, report and executive edition

### V14. CAPCOM 品質ゲートの拡張 + データ保存の堅牢化 / Expanded Gates + Robust Data Saving
- **ゲートを 13 → 20 系統に拡張**（後の V20 で 22 系統、V25〜V27 でさらに 37 系統へ）: 4 層ラベルの本文露出（Check 14）・国籍トートロジー（Check 15）・**PPTX 12 項目の機械チェック（Check 16）**・過剰修辞（Check 17）・**別冊の充実度（Check 18）**・**反復・水増し検出（Check 19）**を追加。Check 16 は python-pptx で実 `.pptx` を解析し、出所のモジュール名/ファイルパス混入・薄いスライド・「～」副題・数値不一致・事業ファクトの出所ミス等を自動検出（存在時のみ）。Check 19 は report.typ/別冊の同一文反復・回転名詞テンプレの量産・連番定型見出しを機械検出し、「最低 N 行」をコピペで埋める水増し（literal なモデルで頻発）を **FAIL** で根治 ＋ **スナップショット網羅 WARN（Check 20）**。さらに内容量を行数でなく**文字数**で判定（1文1行の水増し対策） / Gates grown 13→20 (later 22 via V20, then 37 via V25–V27), incl. a 12-point `.pptx` check, a repetition/padding detector, char-based volume, and a snapshot-coverage warning
- **分析の深さを強化**: AI インサイト読了を**最低 8 件**（主要モジュール各 1 件以上）に、クロスモジュール分析を**最低 5 パターン**（gate も 120 行に）へ引き上げ / Deeper analysis: ≥8 AI insights, ≥5 cross-module patterns
- **「選択で上書き」を解消**: 複数の軸/指標/対象を切り替えながら分析しても取りこぼさないよう、**MEGA は軸別**（出願人/IPC/Fターム）、**CREW は指標別トップ**（`top_by_metric`）、**ドリルダウンは対象別ファイル名**で保存（Explorer は従来から種別別）/ Multi-selection outputs saved distinctly so nothing is lost
- **同じマップから複数スナップショットを収集**: 1 枚保存すると、その下に **「➕ 現在の表示を別カットとして追加」ボタン**が出て、表示オプション（上位N・配色・軸・対象など）を変えた別バージョンを連番（`__sN`）で**何枚でも上書きせず追加**できる。ドリルダウンの各クラスタ・MEGA の各軸のように**ビューを切り替えた場合も別カット**として残る。**CREW はセレクタの性質で分離**し、分析モード（発明者ネットワーク／企業アライアンス）は別グラフとして独立保存（新規「Save Snapshot」）、色分け基準（媒介中心性・技術ブローカー等）の変更は同一グラフの別カット（`__sN`）として積む。ボタン直下に「📸 このマップで N 枚 保存済み」と現在の保存枚数を表示。集めた全マップは CAPCOM ZIP に同梱され、レポートで網羅的に分析・掲載される（未参照マップは Check 20 が警告）/ After the first save, a **“➕ add the current view as another cut”** button lets you append as many variations as you like (changing top-N / colors / axis) under sequential `__sN` ids without overwriting; switching views (per cluster / axis) also yields separate cuts. **CREW separates its two selectors by nature**: the analysis mode (inventor vs. corporate network) is saved as a distinct base graph, while changing the node-color metric appends a cut of the same graph. A live saved-count is shown, and every captured map is bundled into CAPCOM (Check 20 flags any unreferenced map)

### V15. レポート可読性「走査層」+ 工程ナレーション節の検出 + セッション・チェックポイント / Report Readability "Scan Layer" + Process-narration Detection + Session Checkpoints
- **可読性の「走査層」（地の文を減らさず読みやすく）**: 密な散文（精読層）はそのまま残し、その上に拾い読みできる層を薄く重ねる設計。①各分析セクション冒頭に結論を 1〜2 行で先出しする**要点ストリップ**（`#point-lead`）②件数・%・倍などの**数値＋単位をテンプレートが自動で強調**（年号・特許番号は対象外＝過剰強調を防ぐ）③段落余白・見出しバー④長い列挙の表化。「要点だけ書いて散文を削る」薄化は内容量（非空白文字数）ゲートで FAIL（`report_style.typ`・`deep_dive_guide.md`）/ A scannable layer (key-point strips, auto-highlighted figures, breathing room, tabulated lists) layered over full prose — readability without thinning
- **工程ナレーション節の検出（Check 8e）**: 「後続分析への接続」「次章への申し送り」のような、他章の ToDo を並べただけの**無意味なメタ節＝水増し**を `phase_d_gate.sh` / `phase_c_gate.sh` で自動 FAIL。章間連携は専用章「クロスモジュール統合分析」が担い、他章参照は「〜で確認された〈事実〉」の過去形に限る / Detects meaningless "to be covered in the next chapter" filler sections as padding
- **セッション・チェックポイント（Codex / Antigravity）**: コンテキスト枯渇の予防として、各フェーズの区切り（Phase A/B 完了・**Phase C の各モジュール完了ごと**・Phase D 着手前）で、ゲート通過＋フェーズ間引き継ぎ日誌（`_carryover.md`）更新の後に「新セッションに切り替えますか？」と能動的に提案して一旦停止。切替時は新セッションが日誌から自動再開する / Proactive session-switch checkpoints at clean phase boundaries to prevent context exhaustion

### V16. VOYAGER レポート品質の強化（量×質の反映 + 証拠供給の拡大）/ VOYAGER Report-quality Boost (Quantity×Quality + Wider Evidence Feed)
- **権利化率（量×質）を VOYAGER に配線**: 従来 VOYAGER のプロンプトは権利化率に一切言及せず、`atlas_grant_rate.json` のデータも Phase 0 の抽出対象外で Gemini に届いていなかった。ATLAS 分析ガイド・モジュール別分析（Analyst）・クロスモジュール分析（権利化率×出願数／×成長率／×クラスタの 3 パターン追加）・統合レポート構成（標準/詳細/市場の 3 モード）・必須引用チャートに権利化率マップを追加し、「出願数が多い＝強い」と短絡せず**量産型と少数精鋭型を峻別**させる。Phase 0 のデータ抽出に権利化率の中央値・主体別内訳（`median_grant_rate` / `groups`）と共願ネットワーク指標（`top_by_metric`）を追加 / Grant-rate (quantity×quality) wired into the VOYAGER prompts **and** the Phase-0 data extraction (both previously omitted), so reports stop equating volume with strength
- **証拠供給の拡大（100 万トークン窓を活用）**: VOYAGER の生成モデル `gemini-2.5-flash` は約 100 万トークンの入力窓を持つのに、ボトルネックはトークンではなく**コード側の控えめな打ち切り**で全モジュールの分析やリスト系データが途中で捨てられていた点だった。スナップショット証拠（5k→16k 字）・モジュール別分析（12k→40k 字＝**8 モジュール全ての分析が統合段階に届く**・従来は 3 モジュール分で打ち切り）・CAPCOM データ要約（8k→40k 字）・個別フィールド（800→2,500 字＝権利化率の全主体一覧や CREW 指標が数社で切れない）へ拡大。最大フェーズでも入力窓の約 1 割に収まる / The bottleneck was conservative in-code truncations, not tokens; loosened them to feed far richer evidence to Gemini (all 8 modules' analyses now reach synthesis), still ~10% of the 1M-token window
- **CREW 指標の明文化**: 媒介中心性（橋渡しの結節点）・技術ブローカー（越境者）・**生産性スコア**（出願数/(次数+1)＝少人数連携の多産度・従来欠落）・急上昇スコアの定義を分析ガイドに明記 / CREW metrics — incl. the previously-missing productivity score — spelled out in the analysis guide
- **位置づけ**: VOYAGER は引き続き**アプリ内の高速な骨格レポート**（本格レポートは CAPCOM）。モデルは速度・コスト最優先で `gemini-2.5-flash` を維持し、品質は主にデータ供給で底上げ / VOYAGER stays the fast in-app skeleton (full reports remain CAPCOM's job); model kept as flash for speed/cost, quality raised mainly via richer data

### V17. OpenALEX 検索の刷新（コマンドライン式デフォルト + AI 作成補助 + 中断 + 構文プレビュー）/ OpenALEX Search Overhaul (Command-line Default + AI Authoring Aid + Interrupt + Syntax Preview)
- **検索モードの既定をコマンドライン検索式に**: `TI=/AB=/TA=/TX=/FT=` + `AND/OR/NOT` + 近傍 `nearN/adjN` + ワイルドカードのフィールド指定モードを既定にし、ラジオの並びも左端に配置。旧「標準検索」は「キーワード検索」に改称 / Command-line query syntax is now the default mode (placed first); the former "standard search" is renamed "keyword search"
- **AI にクエリ作成を依頼するプロンプトを表示**: 検索テーマを入力すると、文法を厳守して貼り付け可能な検索式を AI に出力させるプロンプトを生成（コピー用）。OR 選択肢の掛け算で候補クエリ＝API 消費が増えるため「式を短く保つ（候補 6 件以内目安）」制約を内蔵。論文種別の除外は検索式ではなく種別フィルタで行う旨も明記 / Generates a copy-paste prompt that asks an AI to author a grammar-compliant query; the prompt embeds a "keep it short (≤6 candidate queries)" constraint since ORed alternatives multiply into more candidate queries (= API spend), and notes that publication-type exclusion belongs in the type filter, not the query
- **構文チェック / OpenAlex 候補式プレビュー**: 入力式を `compile_command_query` で解析し、構文の可否・展開後の式・実際に OpenAlex へ投げる候補検索式（件数・scope）・near/adj 等のローカル厳密照合の有無を表示 / Parses the input with `compile_command_query` and previews validity, the expanded form, the actual candidate queries sent to OpenAlex (count & scope), and which terms are enforced by local strict matching
- **「🛑 検索を終了」ボタン**: 実行中の OpenALEX 検索を中断（次の進捗更新で再実行を割り込み）。複数クエリの OR 検索にも進捗表示を追加 / A "🛑 Stop search" button interrupts a running OpenALEX search (at the next progress tick); progress is also shown for multi-query OR searches
- **要旨の残存タグ除去を強化**: `&lt;p&gt;`（エスケープされたタグ）や `&#x0D;`（16 進数値参照）、`<p class="E-JOURNAL...">` が残る不具合を、`html.unescape`→タグ除去の順序修正＋数値参照対応で根治 / Hardened abstract tag-stripping: escaped tags (`&lt;p&gt;`), hex numeric references (`&#x0D;`), and leftover `<p class="E-JOURNAL...">` are now removed by fixing the `html.unescape`→strip order and handling numeric references

### V18. NEBULA の表示・書き出し改善 / NEBULA Display & Export Fixes
- **急上昇キーワードの全ラベル表示**: バー数に応じて図の高さを動的化し、Plotly の y 軸ラベル間引き（ラベルが飛ばされる現象）を解消 / Trending-keyword chart height now scales with the bar count, eliminating Plotly's y-axis label thinning so all labels show
- **スライド/レポート用 PNG 書き出しを全図に**: 従来は 1 枚目（棒グラフ）しか書き出せなかったが、スナップショットに複数図がある場合（急上昇キーワード＋共起ネットワーク等）に各図へ書き出しボタンを表示（全モジュール共通の修正）/ The slide/report PNG export button now appears for each figure when a snapshot holds multiple figures (e.g. trending keywords + co-occurrence network) — a fix shared across all modules (previously only the first figure could be exported)
- **学術論文ランドスケープの初期表示サイズを是正**: Saturn V と同一のアスペクト設定（`height=1200`・yaxis にも `constrain="domain"`）に揃え、初回レンダリングで小さく表示される不具合を解消 / Academic landscape now matches Saturn V's aspect settings (`height=1200`, `constrain="domain"` on the y-axis too), fixing the too-small initial render
- **学術クラスタのラベル命名 AI プロンプトを学術ドメインに**: 「特許情報アナリスト / 代表的特許リスト」固定だったのを、NEBULA 学術では「学術文献アナリスト / 代表的な論文リスト」に切替（`domain` 引数）/ The AI label-naming prompt for academic clusters now uses an academic persona ("academic-literature analyst / representative paper list") via a `domain` argument, instead of the patent-fixed wording

### V19. CAPCOM PPTX を Claude Design 準拠に再設計 + ヘルパーをモジュール化 / CAPCOM PPTX Redesigned to Claude Design + Helpers Modularized
- **主張骨格**: 各コンテンツ面を「アイブロウ（章/モジュール名）＋主張見出し（結論の名詞句）＋リード文（核心主張の完結文）＋根拠＋締め文（So What）」で構成。断片箇条書きでなく内容のある文で 1 スライド 1 主張を論証（`slides_spec.md §0.9-A0`）/ Each content slide follows a claim skeleton (eyebrow + claim headline + lead sentence + evidence + closing So-What), arguing one claim per slide with substantive sentences rather than fragment bullets
- **役割タイポ階層（ゴシック）**: `add_title_shape(..., eyebrow=...)` でタイトル直上にアイブロウ（字間広め・ミュート色）。Noto Sans JP のウェイトで階層化（明朝/欧文サンセリフは使わない）/ Role-based type hierarchy via `add_title_shape(..., eyebrow=...)` (tracked, muted eyebrow above the title); hierarchy expressed through Noto Sans JP weights (no Mincho / Latin sans-serif)
- **章構成スライド 4 種を追加**: 重心移動 PAST→PRESENT（`add_shift_slide`）・N 手法が同じ結論へ収束（`add_convergence_slide`）・優先度別アクション（`add_priority_actions_slide`）・アクションアイテム☐（`add_action_items_slide`）/ Four narrative-structure slide types added: PAST→PRESENT shift, N-method convergence, priority actions, and action-item checklist
- **★ヘルパーを実モジュール化**: 仕様書 `slides_spec.md` は 74% がコピー用 Python コードだった → 実モジュール **`capcom_schema/templates/apollo_slides.py`**（import 運用・全 `add_*_slide`）に抽出。`slides_spec.md` は **2,715 行 → 約 600 行の設計ガイド**に凝縮（規律は温存）。AI が読む量が大幅減。生成スクリプトは `from apollo_slides import *` で使う / The spec (74% copy-paste Python) was extracted into a real module **`capcom_schema/templates/apollo_slides.py`** (imported, all `add_*_slide`), shrinking `slides_spec.md` from 2,715 to ~600 lines of design guidance — far less for the AI to read; generation scripts now `from apollo_slides import *`
- コネクタ（破損要因）を使っていた 2 関数をオートシェイプ線に置換。SKILL/ワークフローの「コピー」運用を「import」運用に更新 / Two functions that used connectors (a corruption risk) now use autoshape lines; SKILL/workflow docs updated from "copy" to "import"

### V20. CAPCOM レポート品質の追加根治（実セッションで再発した問題）/ Additional CAPCOM Report-quality Fixes (Issues Recurring in Real Sessions)
- **見出し・目次の数値強調漏れの根治**: 本文の「数値＋単位」自動ネイビー強調を打ち消す `no-num-hl` が機能しておらず（`show num-rx: c=>c` の出力を大域規則が再マッチ）、見出し・目次の数値だけ色が浮いていた。**U+2060（不可視の語結合子）を数字と単位の間に挿入して再マッチを断つ**方式に根治（`report_style.typ`）。範囲表記「20〜50 件」で後半だけ太字になる件も正規表現で範囲全体を拾うよう修正 / Fixed figure-highlighting leaking into headings/TOC: the `no-num-hl` canceller didn't work (the global rule re-matched the `c=>c` output), so figures in headings/TOC stayed navy. Now a U+2060 word joiner is inserted between digit and unit to break the re-match; ranges like "20〜50 件" are also captured whole so only-the-tail-bold is fixed
- **特許番号のプレースホルダ／捏造を防止**: 代表特許に実番号が無く執筆者が「特開2023-XXXXXX」を捏造していた。`phase_d_gate.sh` に **Check 21**（プレースホルダ検出で FAIL）を追加。アプリ側も代表特許の識別子を **公開番号→（無ければ）出願番号** に統一（`best_patent_ref`）し、番号を省いていた整形（NEBULA / Explorer / Saturn V）に番号を付与 / Prevents patent-number placeholders/fabrication (writers invented "特開2023-XXXXXX"): `phase_d_gate.sh` gains **Check 21** (FAIL on placeholders), and the app unifies the representative-patent identifier to publication number → application number (`best_patent_ref`), adding numbers where NEBULA / Explorer / Saturn V had omitted them
- **自明な一般論での字数稼ぎを禁止**: 反復でなくても「データを見なくても言える一般論」で字数を稼ぐのを水増しと定義（`terminology.md`・3 コピー SKILL §0 第 6 項）/ Padding now explicitly includes generic statements "you could write without looking at the data," even when not repetitive (`terminology.md`, item 6 of §0 in all 3 SKILL copies)
- **依存同梱の修正**: スライド/解析に必要な `pandas` が `requirements-session.txt` に無く、SKILL にインストール手順も無かった → `pandas` 追加＋各 SKILL に「環境準備（依存インストール）」を明記 / `pandas` (needed for slides/analysis) was missing from `requirements-session.txt` with no install step in SKILL — added `pandas` and documented an "environment setup (install dependencies)" step in each SKILL

### V21. ドリルダウン自動最適化をメインマップと統一 / Drill-down Auto-optimization Unified with Main Map
- **目標クラスタ数の件数適応**: Saturn V / MEGA / EAGLE のドリルダウンの HDBSCAN 2 パラメータ自動掃引で、目標サブクラスタ数を固定値からメインと同じ `suggest_target_k(対象件数)` に変更（上限もメインに統一）/ Drill-down's HDBSCAN auto-sweep now derives the target sub-cluster count from the subset size (`suggest_target_k`) like the main map, instead of a fixed value
- **結果の常時再表示**: 自動決定（最小クラスタサイズ / 最小サンプル数 / クラスタ数 / ノイズ / DBCV）を `session_state` に保持し、メイン同様「前回の自動決定」を常時表示。手動値は自動 ON 時にグレーアウト / The auto-decided parameters (incl. DBCV) persist and are shown continuously; manual values gray out when auto is on
- **挙動差の解消**: 目標クラスタ数が件数に適応しない・結果が残らない・手動値がグレーアウトしないというメインとの差を解消 / Resolves the differences from the main map (target not adapting to size, results not persisting, manual values not graying out)

### V22. 分析の立場（叙述スタンス）の確定 / Narrative Stance
- **立場を母集団タイプと独立に確定**: 提言を「誰の意思決定のために書くか」を Phase A STOP-GATE C で確定（`self`=自社「当社」/ `competitor`=競合 / `neutral`=投資家・アナリスト）。単一企業データでも自動で「当社」にしない（曖昧なら `neutral` 既定で確認）/ Determines whose decision the report serves, independent of population type; single-company data no longer auto-implies "our company" (defaults to neutral/third-person when unclear)
- **立場で分析・提言が変わる**: 呼称だけでなく、洞察の力点（self=内省 / competitor=機会探索 / neutral=客観評価）と提言の型（self=打ち手 / competitor=対抗・参入 / neutral=評価・予測）、サブクエスチョンの観点も立場に合わせる / Stance shapes not just naming but the analytical angle, recommendation type, and sub-question framing
- **gate Check 11s**: 立場が `neutral` なのに「当社」が出れば FAIL。`terminology.md §6-2-B` が呼称・分析ロジックの正本 / Check 11s fails on first-person "our company" under a neutral stance

### V23. PPTX 情報密度の底上げ / PPTX Information-density Boost
- **初回から十分な密度**: 1 スライドの本文を 180〜320 字 → **250〜400 字**・根拠注釈 4〜6 点に引き上げ、図がある面も本文（リード文＋読み取り注釈）を薄くしない（`slides_spec.md §0.9-B`）/ Raised per-slide body to 250–400 chars with 4–6 evidence points; slides with figures still need a full lead + annotations
- **gate Check 16e 強化**: 薄い面の判定を「図なし 40 字未満」のみ → **図あり<80字 / 図なし<150字**、薄い面がコンテンツ面の 3 割超なら **FAIL**（初回生成で薄くなる失敗を止める）。章扉は番号で識別して除外し誤検出を防止 / Check 16e now flags thin content faces (figure<80 / no-figure<150 chars) and FAILs when >30% are thin; section dividers are excluded by number to avoid false positives

### V24. CAPCOM スナップショット重複の根治 + Export の安定化 / CAPCOM Snapshot-duplication Fix + Export Robustness
- **マップ重複（`voyager_ev` 二重出力）の根治**: 撮影時に本来名（`atlas_*` / `saturn_*` / `core_*` 等）で保存済みのスナップショットを、CAPCOM Export が `voyager_ev*` として**再保存**していたため、同一マップが ZIP に二重に入りレポートで重複していた。Export は再保存せず、Evidence の画像参照を本来名スナップショットへ向けるよう変更（`capcom.snapshot_logical_path`）。手本（exemplar / テンプレート）の `voyager_ev` 参照も本来名へ修正 / Snapshots already saved under canonical ids were being **re-saved** as `voyager_ev*` by Export, so the same map entered the ZIP twice and looked duplicated in reports. Export no longer re-saves; Evidence now points to the canonical snapshots (`capcom.snapshot_logical_path`), and exemplar/template references were corrected too
- **ZIP に `voyager/` が確実に入る**: `voyager_ev` を出力しなくなった結果、Export してもスナップショット件数が変わらず、件数ベースの ZIP キャッシュが更新されないため `voyager/`（mission / evidence / context）が古い ZIP に入らないことがあった。キャッシュキーに **voyager の Export 状態**を加えて根治 / Since `voyager_ev` is no longer emitted, the snapshot count stops changing across Export, so the count-based ZIP cache wasn't refreshed and `voyager/` could be missing from the download. The cache key now includes the voyager export state
- **Export 再実行時の Evidence 累積を一掃**: スナップショットを削除/再生成して再 Export すると `ev{N}_{module}.json` が上書きされず累積し、id 重複（mission.json との不整合）が起きていた。Export 冒頭で古い Evidence を一掃するよう修正（`capcom.clear_voyager_evidence`）/ Re-exporting after deleting/regenerating snapshots accumulated `ev{N}_{module}.json` with duplicate ids; Export now clears stale Evidence first (`capcom.clear_voyager_evidence`)
- **MEGA 再描画時の `axis_col` エラー修正**: 「動態分析マップを描画」ボタンを押さない再描画で `axis_col` が未定義となり CAPCOM へのデータ保存が失敗していた（`name 'axis_col' is not defined`）。`session_state` から復元するよう修正 / A re-render without pressing "draw the dynamics map" left `axis_col` undefined and broke the CAPCOM data save; it is now restored from `session_state`

### V25. 構造化分析技法 — 結論の検証を標準工程に / Structured Analytic Techniques — Conclusion Verification as a Standard Step
- **「正しく見えるが間違っている」への対抗**: CIA の分析教本として知られるリチャーズ・ホイヤー『インテリジェンス分析の心理学』の構造化分析技法をレポート工程に移植（新ガイド `capcom_schema/analysis/structured_techniques.md`）。設計原則は「機械の仕事は正しさの判定ではなく**注意の誘導**」＝ゲートが保証するのは検証の痕跡（監査可能性）であり、意味の番人は人間 / Ports Richards Heuer's structured analytic techniques (Psychology of Intelligence Analysis) into the report pipeline to counter plausible-but-wrong reports. Design principle: the machine's job is **directing attention**, not judging truth — gates guarantee auditability, humans remain the guardians of meaning
- **ACH（競合仮説分析）**: 主要結論ごとに「実際に支持のあった最も強い対立仮説＋それと食い違う決め手」を Typst ヘルパー `#competing-hypotheses` で外部化し、報告書の中心的な問いには ○/×/— セルの仮説×材料マトリクス（`#ach-matrix`）を描く。結論は「最も矛盾の少ない解釈」として提示（Check 30/30b/33）/ Every major conclusion externalizes its strongest genuinely-supported rival hypothesis plus the discriminating evidence via `#competing-hypotheses`; the central question gets a full hypothesis-by-evidence matrix (`#ach-matrix`); conclusions are presented as "the interpretation with the fewest contradictions"
- **リンチピン分析**: 結論の「要の前提・確認データ・**見直すべきサイン**（崩れる条件）」を `#linchpin` で明示（Check 31/31b）。「新しい情報が出たら」式の空文は不可＝観測できる出来事で書く / Each headline conclusion states its key premise, verifying data, and **observable overturn conditions** via `#linchpin` — vague "if new information emerges" phrasing is rejected
- **ミラーイメージング点検**: 分析の立場が competitor/buyer/supplier のとき「相手の立場から見た合理性」の一段落を義務化（Check 32）。実地テストでは、この点検が提言を実際に修正した（価格圧力 → 関係深化）/ When the stance is competitor/buyer/supplier, one paragraph on "rationality seen from the subject's side" is mandatory — in field testing this check actually changed a recommendation (price pressure → relationship deepening)
- **「三つの環」で分析の芝居を防ぐ**: 立派な検証マトリクスの脇で結論・提言が別の筋書きで書かれる「分析の芝居」を封じるため、マトリクス→結論→提言→リンチピンの連結を型として義務化。各提言は採用した結論を名指しで前提に置き、リンチピンの崩れる条件には「退けた対立仮説の復活」を書く（Check 34 が接続切れを検出）/ A mandated matrix→conclusion→recommendation→linchpin chain prevents "analytical theater" — writing conclusions from an unrelated storyline next to an impressive matrix; each recommendation must name its adopted conclusion as premise (Check 34 detects disconnection)
- **読者にはビジネス日本語で**: ACH・リンチピン等の技法名はスキーマ内部用語とし、レポート本文は「別解釈」「決め手」「結論の前提と見直しのサイン」「相手の立場から見た合理性」等の読者向け呼称に統一（`terminology.md §2-F`・Check 8f が露出を警告）。Typst ヘルパーのラベルも読者向け呼称で自動描画 / Technique jargon stays schema-internal; the report body uses reader-facing wording (mapping table in `terminology.md §2-F`, leakage warned by Check 8f)

### V26. 代表特許の決定的選定 / Deterministic Representative-patent Selection
- **「つまみ食い」の構造的排除**: どの特許を代表として引用するかを、モデルの自由選択から `capcom_schema/scripts/select_representatives.py` の決定的手順に移管。結論に都合の良い特許だけを引く恣意性を排除する / Which patents get cited as "representative" moves from the model's free choice to a deterministic script, structurally eliminating cherry-picking of conclusion-friendly patents
- **モジュール別の固定規則（タイブレークまで）**: Saturn V＝SBERT 全次元空間の重心近傍 / MEGA＝各象限内で出願年昇順→公開番号昇順 / Explorer・CREW＝中心性上位ノードに対応する最先の特許 / ATLAS＝出願年四分位 4 期それぞれの先頭（各時代の幕開けの特許）/ CORE＝技術×課題クロス集計の件数上位セル。Phase C 冒頭に 1 回実行して `reports/representative_patents.json` を生成する / Fixed per-module rules down to tie-breaks (Saturn V: SBERT centroid neighbors; MEGA: filing-year order within each quadrant; Explorer/CREW: earliest patents of top-centrality nodes; ATLAS: the opening patents of each filing-year quartile; CORE: top cross-tab cells). Run once at the start of Phase C
- **機械照合**: ミクロ分析はこの JSON に載った番号だけを引用でき、リスト外の引用は Check 35 が検出。ドリルダウン図の代表特許も選定対象に含む（図 1 枚あたり最大 6 件）/ Micro-analysis may only cite listed numbers (Check 35 flags out-of-list citations); drill-down figures are covered too (up to 6 per figure)
- **正直な限界も明記**: 本手順が潰すのは「選択の恣意性」のみで、引用後の解釈の歪みは人間の監査に委ねる、とガイド自身が明記 / The guide states its honest limit: it kills selection bias only — interpretive distortion around a cited patent still requires human audit

### V27. レポート読者体験の改善（マップ・章末まとめ・壁テキスト）/ Report Reader-experience Overhaul (Maps, Chapter Summaries, Walls of Text)
- **マップは「貼る」から「論じる」へ**: 撮った全スナップショットを 1 枚 1 行・全幅で掲載し（grid 横並べ・縮小の禁止＝Check 28）、隣接段落は「俯瞰図（下図）を見ると〜」のような指示語で図を**名指し**し、図から読み取った事実を最低 1 つ書く（Check 29/29b/29c の 3 段検査）。見出し直後にマップを羅列するアンチパターンも検出（Check 23）/ Maps go from "pasted" to "argued": full-width, one per row (no grid side-by-side, Check 28); adjacent paragraphs must explicitly point at the figure and state at least one fact read from it (3-layer Checks 29/29b/29c); front-of-chapter map dumps are detected (Check 23)
- **章末まとめボックス**: 各モジュール章の最後に `#chapter-summary`（📌 本章のまとめ・薄黄ボックス）を置いて章を閉じる。位置ズレは検出でなく執筆順ルール「本文を全部書き終えてから最後にまとめを書く」で根治し、Check 25/26 を backstop に / Each module chapter closes with a summary box; placement is fixed by a write-order rule (finish the body, then write the summary) with Checks 25/26 as backstop
- **ワードクラウドも定量分析**: 画像を貼って終わりにせず、「語（N回）」形式で実際の出現回数（`data/*_wordcloud.json` の実数）に言及する分析を義務化（Check 27）/ Word-cloud chapters must cite actual word counts ("term (N times)") from the data, not just paste the image (Check 27)
- **壁テキストの禁止**: 地の文の連続は 4 段落まで。クロスパターン統合などのサマリ系の節は、冒頭に統合表（パターン×主要発見×含意）を置いて全体を一望させてから散文を書く「表 → 散文」構成を強制。構造化要素なしの 3,000 字超連続は Check 37 が検出 / Prose may run at most 4 consecutive paragraphs; summary sections must open with an overview table before any narrative ("table first, prose second"); runs of 3,000+ chars without structure are detected (Check 37)
- **Explorer 章の捏造根治**: 手本（exemplar）に紛れていた「実在しない成長率×中心性の 4 象限マトリクス」を除去し、Check 24（FAIL）で機械封鎖。マップ種別はファイル名パターン→読み方の対応表を整備し「種別不明」を未掲載の言い訳にできないように / A fabricated quadrant chart lurking in the Explorer exemplar was removed and machine-blocked (Check 24, FAIL); a filename→map-type→how-to-read lookup table means "unknown type" can no longer excuse leaving a snapshot unanalyzed

### V28. 経営層向け別冊に「結論の確からしさ」+ 立場 5 分類 / Executive Edition Gains "How Solid Is This" + 5 Narrative Stances
- **各要点に検証を凝縮**: 別冊の各要点（8-12 行）に「検討した別解釈と決め手」を 1-2 行で組み込み、**別冊単体で「この結論はどこまで信じてよいか」が読める**ようにした（Check 18b）/ Each key finding carries a condensed 1-2 line verification (the alternative considered + the deciding fact), so executives can judge from the booklet alone how much to trust each conclusion (Check 18b)
- **「この提言を見直すべきサイン」ボックス**: 戦略的インプリケーション章の末尾に、観測可能な出来事 2-3 件＋そのとき動きをどう変えるか各 1 行を必須化。「何が起きたら方針転換か」を経営層に直接届ける / The implications chapter must end with 2-3 observable trigger events, each paired with a one-line response — telling executives exactly what event should trigger a course change
- **分析の立場を 5 分類に拡張**: self / competitor / **buyer / supplier**（垂直方向の関係者を新規サポート）/ neutral。関係性立場では自社名を尋ねて自社の Web 調査を行い、対象企業と名指しで対比する提言に昇格（Check 11s'）/ The narrative stance grows to 5 types (adding buyer/supplier for vertical relationships); relational stances ask for your company name and research it on the web, upgrading recommendations to named comparisons
- **HHI を交渉力のシグナルとして解釈**: 出願人集中度（HHI）を Porter の買い手/売り手交渉力の文脈で読む規約を追加（単一企業母集団＝タイプ C では算出無意味と明記）。Porter 5F の単体機能は不採用＝単一母集団での市場断定を避け、水平 3 力は既存モジュール・垂直 2 力はチャネル立場＋HHI で代替 / Applicant HHI is now read as a Porter bargaining-power signal (explicitly meaningless for single-company populations); a standalone Five-Forces feature was deliberately rejected to avoid market-wide claims from a single population

### V29. スキーマ軽量化 — 規律の一本化（−17%）/ Schema Slimming — Single Source of Truth (−17%)
- **SSoT（正本＋ポインタ）方式**: 同じ規律を複数ファイルに書かず、正本 1 箇所＋他所は 1-2 行のポインタに統一。Phase 必読ガイド群を **175,614 → 145,631 字（−17.1%）** に削減し、CAPCOM セッションのトークン消費と読み飛ばしリスクを低減。定量チェックコマンドの二重管理も解消（gate スクリプトを唯一の正本に）/ Each discipline now lives in one canonical home with 1-2 line pointers elsewhere; the must-read guides shrank from 175,614 to 145,631 chars (−17.1%), cutting token cost and skim risk; duplicated quantitative-check commands were consolidated into the gate scripts
- **品質フロアは不変**: `phase_d_gate.sh` は 1 文字も変更せず、回帰テストで圧縮前後の結果が完全一致（GATE PASSED）することを確認 / The gate scripts were untouched — regression testing confirmed byte-identical results before and after compression
- **段階的読み込み**: `structured_techniques.md` は Phase C の統合インサイト節・Phase D の結論章の執筆直前にのみ読む設計（常時コンテキストを圧迫しない）。旧スキーマ世代のセッションでは Check 30/31 を自動スキップする後方互換ガードも実装 / The techniques guide loads just-in-time before the sections that need it; a backward-compatibility guard auto-skips Checks 30/31 for old-schema sessions
- **実地テスト 2 回合格**: 実データ母集団のフル生成（Opus 4.8）で GATE PASSED を 2 回確認してから v9.0.0 をリリース / Two full-generation field tests on real patent data passed the gate before the v9.0.0 release

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
- **コマンドライン検索式が既定**: `TI=/AB=/TA=/TX=/FT=` + `AND/OR/NOT` + 近傍 `nearN/adjN` + ワイルドカードのフィールド指定モードを既定化（旧「標準検索」は「キーワード検索」に改称）/ Command-line query syntax is the default mode (the former "standard search" → "keyword search")
- **AI クエリ作成補助 + 構文プレビュー + 検索中断**: 検索式を AI に書かせるコピー用プロンプト、`compile_command_query` による構文チェック／OpenAlex 候補式プレビュー、「🛑 検索を終了」ボタンを追加 / AI query-authoring prompt, syntax check / candidate-query preview (`compile_command_query`), and a "🛑 Stop search" button
- **論文種別複数選択**: article / review / book-chapter / preprint / dissertation など10種 / 10 publication types (multi-select)
- **検索結果 CSV ダウンロード**: 取得した論文をそのまま CSV で保存可能 / CSV download of search results
- **未選択＝全種別**: デフォルト挙動を維持 / Default = all types (backward compatible)

### 5. CORE ヒートマップの可読性 / CORE Heatmap Readability
- **塗りつぶし維持（白線なし）**: `xgap=0, ygap=0` でマス間にグリッド線を入れず、件数ラベル（`text_auto`）と配色で可読性を確保。セルクリックは塗りつぶし上に重ねた透明レイヤで受ける / No white grid lines (`xgap=0, ygap=0`); readability via solid fill + value labels, with clicks handled by a transparent overlay
- **ヒートマップ／バブル切替**: 同じクロス集計をヒートマップとバブルで切り替え可能 / Toggle the same cross-tab between heatmap and bubble views

### 6. 経営層向け要約版（別冊）の同時生成 / Executive Summary Edition
- **Phase A の STOP-GATE で確認**: 「レポート書いて」と依頼すると、エージェントがまず `AskUserQuestion` で「本編に加えて別冊（8-12ページの経営層向け要約版）も生成しますか？」と確認 / Phase A STOP-GATE: when user asks for a report, the agent first confirms via `AskUserQuestion` whether to also generate an 8-12 page executive summary edition
- **刈り取り禁止**: 別冊は本編の段落を短縮したものではなく、**本分析の Mission Objective と `query_intent` から導かれる「今回の意思決定テーマ」に即して**エッセンスを再構成した凝縮版。定型の分類軸を機械的に当てはめるのは不可 / Not a truncation: the executive edition is **re-synthesized around the specific decision theme** derived from this analysis's Mission Objective and `query_intent`, not a shortened copy. A fixed set of categories must not be forced on every report
- **So What 原則**: 各段落は「経営判断に何を意味するか」を必ず含み、手法詳細（SBERT/UMAP 等）は混入させない / Each paragraph must answer "So What for executives"; methodology details are excluded
- **結論の確からしさと見直しサイン（v9）**: 各要点に「検討した別解釈と決め手」を凝縮し、提言末尾に「この提言を見直すべきサイン」（観測可能な出来事→対応）を必須化（Check 18b）/ Each key finding carries its verification essence (alternative considered + deciding fact), and an observable "signs to revisit" box closes the recommendations (Check 18b)
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

### 17. OpenALEX コマンドライン検索式 + API キー / Command-line Search Syntax + API Key

**API キー（必須・無料）**: OpenALEX は 2026-02-13 より API キーが必須化されました（従来の `mailto` / polite pool 方式は廃止）。[openalex.org/settings/api](https://openalex.org/settings/api) で無料取得し、検索 UI の「🔑 OpenAlex API キー設定」に貼り付けてください。**本アプリは API キー必須**で、未入力では検索ボタンが無効化されます（キーなしのお試し枠運用は廃止）。キーは**各ユーザーがセッションごとに入力**する方式で、サーバ既定キー（環境変数）には対応しません（公開時に全訪問者が運用者のキー・コストを共有してしまうのを防ぐため）。「接続テスト」ボタンでキーの有効性を確認できます。

**A (free) API key is required.** OpenALEX has required an API key since 2026-02-13 (the old `mailto` / polite-pool method is discontinued). Get one free at [openalex.org/settings/api](https://openalex.org/settings/api) and paste it into the "🔑 OpenAlex API key" field. **The key is mandatory** — the search button is disabled until one is entered (the keyless trial-pool mode has been removed). Each user supplies their own key per session; there is **no server-default key (no env-var fallback)** — this prevents all visitors from sharing the operator's key/cost on a public deployment. Use the "接続テスト / Test connection" button to verify.

**コマンドライン検索式モード**: 標準検索（1 行 1 クエリ・複数行で OR）に加え、特許文献検索風のフィールド指定モードを搭載:

| 要素 / Element | 構文 / Syntax | 説明 / Description |
|---|---|---|
| フィールド / Field | `TI=` `AB=` `TA=` `TX=` `FT=` | タイトル / 要旨 / タイトル+要旨 / OpenALEX 総合 / 全文索引 |
| 論理演算 / Boolean | `AND` `OR` `NOT` | 大文字。`(...)` でグループ化 |
| 近傍 / Proximity | `nearN`（順不同・間に N 語以内）/ `adjN`（順序固定・左→右で N 語以内） | 例: `solid adj3 electrolyte` |
| ワイルドカード / Wildcard | `*`（任意長）/ `?`（1 文字） | 例: `electrol*`、`wom?n` |
| 複数行結合 / Multi-line | `#01 ...` `#02 ...` + `T=(#01 AND #02) OR #03` | 1 行のみなら `T=` 式は省略可 |

**2 フェーズ照合 / Two-phase matching**: ① 検索式を AST に解析し OpenALEX 索引で候補を取得、② `TI`/`AB`/`TA` の near/adj・NOT・ワイルドカードは取得後にローカルで厳密照合して絞り込みます（`TX`/`FT` は OpenALEX 索引による候補取得のみ）。Parse the query into an AST and fetch candidates from the OpenALEX index, then strictly re-match near/adj · NOT · wildcards locally for `TI`/`AB`/`TA` (`TX`/`FT` are candidate-retrieval only).

**年別取得モードと併用可 / Works with year-by-year mode**: セクション 14 の「📅 年別取得モード」はコマンドライン検索式でも有効です。ON にすると各候補クエリを年別に取得（10,000 件/クエリ制限を回避）してから②のローカル厳密照合を適用します。The "📅 Year-by-year mode" (Section 14) also applies to command-line queries: each candidate query is fetched per year (bypassing the 10k/query cap), then phase-② local matching is applied.

> ⚠️ **ワイルドカードの注意 / Wildcard caveat**: OpenALEX 索引は前方一致に非対応のため、ワイルドカード語は候補取得には使えず**ローカル照合でのみ判定**されます。したがってワイルドカード語は**必ず具体語との AND/OR で併用**してください（例: `TI=(battery AND electrol*)`）。ワイルドカードのみ・NOT のみの検索式は候補取得に使える具体語が無いため実行できず、明確なエラーを返します。
> Since the OpenALEX index has no prefix matching, wildcard terms cannot drive candidate retrieval and are **enforced only by local matching**. Always pair a wildcard term with a concrete term via AND/OR (e.g. `TI=(battery AND electrol*)`). A wildcard-only or NOT-only query has no retrievable concrete term and is rejected with a clear error.

### 18. Janome 例外防御層 / Janome Exception Guards (4 modules)

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

### 19. ラベル編集 UI の大規模対応 / Large-scale Label Editor UI

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

さらに **`.streamlit/config.toml` で `maxMessageSize = 500`**（既定 200MB → 500MB）に引き上げ、大量クラスタ編集と大きなマップ/動態マップを同一ページで再描画する際の WebSocket メッセージ破損（上記と同じクラッシュ）を緩和している / `.streamlit/config.toml` also raises `maxMessageSize` to 500 to prevent the same WebSocket crash when re-rendering large maps alongside bulk label editing.

### 20. AI ラベルサジェスト / AI Label Suggestion

**設計思想**: LLM に大量クラスタの JSON を強制すると中断・省略・括弧エラーが起きやすい。そこで TSV を推奨形式とし、JSON / Markdown / 平文も自動判別。部分再提案の追記マージも可能にすることで「大量編集」を「部分的な繰り返し」に変える。

**仕様**:

**プロンプト側**: 推奨フォーマットは **TSV**
```
# 出力フォーマット（TSV: クラスタID<TAB>ラベル）
0	全固体電池の固体電解質
1	画像認識による異常検知
```
「部分応答 OK」を明記し、LLM 側の出力負担を軽減。

**パーサ側**: 4 形式を自動判別（TSV / JSON / Markdown 表 / 平文）。JSON はコードフェンス（```json … ```）付きでも前処理でフェンスを除去して受理する。クォート（`"..."` / `'...'` / `「...」` / `『...』`）も自動除去。

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
| テキスト埋め込み / Text Embedding | sentence-transformers（Mission Control フェーズ4 で**モデル選択可**: ⚡高速 `paraphrase-multilingual-MiniLM-L12-v2`（384次元・軽量・既定）/ 🎯高精度 `intfloat/multilingual-e5-base`（768次元・多言語E5・`passage:` 接頭辞を自動付与）。重くて止まる環境では高速を選択） |
| 日本語処理 / Japanese NLP | Janome(形態素解析) |
| 可視化 / Visualization | Plotly, Matplotlib, WordCloud |
| 学術 API / Academic API | **OpenALEX API**（2026-02-13 より API キー必須・無料／論文種別10種選択・CSV DL・コマンドライン検索式 — API key required as of 2026-02-13） |
| レポート生成 / Report Generation | **Typst**（PDF コンパイル、別途インストールが必要：[typst/typst](https://github.com/typst/typst)） |
| AI エージェント連携 / AI Agent Integration | **Claude Code / Codex CLI / Antigravity IDE**（マルチエージェント対応・本体レポートは Claude Code 推奨）|
| 品質ゲート / Quality Gates | **bash スクリプト**（Phase C / Phase D 自動検証・用語統一・反復水増し検出・構造化分析技法（結論検証）など **Check 1〜37**）+ `select_representatives.py`（代表特許の決定的選定） |

---

## 📁 プロジェクト構成 / Project Structure

```
apollo_v8/
├── Home.py                  # Mission Control（データ取込・前処理・CAPCOM セッション開始・OpenALEX 統合）
├── utils.py                 # 共通ユーティリティ（描画・サイドバー・スナップショット・クラスタ動態・ラベル編集・AI サジェスト）
├── utils_ai.py              # AI プロンプト生成 / AI prompt generation
├── utils_spatial.py         # 空間分析（patiroha 委譲）
├── capcom.py                # CAPCOM 通信モジュール（In-Memory セッション + selected_tools パッチ自動同梱 + ZIP エクスポート）
├── openalex.py              # OpenALEX API クライアント（API キー対応・学術論文検索、年別取得・要約フィルタ・言語フィルタ・コマンドライン検索式の候補取得＋ローカル照合・4xx 即時失敗）
├── openalex_query.py        # コマンドライン検索式エンジン（TI=/AB=/TA=/TX=/FT= + AND/OR/NOT + near/adj + ワイルドカードの AST 解析・候補クエリ構築・ローカル厳密照合、純 Python）
├── apollo_kw_worker.py      # キーワード抽出のプロセス並列ワーカー（streamlit 非依存・picklable、utils.extract_keywords_batch から参照）
├── pages/                   # 10 の分析モジュール
│   ├── 1_🌍_ATLAS.py        # 基本統計 + 多様性指標（HHI / Entropy / Gini）
│   ├── 2_💡_CORE.py          # AND/OR/NEAR/ADJ 論理式分類 + クロス集計 + ヒートマップ
│   ├── 3_🚀_Saturn_V.py     # 俯瞰図分析 + ノイズ分析 + クラスタ動態マップ
│   ├── 4_📈_MEGA.py          # PULSE 4 象限動態分析（CAGR × 活動量）+ TELESCOPE ドリルダウン
│   ├── 5_🧭_Explorer.py     # 共起ネットワーク + トレンドキーワード
│   ├── 6_🔗_CREW.py          # 発明者・出願人ネットワーク + 媒介中心性
│   ├── 7_🦅_EAGLE.py         # 投げ縄ツールで手動クラスタ + クラスタ動態マップ
│   ├── 8_📝_VOYAGER.py      # スナップショット収集 + Mission Objective + Markdown レポート
│   ├── 9_🌌_NEBULA.py       # OpenALEX + Hype Cycle + 学術ランドスケープ（Saturn V デザイン統一 + CSV DL）
│   └── 10_📡_CAPCOM.py      # 母集団メタ 4 項目入力 + マルチエージェント複数選択 + ZIP エクスポート
├── capcom_schema/           # CAPCOM スキーマ・テンプレート・手順書
│   ├── SKILL.md             # 4 フェーズ手順（Phase A STOP-GATE + Phase D 品質ゲート Check 1〜37）
│   ├── analysis/            # 分析フレームワーク
│   │   ├── terminology.md              # 用語統一ルール（内部識別子の露出禁止、スコープ限定ルール、サブクエスチョン化）
│   │   ├── executive_summary_guide.md  # 経営層向け要約版（別冊）執筆ガイド
│   │   ├── query_logic_reading.md      # 7 DB 構文リファレンス + 意図整合性検査 + データ逆読み
│   │   ├── population_type_metrics.md  # 母集団 5 タイプ分類 + 指標解釈 + `_phase_a_decisions.json` 仕様
│   │   ├── common_framework.md         # 4 層分析モデル + 母集団タイプ別運用
│   │   ├── data_notes.md               # 特許 / NPL 非対称性 + 全章共通のスコープ明示ルール
│   │   ├── report_structure.md         # report.typ 構造 + 付録 + NEBULA 3 モード分岐 + 「分析過程で確認された追加的事項」章
│   │   ├── quality_checklist.md        # Phase A/C/D の全ゲート項目
│   │   ├── structured_techniques.md    # 構造化分析技法（ACH 競合仮説分析・リンチピン・ミラーイメージング＝結論の検証、Phase C/D で段階的読み込み）
│   │   ├── deep_dive_guide.md / cross_module.md / map_reading.md / patent_citation.md / noise_analysis.md
│   ├── references/          # JSON スキーマ解説（.md 形式・各モジュール別 + metadata_schema.md / wordcloud_schema.md 等の共通スキーマ）
│   ├── exemplars/           # レポート見本（Typst 7 種: ATLAS/CORE/CREW/Explorer/MEGA/NEBULA/Saturn V）
│   ├── templates/           # report.typ（本編雛形）/ report_style.typ / slides_spec.md（PPTX 設計ガイド・約 600 行）/ apollo_slides.py（PPTX 作図ヘルパー本体・全 add_*_slide を import 運用）/ carryover_template.md（引き継ぎ日誌）/ apollo_template.pptx / create_ppt_template.py
│   └── scripts/             # 品質ゲート + 決定的選定
│       ├── phase_c_gate.sh  # Phase C deep_dive の文字数（非空白）検証 + 機械生成・工程ナレーションの早期検出
│       ├── select_representatives.py  # 代表特許の決定的選定（Phase C 冒頭に 1 回実行 → reports/representative_patents.json、Check 35 の照合元）
│       └── phase_d_gate.sh  # Phase D 統合レポートの Check 1〜37（定量・用語・スコープ・母集団タイプ・分析の立場・設計意図・NEBULA 戦略・修辞・反復水増し・PPTX・特許番号プレースホルダ・マップ掲載・章末まとめ・構造化分析技法・壁テキスト）
├── capcom_schema_patches/   # マルチエージェント用オーバーレイ資材
│   ├── README.md
│   ├── codex/               # Codex CLI 用（AGENTS.md + .codex/skills/ + exec_mode_addendum.md）
│   └── antigravity/         # Antigravity IDE 用（GEMINI.md + .agent/ + artifacts_templates/）
├── .claude/skills/          # Claude Code スキル
│   └── apollo-pptx/         # コンサル品質 PowerPoint 生成
├── assets/icons/            # 各モジュールの写実ページアイコン（utils.module_icon が参照・CREDITS.md に出典）
├── .streamlit/config.toml   # WebSocket メッセージ上限引き上げ（maxMessageSize=500・大規模再描画の安定化）
├── requirements.txt
├── packages.txt             # HF Spaces のシステム依存（CJK フォント + chromium/PNG 書き出し用）
├── CLAUDE.md                # プロジェクト設計思想（用語ルール + マルチエージェント方針）
└── README.md                # ← 本ファイル / this file
```

---

## 🤔 FAQ

**Q: APOLLO v8 から何が変わった?（v9 の新機能）**
**What's new in v9 (compared to v8)?**

A: v9 では「分析の深さ」「レポート/スライド化」「初心者のしやすさ」を中心に強化しました（詳細は「✨ v9.0.0 の新機能 詳解」セクション参照）:

1. **CORE 分類設計アシストの刷新**: 分類ルール設計を支援するクラスタリングに SBERT・HDBSCAN・最適 k 自動分析を追加し Phase 1 UI を再構築（ルールベース分類とクロス集計マトリクス自体は従来からの中核機能）
2. **権利化率分析（量×質）**: 出願数 × 権利化率の 4 象限で「件数先行型」と「真の強者」を識別（ATLAS にネイティブ。他モジュールでも任意の一観点として重ねられるが必須の主軸ではない）
3. **クラスタ動態マップの AI インサイト**: 累積件数 × CAGR の 4 象限を自動解釈（Saturn V / EAGLE / NEBULA）
4. **レポート/スライド用ビジュアル**: クリーン高解像度 PNG + 整理版ランドスケープを CAPCOM へ自動同梱
5. **初心者向けヘルプ**: 全パラメータに「？」ツールチップ約 150 箇所
6. **クラスタ数の自動最適化（DBCV 掃引）**: `min_cluster_size` × `min_samples` を自動掃引し DBCV で最良値を提示（Saturn V / NEBULA / CORE の「🤖 自動最適化」）
7. **文埋め込みモデルの選択**: Mission Control で fast=MiniLM（384 次元）/ quality=multilingual-e5-base（768 次元）を切替
8. **MEGA 動態の 3 軸対応**: 出願人 / IPC / F ターム の 3 軸で 4 象限を実行し、軸別ファイルで保存（取り違え防止）
9. **構造化分析技法（結論の検証）**: ホイヤー流の ACH（競合仮説分析）・リンチピン分析・ミラーイメージング点検を標準工程化。代表特許は `select_representatives.py` が決定的に選定（つまみ食い防止）し、経営層向け別冊にも「結論の確からしさ」と「見直すべきサイン」を必須化
10. **その他**: 用語「俯瞰図分析」へ統一 / Mission Control 高速化（SBERT GPU + 並列）/ CAPCOM exemplar 標準化 / スキーマ軽量化（必読ガイド −17%）/ 全体レビューによるバグ修正

v8 で確立した 4 本柱（母集団設計の一貫管理 / マルチエージェント CAPCOM / 13 品質ゲート → v9 で 37 系統に拡張 / UX・耐障害性）は v9 にもすべて継承されています。

v9 focuses on deeper analysis, report/slide-readiness, and beginner-friendliness (see "What's New in v9.0.0 (In Detail)"): (1) CORE classification-assist rework (SBERT/HDBSCAN + auto-k added to the rule-design clustering; the rule-based classification and cross-tab matrix themselves are existing features), (2) grant-rate analysis (applications × grant-rate quadrant; native to ATLAS and optionally overlaid elsewhere — one lens, not a mandatory theme), (3) AI insight on cluster-dynamics maps (Saturn V / EAGLE / NEBULA), (4) report/slide-ready visuals (clean high-res PNG + curated landscape) auto-bundled into CAPCOM, (5) ~150 "?" tooltips for beginners, (6) cluster-count auto-optimization (DBCV sweep of min_cluster_size × min_samples in Saturn V / NEBULA / CORE), (7) sentence-embedding model selection (fast MiniLM 384d / quality multilingual-e5-base 768d), (8) MEGA 3-axis dynamics (applicant / IPC / F-term, saved per-axis), (9) structured analytic techniques for conclusion verification (Heuer's ACH / linchpin / mirror-imaging, deterministic representative-patent selection via `select_representatives.py`, and "how solid is this" verification in the executive edition), (10) "landscape overview" terminology, faster Mission Control (SBERT GPU + parallel), CAPCOM exemplar standardization, schema slimming (must-read guides −17%), and bug fixes from a full review. The four v8 pillars (population-design management, multi-agent CAPCOM, quality gates — grown to 37 checks in v9, UX & fault tolerance) all carry over into v9.

**Q: では v7 から v8 では何が変わった?**
**And what changed from v7 to v8?**

A: 主な進化は以下の 4 つの柱です:

1. **母集団設計の一貫管理**: 設計意図・検索式・収録年・DB 名の文書化、7 DB 構文読解、5 タイプ分類、スコープ限定ルール、`_phase_a_decisions.json` 永続化。これにより「母集団と業界全体の誤読」「無意味な指標算出（例: 単一企業で HHI）」「設計意図が各章でバラバラ」を構造的に防止
2. **マルチエージェント CAPCOM**: Claude Code / Codex CLI / Antigravity IDE を複数選択可能、パッチ自動同梱、3 ツール共通の bash 品質ゲート
3. **レポート品質ゲート（v8: 13 → v9: 37 系統）**: 用語統一 + スコープ + 母集団タイプ別禁止表現 + 設計意図一貫性 + NEBULA 戦略 + 数値/代表特許/クロス分析 + 修辞 + 別冊充実度 + 反復・水増し検出 + 工程ナレーション検出 + PPTX + 特許番号プレースホルダ検出 + マップ掲載・章末まとめ + 構造化分析技法（結論検証）+ 壁テキスト などを `phase_d_gate.sh` で客観判定
4. **UX と耐障害性**: OpenALEX 年別取得 / 要約ありフィルタ / 英語のみフィルタ（タイトル二次判定）、ラベル編集の `st.data_editor` 自動切替（大量クラスタ対応）、AI ラベルサジェスト（TSV 推奨 + 4 形式自動判別 + 追記マージ）、Janome 例外防御（異常入力耐性）

The four pillars of APOLLO v8: (1) **Consistent population-design management** — documentation of intent/query/coverage/DB name, 7-DB syntax reading, 5 population types, scope-limiting rule, persistence via `_phase_a_decisions.json`. Structurally prevents misreading ("population vs industry-wide"), meaningless metrics (e.g. HHI on a single-company population), and fragmented design intent. (2) **Multi-agent CAPCOM** — select any of Claude Code / Codex CLI / Antigravity IDE, patches auto-bundled, shared bash quality gates across 3 tools. (3) **Report quality gate (v8: 13 → v9: 37 checks)** — terminology + scope + population-type-specific forbidden expressions + design-intent consistency + NEBULA strategy + quantitative checks + rhetoric + executive-edition depth + repetition/padding detection + process-narration detection + a PPTX machine-check + patent-number placeholder detection + map-placement & chapter-summary checks + structured-analytic-technique (conclusion verification) checks + wall-of-text detection, all objectively judged by `phase_d_gate.sh`. (4) **UX & fault tolerance** — OpenALEX year-by-year retrieval / abstract-only / English-only (with title check), label editor auto-switching to `st.data_editor` for large clusters, AI label suggestion (TSV preferred + 4-format auto-detect + partial-merge), Janome exception guards.

**Q: 権利化率分析（量×質）はどう使う?**
**How do I use the grant-rate analysis (quantity × quality)?**

A: Mission Control で「ステータス（権利状況）」列をマッピングすると有効になります。ATLAS の「権利化率マップ」が**出願数 × 権利化率**の 4 象限を描き、「量で先行するが権利化率は低い件数主義型」「量・質を両立する真の強者」「少数だが権利化率が高い要警戒プレイヤー」を見分けられます。権利化成功は「一度でも登録された出願（権利継続 ＋ 失効）」で定義し、両軸の中央値で象限を区切ります。MEGA・CORE・CREW・Saturn V でも同じ量×質の視点でクロス参照できます。

Map the **status (legal status)** column in Mission Control to enable it. ATLAS's grant-rate map plots **applications × grant-rate** in 4 quadrants, distinguishing volume-first players (high count, low grant rate) from true leaders (both high) and high-grant-rate specialists. Grant success = ever-registered (active + lapsed); quadrants split by the median of each axis (computed over the displayed players, so the crosshair always centers the shown cloud). The same quantity×quality lens is read alongside (analyst-side) in MEGA, CORE, CREW, and Saturn V.

**Q: CORE が全面刷新されたと聞いたが、何ができる?**
**What can the rebuilt CORE do?**

A: まず前提として、**ルールベース分類（AND/OR/NEAR/ADJ 論理式で「技術 / 課題 / 解決手段」に振り分け）と、技術×課題のクロス集計マトリクス（ヒートマップ／バブル・ホワイトスペース可視化・セルクリックで特許到達）は CORE の従来からの中核機能**です（v9 で新しくなったものではありません）。**v9 で刷新されたのは「分類ルールの設計を支援するクラスタリング・アシスト」部分**で、具体的には: ①叩き台を作るクラスタリングに特徴量 SBERT・手法 HDBSCAN を追加（従来は KMeans のみ。HDBSCAN は UMAP で低次元化してから適用し、失敗時は KMeans に自動フォールバック）、②最適クラスタ数 k の自動分析（スコア曲線＋推奨 k、スライダーが推奨値に自動追従）、③「その他（未分類）」の再分析にも SBERT / HDBSCAN / k 自動を追加、④AI インサイトをマトリクス表＋軸別の構造化 markdown に整形。これらは「AI に分類カテゴリ設計を依頼するプロンプトの叩き台」を作るための高度化です。

Note first: CORE's **rule-based classification (AND/OR/NEAR/ADJ logic across technology / problem / solution) and the tech×problem cross-tab matrix (heatmap/bubble, white-space, click-to-patent) are long-standing core features — not new in v9**. **What v9 reworked is the clustering assist that helps you design the rules**: (1) added SBERT features and HDBSCAN to the assist clustering (previously KMeans only; HDBSCAN runs after UMAP reduction with automatic KMeans fallback), (2) automatic optimal-k analysis (score curve + recommended k, slider auto-follows), (3) the "Other (unclassified)" re-analysis also gained SBERT / HDBSCAN / auto-k, and (4) AI insight reformatted as a matrix table + structured markdown. These improve the draft that seeds the AI prompt for designing classification categories.

**Q: 「俯瞰図分析」とは? 以前の「AI ランドスケープ」と同じ?**
**What is "landscape overview"? Is it the same as the old "AI landscape"?**

A: 同じ機能です。Saturn V の SBERT + UMAP + HDBSCAN による全体俯瞰マップを、特許レポートで自然な呼称になるよう v9 で「**俯瞰図分析**」に統一しました（旧称「AI ランドスケープ」は使いません）。もう一方の正式呼称「Saturn V TELESCOPE 分析」も併用できます。

Same feature. Saturn V's SBERT + UMAP + HDBSCAN overview map was renamed to **"landscape overview"** in v9 for more natural wording in patent reports (the old "AI landscape" is no longer used). The alternative canonical name "Saturn V TELESCOPE analysis" is also valid.

**Q: スライドやレポートに貼れるきれいな図がほしい**
**I want clean figures I can paste into slides/reports**

A: 各マップの下にある「🎨 スライド/レポート用に書き出す」から、白背景・大きな文字の高解像度 PNG（16:9 / 4:3 / 正方形）を書き出せます。クラスタが多くてラベルが重なる場合は「🖼️ 整理版ランドスケープ」で件数上位クラスタだけを大きく示せます（領域 / 密度 / 散布の 3 モード）。CAPCOM がアクティブなら整理版は ZIP に自動同梱され、レポート/スライド生成エージェントが要点・出典を付けて使います。

Use "🎨 Export for slides/reports" under each map to get a clean high-res PNG (white bg, large fonts; 16:9 / 4:3 / square). If clusters overlap, the "🖼️ Curated landscape" shows only the top-N clusters (hull / density / scatter modes). When CAPCOM is active, the curated figure is auto-bundled into the ZIP so the report/slide agent can add captions and sources.

**Q: パラメータの意味が分からない（初心者です）**
**I'm a beginner and don't understand the parameters**

A: 各設定項目の横にある「**？**」アイコンにマウスを乗せると、その項目が「**何を決めるか・大きく/小さくするとどう変わるか・推奨値**」を日本語で表示します（最小クラスタサイズ、Jaccard 閾値、CAGR 期間、特徴量 TF-IDF か SBERT か 等、約 150 箇所）。まずは推奨値のまま実行し、結果を見てから調整するのがおすすめです。

Hover the "**?**" icon next to each setting for a plain-language explanation of what it controls, the effect of increasing/decreasing, and recommended values (~150 tooltips). Start with the defaults, then adjust based on what you see.

**Q: クラスタ数が多すぎ/少なすぎる。いくつにすればいい?**
**There are too many / too few clusters — how many should I use?**

A: 手で `min_cluster_size` を調整しなくても、**「🤖 自動最適化」にチェック**を入れて目標クラスタ数（例: 15）を指定すれば、APOLLO が `min_cluster_size` × `min_samples` を自動掃引し、**DBCV（クラスタの分かれ方の良さを測る指標）が最良**になる組み合わせを選びます（Saturn V / NEBULA 学術 / CORE で対応）。DBCV の目安は「**？**」popover に表示（おおむね 0.3 前後＝良好 / 0.15 以上＝実用 / 0 以下＝要再検討。データ間で比べる相対指標）。まず自動最適化で当たりを付け、必要なら目標数を変えて再実行するのが簡単です。

A: Instead of hand-tuning `min_cluster_size`, tick **"🤖 auto-optimize"** and set a target cluster count (e.g. 15). APOLLO sweeps `min_cluster_size` × `min_samples` and picks the combo with the best **DBCV** (a density-based cluster-validity score), in Saturn V / NEBULA (academic) / CORE. A "?" popover explains DBCV (≈0.3 good / ≥0.15 usable / ≤0 reconsider; relative). Start with auto-optimize, then re-run with a different target if needed.

**Q: 分析が遅い。速くできる?**
**Analysis is slow — can I speed it up?**

A: v9 では SBERT が GPU を自動検出（CUDA > MPS > CPU）し、キーワード抽出をマルチコア並列化しています。GPU 搭載機（NVIDIA / Apple Silicon）では前処理が大幅に短縮されます。MPS が不安定な場合は環境変数 `APOLLO_FORCE_CPU=1` で CPU 実行に固定できます。なお初回はモデルロードで時間がかかり、2 回目以降はキャッシュで高速化します。

In v9, SBERT auto-detects a GPU (CUDA > MPS > CPU) and keyword extraction runs in parallel across cores, greatly shortening preprocessing on NVIDIA / Apple Silicon machines. If MPS is unstable, set `APOLLO_FORCE_CPU=1` to force CPU. The first run is slower due to model loading; later runs are cached.

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

A: 3 ツール共通で `bash capcom_schema/scripts/phase_c_gate.sh` / `phase_d_gate.sh` が同一基準で動作し、**品質の「下限」を機械的に保証**します（計 37 系統: 内容量＝非空白文字数・代表特許数・クロス分析量・用語統一・スコープ限定・反復/水増し検出・工程ナレーション節検出・PPTX・特許番号プレースホルダ検出・マップ掲載・構造化分析技法（結論検証）など）。`terminology.md` により、どのエージェントでも同じ呼称が使われます。

ただし **ゲートは「失敗を止める backstop」であって、分析の深さ・洞察の質そのものは機械では測りきれません**。特に Codex は指示を逐語的・最小コストで満たす傾向があり、過去に水増し（同一文の反復／接続句だけ変えた重複／本文を Python スクリプトで生成／1 文 1 行での行数稼ぎ／「後続分析への接続」等の工程ナレーション節）が観測されました。これらは Check 19/19a・**文字数判定**・Check 8e で個別に塞いでいます。Codex / Antigravity ではコンテキスト枯渇を防ぐため、各フェーズ境界で**セッション・チェックポイント**（新セッションへの切替提案）も入ります。

**推奨**: 3 ツールいずれもゲートを通過するレポートを生成できますが、**レポート本体（深い分析・論証の一貫性）の品質は Claude Code が最も安定**します。Codex CLI / Antigravity IDE も利用可能ですが、その場合は `model_reasoning_effort=high`（IDE のモデル選択では「高」。**xhigh/「非常に高い」は枯渇しやすいので避ける**）・1 フェーズ 1 セッション・引き継ぎ日誌（`_carryover.md`）の運用を前提にしてください。

The same `phase_c_gate.sh` / `phase_d_gate.sh` run identically on all 3 tools, mechanically enforcing a **quality floor** (37 checks: char-based volume, representative-patent count, cross-module depth, unified terminology, scope-limiting, repetition/padding detection, process-narration detection, a PPTX machine-check, map-placement checks, structured-analytic-technique (conclusion verification) checks, etc.). `terminology.md` keeps terms identical across agents.

But **gates are a backstop that stops failures — they cannot fully measure analytical depth or insight.** Codex in particular tends to satisfy instructions literally and at minimum cost; past runs showed padding (repeated sentences / duplicates with only connectors changed / generating the body via a Python script / one-sentence-per-line line-count inflation / meaningless "to be covered in the next chapter" sections). These are individually blocked by Check 19/19a, char-based judging, and Check 8e. On Codex / Antigravity, a **session checkpoint** is also proposed at each phase boundary to avoid context exhaustion.

**Recommendation: all three can produce gate-passing reports, but Claude Code is the most consistent for the report body** (deep analysis, argument coherence). Codex CLI / Antigravity IDE work too — just use `model_reasoning_effort=high` (pick "high", avoid xhigh), one-phase-per-session, and the carryover diary (`_carryover.md`).

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

A: エージェントがレポート生成で「効率のために省略しよう」と判断するのを防ぐ仕組みです。Phase C（deep_dive の内容量＝非空白文字数）・Phase D（レポート品質 + 用語統一 + 結論検証、Check 1〜37）で bash スクリプトが客観的合否を判定し、不合格なら該当 Phase に戻って補強します。内部識別子（`spatial_context`, `cluster_dynamics`, `*.json`, `*.md`）のレポート本文露出も自動検出します。

Mechanisms that prevent the agent from skipping steps for efficiency. Bash scripts enforce objective pass/fail on deep_dive volume (non-whitespace characters, Phase C) and report quality + terminology + conclusion verification (Phase D, Checks 1-37). Failures trigger mandatory loop-backs. Internal-identifier leakage (`spatial_context`, `cluster_dynamics`, `*.json`, `*.md`) into the report body is also auto-detected.

**Q: AI が書いたレポートの「結論」はどこまで信用できる?**
**How much can I trust the conclusions of an AI-written report?**

A: v9 では「もっともらしいが間違っている」結論への対策として、CIA の分析教本で知られるホイヤーの**構造化分析技法**を標準工程にしました。主要な結論には必ず次が伴います: (1) **別解釈との比較** — 実際に支持のあった最も強い対立仮説と、それを退けた決め手を明示。中心的な問いには仮説×材料の検証マトリクスを掲載し、結論は「最も矛盾の少ない解釈」として提示。(2) **結論の前提と見直しのサイン** — 何が観測されたらこの結論を見直すべきかを具体的な出来事で明記。(3) **相手の立場から見た合理性の点検**（競合・取引先の立場で書く場合）。加えて代表特許は決定的選定スクリプトが固定規則で選ぶため、結論に都合の良い特許だけを引く「つまみ食い」ができません。これらの痕跡は Check 30〜36 で機械検査されます。ただし機械が保証するのは**検証の痕跡が残っていること（監査可能性）**であり、内容の正しさそのものではありません — 最終判断は人間の監査が前提です。

In v9, conclusions are defended against "plausible but wrong" by Heuer's **structured analytic techniques** (known from CIA analytical tradecraft), applied as a standard step. Every major conclusion ships with: (1) **a comparison against alternatives** — the strongest genuinely-supported rival hypothesis and the deciding fact that rejected it, plus a hypothesis-by-evidence verification matrix for the central question, presenting the conclusion as "the interpretation with the fewest contradictions"; (2) **its premises and observable signs to revisit it**; and (3) **a rationality check from the subject's side** (for competitor/partner stances). Representative patents are chosen by a deterministic script under fixed rules, so cherry-picking conclusion-friendly patents is impossible. These traces are machine-audited by Checks 30-36 — but the machine guarantees **auditability** (that verification traces exist), not truth itself; final judgment remains with the human reader.

**Q: APOLLO SPACE とどう違うの?**
**How does this differ from APOLLO SPACE?**

A: 用途と規模が異なります:
- **APOLLO v9(本ツール)**: 本格分析向け。10 モジュール × Streamlit × マルチエージェント CAPCOM 連携で深い分析とレポート生成
- **APOLLO SPACE**: 入門者・初心者向け。単一 HTML で環境構築ゼロ、Gemini API のみで完結

Different use cases and scales:
- **APOLLO v9 (this tool)**: For serious analysis. 10 modules × Streamlit × multi-agent CAPCOM integration for deep analysis and report generation.
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

A: **TSV 推奨 + 4 形式自動判別 + 追記マージモード** で対応しています:
1. プロンプトの推奨形式は **TSV (`0\tラベル`)**。LLM の出力が大幅に安定
2. それでも JSON / Markdown 表 / 平文 で応答してくれば **自動判別して取り込み**
3. 「クラスタ 5, 12, 47 だけ再提案して」と LLM に指示して、**部分応答を追記マージ**できる（既存マップは保持）
4. AI 提案は `st.data_editor` の「AI 提案」列に自動反映、一括コピーボタンで「編集後ラベル」へ転記可能

Solved with **TSV-preferred + 4-format auto-detection + partial-merge mode**: (1) Prompt recommends TSV over JSON for stable LLM output. (2) JSON/Markdown/plain text responses are auto-detected. (3) Partial re-suggestions like "regenerate only clusters 5, 12, 47" merge into the existing map. (4) AI suggestions populate the "AI 提案" column in `st.data_editor` with a bulk-copy button.

---

## 📄 ライセンス / License

Apache License 2.0（v9.0.0 より。v8 以前は MIT License）— 詳細は [LICENSE.txt](LICENSE.txt) と [NOTICE](NOTICE) を参照。

Apache License 2.0 (as of v9.0.0; v8 and earlier were MIT-licensed) — see [LICENSE.txt](LICENSE.txt) and [NOTICE](NOTICE).

---

## 🔗 関連リポジトリ / Related Repositories

- **APOLLO CAPCOM v1.0** — 公開中の安定版 / Stable public release: [GitHub](https://github.com/shibayamalicht/apollo-patent-analysis-capcom)
- **APOLLO SPACE** — 単一HTML版(入門者向け)/ Single-HTML edition for beginners
- **APOLLO Lite** — 軽量版(PyScript)/ Lightweight PyScript edition
- **KATHERINE** — AI対話型分析設計 / AI conversational analysis designer
- **patiroha** — コアライブラリ / Core library

---

© 2025-2026 しばやま
