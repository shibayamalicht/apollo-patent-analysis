---
title: APOLLO v7
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.41.1
app_file: Home.py
pinned: false
short_description: Patent Analysis × Claude Code — End-to-End from Data to Strategic Report
license: mit
---

# 🚀 APOLLO v7.0.0

**特許情報分析 × Claude Code — 萌芽技術から戦略レポートまで、全部おまかせ。**

**Patent Analysis × Claude Code — From emerging tech detection to strategic reports, fully automated.**

> "Houston, we have ~~a problem~~ a report — and this time, with quality gates." — APOLLO v7

---

## これは何？ / What is this?

**APOLLO v7** は、APOLLO CAPCOM v1.0 を**全面刷新した次世代版**。10モジュールで特許データを多角的に分析し、**CAPCOM** が結果を Claude Code に橋渡しし、Claude Code が**品質ゲート付きの戦略レポート**を執筆します。

**APOLLO v7** is the fully revamped successor to APOLLO CAPCOM v1.0. It analyzes patent data through 10 specialized modules, **CAPCOM** bridges the results to Claude Code, and Claude Code writes **strategic reports with built-in quality gates**.

```
CSV/Excel  →  APOLLO v7(分析・可視化)  →  CAPCOM(In-Memory セッション)  →  ZIP DL  →  Claude Code(レポート執筆)
              Analysis & Viz              In-Memory Session                         Report Writing
                                                                                          ↓
                                                                                   Typst PDF 完成 🎉
```

APOLLO CAPCOM v1.0 からの主な進化:
- 🌱 **萌芽技術の自動発見**(ノイズ分析・クラスタ動態マップ・多様性3指標)
- 🌌 **学術・ニュース・政策の統合環境分析**(OpenALEX API + Hype Cycle)
- 📡 **4フェーズ + 品質ゲート**の構造化レポート生成
- ☁️ **Hugging Face Spaces / Streamlit Cloud で動く**(In-Memory 化)
- 🧪 **コアライブラリ patiroha**(pytest 84件で品質保証)

Evolution highlights from APOLLO CAPCOM v1.0:
- 🌱 **Auto-detection of emerging tech** (noise analysis, cluster dynamics map, 3 diversity indices)
- 🌌 **Integrated environmental analysis** of academia/news/policy (OpenALEX API + Hype Cycle)
- 📡 **4-phase + quality-gated** report generation
- ☁️ **Runs on Hugging Face Spaces / Streamlit Cloud** (in-memory architecture)
- 🧪 **Core library patiroha** (84 pytest-covered functions)

---

## 🚀 クイックスタート / Quick Start

### A. Hugging Face Spaces(推奨・環境構築ゼロ)/ Hugging Face Spaces (recommended, zero setup)

```
1. Hugging Face Spaces で APOLLO v7 を開く
   Open APOLLO v7 on Hugging Face Spaces

2. CSV/Excel の特許データをアップロード
   Upload patent CSV/Excel data

3. 各モジュールで分析 → CAPCOM で ZIP ダウンロード
   Analyze across modules → Download ZIP from CAPCOM

4. ZIP をローカル展開 → Claude Code でレポート生成
   Extract ZIP locally → Generate report in Claude Code
```

> ⚠️ **セッションはブラウザを閉じると消失します**。必ず CAPCOM ページから ZIP をダウンロードしてください。
>
> ⚠️ **Session data is lost when the browser closes.** Always download the ZIP from the CAPCOM page before leaving.

### B. ローカル実行 / Local execution

```bash
pip install -r requirements.txt
streamlit run Home.py
# http://localhost:8501 でコーヒー片手にどうぞ ☕
# Open http://localhost:8501 — grab a coffee while you're at it ☕
```

### 基本ワークフロー / Basic Workflow

1. **Home.py** で特許データをアップロード → 前処理(SBERT + TF-IDF + メタデータ正規化)
   Upload patent data in Home.py → Preprocess (SBERT + TF-IDF + metadata normalization)
2. **CAPCOMセッション開始** → 以降の分析結果は自動的に In-Memory ストアに蓄積
   Start a CAPCOM session → Analysis results are auto-saved to the in-memory store
3. 各モジュール(ATLAS/Saturn V/MEGA/Explorer/CREW/EAGLE/NEBULA など)で分析・可視化
   Analyze & visualize across 10 modules
4. 気になるチャートを **VOYAGER** or 各モジュールの **📸 Snapshot** で収集
   Collect key charts as snapshots
5. **CAPCOM** で Mission Objective を設定 → **ZIP ダウンロード**
   Set Mission Objective in CAPCOM → Download ZIP
6. **Claude Code** で ZIP を展開 → 4フェーズで戦略レポート生成(品質ゲート自動検証付き)
   Extract ZIP in Claude Code → Generate reports through 4 phases with auto quality gates

---

## 🧩 10 の分析モジュール / 10 Analysis Modules

APOLLO v7 は 10 モジュールで特許データを多角的に分析します。**太字**は APOLLO CAPCOM v1.0 比で強化・新規追加された機能。

APOLLO v7 analyzes patent data across 10 modules. **Bold items** are enhanced or newly added in v7.

| # | モジュール / Module | 概要 / Description |
|---|----------|---------|
| 1 | 🌍 **ATLAS** | 基本統計 + **多様性指標(HHI + Entropy + Gini の3指標)** — Basic stats + **3 diversity indices** |
| 2 | 💡 **CORE** | AND/OR/NEAR/ADJ 論理式での分類 + クロス集計マトリクス — Rule-based classification + cross-tab matrix |
| 3 | 🚀 **Saturn V** | AIランドスケープ + **ノイズ分析(萌芽技術)** + **クラスタ動態マップ(4象限)** — AI landscape + **noise analysis (emerging tech)** + **cluster dynamics map** |
| 4 | 📈 **MEGA** | PULSE 4象限動態分析 + **クラスタ動態マップ** — PULSE quadrant analysis + **cluster dynamics** |
| 5 | 🧭 **Explorer** | 共起ネットワーク + **急上昇キーワード** + **トルネードチャート競合比較** — Co-occurrence + **trending keywords** + **tornado competitor comparison** |
| 6 | 🔗 **CREW** | 発明者・出願人ネットワーク + 媒介中心性 + コミュニティ検出 — Inventor/applicant networks + betweenness + community detection |
| 7 | 🦅 **EAGLE** | 投げ縄ツールで手動クラスタ + **クラスタ動態マップ** — Lasso-based manual clusters + **cluster dynamics map** |
| 8 | 📝 **VOYAGER** | スナップショット収集 + Mission Objective 設定 + Markdown レポート骨格生成 — Snapshot collection + Mission Objective + Markdown report skeleton |
| 9 | 🌌 **NEBULA** | **OpenALEX API 統合** + **Hype Cycle(3軸)** + **学術ランドスケープ + クラスタ動態** + **4タイプ NPL 統合** — **OpenALEX integration** + **Hype Cycle** + **academic landscape** + **4 NPL types** |
| 10 | 📡 **CAPCOM** | In-Memory セッション管理 + **独立 Mission Objective** + ZIP エクスポート + Claude Code 連携 — In-memory session mgmt + **independent Mission Objective** + ZIP export + Claude Code bridge |

---

## 📡 CAPCOM — Claude Code への橋渡し(v7 完全刷新)/ Bridge to Claude Code (fully revamped in v7)

**CAPCOM** (Capsule Communicator) は APOLLO と Claude Code を繋ぐ通信モジュール。v7 では **In-Memory アーキテクチャ**に刷新され、Hugging Face Spaces や Streamlit Cloud でもそのまま動きます。

**CAPCOM** (Capsule Communicator) bridges APOLLO and Claude Code. In v7, CAPCOM has been rebuilt with an **in-memory architecture** that runs on Hugging Face Spaces and Streamlit Cloud out of the box.

### セッション構造(ZIP 展開後)/ Session Structure (after ZIP extraction)

```
session_YYYYMMDD_HHMMSS_<uuid>/
├── data/                    # 全分析データ / All analysis data
│   ├── patents.csv          # 特許データ(クラスタ情報付き) / Patents with cluster info
│   ├── atlas_statistics.json
│   ├── saturnv_clusters.json      # ← cluster_dynamics フィールド含む / includes cluster_dynamics
│   ├── mega_momentum.json
│   ├── explorer_global_network.json
│   ├── nebula_hype_cycle.json
│   ├── nebula_academic_clusters.json
│   └── ...
├── voyager/                 # 戦略ストーリー / Strategic narrative
│   ├── mission.json         # Mission Objective + Evidence 一覧
│   ├── evidence/            # モジュール横断の Evidence 群
│   └── context.json         # データセットメタ情報
├── snapshots/               # チャート画像 (PNG)
├── prompts/                 # AI インサイト (Markdown)
├── reports/                 # ← Claude Code がここにレポートを生成
├── capcom_schema/           # 分析スキーマ・テンプレート・品質ゲートスクリプト
│   ├── SKILL.md             # 4フェーズ手順 + 絶対遵守ゲートルール
│   ├── analysis/            # common_framework / cross_module / deep_dive_guide ...
│   ├── references/          # 各モジュールの JSON スキーマ
│   ├── exemplars/           # Typst レポート見本(NEBULA / Saturn V / Explorer / MEGA / ATLAS)
│   ├── templates/           # report_style.typ / slides_spec.md / apollo_template.pptx
│   └── scripts/             # 🆕 phase_c_gate.sh / phase_d_gate.sh(bash 品質ゲート)
├── .claude/skills/          # Claude Code スキル(apollo-pptx など)
├── CLAUDE.md                # プロジェクト設計思想
└── metadata.json
```

### Claude Code での使い方 / How to use with Claude Code

Claude Code でセッションフォルダを開いたら、2 つのモードが使えます。

Open the session folder in Claude Code and choose from two modes:

**🔍 自由分析モード / Free Analysis Mode** — 何でも聞ける / Ask anything
```
「このデータセットで最も成長しているクラスタは?」
「クラスタ動態マップで『新興クラスタ』に属する特許を教えて」
「ノイズ特許から見える萌芽テーマを5つ抽出して」
```

**📄 レポート生成モード / Report Generation Mode** — 4フェーズ + 品質ゲート

v7 では**品質ゲート付きの 4 フェーズ構造**に進化しました:

| Phase | タスク | 絶対遵守ゲート |
|-------|-------|--------------|
| **A** | ミッション理解 + データ精読 | 主要数値把握・AIインサイト 3 件以上読了 |
| **B** | エビデンス精読 + クロス分析 | 13 種クロスパターンから 3 つ以上選定 + Web 調査テーマのユーザー確認 |
| **C** | モジュール別 Deep Dive(7 モジュール) | `bash capcom_schema/scripts/phase_c_gate.sh` で行数自動検証 |
| **D** | 統合レポート + 品質検証 | `bash capcom_schema/scripts/phase_d_gate.sh` で定量チェック自動実行 |

> Claude Code が「効率のために省略しよう」と判断するのを**構造的に防ぐ**設計です。bash スクリプトが客観的な合否を判定するため、主観的な「実質 OK」判断で量的基準を上書きできません。
>
> By design, Claude Code **cannot skip steps for efficiency**. Bash scripts enforce objective pass/fail criteria, so subjective "good enough" judgments cannot override quantitative requirements.

---

## 🌟 v7 の主な強化機能 / Main v7 Enhancements

### 1. 萌芽技術・動態分析 / Emerging Tech & Dynamics Analysis
- **ノイズ分析**: HDBSCAN のノイズ点を萌芽技術として積極分析 / HDBSCAN noise points analyzed as emerging tech
- **クラスタ動態マップ**: X = 累積件数、Y = CAGR の 4 象限分類 / Cluster dynamics map: cumulative count × CAGR quadrants
- **多様性指標の 3 指標化**: HHI + Entropy + Gini / Three diversity indices

### 2. NEBULA の大幅強化 / Major NEBULA Upgrade
- **OpenALEX API 統合**: 学術論文を直接検索・取得 / Direct search and fetch of academic papers
- **Hype Cycle**: 特許 × 論文 × ニュースの 3 軸トレンド比較 / 3-axis trend comparison
- **学術ランドスケープ**: 論文クラスタ化 + クラスタ動態マップ / Academic paper clustering + dynamics map

### 3. CAPCOM ワークフロー構造化 / Structured CAPCOM Workflow
- **4 フェーズ + Stop-Gate**: 省略不可のゲートルール / Non-skippable gates
- **bash 品質ゲート**: `phase_c_gate.sh` / `phase_d_gate.sh` で客観検証 / Objective validation via bash scripts
- **独立 Mission Objective**: VOYAGER を使わず CAPCOM 単独利用にも対応 / Standalone CAPCOM support

### 4. Web 公開対応 / Web-Ready
- **In-Memory セッション**: ephemeral storage で動作 / Runs on ephemeral storage
- **ZIP 動的構築**: ファイルシステム書き込み不要 / Dynamic ZIP assembly, no filesystem writes
- **マルチユーザー安全**: UUID セッション ID でブラウザ間衝突なし / UUID-based session isolation

### 5. コアライブラリ patiroha / Core Library patiroha
- **pytest 84 件**の品質保証 / 84 pytest cases
- テキスト処理・統計・クラスタリング・ネットワーク分析を統合 / Unified text processing, statistics, clustering, network analysis
- 他プロジェクトへの再利用容易 / Easily reusable

---

## 🏗️ 技術スタック / Tech Stack

| カテゴリ / Category | ライブラリ / Libraries |
|---------|-----------|
| フレームワーク / Framework | Streamlit 1.41.1 |
| コアライブラリ / Core Library | **patiroha[all]** (pandas, janome, sklearn, SBERT, UMAP, HDBSCAN, NetworkX) |
| テキスト埋め込み / Text Embedding | sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`) |
| 日本語処理 / Japanese NLP | Janome(形態素解析) |
| 可視化 / Visualization | Plotly, Matplotlib, WordCloud |
| 学術 API / Academic API | **OpenALEX API** (NEBULA 用) |
| レポート生成 / Report Generation | Typst(PDF) |
| 品質ゲート / Quality Gates | **bash スクリプト**(Phase C / Phase D 自動検証) |

---

## 📁 プロジェクト構成 / Project Structure

```
apollo_v7/
├── Home.py                  # Mission Control(データ取込・前処理・CAPCOMセッション開始)
├── utils.py                 # 共通ユーティリティ(描画・サイドバー・スナップショット・クラスタ動態)
├── utils_ai.py              # AIプロンプト生成 / AI prompt generation
├── utils_spatial.py         # 空間分析(patiroha 委譲)
├── capcom.py                # CAPCOM 通信モジュール(In-Memory セッション管理)
├── openalex.py              # 🆕 OpenALEX API クライアント(NEBULA 学術論文検索)
├── pages/                   # 10 の分析モジュール
│   ├── 1_🌍_ATLAS.py
│   ├── 2_💡_CORE.py
│   ├── 3_🚀_Saturn_V.py
│   ├── 4_📈_MEGA.py
│   ├── 5_🧭_Explorer.py
│   ├── 6_🔗_CREW.py
│   ├── 7_🦅_EAGLE.py
│   ├── 8_📝_VOYAGER.py
│   ├── 9_🌌_NEBULA.py
│   └── 10_📡_CAPCOM.py      # 🆕 独立ページ化、独立 Mission Objective
├── capcom_schema/           # CAPCOM スキーマ・テンプレート・手順書
│   ├── SKILL.md             # 🆕 4フェーズ手順 + 絶対遵守ゲートルール
│   ├── analysis/            # 分析フレームワーク(9 ファイル)
│   ├── references/          # JSON スキーマ定義(10 ファイル)
│   ├── exemplars/           # レポート見本(Typst 5 種)
│   ├── templates/           # report_style.typ / slides_spec.md / apollo_template.pptx
│   └── scripts/             # 🆕 phase_c_gate.sh / phase_d_gate.sh(品質ゲート)
├── .claude/skills/          # Claude Code スキル(apollo-pptx 等)
├── requirements.txt
├── CLAUDE.md                # プロジェクト設計思想
└── README.md                # ← 本ファイル / this file
```

---

## 🤔 FAQ

**Q: APOLLO CAPCOM v1.0 から何が変わった?**
**What's new compared to APOLLO CAPCOM v1.0?**

A: 機能面での主な進化は以下の 5 つです:
1. 萌芽技術・動態分析(ノイズ分析・クラスタ動態マップ・多様性 3 指標)
2. NEBULA の大幅強化(OpenALEX API・Hype Cycle・学術ランドスケープ)
3. CAPCOM ワークフロー構造化(4 フェーズ + bash 品質ゲート)
4. Web 公開対応(In-Memory 化で HF Spaces / Streamlit Cloud 対応)
5. コアライブラリ patiroha 統合(pytest 84 件)

The top 5 evolutions: (1) Emerging tech & dynamics analysis, (2) Major NEBULA upgrade, (3) Structured CAPCOM workflow with quality gates, (4) Web-ready in-memory architecture, (5) patiroha core library.

**Q: Hugging Face Spaces で使うときの制約は?**
**Any limitations when running on Hugging Face Spaces?**

A: セッションは**ブラウザを閉じると消失**します。必ず CAPCOM ページから ZIP をダウンロードしてください。また、SBERT モデルのロードで初回起動に 1〜2 分かかります(2 回目以降はキャッシュで高速化)。

Sessions are **lost when the browser closes**. Always download the ZIP from the CAPCOM page. Initial boot takes 1-2 minutes due to SBERT model loading (cached after first run).

**Q: APOLLO 単体でも使える? Claude Code なしでも?**
**Can I use APOLLO without Claude Code?**

A: 分析・可視化・Markdown レポート骨格(VOYAGER)は APOLLO 単体で動きます。ただし**本格的な戦略レポート**が欲しい場合は Claude Code 連携が必要です。

Analysis, visualization, and the VOYAGER Markdown report skeleton work standalone. However, for **full-scale strategic reports**, Claude Code integration is required.

**Q: 日本語以外の特許データも使える?**
**Does it work with non-Japanese patent data?**

A: 日本語特許に最適化していますが、英語データでも動作します。ただし形態素解析(Janome)は日本語専用のため、多言語混在データはおすすめしません。

Optimized for Japanese patents but works with English data too. Mixed-language datasets aren't recommended since Janome (morphological analyzer) is Japanese-only.

**Q: CAPCOM の品質ゲートって何?**
**What are CAPCOM quality gates?**

A: Claude Code がレポート生成で「効率のために省略しよう」と判断するのを防ぐ仕組みです。Phase C(deep_dive 行数)・Phase D(レポート品質)で bash スクリプトが客観的合否を判定し、不合格なら該当 Phase に戻って補強します。

Mechanisms that prevent Claude Code from skipping steps for efficiency. Bash scripts (`phase_c_gate.sh` / `phase_d_gate.sh`) enforce objective pass/fail criteria on deep_dive length (Phase C) and report quality (Phase D). Failures trigger mandatory loops back.

**Q: APOLLO SPACE とどう違うの?**
**How does this differ from APOLLO SPACE?**

A: 用途と規模が異なります:
- **APOLLO v7(本ツール)**: 本格分析向け。10 モジュール × Streamlit × Claude Code 連携で深い分析とレポート生成
- **APOLLO SPACE**: 入門者・初心者向け。単一 HTML で環境構築ゼロ、Gemini API のみで完結

Different use cases and scales:
- **APOLLO v7 (this tool)**: For serious analysis. 10 modules × Streamlit × Claude Code integration for deep analysis and report generation.
- **APOLLO SPACE**: For beginners. Single HTML, zero setup, powered solely by Gemini API.

**Q: PDF レポートはどう生成する?**
**How do I generate PDF reports?**

A: Claude Code で ZIP 展開フォルダを開き、`capcom_schema/SKILL.md` を読ませると 4 フェーズで `report.typ`(Typst)を生成します。その後 `typst compile reports/report.typ reports/report.pdf` で PDF 化。

Open the extracted ZIP folder in Claude Code, have it read `capcom_schema/SKILL.md`, and it will generate `report.typ` (Typst) through the 4 phases. Then compile with `typst compile reports/report.typ reports/report.pdf`.

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
