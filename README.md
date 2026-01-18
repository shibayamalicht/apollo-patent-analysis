---
title: APOLLO v6 Patent Analysis
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.42.0
app_file: Home.py
pinned: false
short_description: AI-Powered Patent Analysis Platform (JP/EN)
license: mit
---

# 🚀 APOLLO v6: Patent Analysis Platform

**APOLLO (Advanced Patent & Overall Landscape-analytics Logic Orbiter)** is an advanced, AI-powered patent analysis platform designed to visualize technology trends, competitor strategies, and inventor networks using state-of-the-art NLP techniques (SBERT, UMAP, HDBSCAN).

**APOLLO (Advanced Patent & Overall Landscape-analytics Logic Orbiter)** は、最先端の自然言語処理技術（SBERT, UMAP, HDBSCAN）を活用し、技術トレンド、競合戦略、発明者ネットワークを可視化・分析するための高度な特許分析プラットフォームです。

---

## 🛰️ Mission Control (Data Hub)

The entry point for all analyses.
全ての分析の出発点です。

1.  **Data Import**: Upload patent data (CSV/Excel).
    * *データインポート*: 特許データ（CSV/Excel）をアップロードします。
2.  **Smart Mapping**: Automatically maps columns (Title, Abstract, Claims, IPC, etc.) based on keywords.
    * *スマートマッピング*: キーワードに基づいてカラム（名称、要約、請求項、IPCなど）を自動的に紐付けます。
3.  **Analysis Engine**: Pre-calculates SBERT vectors, TF-IDF keywords, and normalizes metadata with a real-time progress bar.
    * *分析エンジン*: SBERTベクトル化、TF-IDF計算、メタデータ正規化をバックグラウンドで実行します（リアルタイム進捗表示付き）。
4.  **Stopword Management**: Manage and edit stopwords to refine analysis accuracy.
    * *ストップワード管理*: 分析精度を向上させるため、ストップワードの管理・編集が可能です。

---

## 🧩 Analysis Modules (分析モジュール)

### 1. 🌍 ATLAS (Basic Statistics / 基本統計)
Visualizes basic statistics of the dataset.
データセットの基礎統計を可視化します。
* **Time Series**: Application trends over time. (時系列推移)
* **Rankings**: Top Applicants and IPCs. (出願人・IPCランキング)
* **Tree Maps**: Hierarchical view of IPCs or Applicants. (構成比マップ)
* **Lifecycle Map**: Technology maturity assessment (Applicants vs Applications). (技術ライフサイクル分析)

### 2. 💡 CORE (Rule-based Classification / ルールベース分類)
Classifies patents using user-defined logical rules or AI-suggested topics.
ユーザー定義の論理式、またはAIによる提案に基づいて特許を分類します。
* **AI Assistant**: Suggests classification axes using K-Means. (AIによる分類軸提案)
* **Rule Engine**: Supports complex boolean logic (AND, OR, NEAR, ADJ). (高度な論理式検索)
* **Heatmaps**: Visualizes cross-tabulation (e.g., Problem vs Solution). (ヒートマップ・バブルチャートによるクロス分析)

### 3. 🚀 Saturn V (AI Landscape / AIランドスケープ)
Generates a semantic landscape map using SBERT vectors.
SBERTベクトルを用いた意味論的な技術ランドスケープマップを生成します。
* **TELESCOPE**: Global map using UMAP & HDBSCAN clustering. (UMAPとHDBSCANによる全体マップ)
* **PROBE**: Drill-down analysis into specific clusters. (特定クラスタへのドリルダウン)
* **Auto-Labeling**: Automatically generates labels for clusters using TF-IDF. (TF-IDFによるクラスタ自動ラベリング)

### 4. 📈 MEGA (Trend & Portfolio / 動態・ポートフォリオ分析)
Analyzes macro trends and micro portfolios.
マクロな技術動態とミクロなポートフォリオを分析します。
* **PULSE**: Momentum analysis (CAGR vs Volume) to identify Leaders and Emerging players. (成長率と規模による4象限分析・動態マップ)
* **Trajectory**: Visualize historical shifts of players. (プレイヤーの時系列軌跡)
* **TELESCOPE**: Detailed portfolio mapping for specific applicants/IPCs. (特定対象のポートフォリオ詳細マップ)

### 5. 🧭 Explorer (Keyword Strategy / キーワード戦略)
Explores strategic keywords and competitor differences.
戦略的キーワードと競合他社との差異を探索します。
* **Global Overview**: Keyword co-occurrence networks. (全体共起ネットワーク)
* **Trend Analysis**: Identifies fast-growing keywords. (急上昇キーワード分析)
* **Comparative Strategy**: Tornado charts comparing two companies. (2社間のキーワード比較・トルネードチャート)
* **KWIC**: Keyword-in-Context search. (文脈検索)

### 6. 🔗 CREW (Network Analysis / ネットワーク分析)
Analyzes co-occurrence networks of inventors or applicants.
発明者や出願人の共起ネットワーク（つながり）を分析します。
* **Co-occurrence Graph**: Interactive network visualization. (インタラクティブなネットワーク図)
* **Metrics**: Betweenness Centrality, Brokerage Score, Productivity Score. (媒介中心性、技術ブローカー、生産性スコアなどの指標算出)
* **Community Detection**: Identifies research groups/factions. (コミュニティ・派閥の検出)

### 7. 🦅 EAGLE (Exploratory Landscape / 探索的ランドスケープ)
An interactive exploration module based on Saturn V, featuring manual clustering.
Saturn Vをベースにした、手動クラスタリング可能な探索的分析モジュールです。
* **Lasso Clustering**: Manually select and cluster data points. (自由選択クラスタリング)
* **Drill-down**: Detailed analysis of selected areas. (ドリルダウン分析)
* **Visual Editing**: Edit clusters and labels interactively. (視覚的なクラスタ編集)

### 8. 📝 VOYAGER (Strategic Reporting / 戦略レポート)
Compiles snapshots from all modules into a cohesive strategic narrative.
全モジュールから収集したスナップショット（証拠）を統合し、戦略的なストーリーを構築します。
* **Snapshot Curator**: Collect important charts as "Evidence" across ATLAS, Saturn V, and Explorer. (モジュール横断的な証拠収集・スナップショット機能)
* **Strategic Deep Dive**: Generates CSO-level strategic reports with Scenario Planning (Probable/Best/Risk). (CSO視点の詳細戦略レポート・シナリオプランニング機能)
* **Evidence Download**: Download gathered evidence as images consistent with report references (`Evidence X.png`). (証拠画像のダウンロード)
* **AI-Powered Insight**: Context-aware generation using Gemini 2.5 Flash. (Gemini 2.5 Flashによる文脈認識型インサイト生成)

### 9. 🌌 NEBULA (Environmental Analysis / 環境分析)
Environmental Analysis module that integrates non-patent literature (papers, news, policy documents) with patent data. Visualize gaps and synergies between social trends and technological development.

特許情報だけでなく、論文・ニュース・政策文書までを含めた「環境分析」を行うモジュールです。社会トレンドや市場の期待を統合し、特許データとのギャップやシナジーを可視化します。

---

## 🛠️ Requirements (動作環境)

* Python 3.9+
* **Key Libraries**:
    * `streamlit`
    * `pandas`
    * `sentence-transformers` (AI Vectors)
    * `umap-learn`, `hdbscan` (Dimensionality Reduction & Clustering)
    * `google-generativeai` (Likely required for VOYAGER)
    * `kaleido` (Image Export)
    * `plotly` (Interactive Charts)

## 🚀 How to Run (実行方法)

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the application:
    ```bash
    streamlit run Home.py
    ```

---
© 2025-2026 しばやま