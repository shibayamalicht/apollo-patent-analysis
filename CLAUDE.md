# APOLLO v8.0.0 — 特許分析プラットフォーム

## 言語設定
- 常に日本語で会話する
- コメントも日本語で記述する
- エラーメッセージの説明も日本語で行う
- ドキュメントも日本語で生成する

## プロジェクト概要
APOLLO v8.0.0 は、Streamlitベースの特許分析プラットフォーム。
patirohaライブラリをコアエンジンとし、SBERT・UMAP・HDBSCANによる分析、
Gemini APIによるアプリ内レポート生成（VOYAGER）、
Claude Code / Codex CLI / Antigravity IDE による Deep Diveレポート生成（CAPCOM・マルチツール対応）を統合した次世代版。

## 主な機能
1. **母集団設計の文書化**: CAPCOM ページで以下4項目を任意入力可能にし、分析レポートに反映
   - 母集団論理式の設計意図 → CAPCOM に送信し分析で考慮
   - 母集団論理式 → 分析反映 + レポート付録に掲載
   - 収録年情報 → CAPCOM に送信し分析で考慮
   - 使用した特許データベース名 → 分析注記と付録で反映
2. **CAPCOM マルチツール対応**: Claude Code / Codex CLI / Antigravity IDE を複数選択可能（選択したツール用パッチがZIPに展開済みで同梱）
3. **OpenALEX 拡張**: 論文種別（article / review / book-chapter など10種）の複数選択、検索結果のCSVダウンロード
4. **レポート用語統一**: 内部ファイル名・フィールド名の露出を禁止し、正式な日本語呼称に統一（`capcom_schema/analysis/terminology.md`）
5. **経営層向け要約版（別冊）の任意同時生成**: Phase A で STOP-GATE 確認 → 8-12 ページに凝縮した別冊を `reports/report_executive.typ` として出力（`capcom_schema/analysis/executive_summary_guide.md`）

## 起動方法
```bash
pip install -r requirements.txt
streamlit run Home.py
# http://localhost:8501 でアクセス
```

## ファイル構成

### エントリーポイント・ユーティリティ
| ファイル | 役割 |
|---------|------|
| `Home.py` | Mission Control — データ取込、カラムマッピング、前処理、OpenALEX統合 |
| `utils.py` | 共通ユーティリティ — フォント、サイドバー、テーマ、描画、スナップショット |
| `utils_ai.py` | AIプロンプト生成 — 外部LLM向けプロンプト構築・コピーUI |
| `utils_spatial.py` | 空間分析 — patiroha.generate_spatial_summary への委譲 |
| `capcom.py` | CAPCOM通信 — In-Memory セッション管理(`session_state['capcom_store']`)、ZIPエクスポート時にメモリ上で動的構築 |
| `openalex.py` | OpenALEX API — 学術論文検索・取得モジュール |

### 分析モジュール (`pages/`)
| # | ファイル | モジュール名 | 機能 |
|---|---------|------------|------|
| 1 | `1_🌍_ATLAS.py` | ATLAS | 基本統計 + 多様性指標（HHI/Entropy/Gini） |
| 2 | `2_💡_CORE.py` | CORE | ルールベース分類（AND/OR/NEAR/ADJ） |
| 3 | `3_🚀_Saturn_V.py` | Saturn V | AIランドスケープ + ノイズ分析 + クラスタ動態マップ |
| 4 | `4_📈_MEGA.py` | MEGA | 動態分析（CAGR×活動量 4象限） |
| 5 | `5_🧭_Explorer.py` | Explorer | キーワード戦略（共起ネットワーク） |
| 6 | `6_🔗_CREW.py` | CREW | ネットワーク分析（媒介中心性） |
| 7 | `7_🦅_EAGLE.py` | EAGLE | 探索的ランドスケープ + クラスタ動態マップ |
| 8 | `8_📝_VOYAGER.py` | VOYAGER | Gemini APIレポート生成 + CAPCOM Export |
| 9 | `9_🌌_NEBULA.py` | NEBULA | 環境分析 + 学術クラスタ分析 + クラスタ動態マップ |
| 10 | `10_📡_CAPCOM.py` | CAPCOM | セッション管理 + ZIPエクスポート + Claude Code連携ガイド |

## コアライブラリ: patiroha
テキスト処理・統計・クラスタリング等のコアロジックは `patiroha` ライブラリに委譲する。
utils.pyには描画系（Plotly/Matplotlib）とStreamlit UI系のみを残す。

### patiroha主要API
```python
import patiroha

# ストップワード
sw = patiroha.get_stopwords()          # 特許モード
mgr = patiroha.StopwordManager(include=["general", "patent_terms"])

# キーワード抽出
kw = patiroha.extract_keywords(text, stopwords=sw)

# メタデータ
col_map = patiroha.smart_map_columns(df)
ipc = patiroha.parse_ipc("H01L31/0725")
dates = patiroha.parse_date(df["出願日"])
applicants = patiroha.normalize_applicant("トヨタ自動車株式会社;ソニー")

# 埋め込み・TF-IDF
embedder = patiroha.SBERTEmbedder()
vectors = embedder.encode(df, text_columns=["title", "abstract"])
tfidf_matrix, features = patiroha.build_tfidf(texts)

# クラスタリング
result = patiroha.build_landscape(vectors, min_cluster_size=15)
names = patiroha.auto_label(texts, result.labels, method="c-tfidf")

# 統計
div = patiroha.calculate_diversity(counts)  # HHI + Entropy + Gini
cagr = patiroha.calculate_cagr(df, year_col="year")
reps = patiroha.find_representatives(vectors, df, n=5)
reps_mmr = patiroha.find_representatives_mmr(vectors, df, diversity=0.3)

# ネットワーク
G = patiroha.build_cooccurrence_graph(keyword_lists, similarity="jaccard")
communities = patiroha.detect_communities(G, algorithm="louvain")
hubs = patiroha.get_hub_keywords(G, centrality="pagerank")

# 空間分析
summary = patiroha.generate_spatial_summary(df, "cluster", "umap_x", "umap_y")
```

## レポート生成: 二系統

### VOYAGER（アプリ内・Gemini API）
- 片道通信: スナップショット + Mission Objective → Gemini API → Markdown/PDF
- 2Phase: Analyst（モジュール別分析） → Strategist（統合レポート）
- レポート深度: 骨格（最初の10%）

### CAPCOM（外部・Claude Code）
- 双方向通信: **In-Memoryセッション → ZIPダウンロード → ローカル展開 → Claude Code → Typst/PDF**
- 4フェーズ（ツァーリ・ボンバ対策版）: ミッション+データ → エビデンス+クロス → Deep Dive → 統合
- レポート深度: 本格レポート（残り90%）
- **重要**: Web版(HF Spaces / Streamlit Cloud)対応のため、データは `st.session_state['capcom_store']` に保持されブラウザを閉じると消失する。ユーザーは必ず CAPCOM ページから ZIP をダウンロードし、ローカルで Claude Code を起動して使用する

## CAPCOM トークン効率の制約（ツァーリ・ボンバ対策）
- サブエージェント（Agent tool）を起動しないこと。全処理をメインコンテキスト内で完結させる
- 探索用エージェントの代わりに、Grep/Glob/Readツールを直接使用する
- ファイルの読み込みは必要最小限に。一度読んだ内容は会話内で参照し、再読み込みしない
- capcom_schemaのスキーマはPhase別統合ファイルを参照すること

**重要**: 上記の効率制約は **品質ゲートを犠牲にする理由にはならない**。`capcom_schema/SKILL.md ## 0. 絶対遵守ゲートルール` が最上位。トークンが足りなければ `/compact` を実行するか、分割実施を提案する(効率のためゲート省略は禁止)。

## CAPCOM Web調査（積極推奨）
- 特許データだけでは得られない外部情報（市場動向・企業戦略・政策・学術トレンド等）をWeb調査で積極的に収集する
- NEBULAデータの有無にかかわらず、主要出願人の事業戦略・市場規模・政策動向・萌芽技術の実用化動向を調べる
- Phase Bの完了前に「以下のテーマでWeb調査を行います。よいですか？」とテーマ一覧を提示して確認する
- APOLLOの分析結果（例: クラスタ動態で「新興」判定）をWeb情報で裏付けることで、仮説を結論に昇格させる
- Web情報をレポートに使用する場合、出所情報（URL、サイト名、取得日）を必ず明記する

## 技術スタック
| カテゴリ | ライブラリ |
|---------|-----------|
| フレームワーク | Streamlit 1.41.1 |
| コアライブラリ | patiroha[all]（pandas, janome, sklearn, SBERT, UMAP, HDBSCAN, NetworkX） |
| 可視化 | plotly, matplotlib, japanize-matplotlib, wordcloud |
| レポート生成 | google-generativeai (VOYAGER), Typst (CAPCOM) |
| AI連携 | CAPCOM (Claude Code), python-pptx |
| データ取得 | requests (OpenALEX API) |

## 開発上の注意点

### テキスト前処理パイプライン（patiroha統合版）
1. Unicode正規化 (NFKC) — `patiroha.normalize_text()`
2. N-gramフィルタ（定型句除去）— `patiroha.apply_ngram_filters()`
3. キーワード抽出（複合名詞）— `patiroha.extract_keywords()`
4. TF-IDF用トークナイズ — `patiroha.tokenize_for_tfidf()`

### テスト
- patiroha: `pytest tests/` — 84テスト
- APOLLO: Streamlit動作確認（手動）

## CAPCOM Skills (Claude Code連携)
CAPCOMセッションデータ(ZIP展開後の `session_xxx/` フォルダ)を分析・レポート生成する際:

### 最初に読むファイル
`capcom_schema/SKILL.md`（338行のコア手順書）→ 4フェーズの流れと完了条件

### 各フェーズで追加で読むリファレンスファイル（省略禁止）
| Phase | 読むファイル | 内容 |
|-------|-----------|------|
| Phase A | `analysis/terminology.md` | 用語統一ルール（最優先） |
| Phase B | `analysis/common_framework.md` | 4層分析モデル・数値根拠の書式 |
| Phase B | `analysis/data_notes.md` | 特許/NPLの非対称性・Web調査ルール |
| Phase B | `analysis/cross_module.md` | 13種のクロス分析パターン |
| Phase C | `analysis/deep_dive_guide.md` | Step 0-6の必須セクション・最低行数・ミクロ分析ルール |
| Phase D | `analysis/report_structure.md` | report.typ構造・付録テンプレート |
| Phase D | `analysis/quality_checklist.md` | 品質チェック（定量コマンド + 項目一覧） |
| Phase D | `analysis/executive_summary_guide.md` | 経営層向け要約版（別冊）執筆ガイド（別冊生成フラグが ON の場合のみ） |

**各フェーズにゲートあり**: リファレンスを読んだ内容（セクション数・パターン番号等）をユーザーに報告してから作業を開始すること

### 絶対ルール（省略すると品質不合格）
- **`data/patents.csv`**: 必ず読む。出願人上位・クラスタ別件数・年別件数を把握
- **`prompts/` AIインサイト**: 最低3件読了。読まずにdeep_diveを書くと表面的になる
- **ミクロ分析**: 代表特許15件以上（公開番号・タイトル・出願人）、出願人5社以上（各5行以上）
- **クロスモジュール分析**: 最低3パターン、各パターン仮説→検証→結論の15-20行
- **v7/v8新機能**: ノイズ分析・クラスタ動態マップ・多様性指標(Entropy/Gini)・学術ランドスケープの解釈を必ず含める
- **用語ルール**: レポート本文には内部ファイル名（`saturnv_clusters.json` 等）・内部フィールド名（`spatial_context`, `cluster_dynamics` 等）・内部ガイドファイル名（`*.md`）を**書いてはいけない**。`capcom_schema/analysis/terminology.md` の正式日本語呼称を使うこと
- **Web調査**: Phase B完了前にテーマ一覧を提示してユーザー確認を得る
