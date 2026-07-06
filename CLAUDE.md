# APOLLO v9.0.0 — 特許分析プラットフォーム

## 言語設定
- 常に日本語で会話する
- コメントも日本語で記述する
- エラーメッセージの説明も日本語で行う
- ドキュメントも日本語で生成する

## プロジェクト概要
APOLLO v9.0.0 は、Streamlitベースの特許分析プラットフォーム。
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
5. **経営層向け要約版（別冊）の任意同時生成**: Phase A で STOP-GATE 確認 → 8-12 ページに凝縮した別冊を `reports/report_executive.typ` として出力。各要点に「別解釈と決め手」、提言末尾に「見直しのサイン」ボックスを必須化（`capcom_schema/analysis/executive_summary_guide.md`、gate Check 18/18b）
6. **構造化分析技法による結論検証（v9）**: 主要結論に「結論の検証（別解釈＋決め手）」「結論の前提と見直しのサイン」、関係性立場では「相手の立場から見た合理性」点検を課す（`capcom_schema/analysis/structured_techniques.md`）。本文では読者向け呼称を使い、技法名（ACH 等）の露出は禁止（`terminology.md §2-F`・Check 8f）
7. **代表特許の決定的選定（v9）**: Phase C 冒頭で `capcom_schema/scripts/select_representatives.py` を 1 回実行し、ミクロ分析 A は `reports/representative_patents.json` の番号のみ引用（つまみ食い防止・Check 35）
8. **品質ゲート Check 1〜37（v9）**: `capcom_schema/scripts/phase_d_gate.sh` 1 本で内容量・引用・用語・スコープ・母集団タイプ・立場・水増し・マップ掲載・構造化分析・裏付けを統合判定
9. **分析の立場（narrative_stance）の確定**: Phase A STOP-GATE C で self / competitor / buyer / supplier / neutral の 5 分類をユーザー確認し、呼称・分析の力点・提言の型を全編で一貫させる（Check 11s）

> 📌 **CAPCOM セッション（ZIP 展開後のレポート生成）でこのファイルを読んでいる場合**: 「起動方法」「ファイル構成」「コアライブラリ」「技術スタック」「開発上の注意点」の各節は APOLLO 本体の開発者向け情報であり、レポート生成では参照不要。レポート生成の指示は「レポート生成」「CAPCOM 〜」各節と `capcom_schema/` を正とし、本ファイルと `capcom_schema/` が食い違う場合は `capcom_schema/SKILL.md §0` と各 analysis/ 正本が優先する。

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
| `openalex_query.py` | OpenALEX 検索式エンジン — コマンドライン構文（TI=/AB=/TA=/TX=/FT= + AND/OR/NOT + near/adj + ワイルドカード）の AST 解析・候補クエリ構築・ローカル厳密照合 |
| `apollo_kw_worker.py` | キーワード抽出のプロセス並列ワーカー（streamlit 非依存・utils.extract_keywords_batch から参照） |

### 分析モジュール (`pages/`)
| # | ファイル | モジュール名 | 機能 |
|---|---------|------------|------|
| 1 | `1_🌍_ATLAS.py` | ATLAS | 基本統計 + 多様性指標（HHI/Entropy/Gini） |
| 2 | `2_💡_CORE.py` | CORE | ルールベース分類（AND/OR/NEAR/ADJ） |
| 3 | `3_🚀_Saturn_V.py` | Saturn V | 俯瞰図分析 + ノイズ分析 + クラスタ動態マップ |
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

### CAPCOM（外部・Claude Code / Codex CLI / Antigravity IDE）
- 双方向通信: **In-Memoryセッション → ZIPダウンロード → ローカル展開 → 選択した AI ツール → Typst/PDF**
- 4フェーズ（ツァーリ・ボンバ対策版）: ミッション+データ → エビデンス+クロス → Deep Dive → 統合+品質検証
- レポート深度: 本格レポート（残り90%）
- **重要**: Web版(HF Spaces / Streamlit Cloud)対応のため、データは `st.session_state['capcom_store']` に保持されブラウザを閉じると消失する。ユーザーは必ず CAPCOM ページから ZIP をダウンロードし、ローカルで選択した AI ツール（Claude Code 等）を起動して使用する

## CAPCOM トークン効率の制約（ツァーリ・ボンバ対策）
- サブエージェント（Agent tool）を起動しないこと。全処理をメインコンテキスト内で完結させる
- 探索用エージェントの代わりに、Grep/Glob/Readツールを直接使用する
- ファイルの読み込みは必要最小限に。一度読んだ内容は会話内で参照し、再読み込みしない
- capcom_schemaのスキーマはPhase別統合ファイルを参照すること
- patents.csv の統計（出願人上位・クラスタ別・年別）は Phase A のワンショット統計スクリプトで1回だけ算出し、CSVの再読み込み・カラム名の試行錯誤をしない（コンテキスト枯渇の主因）
- コンテキストが厳しい場合は1スレッド=1フェーズに分割してよい（成果物はディスクに残る）。再開時は `reports/_carryover.md`（引き継ぎ日誌）・`reports/_phase_a_decisions.json`・`reports/representative_patents.json` の正本3点と `ls reports/` から現在地を復元する
- STOP-GATE はコンテキスト限界でも省略禁止。`AskUserQuestion` を実際に呼ばずに「回答を受信した」と仮定して進むのは重大違反（先に日誌へ保存 → `/compact` → 再開後に必ず質問する）

**重要**: 上記の効率制約は **品質ゲートを犠牲にする理由にはならない**。`capcom_schema/SKILL.md ## 0. 絶対遵守ゲートルール` が最上位。トークンが足りなければ `/compact` を実行するか、分割実施を提案する(効率のためゲート省略は禁止)。

## CAPCOM Web調査（積極推奨）
- 特許データだけでは得られない外部情報（市場動向・企業戦略・政策・学術トレンド等）をWeb調査で積極的に収集する
- NEBULAデータの有無にかかわらず、主要出願人の事業戦略・市場規模・政策動向・萌芽技術の実用化動向を調べる
- Phase B 本体作業前に STOP-GATE でテーマ一覧を提示して確認する。NEBULA 未実行時は Phase A で Web補完/省略をユーザー選択し、Web補完モードでは4カテゴリ（市場規模・政策規制・学術動向・主要企業動向）が必須でスキップ不可（Check 13）
- 分析の立場が competitor/buyer/supplier で自社名が確定している場合、自社の事業・技術・特許ポジションの Web 調査テーマを必ず含める
- APOLLOの分析結果（例: クラスタ動態で「新興」判定）をWeb情報で裏付けることで、仮説を結論に昇格させる
- Web情報をレポートに使用する場合、`#footnote[サイト名, 完全URL, 取得日]` の3要素で出所を明記し、調査ヒットごとに引き継ぎ日誌（`reports/_carryover.md`）の WEB 出所台帳へ即記録する

## 技術スタック
| カテゴリ | ライブラリ |
|---------|-----------|
| フレームワーク | Streamlit 1.41.1 |
| コアライブラリ | patiroha[all]（pandas, janome, sklearn, SBERT, UMAP, HDBSCAN, NetworkX） |
| 可視化 | plotly, matplotlib, japanize-matplotlib, wordcloud |
| レポート生成 | google-generativeai (VOYAGER), Typst (CAPCOM) |
| AI連携 | CAPCOM (Claude Code / Codex CLI / Antigravity IDE), python-pptx |
| データ取得 | requests (OpenALEX API) |

## 開発上の注意点

### テキスト前処理パイプライン（patiroha統合版）
1. Unicode正規化 (NFKC) — `patiroha.normalize_text()`
2. N-gramフィルタ（定型句除去）— `patiroha.apply_ngram_filters()`
3. キーワード抽出（複合名詞）— `patiroha.extract_keywords()`
4. TF-IDF用トークナイズ — `patiroha.tokenize_for_tfidf()`

### テスト
- patiroha: patiroha リポジトリ側で `pytest tests/` を実行（テスト数は同リポジトリ参照）
- APOLLO: Streamlit動作確認（手動）

## CAPCOM Skills (AI ツール連携)
CAPCOMセッションデータ(ZIP展開後の `session_xxx/` フォルダ)を分析・レポート生成する際:

**環境準備**: 最初に依存を確認 — `python3 -c "import pandas, pptx, PIL"` → 無ければ `pip install -r requirements-session.txt`

### 最初に読むファイル
`capcom_schema/SKILL.md`（コア手順書）→ 4フェーズの流れ・絶対遵守ゲートルール（§0）・完了条件

### 各フェーズで追加で読むリファレンスファイル（省略禁止）
| Phase | 読むファイル | 内容 |
|-------|-----------|------|
| Phase A | `analysis/terminology.md` | 用語統一ルール（最優先） |
| Phase A | `analysis/query_logic_reading.md` | 検索式の構造化読解・意図↔論理整合・データ逆読み（STOP-GATE A/B/C） |
| Phase A | `analysis/population_type_metrics.md` | 母集団5タイプ判定・タイプ別禁止表現（STOP-GATE C） |
| Phase B | `analysis/common_framework.md` | 4層分析モデル・数値根拠の書式 |
| Phase B | `analysis/data_notes.md` | 特許/NPLの非対称性・Web調査ルール |
| Phase B | `analysis/cross_module.md` | 13種のクロス分析パターン |
| Phase B | `analysis/map_reading.md` | マップ読解手順（該当セクションのみ） |
| Phase C | `analysis/deep_dive_guide.md` | Step 0-6の必須セクション・最低文字数（非空白）・ミクロ分析ルール |
| Phase C | `analysis/noise_analysis.md` | ノイズ特許の5手法分析（Step 1） |
| Phase C/D | `analysis/structured_techniques.md` | 構造化分析技法＋決定的選定の原則（統合インサイト節・結論章の執筆**直前**に読む。常時読み込み不要） |
| Phase D | `analysis/report_structure.md` | report.typ構造・付録テンプレート |
| Phase D | `analysis/patent_citation.md` | 代表特許の引用書式（セクション2-3のみ） |
| Phase D | `analysis/quality_checklist.md` | 品質チェック（gate スクリプト Check 1〜37 + 人間確認項目） |
| Phase D | `analysis/executive_summary_guide.md` | 経営層向け要約版（別冊）執筆ガイド（別冊生成フラグが ON の場合のみ） |

**各フェーズにゲートあり**: リファレンス読了内容（セクション数・パターン番号等）の報告に加え、STOP-GATE では `AskUserQuestion` でユーザー応答を得るまで次フェーズへ進まない（テキスト報告のみは不可）。Phase C/D 完了時は `capcom_schema/scripts/phase_c_gate.sh` / `phase_d_gate.sh` の実行・合格が必須（「実質的にOK」等の質的判断での上書き禁止。内容量は行数でなく非空白文字数で判定）

### 絶対ルール（省略すると品質不合格）
- **`data/patents.csv`**: 必ず読む。出願人上位・クラスタ別件数・年別件数は Phase A のワンショット統計スクリプトで1回だけ算出（リスト文字列カラムは explode 必須・再読み込み禁止）
- **`prompts/` AIインサイト**: **主要モジュール各1件以上かつ全体で最低8件**読了（件数が多ければ可能な限り全件）。読まずにdeep_diveを書くと表面的になる
- **ミクロ分析**: Phase C 冒頭に `capcom_schema/scripts/select_representatives.py` を1回実行し、代表特許（計15件以上・公開番号/タイトル/出願人＋技術的意義1-2文）は `reports/representative_patents.json` の番号のみ引用（自由選択・プレースホルダ番号・捏造番号は不合格）。出願人5社以上（各5行以上）
- **クロスモジュール分析**: **最低5パターン**、各パターン仮説→検証→結論の15-20行
- **新規分析要件**: ノイズ分析・クラスタ動態マップ4象限・多様性3指標(HHI/Entropy/Gini)・学術-特許クロス分析の解釈を必ず含める
- **構造化分析（v9）**: 結論・提言章の主要結論に「結論の検証（別解釈＋決め手）」必須、主要結論1〜3個に「結論の前提と見直しのサイン」（gate Check 30/31。技法名は本文に書かず読者向け呼称を使う＝Check 8f）
- **走査層（v9）**: 各番号セクション冒頭に `#point-lead`、各章末に `#chapter-summary`（散文の代替にしない。Check 25/26）
- **用語ルール**: レポート本文には内部ファイル名（`saturnv_clusters.json` 等）・内部フィールド名（`spatial_context`, `cluster_dynamics` 等）・内部ガイドファイル名（`*.md`）を**書いてはいけない**。`capcom_schema/analysis/terminology.md` の正式日本語呼称を使うこと
- **Web調査**: Phase B本体作業前にテーマ一覧を提示してユーザー確認を得る
