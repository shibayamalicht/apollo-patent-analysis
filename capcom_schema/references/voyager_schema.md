# VOYAGER スキーマ

## 対象ファイル
- `voyager/mission.json`（問い + Evidence一覧）
- `voyager/evidence/ev{N}_{module}.json`（個別Evidenceデータ）
- `voyager/context.json`（データセットメタ情報）

## モジュール概要
VOYAGER はスナップショット収集とCAPCOM Export を行うモジュール。ユーザーが設定した Mission Objective（分析の問い）に基づき、各モジュールで取得したスナップショットを Evidence としてパッケージングする。CAPCOM Export は、Mission Objective・Evidence群・データセットメタ情報を構造化された3ファイル構成でエクスポートし、Claude Code が一貫した文脈のもとで戦略レポートを生成できるようにする。

## JSONスキーマ

### mission.json（ミッション定義 + Evidence一覧）

#### トップレベル構造

| フィールド名 | 型 | 説明 |
|---|---|---|
| `mission_objective` | string | ユーザーが設定した分析の問い。レポートの主題となる |
| `timestamp` | string | エクスポート時刻。ISO 8601形式（例: `"2024-12-15T14:30:00"`) |
| `evidence_count` | int | Evidence の総数 |
| `evidence_list` | [object] | 各 Evidence の概要情報（配列） |
| `available_data_files` | [string] | `data/` ディレクトリ内の全ファイル名リスト |

#### evidence_list 配列の各要素

| フィールド名 | 型 | 説明 | 例 |
|---|---|---|---|
| `id` | int | Evidence の連番ID（1始まり） | `1` |
| `module` | string | スナップショットのソースモジュール名 | `"ATLAS"`, `"Saturn V"`, `"MEGA"` |
| `title` | string | スナップショットのタイトル | `"出願件数推移 (2015-2023)"` |
| `file` | string | 個別 Evidence ファイルへの相対パス | `"evidence/ev1_atlas.json"` |
| `images` | [string] | 関連する画像ファイルパスのリスト。`snapshots/` ディレクトリ内 | `["snapshots/atlas_trend.png"]` |

#### available_data_files

| 値の型 | 説明 | 例 |
|---|---|---|
| [string] | `data/` ディレクトリに格納された補足データファイルの名前一覧 | `["patents.csv", "atlas_statistics.json", "saturnv_clusters.json", "nebula_hype_cycle.json", "nebula_macro_events.json", "saturnv_drill_wordcloud.json"]` |

各モジュールが `capcom.save_data()` で出力したJSONやCSVが含まれる。Evidence だけでは不足する場合の補足データソースとして利用可能。

---

### evidence/ev{N}_{module}.json（個別Evidenceデータ）

`{N}` は連番ID、`{module}` はソースモジュール名の小文字表記（例: `ev1_atlas.json`, `ev3_saturnv.json`）。

#### トップレベル構造

| フィールド名 | 型 | 説明 | 例 |
|---|---|---|---|
| `id` | int | Evidence のID（mission.json の evidence_list と対応） | `1` |
| `module` | string | ソースモジュール名 | `"ATLAS"` |
| `title` | string | スナップショットのタイトル | `"出願件数推移 (2015-2023)"` |
| `description` | string | スナップショットの説明テキスト。ユーザーが入力した補足や自動生成された概要 | `"対象期間の年別出願件数を棒グラフで表示"` |
| `images` | [string] | 関連画像ファイルパスのリスト | `["snapshots/atlas_trend.png"]` |
| `data_summary` | object | モジュール固有のデータサマリー（構造はモジュールにより異なる） | 下記参照 |

#### data_summary オブジェクト

`data_summary` はモジュールごとに異なる構造を持つ。各モジュールのスナップショット取得時に、分析結果の要点が格納される。

| ソースモジュール | data_summary に含まれ得る主なフィールド |
|---|---|
| ATLAS | `trend`（年別件数）、`cagr`、`applicant_ranking`、`hhi` 等 |
| CORE | `categories`（分類結果）、`classification_rules`、`hit_counts` 等 |
| Saturn V | `clusters`（クラスタ情報）、`spatial_context`、`mode`（TELESCOPE/PROBE）等 |
| MEGA | `quadrant_summary`（4象限分布）、`top_entities`、`cagr_values` 等 |
| Explorer | `co_occurrence_network`、`trending_keywords`、`kwic_results` 等 |
| CREW | `network_metrics`、`communities`、`key_persons` 等 |
| EAGLE | `manual_clusters`（手動クラスタ）、`selections` 等 |
| NEBULA | `npl_gap_analysis`、`patent_npl_alignment`、`hype_cycle_phase`、`macro_events` 等 |
| ワードクラウド | `word_frequencies`（単語→出現頻度の辞書）、`source_module`（生成元モジュール名） |

各モジュールの `data_summary` の詳細構造は、対応するモジュールのスキーマファイル（`atlas_schema.md`, `saturnv_schema.md` 等）を参照のこと。スナップショット取得タイミングにより、フルデータの部分集合が格納される場合がある。

---

### context.json（データセットメタ情報）

#### トップレベル構造

| フィールド名 | 型 | 説明 |
|---|---|---|
| `session_id` | string | CAPCOMセッションの一意識別子 |
| `dataset` | object | データセットの基本情報 |
| `modules_used` | [string] | Evidence が収集されたモジュール名のリスト |
| `available_data_files` | object | `data/` 内のファイル名と説明の辞書 |

#### dataset オブジェクト

| フィールド名 | 型 | 説明 | 例 |
|---|---|---|---|
| `total_patents` | int | データセット全体の特許件数 | `2847` |
| `period` | string | データセットの対象期間。`"開始年-終了年"` 形式 | `"2018-2024"` |
| `column_mapping` | object | `col_map` の内容。カラム名のマッピング辞書 | `{"title": "発明の名称", "abstract": "要約", ...}` |
| `preprocessing` | string | 適用された前処理パイプラインの概要 | `"Janome + SBERT(paraphrase-multilingual-MiniLM-L12-v2)"` |
| `tfidf_vocab_size` | int | TF-IDF語彙サイズ（特徴語数） | `5200` |
| `stopwords_count` | int | 適用されたストップワードの総数 | `380` |

#### modules_used

| 値の型 | 説明 | 例 |
|---|---|---|
| [string] | Evidence が1件以上存在するモジュール名の重複なしリスト | `["ATLAS", "Saturn V", "MEGA", "CREW"]` |

#### available_data_files

| キー | 値の型 | 説明 | 例 |
|---|---|---|---|
| ファイル名（string） | string | ファイルの内容説明 | `{"patents.csv": "全特許データ（フィルタ適用後）", "atlas_statistics.json": "ATLAS基本統計", ...}` |

Evidence だけでは分析の深掘りに不足する場合、ここに列挙されたファイルを選択的に読み込むことで補足データを取得できる。

## 読み取り手順

CAPCOM Export パッケージを受け取った場合、以下の順序で読み取ることを推奨する。

1. **mission.json を読み込む** -- `mission_objective` でレポートの主題（分析の問い）を確認する。これがレポート全体の方向性を決定する。
2. **evidence_list を走査する** -- 各 Evidence の `module` と `title` から、どのモジュールのどの分析結果が収集されているかを把握する。`file` フィールドで個別ファイルのパスを取得する。
3. **個別 Evidence ファイルを読み込む** -- `evidence/ev{N}_{module}.json` を開き、`data_summary` でモジュール固有の分析データを確認する。`description` にはスナップショットの文脈情報が含まれる。
4. **context.json を読み込む** -- `dataset` でデータセット全体のメタ情報（件数、期間、前処理方法等）を把握する。`column_mapping` により、各フィールドの元のカラム名を確認できる。
5. **available_data_files を確認する** -- Evidence の `data_summary` だけでは不足する場合、`data/` 内の補足ファイル（`patents.csv`, 各モジュールのJSON等）を選択的に読み込む。`context.json` の `available_data_files` にファイル名と説明の対応が記載されている。

## 解釈ガイドライン

### Mission Objective の位置づけ
- `mission_objective` はレポートの中心軸となる問い。全ての Evidence はこの問いに対する回答材料として解釈すべきである。
- 問いが具体的な場合（例: 「A社の自動運転技術における競争優位性は何か」）は、各 Evidence を問いへの直接的な回答に結びつける。
- 問いが広範な場合（例: 「技術動向の全体像を把握したい」）は、Evidence 間の関連性を重視し、複数の視点から俯瞰的な分析を構成する。

### Evidence の読み方
- **module フィールドで文脈を特定する**: 同じ数値データでも、ATLAS（全体統計）由来か MEGA（動態分析）由来かで解釈が異なる。
- **images は視覚的エビデンス**: スナップショット画像はチャートやグラフを含む。レポートでは画像を直接参照して分析の根拠とする。
- **description はユーザーの意図を反映**: 自動生成のタイトルに加え、ユーザーが付与した説明がある場合、そのスナップショットを取得した目的を示している。

### data_summary の解釈
- **モジュール横断的な一貫性を確認する**: 例えば ATLAS の `total_patents` と Saturn V の `total_patents` が一致するか確認し、フィルタ条件の差異を検出する。
- **部分データの可能性**: スナップショット取得タイミングにより、モジュールの全出力の一部のみが `data_summary` に含まれる場合がある。詳細が必要な場合は `available_data_files` から元データを参照する。
- **null値・欠損値**: 各モジュールのスキーマで定義された null 条件に従う。データ不足を意味する場合が多い。

### context.json の活用
- **preprocessing で前処理パイプラインを把握する**: テキスト分析結果（クラスタリング、キーワード抽出等）の解釈に不可欠。使用モデル（SBERT等）や形態素解析器（Janome等）の情報が含まれる。
- **column_mapping でフィールド名を解決する**: Evidence 内のフィールド参照が元データのどのカラムに対応するかを特定する。
- **modules_used で分析カバレッジを確認する**: 全9モジュールのうちどれが使用されたかにより、分析の網羅性を判断できる。

## レポートでの典型的な言及パターン

### 導入部（本分析の問いの提示）

**⚠️ Mission Objective の原文をレポート本文にベタ貼りしない。** 読解した上で、エージェントが自分の言葉で「本分析が答えようとしている問い」を咀嚼して書き下すこと（原文の語尾・箇条書き構造をそのまま転記しない）。

- OK: 「本レポートは、{問いの咀嚼: 分析で明らかにしようとしている内容を自然な日本語で 1-2 行}という観点から、{evidence_count}件の Evidence に基づき分析を行ったものである」
- OK: 「{modules_used} の {N}モジュールから収集されたデータを横断的に分析した」
- NG: 「本レポートは『{mission_objective 原文をそのまま挿入}』という問いに対し...」（原文ベタ貼り禁止）

### Evidence に基づく論証
- 「Evidence #{id}（{module}: {title}）によれば、{data_summary の要点}」
- 「{module}モジュールの分析結果（Evidence #{id}）は、{mission_objective への回答の一部} を示唆している」

### モジュール横断的な考察
- 「ATLASの出願トレンド（Evidence #{id_a}）とSaturn Vのクラスタ分布（Evidence #{id_b}）を対比すると、{統合的な知見}」
- 「MEGAの4象限分析（Evidence #{id_c}）で特定されたホットプレイヤーは、CREWのネットワーク分析（Evidence #{id_d}）における高媒介中心性ノードと一致しており、{示唆}」

### データセット文脈の補足
- 「分析対象は {total_patents}件（{period}）であり、前処理には{preprocessing}を使用した」
- 「TF-IDF語彙サイズ{tfidf_vocab_size}語、ストップワード{stopwords_count}語の条件下での結果である」

### NEBULA環境分析の位置づけ（v1.1）
NEBULAデータ（`nebula_hype_cycle.json`, `nebula_macro_events.json`）が `available_data_files` に含まれる場合、レポートの冒頭（エグゼクティブサマリー直後）に環境分析を配置する。環境分析で導出された主要仮説は、後続の各モジュール分析の文脈として参照される。詳細は `SKILL.md` Phase 5a Step 0 を参照。

### AIインサイト（prompts/）の活用
`prompts/` ディレクトリにはATLASの各タブのAIインサイトプロンプト結果が格納される。命名規則:
- `atlas_trend_insight.md` — 出願トレンド分析
- `atlas_line_insight.md` — 折れ線推移分析
- `atlas_ranking_insight.md` — 出願人ランキング分析
- `atlas_ipc_insight.md` — IPC分布分析
- `atlas_bubble_insight.md` — バブルチャート分析
- `atlas_treemap_insight.md` — ツリーマップ分析
- `atlas_lifecycle_insight.md` — ライフサイクル分析

これらは Evidence ではなく「分析の参考資料」として位置づけ、deep_diveの該当セクションで内容を反映する。

### 構成の自由度
VOYAGER のレポート構成は Mission Objective に応じて自由に設計できる。以下は典型的なパターンだが、これに限定されない。

- **テーマ別構成**: Mission Objective を複数のサブテーマに分解し、各テーマに関連する Evidence を集約して論じる
- **モジュール順構成**: NEBULA（環境分析）→ ATLAS（全体像） → Saturn V（技術領域） → MEGA（動態） → CREW（ネットワーク）の順に段階的に深掘りする
- **仮説検証型構成**: Mission Objective から仮説を立て、Evidence で検証し、結論を導く
- **比較分析型構成**: 出願人間、技術領域間、期間間の比較を軸に Evidence を配置する
