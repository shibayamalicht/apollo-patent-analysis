---
name: APOLLO CAPCOM Skills
description: >
  APOLLO特許分析プラットフォームのCAPCOMセッションデータを
  解釈し、戦略レポートを生成するための辞書・業務マニュアル。
  output/session_* フォルダ内のデータファイルを読み取る際に参照。
---

> **このファイルは要約版。各フェーズの開始前に指定されたリファレンスファイルを必ず読むこと。**

## 0. 絶対遵守ゲートルール (最優先)

**以下は他の全ルール(トークン効率制約含む)に優先する。例外なく適用する。**

1. **全ゲートは省略不可**: 「ユーザーが短く指示した」「効率上スキップしたい」等の理由でゲートを省略してはならない
2. **ユーザー応答待ち必須**: 「ユーザーに確認」「報告して」と書かれた箇所では、`AskUserQuestion` ツールでユーザー応答を取得するまで次フェーズへ進まない。テキスト出力だけで満足してはならない
3. **不合格時は強制ループ**: Phase 完了条件を満たさない場合、必ず該当 Phase に戻る。「実質的にOK」「内容は保持」等の質的判断で量的基準(行数・件数)を上書きしない
4. **指示の長さで手順を変えない**: ユーザー指示が「レポートを書いて」のように短くても、本 SKILL.md の全手順に従う。短い指示は「省略OK」のサインではなく「SKILL.md 通りに」のサイン
5. **「省略します」と宣言する前に立ち止まる**: 何かを省略する判断をした瞬間、`AskUserQuestion` でユーザーに省略の可否を確認する

このメタルールは下記「トークン効率に関する制約」よりも上位。両者が衝突する場合、本ルールが勝つ。

## トークン効率に関する制約（ツァーリ・ボンバ対策）

**以下のルールはレポートの品質とトークン効率を両立するために厳守すること。**

1. **サブエージェント禁止**: Agent toolを起動しないこと。全処理をメインコンテキスト内で完結させる
2. **ファイル読み込み最小化**: 一度読んだ内容は会話内で参照し、再読み込みしない。必要なスキーマのみ読む
3. **バッチ処理**: 複数のdeep diveをまとめて1回のやり取りで処理する
4. **Phase別スキーマ参照**: references/以下の個別スキーマは非推奨。Phase別統合スキーマを使用する

### 🚨 ゲートとの優先順位

**トークン効率制約は品質ゲートを犠牲にする理由にはならない。** ゲートが優先(`## 0. 絶対遵守ゲートルール` 参照)。

- ✅ サブエージェント禁止 → 守る
- ✅ Web調査ゲート(Phase B) → 守る(省略不可)
- ✅ deep_dive 最低行数(Phase C) → 守る(省略不可)
- ✅ 品質チェック実行(Phase D) → 守る(省略不可)
- ❌ 「効率のためゲート省略」→ 禁止

両者が衝突する場合、**ゲート優先**。トークンが足りなければユーザーに `/compact` 実行を依頼する、または分割実施を提案する。

# APOLLO CAPCOM Skills

## 1. 概要

**APOLLO** は Streamlit ベースの特許分析プラットフォーム。9つのモジュールが特許データを多角的に分析し、可視化・構造化データを生成する。

**CAPCOM** (Capsule Communicator) は APOLLO と Claude Code を繋ぐ通信モジュール。分析結果をファイル出力し、Claude Code がデータを読み取り、自由な分析やレポート生成を行う。

### セッションフォルダ構造

```
output/session_YYYYMMDD_HHMMSS/
├── data/          # patents.csv + 各モジュールJSON
├── voyager/       # VOYAGER Export時のみ（mission.json, evidence/, context.json）
├── snapshots/     # スナップショット画像(PNG)
├── prompts/       # AIプロンプト(Markdown)
├── reports/       # レポート出力先
├── capcom_schema/ # 本スキーマファイル群のコピー
└── metadata.json
```

## 2. 利用モード

### コンテキスト管理の原則（全モード共通）

1. **patents.csvは絶対に全量読み込みしない**: `head -5` でカラム構成を確認し、必要な分析の都度pandasで条件検索する
2. **JSONは必要なモジュールのみ読む**: 全JSONの一括読み込み禁止
3. **references/スキーマは対象モジュールのみ読む**: 全スキーマの一括読み込み禁止
4. **analysis/ガイドは段階的に読む**: まず `common_framework.md` のみ。他は必要な時に該当セクションのみ読む

### 自由分析モード
`data/` 配下のCSV/JSONをユーザーの質問に応じて読み取り、回答する。patents.csvの全量表示（`print(df)`, `cat`）は禁止。常にフィルタリング + `.head()` で制限する。

### レポート生成モード
VOYAGER Export 後に利用。`voyager/mission.json` の Mission Objective に基づく正式レポートを作成する。以下の4フェーズで進行する。

---

## レポート生成 4フェーズ手順

### Phase A: ミッション理解 + データ精読

voyager/mission.json を読み、data/以下のJSONとpatents.csvを把握する。

**全ステップは省略不可。**

1. `voyager/mission.json` を読む（Mission Objective + Evidence一覧）
2. `voyager/context.json` でデータセットのメタ情報を確認する
3. `evidence_list` の全件を走査し、各Evidenceの `module`・`title`・`images` を一覧表で整理する
4. `snapshots/` のファイル一覧を取得する
5. **`data/patents.csv` を読む**: `head -5` でカラム構成 → `wc -l` で件数 → pandasで出願人上位10社・クラスタ別件数・年別件数を把握
6. **`data/` 以下の全JSONファイルを確認**: 各JSONから主要数値（クラスタ数、ノイズ率、HHI/Entropy/Gini、CAGR等）をメモ
7. **`prompts/` のAIインサイトを読む**: 最低3-5件を選定し、1件ずつ読む（一括読み込み禁止）
8. 各AIインサイトから読み取った知見を具体的にメモとして書き出す

コンテキスト管理: `saturn_drill_insight.md`（最大220KB）や `crew_network_insight.md`（最大400KB）は全量読み込み禁止。対象箇所のみ `grep` で部分読み込みすること。

→ **完了条件**: patents.csv統計把握済み / 全JSONから主要数値抽出済み / AIインサイト3件以上読了・メモ作成済み / データセット全体像メモをユーザーに提示済み

### Phase A-2: レポートタイトルの決定

🛑 **STOP-GATE**: 以下を全て実行するまで Phase B へ進むな
- [ ] Mission Objective とデータ特性を踏まえ、タイトル+サブタイトルの **3案** を生成する
  - **タイトル**: 15文字以内の硬派な体言止め。問いかけ型禁止
  - **サブタイトル**: 30文字以内。具体的な件数・期間・分析手法を含める
- [ ] `AskUserQuestion` ツールで 3案を提示し、ユーザーに選択してもらう
- [ ] ユーザーが選択した案(または「Other」で指定された案)を採用
- [ ] **AI 側で勝手にタイトルを決定するのは禁止**(提示だけで満足してはならない)

### prompts/ファイル命名規則

| ファイル名パターン | モジュール | 内容 |
|---|---|---|
| `atlas_*_insight.md` | ATLAS | 各種統計分析 |
| `core_matrix_insight_*.md` | CORE | マトリクス分析 |
| `saturn_main_insight.md` | Saturn V | TELESCOPE全体俯瞰 |
| `saturn_drill_insight.md` | Saturn V | PROBE個別深掘り（**巨大、部分読み込み必須**） |
| `mega_pulse_insight.md` | MEGA | 4象限動態分析 |
| `exp_*_insight.md` | Explorer | 共起ネットワーク分析 |
| `crew_network_insight.md` | CREW | ネットワーク分析（**巨大、部分読み込み必須**） |
| `nebula_insight_*.md` | NEBULA | 特許/学術/ニュース別分析 |

---

### Phase B: エビデンス精読 + クロスモジュール分析

🛑 **STOP-GATE 1 (リファレンス読了 + クロスパターン確認)**: 以下を全て実行するまで Phase B 本体に進むな
- [ ] `analysis/common_framework.md` を読了 → 4層分析モデルと数値根拠の書式を把握
- [ ] `analysis/data_notes.md` を読了 → 特許/NPL 非対称性と Web 調査ルールを把握
- [ ] `analysis/cross_module.md` を読了 → 13種のクロスパターンから3つ以上を選定
- [ ] `AskUserQuestion` ツールで「採用するクロスパターン3つ(例: P1/P4/P13)」をユーザーに提示・確認
- [ ] ユーザー応答を待つ

🛑 **STOP-GATE 2 (Web調査の意思確認)**: Phase B 本体作業前に必須
- [ ] Mission Objective から導出された Web 調査テーマ 3-5 件を提示
- [ ] `AskUserQuestion` ツールで「実施する/しない/テーマ修正」の3択 + Other をユーザーに提示
- [ ] ユーザー応答を待つ。「省略します」「不要と判断しました」等の AI 自己判断は禁止

**Phase A の情報を参照せずに Phase B を進めてはならない。**

1. 上記3ファイルを読む（必読）
2. Evidence全件から優先順位を付ける（Mission Objectiveへの直結度で1-3のランク付け）
3. 優先度の高い5-8件を1件ずつ順次読む
4. 各Evidenceを読む際に: AIインサイトとの照合 / `map_reading.md` の該当セクション読解 / 代表特許の抽出 / スナップショット画像パス記録
5. **代表特許の具体的確認**: `data/patents.csv` をpandasで条件検索し、代表特許のタイトル・出願人・公開番号を**最低15件**取得する
6. `analysis/cross_module.md` の基本原則を読み、最低3パターン（P1-P13から）を選択・実行する
7. クロス分析で得られた洞察を記録する

→ **完了条件**: Evidence 5件以上精読済み / AIインサイト照合メモ作成済み / 代表特許15件以上取得済み / クロス分析3パターン以上の仮説→検証→結論を完了済み
→ **データ特性の注意**: `analysis/data_notes.md`（特許とNPLの非対称性、ギャップ分析の注意）
→ **Web調査ルール**: `analysis/data_notes.md` セクション3

---

### Phase C: モジュール別deep dive ⚠ スキップ禁止

🛑 **STOP-GATE (リファレンス読了 + 計画確認)**: 以下を全て実行するまで deep_dive の執筆を始めるな
- [ ] `analysis/deep_dive_guide.md` を読了 → 各 Step の必須セクション数と最低行数を把握
- [ ] `AskUserQuestion` ツールで「各モジュールの Step 数・最低行数の理解(例: Saturn V 13セクション/250行)を一覧で提示し、これで進めて良いか」をユーザーに確認
- [ ] ユーザー応答を待つ

exemplarsを参照し、全モジュールのdeep_dive.typを生成する。Phase DはPhase Cの出力ファイルを前提とする。

1. **`analysis/deep_dive_guide.md` を読む** → 各Stepの必須セクション数と最低行数を把握
2. 各モジュールのexemplarを読む → deep_dive.typを生成
3. 全deep_diveにミクロ分析A（代表特許15件以上）+ B（出願人5社以上、各5行以上）を含める
4. Step 0: NEBULA → Step 1: Saturn V → Step 2: Explorer → Step 3: MEGA → Step 4: ATLAS → Step 5: CORE → Step 6: CREW の順で処理
5. **Phase C 完了ゲート (必須実行)**: 以下のスクリプトを実行し、exit code が 0 でない場合は Phase D 開始禁止。不足モジュールを補強してから再実行する。

   ```bash
   bash capcom_schema/scripts/phase_c_gate.sh
   ```

   このスクリプトは各 deep_dive ファイルの存在と最低行数を客観的に判定する。**「実質的にOK」等の AI の質的判断による上書きは禁止**(`## 0. 絶対遵守ゲートルール` 第3項)。

→ **完了条件**: deep_dive 4ファイル以上（Saturn V + Explorer + MEGA + ATLAS）、各最低行数を満たす
→ **詳細手順**: `analysis/deep_dive_guide.md`（Step 0-6の必須セクション・最低行数・ミクロ分析ルール全て記載）

#### 最低行数一覧（クイックリファレンス）

| モジュール | 最低行数 | 必須セクション数 |
|-----------|---------|---------------|
| NEBULA | 120行 | 8セクション |
| Saturn V | 250行 | 13セクション |
| Explorer | 200行 | 11セクション |
| MEGA | 120行 | 9セクション |
| ATLAS | 120行 | 9セクション |
| CORE | 80行 | 7セクション |
| CREW | 60行 | -- |

---

### Phase D: 統合レポート + 品質検証

🛑 **STOP-GATE (リファレンス読了 + 構成確認)**: 以下を全て実行するまで report.typ の生成を始めるな
- [ ] `analysis/report_structure.md` を読了 → 章構成と deep_dive コピールールを把握
- [ ] `analysis/quality_checklist.md` を読了 → 定量チェックコマンドとチェック項目を把握
- [ ] `AskUserQuestion` ツールで「report.typ の章構成(例: 10章)と品質チェック項目数(例: 15項目)で進めて良いか」をユーザーに確認
- [ ] ユーザー応答を待つ

全deep_diveを統合し、report.typを生成する。品質チェックリスト確認。

**前提条件**: `reports/` に最低4つの `*_deep_dive.typ` が存在すること（4つ未満ならPhase Cに戻る）。

1. `ls reports/*_deep_dive.typ` でファイル存在を確認する
2. `analysis/patent_citation.md` セクション2-3を読む（引用書式の確認）
3. Phase Cで生成した全deep_diveファイルを読む
4. `report.typ` を生成する（→ `analysis/report_structure.md` セクション1の構造に従う）
5. **deep_diveの全文コピー**: 要約・圧縮・省略は一切禁止（→ `analysis/report_structure.md` セクション2）
6. **品質検証ゲート (必須実行)**: 以下のスクリプトを実行し、結果をユーザーに報告する。exit code が 0 でない場合、不合格項目を修正してから再実行する。

   ```bash
   bash capcom_schema/scripts/phase_d_gate.sh
   ```

   このスクリプトは `analysis/quality_checklist.md` の section 1 にある定量チェックコマンドを全て自動実行する。**「自前のチェックで代替」は禁止**(再現性のないチェックは無効)。

→ **完了条件**: report.typが品質基準を満たす
→ **レポート構造**: `analysis/report_structure.md`（全体構造・deep_diveコピールール・結論章ガイド・付録テンプレート）
→ **品質検証**: `analysis/quality_checklist.md`（定量チェックコマンド・全チェック項目・推奨項目）

---

## 3. モジュール一覧

| モジュール | JSON ファイル | 概要 | スキーマ |
|-----------|-------------|------|---------|
| ATLAS | atlas_statistics.json | 時系列推移、ランキング、ライフサイクル分析 | `references/atlas_schema.md` |
| CORE | core_classification.json | ルールベース特許分類 | `references/core_schema.md` |
| Saturn V | saturnv_clusters.json, saturnv_drilldown.json | AIクラスタリング (TELESCOPE/PROBE) | `references/saturnv_schema.md` |
| MEGA | mega_momentum.json, mega_drilldown.json | 動態分析 (CAGR x 活動量 4象限) | `references/mega_schema.md` |
| Explorer | explorer_global_network.json, explorer_trend.json, explorer_dominance.json | キーワード共起ネットワーク | `references/explorer_schema.md` |
| CREW | crew_network.json | 発明者/出願人ネットワーク (要約版) | `references/crew_schema.md` |
| EAGLE | eagle_clusters.json | 探索的ランドスケープ (手動クラスタリング) | `references/eagle_schema.md` |
| NEBULA | nebula_hype_cycle.json, nebula_macro_events.json | 非特許文献統合・環境分析 | `references/nebula_schema.md` |
| VOYAGER | voyager/mission.json, evidence/, context.json | 戦略レポート用データパッケージ | `references/voyager_schema.md` |
| (共通) | *_wordcloud.json | 各モジュールのワードクラウド単語頻度 | `references/wordcloud_schema.md` |

**スキーマ参照ルール**: `references/` のスキーマファイルは、そのモジュールのJSONを実際に読む直前に参照する。全スキーマの一括読み込みは禁止。

## 4. patents.csv 仕様

全特許データのCSVファイル。サイズ警告: 1,000件で1MB以上。**絶対に全量読み込みしないこと。**

### 推奨アクセスパターン
```python
import pandas as pd
df = pd.read_csv('data/patents.csv')
print(df.columns.tolist()); print(len(df))  # OK
target = df[df['cluster'] == 3][['title', 'applicant_main', 'year']].head(20)  # OK
# print(df)  ← NG（禁止）
```

### カラム構成
- **基本カラム**: title, abstract, app_num, pub_number, applicant_main, inventor_main, year, ipc_main_group
- **Saturn V追加**: cluster, cluster_label, umap_x, umap_y
- **EAGLE追加**: eagle_cluster, eagle_cluster_label
- **ドリルダウン追加**: drill_cluster, drill_cluster_label
- **MEGA追加**: mega_pulse_group, mega_drill_cluster, mega_drill_label
- **CORE追加**: core_{軸名}（ユーザー定義）

> 各モジュール実行後にpatents.csvが随時更新される。未実行モジュールのカラムは存在しない。

## 5. 分析の基本原則

1. **数値根拠**: 全ての主張に具体的な数値を含める（件数、割合、CAGR、HHI等）
2. **特許引用**: 代表特許を具体的に引用する（番号、タイトル、出願人）
3. **クロス検証**: 複数モジュールのデータを組み合わせて結論を補強する。最低3パターン実施（→ `analysis/cross_module.md`）
4. **事実と推論の分離**: 4層分析モデルを適用（→ `analysis/common_framework.md`）
5. **可視化参照（全章必須）**: 全ての章に最低1つの `#snapshot-figure()` を含める
6. **AIインサイト活用**: `prompts/` のAIインサイトを必ず参照し、深い読み取りをレポートに反映する
7. **データソーストレーサビリティ**: 全ての数値に具体的なモジュール名を含むマーカーを付与する
8. **Evidence網羅性**: Evidence総数の半数以上を分析に活用する
9. **Web調査（推奨）**: 外部情報を積極的に収集する（→ `analysis/data_notes.md` セクション3）

## 5.5 データ特性に関する注意事項

→ **詳細**: `analysis/data_notes.md`（特許とNPLの非対称性、ギャップ分析の注意、Web調査ルール）

## 5.6 分析ガイド (analysis/) と AIインサイト (prompts/)

references/ = データの「読み方」（辞書）、analysis/ = 「考え方・書き方」（分析手法）、prompts/ = 「マップからの読み取り結果」（AIインサイト）。

### analysis/ ファイル一覧

| ファイル | 内容 | 使用フェーズ |
|---------|------|-----------|
| `common_framework.md` | 4層分析モデル、数値根拠の書式、データソース明示ルール | Phase B開始時 + Phase D |
| `map_reading.md` | UMAP/共起NW/4象限/人的NW/ライフサイクルの読解手順 | Phase B（該当セクションのみ） |
| `cross_module.md` | 13種のクロスモジュール分析パターン | Phase B（基本原則 + 選択パターンのみ） |
| `patent_citation.md` | 代表特許検索・引用書式・ハルシネーション防止 | Phase D（セクション2-3のみ） |
| `noise_analysis.md` | ノイズ特許の5手法分析フレームワーク | Phase C Step 1 |
| `deep_dive_guide.md` | Step 0-6の必須セクション・最低行数・ミクロ分析ルール | Phase C（必読） |
| `report_structure.md` | report.typ構造・deep_diveコピールール・付録テンプレート | Phase D（必読） |
| `quality_checklist.md` | 定量チェックコマンド・品質チェック全項目・推奨項目 | Phase D（必読） |
| `data_notes.md` | 特許/NPL非対称性・ギャップ分析注意・Web調査ルール | Phase B/C/D |

### exemplars/ ファイル一覧

| ファイル | 内容 | 使用フェーズ |
|---------|------|------------|
| `exemplars/nebula_exemplar.typ` | NEBULA環境分析のお手本 | Phase C Step 0 |
| `exemplars/saturnv_exemplar.typ` | Saturn V / EAGLE分析のお手本 | Phase C Step 1 |
| `exemplars/explorer_exemplar.typ` | Explorer分析のお手本 | Phase C Step 2 |
| `exemplars/mega_exemplar.typ` | MEGA PULSE分析のお手本 | Phase C Step 3 |
| `exemplars/atlas_exemplar.typ` | ATLAS統計分析のお手本 | Phase C Step 4 |

> **お手本の使い方**: exemplar は「どう書くか」を具体例で示す。**exemplarを読まずにdeep_diveを書き始めてはならない。** Step 5-6はexemplarなし。

### 段階的読み込みルール

**analysis/**:
1. Phase B開始時: `common_framework.md` のみ
2. Evidence精読時: `map_reading.md` の対象セクションのみ
3. クロス分析: `cross_module.md` の基本原則 + 使用パターンのみ
4. Phase C: 各モジュールのexemplar + `deep_dive_guide.md`
5. Phase D: `report_structure.md` + `quality_checklist.md` + `patent_citation.md` セクション2-3

**prompts/**:
1. `ls -la prompts/` でファイル一覧とサイズを確認
2. Mission Objective関連の3-5ファイルを選定
3. 50KB以下 → 全量読み込み可。50KB超 → 部分読み込み（grep）
4. `saturn_drill_insight.md`（最大220KB）と `crew_network_insight.md`（最大400KB）は絶対に全量読み込みしない

## 6. データ解釈の共通ルール

### HHI（ハーフィンダール・ハーシュマン指数）
- < 0.15: 分散型 / 0.15-0.25: 中程度の集中 / > 0.25: 高集中型（寡占）

### CAGR
- 形式: パーセント表記（例: +12.3%/年）。始点と終点の出願数から幾何平均成長率

### ネットワーク密度
- < 0.1: 疎 / 0.1-0.3: 中程度 / > 0.3: 密

### MEGA 4象限
- QI (高CAGR・高活動量): 成長期 / QII (高CAGR・低活動量): 新興 / QIII (低CAGR・低活動量): 衰退 / QIV (低CAGR・高活動量): 成熟

### UMAP空間
- 近接するクラスタ: 技術的類似性が高い。UMAPは距離の絶対値より相対的な近接関係が重要

### CREW ネットワーク（要約版）
- ノード: betweenness降順top50 / エッジ: weight降順top200 / コミュニティ: top5メンバー + サイズ

### Explorer ネットワーク
- エッジ: weight降順top100 / metadata内の `n_edges_total` で全体規模を確認

## 7. レポート出力

### Typst PDF
1. `capcom_schema/templates/report_style.typ` を `reports/` にコピー
2. `report.typ` を生成（`#show: apollo-report.with(...)` で開始）
3. スナップショット画像は `#snapshot-figure("../snapshots/xxx.png", caption: "説明")` で挿入
4. テーブルは `#styled-table(columns: ..., header: ([...], [...]), ..body)` でBCG風スタイル適用
5. `typst compile --root ".." reports/report.typ reports/report.pdf`

### 利用可能な関数
- `exec-summary[...]` — エグゼクティブサマリーボックス
- `kpi-dashboard(cols: 3, kpi-card(...), ...)` — KPIダッシュボード（ページまたぎ防止）
- `kpi-card("ラベル", "値", note: "補足")` — KPIカード（**ドル記号禁止**: `$`/`\$` 不可、「ドル」「USD」で表記）
- `evidence-box(番号, "タイトル")[...]` — Evidenceボックス
- `insight-box[...]` — Key Insightボックス
- `note-box[...]` — 注釈ボックス
- `snapshot-figure("パス", caption: "説明")` — スナップショット画像
- `styled-table(columns: ..., header: (...), ..body)` — BCG風テーブル
- `conclusion-box("タイトル")[本文]` — 主要結論ハイライト
- `recommendation-card("高", "タイトル", "説明", timeframe: "短期")` — 優先度付き推奨
- `action-items("アクション1", "アクション2", ...)` — ToDoリスト

**注意**: `report_style.typ` のフォント設定を変更しないこと。`#set text(font: ...)` を report.typ に直接書かないこと。画像パスは `reports/` からの相対パス。typst compile に `--root ".."` を付けること。旧API（`#setup-page()` / `#cover-page(...)` 等）は廃止済み。

### python-pptx PPT
1. `capcom_schema/templates/apollo_template.pptx` を `reports/` にコピーする
2. `capcom_schema/templates/slides_spec.md` を**熟読**する（スライド仕様 v4.1）
3. `Presentation('reports/apollo_template.pptx')` + `slide_layouts[6]`（Blank）
4. **フォント**: `Meiryo UI` 統一。`_apply_font()` で全runに設定
5. **可視化ファースト**: チャート/図が主役。タイトル＝結論（新聞見出し方式）
6. **スライドタイプ**: `add_chart_text_slide()` 40%以上、`add_dual_panel_slide()` 15-20%、テキスト主体15%以下
7. **フォント階層**: 表紙36pt > セクション32pt > タイトル24pt > 本文16pt > 注釈14pt > テーブル13pt
8. `fit_image()` 必須。`reports/presentation.pptx` に出力。推奨25-40枚
9. **多様性ルール**: 同タイプ3枚連続禁止、空きスペースは分析視点で埋める

---

## 8. ユーザー指示の解釈ルール

| ユーザーが言ったこと | 正しい解釈 | 誤った解釈(禁止) |
|---|---|---|
| 「レポートを書いて」 | SKILL.md の全フェーズに従う | 急いでいる→省略OK |
| 「早く」「すぐに」 | 並列処理で速度UP(ゲートは守る) | ゲート省略OK |
| 「簡単でいい」 | 各セクションの記述量を短く | ゲート省略OK |
| 「適当に」 | デフォルト設定で進める | ユーザー確認スキップOK |
| 「次へ」「進めて」 | 当該ステップが完了済みなら次へ | 未完了でも次へ進む |

**省略を許可するのは、ユーザーが明示的に「Phase B は飛ばして」「Web 調査いらない」等と言った時のみ。** AI 側の推測で省略してはならない(`## 0. 絶対遵守ゲートルール` 第5項)。
