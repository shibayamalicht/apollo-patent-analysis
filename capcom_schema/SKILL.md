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

🛑 **STEP 0 (最優先)**: 用語統一ルールの読了と母集団メタ情報の確認
- [ ] `analysis/terminology.md` を**最初に**読む（本ガイド内で最優先の最重要ルール）
  - 内部ファイル名・内部フィールド名・内部ガイドファイル名のレポート露出禁止を理解する
  - 正式な日本語呼称一覧（空間配置分析、クラスタ動態マップ、4 象限サマリ等）を記憶する
  - **Mission Objective のベタ貼り禁止ルール（§4）**を理解する: 原文をレポート本文に転記せず、「本分析が答えようとしている問い」として咀嚼した 1-2 行で書き下す。会話的語尾（「〜してください」等）や原文の構造をそのままコピーしない
- [ ] `voyager/context.json` の `population_meta` フィールドを確認する（全項目任意）:
  - `query_intent`（母集団論理式の設計意図）→ 指定されていれば、**原文を咀嚼して「本分析の視座」として内在化**する（詳細は下記「🎯 query_intent の扱い」）
  - `query_logic`（母集団論理式）→ 指定されていれば、付録 D に `#raw` ブロックで全文掲載する（検索式は原文のまま貼ってよい）
  - `coverage_years`（収録年情報）→ 指定されていれば、付録 A の対象期間欄と時系列分析の解釈で使う
  - `database_name`（使用した特許データベース名）→ 指定されていれば、付録 A とカバレッジ注記で使う。未指定なら「提供された特許データセット」と汎用表記
- [ ] `voyager/context.json` の `capcom_tools.selected` を確認する → レポート付録 A の「CAPCOM モジュール」欄に記載

**重要**: `database_name` が未指定の場合、執筆者が勝手に J-PlatPat 等の具体名を補うことは禁止。汎用表記のみ使用すること。

🛑 **PHASE A STOP-GATE (経営層向け要約版〈別冊〉の生成確認)**:
- [ ] **レポート生成依頼を受けたら、Phase A の早い段階で必ず以下をユーザーに確認する**:
  - `AskUserQuestion` ツールで以下を提示:
    ```
    質問: 本編レポート（60-120ページ相当）に加えて、
          経営層向け要約版（別冊、8-12ページ）も生成しますか？

    選択肢:
    - ✅ 両方生成する（本編 + 別冊） — 経営層への提出を想定している場合に推奨
    - 📘 本編のみ — 詳細分析のみで十分な場合
    - ❓ 相談したい — 別冊の位置づけや粒度を議論してから決定
    ```
- [ ] ユーザー応答を待つ。「省略します」「通常は本編のみで十分と判断しました」等の AI 自己判断は禁止
- [ ] 選択結果を作業メモに記録:
  - 「両方生成」選択時 → **別冊生成フラグ = ON**。Phase D で本編完成後に続けて別冊を生成する
  - 「本編のみ」選択時 → 別冊生成フラグ = OFF。通常通り本編のみ生成
  - 「相談したい」選択時 → 本ガイド `analysis/executive_summary_guide.md` の §0（目的と位置づけ）を要約してユーザーに提示し、改めて確認
- [ ] **別冊は本編の「刈り取り版」ではなく「経営判断に資する核心を再構成した凝縮版」**である点を、フラグ ON 時に作業メモへ明示しておくこと（単なる削減で済ませるのは品質不合格）

詳細ガイド: `analysis/executive_summary_guide.md`（生成ルール・ページ構成・凝縮技法・禁止事項）

🛑 **PHASE A STOP-GATE A (query_logic 構造化読解) — `query_logic` が指定されている場合に限り必須**:

`voyager/context.json` の `population_meta.query_logic` に検索式が入っていた場合、執筆者はまずこの検索式を構造化して読解し、母集団がどのように抽出されているかを把握してからユーザーに確認する。**検索式を付録 D にコピペするだけで済ませるのは禁止**。

- [ ] **`analysis/query_logic_reading.md` を読了**（DB 別構文リファレンスと 4 ステップの読解プロセスを把握）
- [ ] `query_logic_reading.md` §1 の 4 ステップを順に実施:
  1. **DB 識別**: `population_meta.database_name` の値を使う。未指定の場合は検索式の構文特徴（`/TX` なら J-PlatPat、`HTX=` なら JP-NET、`$Wn` なら PatSnap 等）から推測
  2. **構文分解**: AND/OR/NOT で節に分け、各節を「分類条件 / キーワード条件 / 出願人条件 / 日付条件 / その他」に仕分ける
  3. **意図推定**: 各条件について「なぜこの条件がここにあるか」を推定する（例: `NOT A23*/IC` → 食品分野除外）
  4. **ユーザー確認**: `AskUserQuestion` で DB 名・分解結果・意図推定をまとめて提示し、「この読解で合っているか」を確認
- [ ] 選択肢: `AskUserQuestion` で以下を提示
  - ✅ この読解で進める
  - ✏️ 読解に誤りがある（→ 修正内容を受け取り、再度確認）
  - 💬 補足情報を追加（→ 追加情報を統合して再提示）
- [ ] **ユーザーが確定するまで次のステップに進まない**。「検索式は自明なので省略します」等の AI 自己判断は禁止
- [ ] 確定した読解内容を作業メモに固定し、以降の Phase A-B（整合性検査）で参照する

**`query_logic` が未指定の場合**はこの STOP-GATE を省略してよい（その旨をユーザーに 1 行で報告）

詳細: `analysis/query_logic_reading.md` §1（読解プロセス）、§2（DB 別構文リファレンス）

#### 🎯 `query_intent`（設計意図）の扱い — 絶対遵守の 3 原則

**原則 1: ベタ貼り禁止**
- `query_intent` の原文は**レポートのどこにも転記しない**（本分析の前提章・付録 D・各 deep_dive すべて）
- 「本分析の前提」章の「分析の視座」サブセクションでは、原文を読解した上で **3〜5 行の自然な日本語段落**として書き下す。構成要素: ①分析目的、②母集団の輪郭（含めた/除外したもの）、③どの切り口を重視するか
- 用語・語順・箇条書き構造を原文から流用せず、Mission Objective と結び付けた「本分析を読むための視座」として再構成する

**原則 2: ユーザーとの対話確認（STOP-GATE）**

🛑 **PHASE A STOP-GATE (母集団設計の理解確認) — `query_intent` が指定されている場合に限り必須**:
- [ ] `query_intent` の原文を読解し、執筆者自身の言葉で以下 3 点を整理する:
  1. **分析目的の要約**（1 行）: 「本分析は〜を明らかにするために実施される」
  2. **母集団の輪郭**（2-3 行）: 含めた領域・除外した領域・補助的に含めたものを箇条書き
  3. **分析の視座**（1-2 行）: どの切り口を重視するか
- [ ] `AskUserQuestion` ツールで、上記 3 点をユーザーに提示し、「この理解で進めてよいか」を確認する。選択肢は以下：
  - ✅ この理解で進める
  - ✏️ 修正が必要（→ ユーザーから修正内容を受け取り、再度確認）
  - 💬 補足情報を追加（→ ユーザーから追加情報を受け取り、統合して再提示）
- [ ] **ユーザーが確定するまで次のステップに進まない**。「省略します」「この理解で問題ないと判断しました」等の AI 自己判断は禁止
- [ ] 確定した理解を**「本分析の視座」として作業メモに固定**し、以降の全フェーズでこのメモを参照する

**このゲートを省略して自動で書き下すのは絶対禁止**。設計意図の解釈は分析の根幹に関わるため、必ずユーザー合意を取ってから進めること。

`query_intent` が **未指定** の場合はこの STOP-GATE を省略してよい（ただしその旨をユーザーに 1 行で報告すること）。

🛑 **PHASE A STOP-GATE (サブクエスチョン化) — `query_intent` が指定されている場合に限り必須**:

上記の `query_intent` 3 点整理のユーザー合意後、執筆者は 3 点整理を **「本分析が明らかにすべき具体的観点」** に分解する。この分解は執筆者が Phase B/C/D で論点を見失わないための **作業メモ** として機能する。

**⚠️ 絶対制約**: サブクエスチョンは執筆者の内部作業メモ専用であり、**レポート本文には「問い / 答え」「Q1 / A1」の形式で書いてはいけない**。本文は通常の宣言調の論述で書く（詳細: `terminology.md` §5-A-2）

- [ ] `query_intent` の 3 点整理を基に、**「本分析が明らかにすべき具体的観点」を 3-5 個** 箇条書きで起草（問い形式で起草してよい、レポート本文ではない）
- [ ] 各観点から **主要キーワード 1-3 個** を抽出（後の Phase D gate Check 12 で使用）
- [ ] `AskUserQuestion` ツールで以下を提示:
  ```
  質問: 本分析の視座を、より具体的な観点として 3-5 個に分解しました。
        この分解で進めてよいか確認してください。
        ※ この観点は執筆者の内部メモ用であり、レポート本文には問い/答え形式で書きません。

  SQ1: {観点 1}
    キーワード: {keyword-list}
  SQ2: {観点 2}
    キーワード: {keyword-list}
  ...

  選択肢:
  - ✅ この分解で進める
  - ✏️ 修正が必要（内容を指定）
  - 💬 観点を追加・削除（具体指定）
  ```
- [ ] ユーザー応答を待つ。AI 自己判断で分解を確定するのは禁止
- [ ] 確定結果を `reports/_phase_a_decisions.json` の `sub_questions` フィールドに保存:
  ```json
  {
    "sub_questions": [
      {"id": "SQ1", "content": "〜", "keywords": ["〜", "〜"]},
      {"id": "SQ2", "content": "〜", "keywords": ["〜"]},
      ...
    ]
  }
  ```
- [ ] 以降の Phase B/C/D では、各章・各 deep_dive で「どの SQ に答える内容か」を執筆者メモで意識する。**ただしレポート本文に `SQ1` 等の記号は出さない**

**スキップ条件**: `query_intent` が未指定の場合はこの STOP-GATE を省略（その旨をユーザーに 1 行で報告）

詳細: `analysis/terminology.md` §5-A-2（サブクエスチョン化のルールと NG/OK 例）

**原則 3: 全分析を通じた視座として機能させる**
- Phase B（Evidence 精読・クロス分析）・Phase C（各モジュール deep_dive）・Phase D（結論章）**すべてで** 確定した視座を分析の視座として内在化する
- 各章で **最低 1 箇所**は「本分析の視座に照らすと〜」という形で意図を明示的に参照する（機械的なコピペではなく、その章の文脈に溶け込ませる）
- 意図に沿った論点を優先的に掘り下げる
- 意図と整合しない結果（除外したはずの領域が部分的に混入している等）が見つかった場合は、隠さず「意図との乖離」として指摘する
- **設計意図を無視した汎用的な分析は品質不合格**

詳細は `analysis/terminology.md` 第 5-A 節参照。

🛑 **PHASE A STOP-GATE B (意図 ↔ 論理 整合性検査) — `query_intent` と `query_logic` が両方指定されている場合に限り必須**:

STOP-GATE A（`query_logic` 構造化読解）と STOP-GATE（`query_intent` のユーザー合意）の両方が完了した後、執筆者は両者を対比して乖離を検出する。**Critical 乖離を見つけても進行可能**（ユーザー判断を尊重）だが、**改善提案を必ず添える**こと。

- [ ] `analysis/query_logic_reading.md` §4「B: 意図 ↔ 論理 乖離検出チェックリスト」を開く
- [ ] §4-2 の **整合性チェック 8 項目**（技術領域 / 用途・応用 / 対象期間 / 地域・国 / 出願人絞り込み / 除外条件 / 公報種別 / 分類階層）を順に対比
- [ ] 検出した乖離を **3 段階に分類**:
  - 🔴 **Critical**: 意図と論理が矛盾している（例: 意図に「食品除外」とあるが検索式に NOT 条件がない）
  - 🟡 **Warning**: 過剰絞り込み or 不足（例: 意図に記載ない条件が検索式にある）
  - 🔵 **Info**: 解釈の幅がある（例: 意図の「構造材料」が C08L を含むか曖昧）
- [ ] Critical と Warning については **具体的な改善提案**を作成する（例: 「検索式末尾に `* NOT (A23*/IC + A21*/IC)` を追加すると意図に沿う」）
- [ ] `AskUserQuestion` ツールで乖離報告 + 改善提案を提示。以下の選択肢:
  - [A] 検索式を修正して APOLLO で再抽出する（推奨）
  - [B] このまま進め、「本分析の範囲と限界」章で乖離を明記する
  - [C] 乖離は想定内として無視する
  - ✅ 乖離なし、このまま進める
- [ ] **ユーザーが選択するまで次のステップに進まない**。提示なしで「乖離なしと判断しました」等の AI 自己判断は禁止
- [ ] ユーザーの選択結果を作業メモに記録。[B] 選択時は Phase D の「本分析の範囲と限界」章で明示的に記載すること

**スキップ条件**: `query_intent` または `query_logic` のいずれかが未指定なら、この STOP-GATE を省略（その旨をユーザーに 1 行で報告）

詳細: `analysis/query_logic_reading.md` §4（8 項目チェックリストと提示テンプレート）

1. `voyager/mission.json` を読む（Mission Objective + Evidence一覧）
2. `voyager/context.json` でデータセットのメタ情報と population_meta / capcom_tools を確認する
3. `evidence_list` の全件を走査し、各Evidenceの `module`・`title`・`images` を一覧表で整理する
4. `snapshots/` のファイル一覧を取得する
5. **`data/patents.csv` を読む**: `head -5` でカラム構成 → `wc -l` で件数 → pandasで出願人上位10社・クラスタ別件数・年別件数を把握
6. **`data/` 以下の全JSONファイルを確認**: 各JSONから主要数値（クラスタ数、ノイズ率、HHI/Entropy/Gini、CAGR等）をメモ
7. **`prompts/` のAIインサイトを読む**: 最低3-5件を選定し、1件ずつ読む（一括読み込み禁止）
8. 各AIインサイトから読み取った知見を具体的にメモとして書き出す

コンテキスト管理: `saturn_drill_insight.md`（最大220KB）や `crew_network_insight.md`（最大400KB）は全量読み込み禁止。対象箇所のみ `grep` で部分読み込みすること。

🛑 **PHASE A STOP-GATE C (データ側からの母集団実態確認 + 母集団タイプ判定) — 必須（全ケースで実施）**:

データ精読が一通り完了したら、patents.csv から算出した**母集団の実態像**をユーザーに提示し、設計意図・検索式との整合性を最終確認する。**同時に母集団タイプ（A/A'/B/C/D）も判定する**。`query_intent` / `query_logic` が未指定でも、この STOP-GATE は必ず実施する（最小限のユーザー確認は必須）。

**C-1. データ Level 2 逆読み**

- [ ] `analysis/query_logic_reading.md` §5「C: データ側からの逆読み」を開く
- [ ] §5-1 の **Level 2 項目**を算出:
  - 総件数・対象期間・使用 DB
  - 上位 10 出願人（件数・シェア）
  - 主要 IPC/FI 上位 10
  - 出願年分布（年別件数・ピーク年）
  - 出願人集中度（HHI）
  - 国/地域分布（データに含まれる場合）
- [ ] §5-1 の **自動偏り警告閾値** で異常を検出:
  - 上位 1 社が 30% 超 → ⚠️
  - 上位 1 IPC が 40% 超 → ⚠️
  - 直近 2 年に 50% 超集中 → ⚠️
  - HHI > 0.25 → ⚠️（高集中）
  - 特定国が 95% 超 → ⚠️（地理的偏り）

**C-2. 母集団タイプ判定**

- [ ] **`analysis/population_type_metrics.md` を読了**（5 タイプ分類・指標解釈表・タイプ別禁止表現リスト）
- [ ] §4-2 の **判定マトリクス**に基づき候補タイプを推定:
  - 上位 1 社シェア > 90% → **C**（単一企業）
  - 上位 1 社シェア 50-90% + 上位 5 社で 95% 以上 → **B**（競合限定）
  - 上位 5 社で 80-95% + 技術が狭い → **B** または **D**
  - 上位 10 社で 40-70% + 技術特定・出願人絞り込みなし → **A'**（技術領域）
  - 上位 10 社で < 40% + 幅広い技術 → **A**（業界全体）
  - 上位 10 社で > 70% + 複合的絞り込み条件 → **D**（特定製品・技術テーマ）
- [ ] `query_intent` と `query_logic` の内容からも候補タイプを裏付ける
- [ ] 特に注意すべきケース:
  - タイプ C（単一企業）では **出願人 HHI の算出は無意味**（HHI = 1.0 になる）
  - タイプ B/C/D では「市場集中」「業界シェア」「市場構造」等の**市場・業界解釈表現は禁止**
  - タイプ C では MEGA PULSE の 4 象限分析は事実上意味がない（代替案を検討）

**C-3. 統合ユーザー確認（STOP-GATE）**

- [ ] `AskUserQuestion` ツールで以下を統合提示:
  ```
  質問: 本母集団の実態とタイプを確認します。

  【データ実態（Level 2）】
  - 総件数: {N} 件、対象期間: {期間}、DB: {database_name}
  - 上位 10 出願人: {上位10社の件数・シェアリスト}
  - 主要 IPC: {上位10IPCのリスト}
  - 出願人 HHI: {値}
  - 自動偏り警告: {あれば列挙}

  【母集団タイプ推定】
  推定タイプ: {A/A'/B/C/D} — {タイプ名}
  推定根拠: {統計 + query_intent/logic の裏付け}

  選択肢:
  - ✅ この実態・タイプで進める
  - ✏️ タイプは違う（別タイプ名を指定して再判定）
  - 💬 偏りが想定外、このまま進めるが「本分析の範囲と限界」章で明記する
  - 🔙 検索式を修正して再抽出する（Phase A-A に戻る）
  ```
- [ ] **ユーザーが選択するまで Phase A-2（タイトル決定）へ進まない**。提示なしで「実態は想定内と判断しました」等の AI 自己判断は禁止

**C-4. `reports/_phase_a_decisions.json` への保存**

- [ ] 確定した母集団タイプ・禁止表現リスト・ユーザー決定内容を以下の形式で `reports/_phase_a_decisions.json` に保存:
  ```json
  {
    "phase_a_version": "v8.0",
    "phase_a_completed_at": "{ISO8601 タイムスタンプ}",
    "population_type": {
      "code": "{A/A'/B/C/D}",
      "label": "{タイプ名}",
      "reasoning": "{推定根拠・ユーザー確認内容}",
      "confirmed_by_user": true
    },
    "query_logic_structure": { ... },
    "intent_logic_divergences": [ ... ],
    "data_level2_warnings": [ ... ],
    "forbidden_expressions": [
      "市場は寡占", "業界シェア", "業界の集中度", "市場構造", "競争環境は〜", "業界全体で", ...
    ],
    "user_notes": "{ユーザー追加コメント}"
  }
  ```
- [ ] `forbidden_expressions` は `population_type_metrics.md` §3 の該当タイプの禁止表現リストをコピー
- [ ] このファイルは Phase C/D 執筆時に参照され、`phase_d_gate.sh` Check 11 でも自動チェック対象となる

詳細: `analysis/query_logic_reading.md` §5（Level 2 項目）、`analysis/population_type_metrics.md`（5 タイプ分類と指標解釈）

🛑 **PHASE A STOP-GATE D (NEBULA 戦略判定) — 必須（全ケースで実施）**:

APOLLO では NEBULA モジュール（非特許文献分析: 学術論文・ニュース・政策）が未実行の場合、レポートの「環境分析章」が成立しなくなる。このゲートでは、NEBULA データの有無を確認し、未実行なら **Web 調査で補完するか / 省略するか** をユーザーに選択させる。

**D-1. NEBULA データの存在確認**

- [ ] `data/` 配下に NEBULA 関連ファイルが存在するか確認:
  - `data/nebula_hype_cycle.json`
  - `data/nebula_macro_events.json`
  - `data/nebula_academic_clusters.json`
- [ ] 1 つでも存在すれば `nebula_strategy.data_available = true`、全て非存在なら `false`

**D-2. モード判定**

- [ ] `data_available = true` の場合:
  - `selected_mode = "execute"` を自動決定（通常の NEBULA 章を実施）
  - ユーザー確認は不要（そのまま次のフェーズへ）
- [ ] `data_available = false` の場合、`AskUserQuestion` ツールで以下を提示:
  ```
  質問: 本セッションでは NEBULA モジュール（非特許文献分析: 学術論文・ニュース・政策）が
        未実行です。環境分析の扱いをどうしますか?

  選択肢:
  - 🌐 Web 調査で補完する
        → Phase B で以下 4 カテゴリを Web 調査で必須カバー:
          (1) 市場規模 / 業界統計
          (2) 政策・規制動向
          (3) 学術動向 / 研究キーパーソン
          (4) 主要企業動向 / プレスリリース
        → レポートに「外部環境分析（Web 調査）」章を設置
        → 各主張に #footnote[...] で出所明記

  - 📘 NEBULA を省略する
        → レポートに環境分析章を設けない
        → 「本分析の範囲と限界」で「特許情報のみを対象」と明記
        → 学術-特許クロス分析も省略
  ```
- [ ] ユーザー応答を待つ。AI 自己判断禁止

**D-3. `_phase_a_decisions.json` への保存**

- [ ] 確定結果を以下の形式で保存:
  ```json
  {
    "nebula_strategy": {
      "data_available": false,
      "selected_mode": "web_compensation",
      "user_confirmed": true,
      "web_coverage_categories": ["市場規模", "政策・規制", "学術動向", "主要企業動向"],
      "notes": "NEBULA 未実行のため Web 調査で 4 カテゴリを補完"
    }
  }
  ```
- [ ] `selected_mode` は `"execute"` / `"web_compensation"` / `"omit"` のいずれか
- [ ] `web_compensation` モード時、`web_coverage_categories` に 4 カテゴリすべて含める
- [ ] Phase B 以降の Web 調査 STOP-GATE と Phase D で参照される

詳細: `analysis/population_type_metrics.md` §4-3（nebula_strategy フィールド仕様）

→ **完了条件**: terminology.md 読了・population_meta の4フィールド確認済み / patents.csv統計把握済み / 全JSONから主要数値抽出済み / AIインサイト3件以上読了・メモ作成済み / query_logic 構造化読解（指定時）／意図↔論理整合性検査（両方指定時）／データ逆読み(必須)／NEBULA 戦略判定(必須) の 4 STOP-GATE 全て完了 / データセット全体像メモをユーザーに提示済み

### Phase A-2: レポートタイトルの決定

🛑 **STOP-GATE**: 以下を全て実行するまで Phase B へ進むな
- [ ] Mission Objective とデータ特性を踏まえ、タイトル+サブタイトルの **3案** を生成する
  - **タイトル**: **オーソドックス**（標準的・保守的）な体言止め。**20 文字以内**の目安
    - ✅ OK: 「CNF 特許動向分析 2026」「水素貯蔵技術の競合ポジション分析」「次世代半導体製造技術ランドスケープ」「パワー半導体 GaN／SiC 市場動向分析」
    - ❌ NG: 「独断 — CNF の未来」「CNF、敗北の構造」等の扇情的・文学的タイトル
    - ❌ NG: 「CNF はどこへ向かうのか？」等の問いかけ型
    - 指針: 「{技術分野 / 対象企業} の {分析種別}」の単純な組み合わせが基本。クリエイティブなコピーは不要
  - **サブタイトル**: 30 文字以内。具体的な件数・期間・分析軸を含める（例: 「2015-2026年の特許 2,623件を対象とした競合・動態分析」）
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

- [ ] **`reports/_phase_a_decisions.json` の `nebula_strategy.selected_mode` を確認**し、以下のモード別対応を適用:

**モード `execute` の場合**（NEBULA 実行済み）:
- [ ] Mission Objective から導出された Web 調査テーマ 3-5 件を提示
- [ ] `AskUserQuestion` で「実施する / しない / テーマ修正」の 3 択 + Other を提示
- [ ] ユーザーの選択に従い進行

**モード `web_compensation` の場合**（NEBULA 未実行・Web 補完）:
- [ ] Web 調査は **スキップ不可**（Phase A STOP-GATE D でユーザーが補完を選択済み）
- [ ] 以下 **4 カテゴリすべて**をカバーする Web 調査テーマを起草:
  1. **市場規模**: 業界全体の市場規模・成長予測（調査会社レポートの概要）
  2. **政策・規制**: 関連する政策・規制動向・標準化活動（政府機関の公表情報）
  3. **学術動向**: 学術論文の引用動向・キーパーソンの研究活動
  4. **主要企業動向**: 主要出願人の事業戦略・M&A・アライアンス・プレスリリース
- [ ] `AskUserQuestion` で 4 カテゴリ分のテーマ（カテゴリごとに 1-3 件）を一覧提示し、ユーザー確認
- [ ] ユーザーが「テーマ修正 / 追加 / 削除」を選択した場合、`web_coverage_categories` で 4 カテゴリが依然カバーされていることを確認
- [ ] 4 カテゴリが 1 つでも欠ける場合、警告してユーザーに再確認（欠けたまま進めると Phase D gate Check 13 で FAIL）

**モード `omit` の場合**（NEBULA 未実行・省略）:
- [ ] 通常通りの任意 Web 調査として進行（3-5 件のテーマ提示、3 択）
- [ ] 「外部環境分析」章は作らないが、任意のトピックとしての Web 調査は可

- [ ] ユーザー応答を待つ。「省略します」「不要と判断しました」等の AI 自己判断は禁止

詳細: `analysis/population_type_metrics.md` §4-3（nebula_strategy フィールド仕様）

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
- [ ] `analysis/terminology.md` の最終確認（Phase A で読んだ内容を再確認、section 5 の品質チェックコマンドを手元に）
- [ ] `AskUserQuestion` ツールで「report.typ の章構成(例: 11章、本分析の前提章と付録D含む)と品質チェック項目数で進めて良いか」をユーザーに確認
- [ ] ユーザー応答を待つ

全deep_diveを統合し、report.typを生成する。品質チェックリスト確認。

**前提条件**: `reports/` に最低4つの `*_deep_dive.typ` が存在すること（4つ未満ならPhase Cに戻る）。

1. `ls reports/*_deep_dive.typ` でファイル存在を確認する
2. `analysis/patent_citation.md` セクション2-3を読む（引用書式の確認）
3. Phase Cで生成した全deep_diveファイルを読む
4. `report.typ` を生成する（→ `analysis/report_structure.md` セクション1の構造に従う）
   - ****: 「本分析の前提」章を先頭に配置（`population_meta` の任意項目のみ記載）
   - ****: 付録 A に `database_name` / `coverage_years` / CAPCOM モジュール行を追加
   - ****: `population_meta.query_logic` が指定されていれば付録 D「母集団検索式」を追加
5. **deep_diveの全文コピー**: 要約・圧縮・省略は一切禁止（→ `analysis/report_structure.md` セクション2）
6. **用語統一チェック**: 以下のコマンドで禁止語の混入を検出し、ヒットがゼロになるまで修正する

   ```bash
   # 内部ファイル名の混入チェック
   grep -nE "(saturnv_clusters|mega_momentum|nebula_[a-z]+|atlas_statistics|explorer_[a-z]+|eagle_[a-z]+|core_classification|crew_network)\.json" reports/report.typ && echo "❌ 内部JSONファイル名がレポートに残存" || echo "✅ 内部JSON名クリア"

   # 内部フィールド名の混入チェック
   grep -nE "\b(spatial_context|cluster_dynamics|noise_analysis|quadrant_summary|time_series|hype_cycle_phase|macro_events|npl_gap_analysis)\b" reports/report.typ && echo "❌ 内部フィールド名がレポートに残存" || echo "✅ 内部フィールド名クリア"

   # 内部ガイドファイル名の混入チェック
   grep -nE "(SKILL|common_framework|data_notes|cross_module|deep_dive_guide|report_structure|quality_checklist|terminology|map_reading|noise_analysis|patent_citation|executive_summary_guide)\.md" reports/report.typ && echo "❌ 内部ガイドファイル名がレポートに残存" || echo "✅ 内部ガイドファイル名クリア"

   # 内部プロセス用語（deep_dive / Phase A-D 等）の混入チェック
   grep -nE "\b(deep[-_ ]dive|Phase\s+[ABCD]\b|Phase\s+5a)\b" reports/report.typ && echo "❌ 内部プロセス用語がレポートに残存" || echo "✅ プロセス用語クリア"

   # J-PlatPat 等の具体DB名がユーザー未指定にも関わらず補われていないか
   # （database_name が未指定なら本文に J-PlatPat が出てはいけない）
   grep -n "J-PlatPat\|JPlatPat" reports/report.typ

   # スコープ限定ルール（terminology.md §6）: 本母集団と業界全体の混同を防ぐ
   # 限定語（本母集団/本分析の/本対象特許群 等）が十分多く、
   # 無限化語（業界では/市場では/全体として 等）が相対的に少ないか
   grep -cE "本母集団|本分析の特許群|本対象特許群|本検索式|本データセット|本分析の対象|本分析では|本分析において|本母集団に" reports/report.typ
   grep -cE "業界では|業界全体|市場では|市場全体|産業では|全体として|業界の主流|業界の標準|主流技術|業界の|市場の" reports/report.typ
   # 合格基準: 限定語 ≥ 5件 かつ 無限化語 ≤ 限定語 × 0.3
   ```

7. **品質検証ゲート (必須実行)**: 以下のスクリプトを実行し、結果をユーザーに報告する。exit code が 0 でない場合、不合格項目を修正してから再実行する。

   ```bash
   bash capcom_schema/scripts/phase_d_gate.sh
   ```

   このスクリプトは `analysis/quality_checklist.md` の section 1 にある定量チェックコマンドを全て自動実行する。**「自前のチェックで代替」は禁止**(再現性のないチェックは無効)。

8. **別冊（経営層向け要約版）生成 — Phase A で「両方生成」が選択されていた場合に必須**:
   - [ ] `analysis/executive_summary_guide.md` を読了 → ページ構成・凝縮技法・禁止事項を把握
   - [ ] 本編 `reports/report.typ` が合格している（上記 Step 7 をパス）ことを前提に、`reports/report_executive.typ` を生成する
   - [ ] **刈り取り禁止**: 本編からコピーして短縮するのではなく、**本分析の Mission Objective と `query_intent` から導かれる「今回の意思決定テーマ」に即して再構成**する（詳細は `executive_summary_guide.md` §3 参照）。定型の分類軸を機械的に当てはめるのは不可
   - [ ] ページ数は 8-12 ページ厳守（行数目安: 250-500 行）。13 ページ以上 or 7 ページ以下は不合格
   - [ ] サブタイトルに必ず「— 経営層向け要約版」を追加。本編と完全に同一のタイトルは不可
   - [ ] 手法詳細（SBERT / UMAP / HDBSCAN / min_cluster_size 等）は別冊に混入させない（合格基準: これらの言及が 3 件以下）
   - [ ] 本編と同じ `terminology.md` の用語統一ルールを遵守（内部識別子の露出禁止）
   - [ ] 結論・数値は本編と整合させる（別冊独自の再集計・独自結論は不可）
   - [ ] 生成後、以下のコマンドで別冊品質を確認:
     ```bash
     wc -l reports/report_executive.typ
     # 目安: 250-500 行

     # 用語統一（本編と同じルール）
     grep -nE "(saturnv_clusters|mega_momentum|nebula_[a-z]+|atlas_statistics|explorer_[a-z]+|eagle_[a-z]+|core_classification|crew_network)\.json" reports/report_executive.typ
     grep -nE "\b(spatial_context|cluster_dynamics|noise_analysis|quadrant_summary|time_series|hype_cycle_phase|macro_events|npl_gap_analysis)\b" reports/report_executive.typ

     # 手法詳細の混入過多チェック
     grep -c "SBERT\|UMAP\|HDBSCAN\|min_cluster_size\|n_neighbors" reports/report_executive.typ
     # 合格基準: 3 件以下

     # サブタイトルに「経営層向け要約版」が含まれているか
     grep "経営層向け要約版" reports/report_executive.typ
     ```
   - [ ] 別冊完成をユーザーに報告し、PDF 化コマンドを案内:
     ```
     本編: typst compile reports/report.typ reports/report.pdf
     別冊: typst compile reports/report_executive.typ reports/report_executive.pdf
     ```

→ **完了条件**: report.typが品質基準を満たす + 用語チェックが全てゼロヒット + 別冊フラグが ON なら report_executive.typ も品質基準を満たす
→ **レポート構造**: `analysis/report_structure.md`（全体構造・deep_diveコピールール・結論章ガイド・付録テンプレート・v8 母集団メタ反映）
→ **別冊構造**: `analysis/executive_summary_guide.md`（。経営層向け要約版の執筆ルール・ページ構成・凝縮技法）
→ **用語統一**: `analysis/terminology.md`（内部識別子の露出禁止ルールと正式日本語呼称）
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
10. **スコープ限定（絶対遵守）**: 「本母集団内の観察」と「業界全体の傾向」を明確に区別する。「本母集団では〜」「本分析の特許群では〜」等の限定修飾を必須とし、業界全体への一般化は Web 調査の外部裏付け (`#footnote[...]`) を添える（→ `analysis/terminology.md` §6）

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
| `query_logic_reading.md` | 母集団検索式の読解（DB 別構文・乖離検出・データ逆読み） | Phase A（STOP-GATE A/B/C で必読） |
| `population_type_metrics.md` | 母集団 5 タイプ分類と指標解釈ルール（タイプ B/C/D の市場・業界表現禁止） | Phase A STOP-GATE C、Phase C/D 執筆時 |

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
