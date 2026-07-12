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

6. **水増し・反復・すり抜けリライトの禁止**: 「最低◯◯行/◯件」は深さの目安であり、合否は内容の固有性＋**非空白文字数**（`wc -l` の行数ではない）で決まる。禁止: ①同一文・定型文の反復、および本母集団を見なくても言える自明な一般論での字数稼ぎ ②本文（`deep_dive.typ`・`report.typ`）の Python 等によるテンプレート生成・つなぎ文 ③ゲート回避目的で接続詞・語順・文体だけ変えて重複を温存するリライト（正しい対処は重複削除と固有内容への置換）④1文ずつ改行して行数を稼ぐこと ⑤工程ナレーション節・他章への前向き申し送り（章間連携は『クロスモジュール統合分析』章で。他章言及は過去形の根拠引用に限る）。行数不足時は新しい代表特許・数値根拠・クロスパターン・Web裏付けを足し、各段落に本母集団固有の事実（数値・公開番号・クラスタ名・出願人名のいずれか）を最低1つ含める。**正本: `analysis/deep_dive_guide.md`「記述品質の絶対基準」（`phase_d_gate.sh` Check 19/19a/8e で自動 FAIL）**

7. **STOP-GATE はコンテキスト限界でも死守（捏造・先送り厳禁）**: STOP-GATE（`AskUserQuestion` での確認）を**実際に呼ぶ前**に次フェーズへ進んではならない。**STOP-GATE に到達する前にコンテキストが限界に近づいたら**: ① `reports/_carryover.md` に現在地・確定値を保存 → ② ユーザーに「コンテキストが厳しいので一旦 `/compact` します。再開後に必ず STOP-GATE（母集団タイプ・分析の立場・別冊・タイトル・重点）を質問します」と**告げてから** `/compact` → ③ 再開後、**最初に `AskUserQuestion` で STOP-GATE を出す**。**`AskUserQuestion` を実際に呼んでいないのに「ユーザーの回答を受信した／受信できなかった」と仮定して進むのは厳禁**（存在しない回答の捏造＝重大違反。ユーザーが実際に答えるまでフェーズを進めない）。そもそも枯渇させないため、統計は **C-1 のワンショットスクリプトで1回だけ算出**し、CSV の試行錯誤・再読み込みをしない。

このメタルールは下記「トークン効率に関する制約」よりも上位。両者が衝突する場合、本ルールが勝つ。

## トークン効率に関する制約（ツァーリ・ボンバ対策）

**以下のルールはレポートの品質とトークン効率を両立するために厳守すること。**

1. **サブエージェント禁止**: Agent toolを起動しないこと。全処理をメインコンテキスト内で完結させる
2. **ファイル読み込み最小化**: 一度読んだ内容は会話内で参照し、再読み込みしない。必要なスキーマのみ読む
3. **バッチ処理**: 複数のdeep diveをまとめて1回のやり取りで処理する
4. **Phase別スキーマ参照**: references/以下の個別スキーマは非推奨。Phase別統合スキーマを使用する

### 🚨 ゲートとの優先順位

**トークン効率制約は品質ゲートを犠牲にする理由にはならない。** 両者が衝突する場合はゲート優先（各ゲートの内訳は `## 0. 絶対遵守ゲートルール` 第1-2項の再掲につき同節参照）。トークンが足りなければユーザーに `/compact` 実行を依頼するか、分割実施を提案する（効率のためのゲート省略は禁止）。

> **本フローはディスクから再開可能**: 各フェーズの成果物（`reports/_phase_a_decisions.json`、`reports/<module>_deep_dive.typ`、`reports/report.typ`）がディスクに残るため、コンテキストが厳しいツール（Codex 等）では **1スレッド=1フェーズ（Phase C は1モジュールずつ）に分割**し、新スレッドで `ls reports/` を見て続きから再開してよい。1スレッドで全フェーズを通そうとして枯渇するより確実。
>
> ### 🧠 フェーズ間引き継ぎ日誌（`reports/_carryover.md`）— 分割しても分析の記憶を失わない
> 成果物と確定値だけでは失われる「仮説検証過程・Web調査の出所（URL/取得日）・判断理由」を残す追記式台帳（内部作業メモ・**レポート本文へ転載厳禁**）。
> - **作成**: Phase A 開始時に `capcom_schema/templates/carryover_template.md` を `reports/_carryover.md` にコピーする（既にあれば上書きしない）
> - **書く（append-only・各追記に `[日付/Phase/thread]`）**: Phase A 完了時（`_phase_a_decisions.json` 保存とペア）／**Web調査は1件ヒットごとに即 WEB出所台帳へ1行**（URL・サイト名・取得日。後回し禁止）／クロス1パターン確定ごと／deep_dive 1本完了ごと（ポインタのみ・本文は二重保存しない）／**スレッドを閉じる・`/compact` する直前に現在地と直近の固有事実をフラッシュ**
> - **読む（新スレッド開始時の固定手順）**: ① `ls reports/` で到達点判定 → ② `_carryover.md` の STATUS/RESUME・直近フェーズ節・WEB出所台帳・申し送りを読む → ③ `_phase_a_decisions.json` を読む → ④ 着手フェーズの `analysis/` ガイドを読み直す → ⑤ STOP-GATE で「日誌と決定ファイルを読了し現在地を復元した」と1行報告してから着手。**日誌に既にある情報（AIインサイト・Evidence）は再読しない**（再読はトークン枯渇の主因）
> - **役割分担**: 機械可読の確定値は `_phase_a_decisions.json`（gate が読む正本）、散文の記憶・Web台帳は `_carryover.md`。二重管理しない（矛盾時は JSON が正本）
> - **Web→脚注**: Phase D で本文に主張を書く際、WEB出所台帳の該当行から `#footnote[サイト名 (URL), 取得日: YYYY-MM-DD]` を生成し「footnote化」列を「済」にする（全行「済」＝Web反映漏れなし。Check 6/13 を構造的に満たす）

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

### 対話型レポート作成モード（KATHERINE）
`voyager/context.json` の `report_mode` が `"interactive"` の場合、またはユーザーが対話型での進行を明示した場合に使用。**`capcom_schema/interactive/SKILL_INTERACTIVE.md` を読み、それに従って進行する**（変わるのは進行様式のみ。品質ゲート・成果物形式・本ファイル §0 の絶対遵守ゲートルール・トークン効率制約は上記レポート生成モードと同一に適用される）。`report_mode` が未指定・`"autonomous"` で、ユーザーの明示指示もない場合は、従来どおり上記レポート生成モードで進行する。

---

## 環境準備（依存インストール・最初に1回）

レポート生成は **patents.csv の解析に `pandas`、スライド生成に `python-pptx` / `Pillow`** を使う（セッションフォルダ直下の **`requirements-session.txt`** に列挙済み）。**Phase A のデータ精読に入る前に、依存を必ず確認・導入すること**（未導入のままだと `ModuleNotFoundError` で止まる）:

```bash
# セッションフォルダ直下で実行（揃っていればスキップ、無ければ一括導入）
python3 -c "import pandas, pptx, PIL" 2>/dev/null && echo "依存OK" || pip install -r requirements-session.txt
```

- `pip` が無ければ `python3 -m pip install -r requirements-session.txt`（権限エラーは末尾に `--user`）。仮想環境を使うなら `python3 -m venv .venv && source .venv/bin/activate` の後にインストールし、以降の `python3` も同じシェルで実行する。
- ネットワーク制限等で `pip install` が通らない場合は、依存が無いまま分析を始めず、ユーザーにセッションフォルダでのインストールと再開を依頼して一旦停止する。

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
- [ ] **レポート生成依頼を受けたら、Phase A の早い段階で必ずユーザーに確認する**: `AskUserQuestion` で「本編レポート（60-120ページ相当）に加えて、経営層向け要約版（別冊、8-12ページ）も生成しますか？」を提示。選択肢: 「✅ 両方生成する（本編 + 別冊）— 経営層への提出想定なら推奨 ／ 📘 本編のみ — 詳細分析のみで十分な場合 ／ ❓ 相談したい — 別冊の位置づけや粒度を議論してから決定」
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
- [ ] `query_logic_reading.md` §1 の 4 ステップを順に実施（下記は要約。正本は `analysis/query_logic_reading.md` §1、乖離時は正本を優先）:
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

**⚠️ 絶対制約**: サブクエスチョンは執筆者の内部作業メモ専用であり、**レポート本文には「問い / 答え」「Q1 / A1」の形式でも `SQ1` 等の記号でも書いてはいけない**。本文は通常の宣言調の論述で書く（正本: `terminology.md` §5-A-2）

- [ ] `query_intent` の 3 点整理を基に、**「本分析が明らかにすべき具体的観点」を 3-5 個** 箇条書きで起草（問い形式で起草してよい、レポート本文ではない）
- [ ] **確定した立場（`narrative_stance`）の観点で SQ を点検する**（`query_intent` を最優先しつつ、立場が示唆する観点の抜けを確認）: `self`=自社の弱点・空白・強化領域／`competitor`=対象企業の隙・牙城・参入余地／`neutral`=強み・リスク・投資/提携の妙味。立場は `query_intent` に既に織り込まれていることが多い（機械的に SQ を立場で置き換えるのではなく、抜けの点検に使う）。立場と `query_intent` が食い違う場合は STOP-GATE C の立場確認で解消する。
- [ ] 各観点から **主要キーワード 1-3 個** を抽出（後の Phase D gate Check 12 で使用）
- [ ] `AskUserQuestion` ツールで、SQ1〜SQn の一覧（各観点＋キーワード）と「この観点は内部メモ用で、レポート本文には問い/答え形式で書かない」旨を提示し、選択肢「✅ この分解で進める ／ ✏️ 修正が必要（内容を指定） ／ 💬 観点を追加・削除（具体指定）」で確認する
- [ ] ユーザー応答を待つ。AI 自己判断で分解を確定するのは禁止
- [ ] 確定結果を `reports/_phase_a_decisions.json` の `sub_questions` フィールドに保存（各要素は `{"id": "SQ1", "content": "〜", "keywords": ["〜", "〜"]}` の形式）
- [ ] 以降の Phase B/C/D では、各章・各 deep_dive で「どの SQ に答える内容か」を執筆者メモで意識する

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
- [ ] §4-2 の **整合性チェック 8 項目**（技術領域 / 用途・応用 / 対象期間 / 地域・国 / 出願人絞り込み / 除外条件 / 公報種別 / 分類階層）を順に対比（下記項目は要約。正本は `analysis/query_logic_reading.md` §4-2、乖離時は正本を優先）
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
   - ⚠️ **モジュールJSONが欠落している場合のフォールバック**: あるモジュールの構造化JSON（例 `mega_momentum_<軸>.json`）が `data/` に無くても、**同名モジュールのプロンプト `prompts/<module>_*.md` の「# データ (Data)」節に同じ生データ表が埋め込まれている**（出願人/IPC等 × CAGR・活動量・総数・象限の表）。JSONが無い時はこの表をそのままデータ源として使い、deep_dive を書く。スナップショット（`snapshots/`）も併用する。**欠落と代替使用を `reports/_carryover.md` に1行記録**し、レポートには「分析の範囲と限界」で軽く注記する（JSONが無い＝そのモジュールを省略、ではない）
7. **`prompts/` のAIインサイトを読む**: **主要モジュール（Saturn V/MEGA/ATLAS/Explorer/CREW/NEBULA/CORE）各1件以上、かつ全体で最低8件**を、1件ずつ読む（一括読み込み禁止）。インサイトが少なければ全件読む。読了数が少ないと deep_dive が表面的になる。**`prompts/` の各ファイルは「役割・メタデータ・# データ (Data) 表・分析指示」を含むプロンプト**であり、その「# データ (Data)」表は対応するモジュールJSONと同じ生データなので、数値根拠の一次ソースとして使える
8. 各AIインサイトから読み取った知見を具体的にメモとして書き出す

コンテキスト管理: 巨大インサイト（`saturn_drill_insight.md`・`crew_network_insight.md`）は全量読み込み禁止・対象箇所のみ `grep`（サイズと規則の正本は §5.6 段階的読み込みルール）。

🛑 **PHASE A STOP-GATE C (データ側からの母集団実態確認 + 母集団タイプ判定) — 必須（全ケースで実施）**:

データ精読が一通り完了したら、patents.csv から算出した**母集団の実態像**をユーザーに提示し、設計意図・検索式との整合性を最終確認する。**同時に母集団タイプ（A/A'/B/C/D）も判定する**。`query_intent` / `query_logic` が未指定でも、この STOP-GATE は必ず実施する（最小限のユーザー確認は必須）。

**C-1. データ Level 2 逆読み**

**⚠️ patents.csv の実カラム（試行錯誤＝コンテキスト枯渇を避けるため最初に把握する）**: カラムは **処理済み**（`applicant_main`=主出願人 / `inventor_main` / `year`=出願年 / `ipc_main_group` / `cluster`〈整数〉 / `cluster_label`〈`'[3] 半導体記憶, メモリセル, 半導体'` 形式〉 / `umap_x`,`umap_y` / `core_{軸名}` 分類）と **原データ**（`発明名称`〈先頭に BOM あり〉/ `要約` / `出願番号` / `公開番号`）の**混在**。⚠️ **`applicant_main` / `inventor_main` / `ipc_main_group` はリスト文字列**（`"['キオクシア', '東芝']"` 形式）なので集計には展開（explode）必須＝下記スクリプトの `listcol` が対応（しないと共同出願を1社と誤カウント）。**`voyager/context.json` の `column_mapping` は元 CSV 名で実カラムと一致しない**ので照合に使わない。**ステータス（権利状況）列は無い** — 権利化率は `prompts/atlas_*_insight.md` のステータス内訳から読む。

**ワンショット統計スクリプト**（実データで検証済み・BOM/リスト文字列対応。**heredoc の多重実行・カラム名の探り当て・Unicode 正規化の試行は禁止＝今までの枯渇の主因**）:
```python
import pandas as pd, ast
df = pd.read_csv("data/patents.csv", encoding="utf-8-sig")
def listcol(col):  # "['A','B']" 形式のリスト文字列を展開する
    def parse(x):
        try:
            v = ast.literal_eval(x) if isinstance(x, str) and x.strip().startswith('[') else x
            return v if isinstance(v, list) else [v]
        except Exception:
            return [x]
    return df[col].dropna().apply(parse).explode().astype(str).str.strip()
print("総件数:", len(df), "| 期間:", int(df['year'].min()), "-", int(df['year'].max()))
ap = listcol('applicant_main')
print("\n[上位出願人]\n", ap.value_counts().head(10).to_string())
sh = ap.value_counts(normalize=True); print("\n出願人HHI:", round((sh**2).sum(), 4))
print("\n[年別件数]\n", df['year'].value_counts().sort_index().to_string())
print("\n[上位IPC]\n", listcol('ipc_main_group').value_counts().head(10).to_string())
print("\n[クラスタ別件数]\n", df.groupby(['cluster','cluster_label']).size().to_string())
```
（patents.csv に無い指標〈権利化率・Fターム等〉は `prompts/` や `data/*.json` から得る。CSV を読み直さない）

- [ ] `analysis/query_logic_reading.md` §5「C: データ側からの逆読み」を開き、§5-1 の **Level 2 項目**（総件数・期間・DB／上位10出願人〈件数・シェア〉／主要IPC/FI上位10／出願年分布／出願人HHI／国・地域分布）を算出する（項目の正本: 同 §5-1、乖離時は正本を優先）
- [ ] §5-1 の **自動偏り警告閾値** で異常を検出:
  - 上位 1 社が 30% 超 → ⚠️
  - 上位 1 IPC が 40% 超 → ⚠️
  - 直近 2 年に 50% 超集中 → ⚠️
  - HHI > 0.25 → ⚠️（高集中）
  - 特定国が 95% 超 → ⚠️（地理的偏り）

**C-2. 母集団タイプ判定**

- [ ] **`analysis/population_type_metrics.md` を読了**（5 タイプ分類・指標解釈表・タイプ別禁止表現リスト）
- [ ] §4-2 の **判定マトリクス**に基づき候補タイプを推定（下記は要約。正本は `analysis/population_type_metrics.md` §4-2、乖離時は正本を優先）:
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

**C-2'. 分析の立場（叙述スタンス）判定 — 母集団タイプとは独立に必ず判定**

母集団タイプ（データの構成）と「**誰の意思決定のためのレポートか**（＝提言・主張を語る立場）」は**別物**である。母集団が単一企業（タイプ C）でも読者＝依頼主は競合・投資家・アナリストのこともあるため、**`population_type` が C だからといって対象企業を自動的に「当社」と呼んではならない**（立場の取り違え＝対象企業を勝手に「当社」と書く誤りが実テストで頻発）。

- [ ] `query_intent` / Mission Objective から **分析の立場** を推定する（5 分類）:

  | コード | 立場 | 対象企業の呼び方 | 一人称「当社/弊社」 |
  |---|---|---|---|
  | **self** | 自社視点（当事者本人が自社を点検） | 「当社」 | 可 |
  | **competitor** | 競合・他社視点（水平。対象企業をベンチマーク） | 企業名で三人称（「キオクシアは」） | 提言で読み手＝競合自身を指す時のみ可 |
  | **buyer** | **自社＝買い手／対象＝供給元**（自社が対象から仕入れる。例: 自社=Apple, 対象=キオクシア） | 企業名で三人称 | 読み手＝買い手自身を指す時のみ可 |
  | **supplier** | **自社＝供給元／対象＝顧客**（自社が対象に納入する。例: 自社=東京エレクトロン, 対象=キオクシア） | 企業名で三人称 | 読み手＝供給側自身を指す時のみ可 |
  | **neutral** | 中立・第三者視点（投資家・アナリスト・調査） | 企業名で三人称 | 不可 |

- [ ] 推定の手がかり: `query_intent` に「自社」「当社」「我々の」→ **self**／「競合」「ベンチマーク」「対抗」→ **competitor**／「調達」「サプライヤー選定」「供給元」「仕入れ」→ **buyer**／「販売先」「納入先」「顧客の技術動向」→ **supplier**／「投資判断」「評価」「調査」や立場の記述なし → **neutral**（既定）。**手がかりが弱ければ neutral を仮置きし、C-3 で必ずユーザーに確認する**（勝手に self にしない）。
- [ ] 確定した立場は提言・主張・エグゼクティブサマリー・別冊の**全セクションで一貫**させる。**呼称だけでなく分析の力点・提言のロジックも立場に合わせる**（`self`=自社の打ち手／`competitor`=競合の対抗・参入／`buyer`=調達戦略・依存/供給リスク／`supplier`=供給戦略・内製化リスク／`neutral`=第三者の評価・予測。呼称を三人称にしただけの「べき論」にしない）。`self` 以外では対象企業を三人称で呼び「当社/弊社/我が社」を使わない（正本: `terminology.md` の「分析の立場」節）。
- [ ] **立場が `competitor` / `buyer` / `supplier`（＝関係性立場・自社 ≠ 対象）の場合は「自社（分析を行う側の企業）」の特定が必須**: 対象企業（`subject_company`）に加え、分析を行う**自社の社名**を C-3 でユーザーに尋ね `narrative_stance.own_company` に記録する。自社は Phase B で **Web 調査**し、対象企業との**対比**（自社の強み/弱み/空白 vs 対象企業）に用いる（正本: `analysis/data_notes.md` §3、`analysis/terminology.md` §6-2-B）。`buyer`/`supplier` の依存・交渉力の読み方は**母集団タイプ次第**: タイプ A/A'/B なら出願人 HHI・上位集中を供給側/需要側の集中シグナルとして読める（特許≠市場につき Web 裏付け）が、**⚠️ 対象が単一企業（タイプ C）なら HHI は無意味**（出願人ほぼ1社＝HHI≈1）——依存・交渉力は **Web 調査（対象の市場シェア・取引構造）**から読む。`self` では `subject_company` が自社そのもの、`neutral` では自社は存在しない（`own_company` は空文字）。

**C-3. 統合ユーザー確認（STOP-GATE）**

- [ ] `AskUserQuestion` ツールで以下を統合提示する（質問: 「本母集団の実態とタイプを確認します」）:
  - **データ実態（Level 2）**: 総件数・対象期間・DB／上位10出願人（件数・シェア）／主要IPC／出願人HHI／自動偏り警告（あれば列挙）
  - **母集団タイプ推定**: {A/A'/B/C/D} — タイプ名＋推定根拠（統計 + query_intent/logic の裏付け）
  - **分析の立場推定**: {self/competitor/buyer/supplier/neutral}＋根拠（query_intent からの推定）。「この立場で対象企業を『当社』と呼ぶか三人称かが決まる」「関係性立場（competitor/buyer/supplier）なら分析を行う『自社』の社名も確認する（Web 調査で対比に用いるため）」旨を注記
  - **選択肢（5つ全て提示）**: ✅ この実態・タイプ・立場で進める ／ ✏️ タイプが違う（別タイプ名を指定して再判定） ／ 👤 立場が違う（自社／競合／取引先・買い手／サプライヤー／中立 のいずれかを指定） ／ 💬 偏りが想定外、このまま進めるが「本分析の範囲と限界」章で明記する ／ 🔙 検索式を修正して再抽出する（Phase A-A に戻る）
- [ ] **⚠️ 立場を独立の `AskUserQuestion`（複数質問に分ける場合）として聞く時の選択肢（必須）**: 選択肢は**最大4つ**のため5立場を全部は並べられないが、**`competitor` / `buyer` / `supplier` を『その他』に畳んではいけない**（見えないと選べない＝関係性立場が死ぬ）。次の**4択**で提示する（推定した立場を先頭に置き `(推奨)` を付す）:
  1. **中立**（第三者・投資家/アナリスト。対象を三人称で評価）
  2. **自社視点**（当事者本人＝対象企業の社内向け。対象を「当社」）
  3. **競合視点**（対象を競合としてベンチマーク。選ぶと自社名を確認）
  4. **取引先・買い手／サプライヤー視点**（自社と対象の垂直取引関係。選ぶと**取引の向き**＋自社名を確認）
  選択肢 3・4（関係性立場）が選ばれたら、続く `AskUserQuestion` で具体的な立場＋**自社名（`own_company`）**を確認する。**⚠️ コード名は常に『自社の役割』**なので、4 の確認は buyer/supplier の語で迷わせず**取引の向きで訊く**: 「① 自社が対象**から仕入れる**側 → `buyer`／② 自社が対象**に納入する**側 → `supplier`」（定義と例の正本: 上記 C-2' の 5 分類表・`terminology.md` §6-2-B）。**単一企業母集団（タイプ C）でも 5 立場すべて有効**（単一企業だから self/neutral だけ、と決めつけない）。
- [ ] **ユーザーが選択するまで Phase A-2（タイトル決定）へ進まない**。提示なしで「実態は想定内と判断しました」等の AI 自己判断は禁止
- [ ] **立場が `competitor` / `buyer` / `supplier`（関係性立場）に確定したら、続けて自社名を尋ねる（必須）**: 未取得なら追加の `AskUserQuestion` で「本分析を行う『自社』の社名」を確認し、`narrative_stance.own_company` に保存する。**Phase B STOP-GATE の Web 調査テーマ提示に「自社（{社名}）の事業・技術・特許ポジション」を必ず含める**（→ `analysis/data_notes.md` §3）。ユーザーが「自社名は伏せる／一般的な視点でよい」と答えた場合は `own_company` を空文字にし、その旨を 1 行報告して従来どおり進める。

**C-4. `reports/_phase_a_decisions.json` への保存**

- [ ] 確定した母集団タイプ・禁止表現リスト・ユーザー決定内容を以下の形式で `reports/_phase_a_decisions.json` に保存:
  ```json
  {
    "phase_a_version": "v9.0",
    "phase_a_completed_at": "{ISO8601 タイムスタンプ}",
    "population_type": {
      "code": "{A/A'/B/C/D}",
      "label": "{タイプ名}",
      "reasoning": "{推定根拠・ユーザー確認内容}",
      "confirmed_by_user": true
    },
    "narrative_stance": {
      "code": "{self/competitor/buyer/supplier/neutral}",
      "label": "{自社視点（当社）/競合視点/中立（投資家・アナリスト）}",
      "subject_company": "{分析の主役（対象/ベンチマーク先）企業名・特定企業がなければ空文字}",
      "own_company": "{competitor/buyer/supplier 視座で分析を行う『自社』の社名。self では subject_company と同一。neutral・自社名を伏せる場合は空文字}",
      "first_person_allowed": {true（self のみ）/false},
      "reasoning": "{立場の推定根拠・ユーザー確認内容}",
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
- [ ] このファイルは Phase C/D 執筆時に参照され、`phase_d_gate.sh` Check 11（母集団タイプ）・Check 11s（分析の立場 `narrative_stance`）でも自動チェック対象となる

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

→ **完了条件**: terminology.md 読了・population_meta の4フィールド確認済み / patents.csv統計把握済み / 全JSONから主要数値抽出済み / AIインサイト8件以上（主要モジュール各1件以上）読了・メモ作成済み / query_logic 構造化読解（指定時）／意図↔論理整合性検査（両方指定時）／データ逆読み(必須)／NEBULA 戦略判定(必須) の 4 STOP-GATE 全て完了 / データセット全体像メモをユーザーに提示済み

### Phase A-2: レポートタイトルの決定

🛑 **STOP-GATE**: 以下を全て実行するまで Phase B へ進むな
- [ ] Mission Objective とデータ特性を踏まえ、タイトル+サブタイトルの **3案** を生成する
  - **タイトル**: **オーソドックス**（標準的・保守的）な体言止め。**20 文字以内**の目安
    - ✅ OK: 「CNF 特許動向分析 2026」「水素貯蔵技術の競合ポジション分析」等
    - ❌ NG: 「独断 — CNF の未来」「CNF、敗北の構造」等の扇情的・文学的タイトル／「CNF はどこへ向かうのか？」等の問いかけ型
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
- [ ] `analysis/cross_module.md` を読了 → 13種のクロスパターンから**5つ以上**を選定（Phase B 完了条件・gate Check 4 の最低数と同じ）
- [ ] `AskUserQuestion` ツールで「採用するクロスパターン5つ以上(例: P1/P4/P7/P9/P13)」をユーザーに提示・確認
- [ ] ユーザー応答を待つ

🛑 **STOP-GATE 2 (Web調査の意思確認)**: Phase B 本体作業前に必須

- [ ] **`reports/_phase_a_decisions.json` の `narrative_stance` を確認 — 立場が `competitor` / `buyer` / `supplier`（関係性立場）かつ `own_company` が非空なら（下記モードに関わらず必須）**: 提示する Web 調査テーマに「**自社（{own_company}）の事業内容・主要製品・技術/特許ポジション・市場での立ち位置**」を必ず 1 件以上含める（目的・使い方の正本: `analysis/data_notes.md` §3、`analysis/terminology.md` §6-2-B）。取得情報は脚注（サイト名・URL・取得日）を付し、少なくとも 1 つのモジュール章または提言章で自社 vs 対象企業の対比に使う。
- [ ] **`reports/_phase_a_decisions.json` の `nebula_strategy.selected_mode` を確認**し、以下のモード別対応を適用:

**モード `execute` の場合**（NEBULA 実行済み）:
- [ ] Mission Objective から導出された Web 調査テーマ 3-5 件を提示
- [ ] `AskUserQuestion` で「実施する / しない / テーマ修正」の 3 択 + Other を提示
- [ ] ユーザーの選択に従い進行

**モード `web_compensation` の場合**（NEBULA 未実行・Web 補完）:
- [ ] Web 調査は **スキップ不可**（Phase A STOP-GATE D でユーザーが補完を選択済み）
- [ ] **4 カテゴリすべて**（①市場規模 ②政策・規制 ③学術動向 ④主要企業動向 — 各内容の正本は Phase A STOP-GATE D-2 の提示文）をカバーするテーマをカテゴリごとに 1-3 件起草し、`AskUserQuestion` で一覧提示してユーザー確認
- [ ] ユーザーが「テーマ修正 / 追加 / 削除」を選択した場合も、`web_coverage_categories` で 4 カテゴリが依然カバーされていることを確認。1 つでも欠ける場合は警告して再確認（欠けたまま進めると Phase D gate Check 13 で FAIL）

**モード `omit` の場合**（NEBULA 未実行・省略）:
- [ ] 通常通りの任意 Web 調査として進行（3-5 件のテーマ提示、3 択）
- [ ] 「外部環境分析」章は作らないが、任意のトピックとしての Web 調査は可

- [ ] ユーザー応答を待つ。「省略します」「不要と判断しました」等の AI 自己判断は禁止

**Phase A の情報を参照せずに Phase B を進めてはならない。**

1. Evidence全件から優先順位を付ける（Mission Objectiveへの直結度で1-3のランク付け）
2. 優先度の高い5-8件を1件ずつ順次読む
3. 各Evidenceを読む際に: AIインサイトとの照合 / `map_reading.md` の該当セクション読解 / 代表特許の抽出 / スナップショット画像パス記録
4. **代表特許の具体的確認**: `data/patents.csv` をpandasで条件検索し、代表特許のタイトル・出願人・公開番号を**最低15件**取得する
5. `analysis/cross_module.md` の基本原則を読み、**最低5パターン**（P1-P13から）を選択・実行する
6. クロス分析で得られた洞察を記録する

→ **完了条件**: Evidence 5件以上精読済み / AIインサイト照合メモ作成済み / 代表特許15件以上取得済み / クロス分析5パターン以上の仮説→検証→結論を完了済み
→ **データ特性・Web調査ルール**: `analysis/data_notes.md`（特許/NPL非対称性・ギャップ分析の注意・Web調査ルールは同 §3）

---

### Phase C: モジュール別deep dive ⚠ スキップ禁止

🛑 **STOP-GATE (リファレンス読了 + 計画確認)**: 以下を全て実行するまで deep_dive の執筆を始めるな
- [ ] `analysis/deep_dive_guide.md` を読了 → 各 Step の必須セクション数と最低行数を把握
- [ ] `AskUserQuestion` ツールで「各モジュールの Step 数・最低行数の理解(例: Saturn V 13セクション/250行)を一覧で提示し、これで進めて良いか」をユーザーに確認
- [ ] ユーザー応答を待つ

exemplarsを参照し、全モジュールのdeep_dive.typを生成する。Phase DはPhase Cの出力ファイルを前提とする。

1. 各モジュールのexemplarを読む → deep_dive.typを生成
2. 全deep_diveにミクロ分析A（代表特許15件以上）+ B（出願人5社以上、各5行以上）を含める
3. Step 0: NEBULA → Step 1: Saturn V → Step 2: Explorer → Step 3: MEGA → Step 4: ATLAS → Step 5: CORE → Step 6: CREW の順で処理
4. **Phase C 完了ゲート (必須実行)**: 以下のスクリプトを実行し、exit code が 0 でない場合は Phase D 開始禁止。不足モジュールを補強してから再実行する。

   ```bash
   bash capcom_schema/scripts/phase_c_gate.sh
   ```

   このスクリプトは各 deep_dive ファイルの存在と内容量（**非空白文字数**。行数 `wc -l` は1文1行で水増し可能なため文字数で判定）を客観的に判定する。**「実質的にOK」等の AI の質的判断による上書きは禁止**(`## 0. 絶対遵守ゲートルール` 第3項)。

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
6. **用語統一・スコープ限定チェック**: 内部JSONファイル名・内部フィールド名・内部ガイドファイル名・内部プロセス用語（deep_dive / Phase A-D 等）の混入ゼロ、未指定DB名（J-PlatPat 等）の補完なし、スコープ限定語 ≥ 5 件かつ無限化語 ≤ 限定語 × 0.3（`terminology.md` §6）を満たすこと。**個別 grep コマンドの正本は gate スクリプト**（次項で一括実行）

7. **品質検証ゲート (必須実行)**: 以下のスクリプトを実行し、結果をユーザーに報告する。exit code が 0 でない場合、不合格項目を修正してから再実行する。

   ```bash
   bash capcom_schema/scripts/phase_d_gate.sh
   ```

   このスクリプトが `analysis/quality_checklist.md` section 1 の定量チェック（Check 1〜37。用語統一・スコープ限定・別冊チェックを含む）を統合実行する。**「自前のチェックで代替」は禁止**(再現性のないチェックは無効)。

8. **別冊（経営層向け要約版）生成 — Phase A で「両方生成」が選択されていた場合に必須**:
   - [ ] `analysis/executive_summary_guide.md` を読了 → ページ構成・凝縮技法・禁止事項を把握
   - [ ] 本編 `reports/report.typ` が合格している（上記 Step 7 をパス）ことを前提に、`reports/report_executive.typ` を生成する
   - [ ] **刈り取り禁止**: 本編からコピーして短縮するのではなく、**本分析の Mission Objective と `query_intent` から導かれる「今回の意思決定テーマ」に即して再構成**する（詳細は `executive_summary_guide.md` §3 参照）。定型の分類軸を機械的に当てはめるのは不可
   - [ ] ページ数は 8-12 ページ厳守（行数目安: 250-500 行）。13 ページ以上 or 7 ページ以下は不合格
   - [ ] サブタイトルに必ず「— 経営層向け要約版」を追加。本編と完全に同一のタイトルは不可
   - [ ] 手法詳細（SBERT / UMAP / HDBSCAN / min_cluster_size 等）は別冊に混入させない（合格基準: これらの言及が 3 件以下）
   - [ ] 本編と同じ `terminology.md` の用語統一ルールを遵守（内部識別子の露出禁止）
   - [ ] 結論・数値は本編と整合させる（別冊独自の再集計・独自結論は不可）
   - [ ] 生成後の別冊品質確認（行数 250-500 目安・用語統一・手法詳細 3 件以下・サブタイトル「経営層向け要約版」）は `bash capcom_schema/scripts/phase_d_gate.sh` を再実行して確認する（別冊存在時に自動チェック。個別コマンドの正本は gate スクリプト）
   - [ ] 別冊完成をユーザーに報告し、PDF 化コマンドを案内: 本編 `typst compile reports/report.typ reports/report.pdf` ／ 別冊 `typst compile reports/report_executive.typ reports/report_executive.pdf`

→ **完了条件**: report.typが品質基準を満たす + 用語チェックが全てゼロヒット + 別冊フラグが ON なら report_executive.typ も品質基準を満たす
→ **レポート構造**: `analysis/report_structure.md`（全体構造・deep_diveコピールール・結論章ガイド・付録テンプレート・v8 母集団メタ反映）
→ **別冊構造**: `analysis/executive_summary_guide.md`（。経営層向け要約版の執筆ルール・ページ構成・凝縮技法）
→ **用語統一**: `analysis/terminology.md`（内部識別子の露出禁止ルールと正式日本語呼称）
→ **品質検証**: `analysis/quality_checklist.md`（定量チェックコマンド・全チェック項目・推奨項目）

---

## 3. モジュール一覧

| モジュール | JSON ファイル | 概要 | スキーマ |
|-----------|-------------|------|---------|
| ATLAS | atlas_statistics.json, atlas_grant_rate.json | 時系列推移、ランキング、ライフサイクル、権利化率（出願数×権利化率の象限） | `references/atlas_schema.md` |
| CORE | core_classification.json | ルールベース特許分類 | `references/core_schema.md` |
| Saturn V | saturnv_clusters.json, saturnv_drilldown_<クラスタ>.json | AIクラスタリング (TELESCOPE/PROBE) | `references/saturnv_schema.md` |
| MEGA | mega_momentum_<軸>.json (applicant/ipc/fterm), mega_drilldown_<対象>.json | 動態分析 (CAGR x 活動量 4象限・軸別) | `references/mega_schema.md` |
| Explorer | explorer_global_network.json, explorer_trend.json, explorer_dominance.json | キーワード共起ネットワーク | `references/explorer_schema.md` |
| CREW | crew_network.json | 発明者/出願人ネットワーク (要約版) | `references/crew_schema.md` |
| EAGLE | eagle_clusters.json | 探索的ランドスケープ (手動クラスタリング) | `references/eagle_schema.md` |
| NEBULA | nebula_hype_cycle.json, nebula_macro_events.json | 非特許文献統合・環境分析 | `references/nebula_schema.md` |
| VOYAGER | voyager/mission.json, evidence/, context.json | 戦略レポート用データパッケージ | `references/voyager_schema.md` |
| (共通) | *_wordcloud.json | 各モジュールのワードクラウド単語頻度 | `references/wordcloud_schema.md` |

**スキーマ参照ルール**: `references/` のスキーマファイルは、そのモジュールのJSONを実際に読む直前に参照する。全スキーマの一括読み込みは禁止。

## 4. patents.csv 仕様

全特許データのCSVファイル。サイズ警告: 1,000件で1MB以上。**絶対に全量読み込みしないこと。** 推奨アクセスは条件検索＋`.head()`（例: `df[df['cluster'] == 3][['title', 'applicant_main', 'year']].head(20)`）。`print(df)` は禁止。

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
3. **クロス検証**: 複数モジュールのデータを組み合わせて結論を補強する。**最低5パターン**実施（→ `analysis/cross_module.md`）
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
| `structured_techniques.md` | 構造化分析技法（ACH 競合仮説・リンチピン・ミラーイメージング点検）＋代表特許の決定的選定の原則 | Phase C 統合インサイト節・Phase D 結論章の執筆**直前**（常時読み込み不要） |

### exemplars/ ファイル一覧

| ファイル | 内容 | 使用フェーズ |
|---------|------|------------|
| `exemplars/nebula_exemplar.typ` | NEBULA環境分析のお手本 | Phase C Step 0 |
| `exemplars/saturnv_exemplar.typ` | Saturn V / EAGLE分析のお手本 | Phase C Step 1 |
| `exemplars/explorer_exemplar.typ` | Explorer分析のお手本 | Phase C Step 2 |
| `exemplars/mega_exemplar.typ` | MEGA PULSE分析のお手本 | Phase C Step 3 |
| `exemplars/atlas_exemplar.typ` | ATLAS統計分析のお手本（権利化率分析 §10 含む） | Phase C Step 4 |
| `exemplars/core_exemplar.typ` | CORE ルールベース分類分析のお手本 | Phase C Step 5 |
| `exemplars/crew_exemplar.typ` | CREW 人的ネットワーク分析のお手本（指標別の解釈） | Phase C Step 6 |

> **お手本の使い方**: exemplar は「どう書くか」を具体例で示す。**exemplarを読まずにdeep_diveを書き始めてはならない。**（全7モジュールに exemplar あり）

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

### 利用可能な関数（使い方・引数の正本は `templates/report_style.typ` の各関数 doc コメント）
- `exec-summary[...]` — エグゼクティブサマリーボックス
- `kpi-dashboard(cols: 3, kpi-card(...), ...)` — KPIダッシュボード（ページまたぎ防止）
- `kpi-card("ラベル", "値", note: "補足")` — KPIカード（**ドル記号禁止**: `$`/`\$` 不可、「ドル」「USD」で表記）
- `evidence-box(番号, "タイトル")[...]` — Evidenceボックス
- `insight-box[...]` — Key Insightボックス
- `note-box[...]` — 注釈ボックス
- `point-lead[...]` — 要点ストリップ（見出し直後の結論先出し1〜2行。散文の代替ではない）
- `hl[...]` — インライン強調（1セクション数語まで・多用禁止）
- `snapshot-figure("パス", caption: "説明")` — スナップショット画像
- `styled-table(columns: ..., header: (...), ..body)` — BCG風テーブル
- `conclusion-box("タイトル")[本文]` — 主要結論ハイライト
- `recommendation-card("高", "タイトル", "説明", timeframe: "短期")` — 優先度付き推奨
- `action-items("アクション1", "アクション2", ...)` — ToDoリスト

> 📐 **読みやすさ（走査層）— 詳細は `analysis/deep_dive_guide.md`「読みやすさ（走査層）」**: 地の文の塊を減らすため、①**各番号セクションの冒頭に `#point-lead[...]` を1個**置き結論を1〜2行で先出しする（散文は下にそのまま書く＝薄くしない）②件数・%・倍などの**数値＋単位はテンプレートが自動で太字強調**するので**手動で数字を太字化しない**③段落余白・見出しバーも自動。要点だけ書いて散文を削るのは Check 1（文字数）で不合格。

**注意**: `report_style.typ` のフォント設定を変更しないこと。`#set text(font: ...)` を report.typ に直接書かないこと。画像パスは `reports/` からの相対パス。typst compile に `--root ".."` を付けること。旧API（`#setup-page()` / `#cover-page(...)` 等）は廃止済み。

### python-pptx PPT
0. **🔑 デッキは完成レポート `reports/report.typ` を土台に作る** — evidence の短い説明文の寄せ集めでなく、レポート各章の主張→根拠（数値）→示唆を凝縮し、章順（前提→サマリー→環境→俯瞰→動態→競争→クロス統合→仮説検証→結論・提言→将来）に沿わせる。出所は分析モジュール名でなくデータ（特許データセット／Web実出所）にする（詳細は slides_spec §0.9）
1. `capcom_schema/templates/apollo_template.pptx` を `reports/` にコピーする
2. `capcom_schema/templates/slides_spec.md` を**設計ガイドとして熟読**する（Section 0〜6。とくに §0.9「レポートを土台にする」）。いつ・どのヘルパーを・どんな主張骨格で使うかを把握する文書であり、ヘルパー実装をここから写経するのではない
3. **🔑 ヘルパーは import して使う（コピーしない）**: 生成スクリプトの冒頭で `import sys; sys.path.insert(0, "capcom_schema/templates"); from apollo_slides import *` し、`add_title_shape` / `add_sub_message` / `add_kpi_slide` / `add_matrix_2x2_slide` / `_apply_font` 等のヘルパーを呼ぶ。**自前で pptx のフォント・色・レイアウトを書き起こさない**（過去の Codex「単一ウェイト・平板」の原因）。`Presentation('reports/apollo_template.pptx')` + `slide_layouts[6]`（Blank）でスライドを生成する
4. **フォント**: `Noto Sans JP` 統一。`_apply_font(run, weight=...)` で多段ウェイト（見出し=Black/本文=Regular/出典=Light、強調=Bold）を使い分け
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
