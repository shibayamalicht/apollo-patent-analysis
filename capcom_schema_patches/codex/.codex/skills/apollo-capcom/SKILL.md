---
name: apollo-capcom
description: >
  APOLLO特許分析プラットフォームのCAPCOMセッションデータを
  解釈し、戦略レポートを生成するための辞書・業務マニュアル（Codex CLI版）。
  session_* フォルダ内のデータファイルを読み取る際に参照。
---

> **このファイルは要約版。各フェーズの開始前に指定されたリファレンスファイルを必ず読むこと。**
> **Codex CLI 専用版**。Claude Code 用の `capcom_schema/SKILL.md` を Codex 仕様に翻案しています。
> 共有資産（`analysis/`, `references/`, `exemplars/`, `templates/`, `scripts/`）は既存の `capcom_schema/` 配下をそのまま参照します。

## 0. 絶対遵守ゲートルール (最優先)

**以下は他の全ルール(トークン効率制約含む)に優先する。例外なく適用する。**

1. **全ゲートは省略不可**: 「ユーザーが短く指示した」「効率上スキップしたい」等の理由でゲートを省略してはならない
2. **ユーザー応答待ち必須**: 「ユーザーに確認」「報告して」と書かれた箇所では、`ask_user_question` ツール（Codex TUI モード）でユーザー応答を取得するまで次フェーズへ進まない。テキスト出力だけで満足してはならない。`codex exec` 非対話モードは本スキルでは非推奨（`exec_mode_addendum.md` 参照）
3. **不合格時は強制ループ**: Phase 完了条件を満たさない場合、必ず該当 Phase に戻る。「実質的にOK」「内容は保持」等の質的判断で量的基準(行数・件数)を上書きしない
4. **指示の長さで手順を変えない**: ユーザー指示が「レポートを書いて」のように短くても、本 SKILL.md の全手順に従う。短い指示は「省略OK」のサインではなく「SKILL.md 通りに」のサイン
5. **「省略します」と宣言する前に立ち止まる**: 何かを省略する判断をした瞬間、`ask_user_question` でユーザーに省略の可否を確認する

6. **水増し（同一文の反復）禁止 — 量より固有性が合否を決める**: 「最低◯◯行/◯件」は深さの目安であり、合否は内容の固有性で決まる。**同一文・同一構文の反復、回転する名詞だけ変えた定型文の量産、「○○観点 1, 2, 3…」式の連番見出しで行数・件数を稼ぐことは禁止＝`phase_d_gate.sh` Check 19 で自動不合格**。**反復でなくても、データ（本母集団）を見なくても言える自明な一般論で字数を稼ぐのも水増しに含む**（例:「特許は権利文書である」「定型語は技術的意味を持たない」）。行数が不足する時は、文を繰り返す代わりに ①新しい代表特許（固有の公開番号）②新しい数値根拠 ③別のクロスパターン ④Web調査の裏付け を足す。各段落は本母集団固有の事実（固有の数値・公開番号・クラスタ名・出願人名のいずれか）を最低1つ含めること（理由: 反復・自明な一般論は読者に無価値で、機械ゲートで弾かれる）。Codex は逐語的に指示に従うため、量だけ満たして反復・一般論で埋めることは特に厳禁

7. **本文のスクリプト生成・ゲート回避の禁止（Codex で実際に起きた失敗）**: レポート本文・`deep_dive.typ`・`report.typ` を Python 等のスクリプトでテンプレート生成してはならない（`reports/generate_*.py` のような本文生成スクリプトは `phase_d_gate.sh` Check 19a で自動不合格）。各文は特許群の固有事実に基づく分析として直接書く。「最低行数を満たすための補助文・つなぎ文」を入れてはならない（行数不足＝分析不足。固有の事実を足すか、その章を短く確定する）。**`phase_d_gate.sh` を読んで反復検出を『回避』する目的で、接続詞・語順・文体だけを変えて内容の重複を温存することは禁止**。ゲートは実在する欠陥（重複）を検出している。正しい対処は重複文の削除と固有内容への置換であって、検出のすり抜けではない。**また「最低◯◯行」を満たすために1文ずつ改行して行数を稼ぐのも水増し**＝ゲートは行数(`wc -l`)でなく**非空白文字数**で判定する（Typst では行内改行は描画上スペースで見た目不変・行数は改行で増やせるが文字は増やせない）。「第一の…である。第二の…である。」式の一文一行の羅列でなく、複数文で論証（主張→根拠→示唆）を組む段落を書くこと

8. **工程ナレーション節・後続章への申し送りの禁止（Codex で実際に発生）**: 「後続分析への接続」「次章への申し送り」のような、**他章でやることのToDoを並べただけのメタ節・段落を作らない＝意味のない水増し**（完成レポートでは各モジュール章が既に存在するため無価値。Codex は `== 14. 後続分析への接続` のような節を実際に量産した）。「Explorer分析では〜を確認する」「後続のCORE分類で確認する必要がある」式の**前向きの申し送り**も禁止。各章は自章の分析と結論で閉じ、章間連携は『クロスモジュール統合分析』章で行う。他章への言及は「〜で確認された〈事実〉」の**過去形・根拠引用**に限る。`phase_d_gate.sh` **Check 8e** で自動検出 FAIL（→ `analysis/deep_dive_guide.md`「工程ナレーション節を作らない」・`analysis/terminology.md` §1-4）

9. **STOP-GATE はコンテキスト限界でも死守（捏造・先送り厳禁）**: STOP-GATE（`ask_user_question` での確認）を**実際に呼ぶ前**に次フェーズへ進んではならない。**STOP-GATE に到達する前にコンテキストが限界に近づいたら**: ① `reports/_carryover.md` に現在地・確定値を保存 → ② ユーザーに「コンテキストが厳しいので一旦 `/compact` します。再開後に必ず STOP-GATE（母集団タイプ・分析の立場・別冊・タイトル・重点）を質問します」と**告げてから** `/compact` → ③ 再開後、**最初に `ask_user_question` で STOP-GATE を出す**。**`ask_user_question` を実際に呼んでいないのに「ユーザーの回答を受信した／受信できなかった」と仮定して進むのは厳禁**（存在しない回答の捏造＝重大違反。ユーザーが実際に答えるまでフェーズを進めない）。Codex は逐語的に処理を進めがちなので特に注意。そもそも Phase A のデータ精読で枯渇させないため、統計は **STOP-GATE C の C-1 ワンショットスクリプトで1回だけ算出**し、CSV の試行錯誤・カラム名の探り当て・再読み込みをしない（今までの枯渇の主因）。`ask_user_question` は TUI でのみ動くため、フルパイプラインは `codex` TUI で起動すること。

このメタルールは下記「トークン効率に関する制約」よりも上位。両者が衝突する場合、本ルールが勝つ。

## トークン効率に関する制約（ツァーリ・ボンバ対策）

**以下のルールはレポートの品質とトークン効率を両立するために厳守すること。**

1. **サブエージェント禁止**: Codex は組込サブエージェントを持たないため、別エージェント委譲を試みない。全処理をメインコンテキスト内で完結させる
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

## ⚠️ 大型レポートのフェーズ分割（コンテキスト枯渇対策・Codex で最重要）

CAPCOM のレポート生成は巨大なタスク（patents.csv＋多数のJSON＋prompts読込 → 800行超の report.typ＋6〜7本の deep_dive＋PPTX）で、**1スレッドで一気に通すとコンテキスト窓を超過する**（"ran out of context" エラー）。本フローは**各フェーズの成果物をディスクに書く設計**なので、スレッドを分けても続きから再開できる。**完走できないと判断したら、無理に1スレッドで進めず、ユーザーにフェーズ分割を提案すること。**

**分割実行のしかた**:
1. **1スレッド＝1フェーズ**を目安にする（Phase C は **1モジュールずつ**＝Saturn V → 別スレッドで Explorer …）
2. 重い生成の前・フェーズの区切りで **`/compact`** を実行する
3. 新スレッドでは、まず `ls reports/` で**どこまで完了したか**を確認してから続きを実行する:
   - `reports/_phase_a_decisions.json` あり → Phase A 完了
   - `reports/<module>_deep_dive.typ` あり → そのモジュールの Phase C 完了（残りモジュールから再開）
   - `reports/report.typ` あり → Phase D 統合済み
4. 各ゲート（`phase_c_gate.sh` / `phase_d_gate.sh`）はディスク状態を見るので、再開後も同じ判定が効く

**記憶の引き継ぎ（`reports/_carryover.md`・最重要）**: Codex はスレッドを跨ぐ記憶を持たないので、この**引き継ぎ日誌が事実上唯一の長期記憶**。成果物 `*.typ` と `_phase_a_decisions.json` だけでは、Phase B の仮説検証過程・**Web調査の出所(URL/取得日)**・「なぜそう判断したか」が新スレッドで失われる。
- 無ければ `capcom_schema/templates/carryover_template.md` を `reports/_carryover.md` にコピーして作る
- **新スレッド冒頭で必ず `reports/_carryover.md` を通読**（STATUS/RESUME・直近フェーズ節・WEB出所台帳・申し送り）し、現在地を1行報告してから着手。日誌にある情報は再読しない（トークン節約）
- **各フェーズ完了時・`/compact` 直前に該当節へ追記**: Phase A の抽出数値とタイプ判定の根拠／**Web調査は1件ヒットごとに即 WEB出所台帳へ**（URL・サイト名・取得日。後回し禁止）／クロス仮説の検証結果／deep_dive 完了状況（本文は二重保存しない）
- **レポート本文へは転載しない**（内部作業メモ。本文へのコピー元は `*_deep_dive.typ` のみ）。詳細は `capcom_schema/SKILL.md` §フェーズ間引き継ぎ日誌

**設定（`~/.codex/config.toml`）**: `model_reasoning_effort = "high"`、`model_verbosity = "low"`。**「非常に高い」(xhigh) はコンテキストを最も消費する**ので、大型レポートでは IDE のモデル選択でも **「高」**を選ぶ（xhigh を避ける）。

**やってはいけない**: 全フェーズを1スレッドで一気に通そうとする ／ 本文生成スクリプトを書いて巨大な文字列をコンテキストに抱える（§0 第7項で禁止・コンテキストも枯渇する）。

### 🔄 セッション・チェックポイント（各フェーズ境界で必須・枯渇の予防）

「完走できそうにない」と感じてから分割するのでは遅い（"ran out of context" は予兆なく起きる）。そこで **各フェーズのキリのいい境界で、ルーチンとしてセッション切替を提案して一旦停止する**。対象境界:

- **Phase A 完了**（`reports/_phase_a_decisions.json` 保存後）
- **Phase B 完了**（エビデンス＋クロス確定後）
- **Phase C の各モジュール完了ごと**（deep_dive 1本ごと＝最も枯渇しやすい区切り）
- **Phase D 着手前**（`report.typ` 統合は重いので、その前で一度切る）

各境界で必ず以下を順に実行する:

1. 当該フェーズ/モジュールの **ゲート・完了条件を満たしたことを確認**（`phase_c_gate.sh` 等）
2. **`reports/_carryover.md` を更新**（STATUS=現在地・RESUME=次にやること・直近の固有事実・Web出所台帳）
3. **ユーザーにチェックポイントを提示して応答を待つ**（テキスト出力だけで満足せず、ユーザーの選択を取得するまで次に進まない）:

   > ✅ **Phase X（／モジュール M）完了・ゲート通過。`reports/_carryover.md` 更新済み。**
   > ここは安全な区切りです。コンテキスト枯渇を避けるため、**新しいセッション（スレッド）への切り替えを推奨します**。
   > 新セッションでは `reports/_carryover.md` を読んで **次（Phase Y ／ 次モジュール）から自動再開**します。
   > - 🔄 **新セッションに切替（推奨）**: いまのスレッドを閉じ、新規スレッドで `ls reports/` ＋ `reports/_carryover.md` を読んで再開
   > - ▶️ **このまま続行**: コンテキストにまだ余裕がある場合のみ（次の境界で再びチェックポイントを出す）
   > - 🗜️ **`/compact` で続行**: 同セッションのまま圧縮（軽い選択肢）

4. ユーザーが **切替** を選んだら、現スレッドはここで終了してよい（成果物と `_carryover.md` はディスクに残る）。**続行/`/compact`** を選んだらそのまま次へ進み、**次の境界で再びチェックポイントを出す**。

**重要**: このチェックポイントは「枯渇しそうな時だけ」ではなく **各境界で必ず出す**（予防が目的）。ユーザーが「最後まで一気に続けて」と明示した場合のみ、以降のチェックポイントを省略してよい。

# APOLLO CAPCOM Skills (Codex CLI版)

## 1. 概要

**APOLLO** は Streamlit ベースの特許分析プラットフォーム。9つのモジュールが特許データを多角的に分析し、可視化・構造化データを生成する。

**CAPCOM** (Capsule Communicator) は APOLLO と AI coding agent を繋ぐ通信モジュール。分析結果をファイル出力し、Codex CLI がデータを読み取り、自由な分析やレポート生成を行う。

### セッションフォルダ構造

```
session_YYYYMMDD_HHMMSS/
├── capcom_schema/  # 共有資産（analysis/ references/ exemplars/ templates/ scripts/ すべてここから読む）
├── data/           # patents.csv + 各モジュールJSON
├── voyager/        # VOYAGER Export時のみ（mission.json, evidence/, context.json）
├── snapshots/      # スナップショット画像(PNG)
├── prompts/        # AIプロンプト(Markdown)
├── reports/        # レポート出力先
├── .codex/         # Codex スキル（本SKILL.mdの置き場所）
├── AGENTS.md       # Codex階層的ルール
└── metadata.json
```

**cwd 規約**: 本スキル実行時は常に `session_*/` ルートで作業する。AGENTS.md にも明記。

## 2. 利用モード

### コンテキスト管理の原則（全モード共通）

1. **patents.csvは絶対に全量読み込みしない**: `head -5` でカラム構成を確認し、必要な分析の都度pandasで条件検索する
2. **JSONは必要なモジュールのみ読む**: 全JSONの一括読み込み禁止
3. **references/スキーマは対象モジュールのみ読む**: 全スキーマの一括読み込み禁止
4. **analysis/ガイドは段階的に読む**: まず `capcom_schema/analysis/common_framework.md` のみ。他は必要な時に該当セクションのみ読む

### 自由分析モード
`data/` 配下のCSV/JSONをユーザーの質問に応じて読み取り、回答する。patents.csvの全量表示（`print(df)`, `cat`）は禁止。常にフィルタリング + `.head()` で制限する。

### レポート生成モード
VOYAGER Export 後に利用。`voyager/mission.json` の Mission Objective に基づく正式レポートを作成する。以下の4フェーズで進行する。

---

## 環境準備（依存インストール・最初に1回・Codex で頻発）

レポート生成は **patents.csv の解析に `pandas`、スライド生成に `python-pptx` / `Pillow`** を使う。これらはセッションフォルダ直下の **`requirements-session.txt`** に列挙済み。**Phase A のデータ精読に入る前に、依存を必ず確認・導入すること。** Codex は指示に逐語的に従い、依存を自発的に入れないため、未導入のまま `import pandas` / `import pptx` して `ModuleNotFoundError`（例: `No module named 'pandas'` / `No module named 'pptx'`）で止まる事例が実際に発生している:

```bash
# セッションフォルダ直下で実行（揃っていればスキップ、無ければ一括導入）
python3 -c "import pandas, pptx, PIL" 2>/dev/null && echo "依存OK" || pip install -r requirements-session.txt
```

- `pip` が見つからなければ `python3 -m pip install -r requirements-session.txt`。書き込み権限エラーなら末尾に `--user` を付す。
- 仮想環境を使うなら、セッションフォルダで `python3 -m venv .venv && source .venv/bin/activate` の後にインストールし、以降の `python3` も同じシェルで実行する。
- ネットワーク制限等で `pip install` が通らない場合は、依存が無いまま分析を始めず、ユーザーに「セッションフォルダで `pip install -r requirements-session.txt` を実行してから再開してください」と伝えて一旦停止する。

## レポート生成 4フェーズ手順

### Phase A: ミッション理解 + データ精読

voyager/mission.json を読み、data/以下のJSONとpatents.csvを把握する。**Phase A は複数の STOP-GATE で構成される**（本家 Claude Code 版と機能的に等価。Codex では `ask_user_question` ツールを使う）。

**全ステップは省略不可。**

🛑 **STEP 0 (最優先)**: 用語統一ルールの読了と母集団メタ情報の確認
- [ ] `analysis/terminology.md` を**最初に**読む（§1-6 すべて: 内部識別子の露出禁止 / Mission Objective ベタ貼り禁止 / 母集団メタ §5 / スコープ限定ルール §6 / サブクエスチョン化 §5-A-2）
- [ ] `voyager/context.json` の `population_meta` 4 フィールドを確認:
  - `query_intent` → 指定されていれば**全分析を貫く「視座」として内在化**
  - `query_logic` → 指定されていれば付録 D に `#raw` ブロックで全文掲載
  - `coverage_years` → 付録 A の対象期間欄に反映
  - `database_name` → 付録 A に記載。**未指定なら「提供された特許データセット」と汎用表記（J-PlatPat 等を勝手に補わない）**
- [ ] `voyager/context.json` の `capcom_tools.selected` を確認 → 付録 A の「CAPCOM モジュール」欄に記載

🛑 **PHASE A STOP-GATE (経営層向け要約版〈別冊〉の生成確認)**:
- [ ] `ask_user_question` で「本編(60-120p)に加えて別冊(8-12p)も生成するか」を確認（選択肢: ✅ 両方 / 📘 本編のみ / ❓ 相談）
- [ ] 「両方生成」選択時 → 作業メモに **別冊生成フラグ = ON** を記録、Phase D で `reports/report_executive.typ` を生成

詳細ガイド: `analysis/executive_summary_guide.md`

🛑 **PHASE A STOP-GATE A (query_logic 構造化読解) — `query_logic` が指定されている場合のみ必須**:
検索式を付録 D にコピペするだけで済ませるのは禁止。4 ステップ:
- [ ] `analysis/query_logic_reading.md` を読了（7 DB 構文: J-PlatPat / JP-NET / Patentfield / Shareresearch / BizCruncher / PatentSQUARE / PatSnap）
- [ ] **Step 1 DB 識別**: `database_name` があれば使用、なければ構文特徴（`/TX` → J-PlatPat、`HTX=` → JP-NET、`$Wn` → PatSnap 等）から推測
- [ ] **Step 2 構文分解**: AND/OR/NOT で節に分け、各節を「分類条件 / キーワード条件 / 出願人条件 / 日付条件 / その他」に仕分け
- [ ] **Step 3 意図推定**: 各条件の意図（例: `NOT A23*/IC` → 食品分野除外）
- [ ] **Step 4 ユーザー確認**: `ask_user_question` で上記を提示、「この読解で合っているか」を確認（✅ 進める / ✏️ 修正 / 💬 補足）

🛑 **PHASE A STOP-GATE (`query_intent` 3 点整理) — `query_intent` が指定されている場合のみ必須**:
- [ ] `query_intent` を読解し、執筆者の言葉で **3 点**を整理: ①分析目的 ②母集団の輪郭 ③分析の視座
- [ ] `ask_user_question` で 3 点整理を提示。ユーザー確定まで進まない
- [ ] **ベタ貼り禁止**: 原文のままレポートに書かず、Phase B 以降の全 deep_dive・クロス分析・結論章で「分析の視座」として内在化
- [ ] **設計意図を無視した汎用分析は品質不合格**

🛑 **PHASE A STOP-GATE (サブクエスチョン化) — `query_intent` が指定されている場合のみ必須**:
3 点整理を **作業メモ** として 3-5 個の観点に分解:
- [ ] 「本分析が明らかにすべき具体的観点」を 3-5 個起草、各観点から **主要キーワード 1-3 個** を抽出。**確定した立場（`narrative_stance`）の観点で SQ の抜けを点検**（self=自社の弱点・空白／competitor=対象企業の隙・参入余地／neutral=強み・リスク・投資妙味）。`query_intent` を最優先（分析者は通常その立場で母集団を設計済み＝機械的置換でなく抜けの点検）
- [ ] `ask_user_question` でサブクエスチョン一覧 + キーワードを提示、ユーザー確認
- [ ] 確定結果を `reports/_phase_a_decisions.json` の `sub_questions` に保存
- [ ] **⚠️ 絶対制約**: サブクエスチョンは**内部メモ専用**。レポート本文に「Q1 / A1 / SQ1 / 問い 1」等の記号・形式は禁止。本文は通常の宣言調で書く（詳細: `terminology.md` §5-A-2）

🛑 **PHASE A STOP-GATE B (意図 ↔ 論理 整合性検査) — `query_intent` と `query_logic` が両方指定されている場合のみ必須**:
- [ ] `analysis/query_logic_reading.md` §4 の **8 項目**で対比（技術領域 / 用途 / 対象期間 / 地域 / 出願人絞り込み / 除外条件 / 公報種別 / 分類階層）
- [ ] 乖離を 3 段階分類: 🔴 Critical / 🟡 Warning / 🔵 Info
- [ ] Critical / Warning には **具体的な改善提案** を作成（例: 「末尾に `* NOT (A23*/IC)` を追加すると意図に沿う」）
- [ ] `ask_user_question` で乖離 + 改善提案を提示（[A] 修正して再抽出 / [B] このまま進めて「範囲と限界」章で明記 / [C] 無視 / ✅ 乖離なし）
- [ ] Critical 検出でも進行可能（ユーザー判断尊重）

1. `voyager/mission.json` を読む（Mission Objective + Evidence 一覧）
2. `voyager/context.json` でデータセットのメタ情報と population_meta / capcom_tools / report_directives（`image_slide_instruction`＝画像・スライドのユーザー指示）を確認する
3. `evidence_list` の全件を走査し、各 Evidence の `module`・`title`・`images` を一覧表で整理する
4. `snapshots/` のファイル一覧を取得する
5. **`data/patents.csv` を読む**: `head -5` でカラム構成 → `wc -l` で件数 → pandas で出願人上位 10 社・クラスタ別件数・年別件数を把握
6. **`data/` 以下の全 JSON ファイルを確認**: 各 JSON から主要数値（クラスタ数・ノイズ率・HHI/Entropy/Gini・CAGR 等）をメモ
7. **`prompts/` の AI インサイトを読む**: 最低 3-5 件を選定し、1 件ずつ読む（一括読み込み禁止）
8. 各 AI インサイトから読み取った知見を具体的にメモとして書き出す

コンテキスト管理: `saturn_drill_insight.md`（最大 220KB）や `crew_network_insight.md`（最大 400KB）は全量読み込み禁止。対象箇所のみ `grep` で部分読み込みすること。

🛑 **PHASE A STOP-GATE C (データ側からの母集団実態確認 + 母集団タイプ判定) — 必須（全ケースで実施）**:

**C-1. データ Level 2 逆読み**

**⚠️ patents.csv の実カラム（試行錯誤＝コンテキスト枯渇を避けるため最初に把握する）**: `data/patents.csv` は APOLLO 処理済みで、カラムは **処理済み**（`applicant_main`=主出願人 / `inventor_main` / `year`=出願年 / `ipc_main_group` / `cluster` / `cluster_label` / `umap_x`,`umap_y` / `core_技術分類`,`core_課題分類`,`core_解決手段分類`）と **原データ**（`発明名称`〈先頭に BOM あり〉/ `要約` / `出願番号` / `公開番号`）の**混在**。⚠️ **`applicant_main` / `inventor_main` / `ipc_main_group` は `"['キオクシア', '東芝']"` のような Python リストの文字列**なので、集計には `ast.literal_eval` での展開（explode）が必須（しないと共同出願を1社と誤カウントする）。`cluster` は整数、`cluster_label` は `'[3] 半導体記憶, メモリセル, 半導体'` 形式の文字列。**`voyager/context.json` の `column_mapping` は元 CSV 名（`applicant`=出願人 等）で、patents.csv の実カラムとは一致しない**ので照合に使わない。**ステータス（権利状況）列は patents.csv に無い** — 権利化率は `prompts/atlas_*_insight.md` のステータス内訳から読む。

**ワンショット統計スクリプト**（実データで検証済み・出願人HHI 等が正しく出る。BOM 対応 `encoding="utf-8-sig"`、リスト文字列は展開、出力は `.to_string()` で1ブロックに収める。**heredoc の多重実行・カラム名の探り当て・Unicode 正規化の試行は禁止＝今までの枯渇の主因**）:
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
（権利化率や Fターム等、patents.csv に無い指標は `prompts/` のインサイトや `data/*.json` から得る。CSV を何度も読み直さない）

- [ ] `analysis/query_logic_reading.md` §5 の **Level 2 項目**を算出（下記は要約。正本は `analysis/query_logic_reading.md` §5-1、乖離時は正本を優先）: 総件数・対象期間・使用 DB / 上位 10 出願人・シェア / 主要 IPC/FI 上位 10 / 出願年分布 / 出願人集中度 HHI / 国・地域分布
- [ ] **自動偏り警告**: 上位 1 社 30% 超 / 上位 1 IPC 40% 超 / 直近 2 年 50% 超集中 / HHI > 0.25 / 特定国 95% 超 を検出

**C-2. 母集団タイプ判定**
- [ ] `analysis/population_type_metrics.md` を読了、5 タイプから候補を推定
  - **A 業界全体** / **A' 技術領域** / **B 競合限定** / **C 単一企業** / **D 特定製品・技術テーマ**
  - 判定目安: 上位 1 社 > 90% → C、上位 5 社で 95% 超 → B、上位 10 社 40-70% → A'、上位 10 社 < 40% → A、複合的絞り込み + 上位 10 社 > 70% → D
- [ ] タイプ C では出願人 HHI 算出無意味（HHI=1.0）、タイプ B/C/D では「市場集中」「業界シェア」等の **市場・業界解釈は禁止**（`population_type_metrics.md` §3）

**C-2'. 分析の立場（叙述スタンス）判定 — 母集団タイプとは独立に必ず判定**

母集団タイプ（データの構成）と「**誰の意思決定のためのレポートか**（＝提言・主張を語る立場）」は**別物**。母集団が単一企業（タイプ C）でも読者＝依頼主は対象企業自身とは限らず、競合・投資家・アナリストのこともある。**`population_type` が C だからといって対象企業を自動的に「当社」と呼んではならない**（取り違えると、対象企業を勝手に「当社」と書く／中立であるべき評価が当事者寄りになる、といった誤りが生じる）。
- [ ] `query_intent` / Mission Objective から **分析の立場** を 5 分類で推定:
  - **self**（自社視点・当事者本人）: 対象企業を「当社」と呼ぶ／一人称可。手がかり: `query_intent` に「自社」「当社」「我々の」
  - **competitor**（競合視点・水平）: 対象企業を企業名で三人称（「キオクシアは」）。一人称は読み手＝競合自身を指す時のみ。手がかり: 「競合」「ベンチマーク」「対抗」
  - **buyer**（**自社＝買い手／対象＝供給元**。自社が対象から仕入れる。例: 自社=Apple・対象=キオクシア）: 対象を三人称。手がかり: 「調達」「サプライヤー選定」「供給元」「仕入れ」
  - **supplier**（**自社＝供給元／対象＝顧客**。自社が対象に納入する。例: 自社=東京エレクトロン・対象=キオクシア）: 対象を三人称。手がかり: 「販売先」「納入先」「顧客の技術動向」
  - ※**コードは常に『自社の役割』**（buyer=自社が買い手／supplier=自社が供給元。対象は相手方）。「buyer=対象が買い手」ではない。判別は取引の向き（自社が対象から買う→buyer／自社が対象に売る→supplier）
  - **neutral**（中立・投資家・アナリスト）: 対象企業を企業名で三人称／一人称「当社」不可。手がかり: 「投資判断」「評価」「調査」や立場の記述なし（**既定**）
- [ ] **手がかりが弱ければ `neutral` を仮置きし、C-3 で必ずユーザーに確認**（勝手に self にしない）
- [ ] 確定した立場は提言・主張・エグゼクティブサマリー・別冊の**全セクションで一貫**させる。**呼称だけでなく分析の力点・提言のロジックも立場に合わせる**（`self`=自社の打ち手／`competitor`=競合の対抗・参入／`buyer`=調達戦略・依存リスク／`supplier`=供給戦略・内製化リスク／`neutral`=第三者の評価・予測。同じ事実でも読み方と打ち手が変わる＝呼称を三人称にしただけの「べき論」にしない）。`self` 以外では対象企業を三人称で呼び「当社/弊社/我が社」を使わない（詳細は `terminology.md` の「分析の立場」節 §6-2-B）
- [ ] **立場が `competitor` / `buyer` / `supplier`（関係性立場・自社 ≠ 対象）なら「自社（分析を行う側）」を特定（必須）**: 対象企業（`subject_company`）だけでなく分析を行う**自社名**を C-3 で尋ね `narrative_stance.own_company` に記録。自社は Phase B で **Web 調査**し（事業・製品・技術/特許ポジション・市場での立ち位置）、対象企業と対比する（提言を一般論でなく「自社は X が強く／Y が手薄 → Z で差別化・参入。buyer=調達戦略／supplier=供給戦略」に具体化）。`buyer`/`supplier` の依存・交渉力は、母集団がドメイン（タイプ A/A'/B）なら出願人 HHI・上位集中で、**単一企業（タイプ C。例: キオクシア）では HHI 無意味なので Web 調査（市場シェア・取引構造）で**読む（**特許 ≠ 市場**）。`self`=`subject_company` が自社／`neutral`=自社なし（空）（詳細: `data_notes.md` §3、`terminology.md` §6-2-B）

**C-3. 統合ユーザー確認**
- [ ] `ask_user_question` で「データ実態 + タイプ推定 + **分析の立場推定**（self/competitor/buyer/supplier/neutral とその根拠）」を統合提示（選択肢: ✅ この実態・タイプ・立場で進める / ✏️ タイプは違う / 👤 立場が違う（自社／競合／取引先・買い手／サプライヤー／中立を指定）/ 💬 偏りあり、範囲と限界に明記 / 🔙 再抽出）。※関係性立場（competitor/buyer/supplier）なら分析を行う『自社』の社名も確認する旨を明示
- [ ] **⚠️ 立場を独立の `ask_user_question` として聞く場合の選択肢（必須）**: 選択肢は最大4つ。**competitor / buyer / supplier を『その他』に畳まない**。4択で「①中立(推奨) ②自社視点 ③競合視点 ④取引先・買い手／サプライヤー視点」を出し、③④選択時に続く `ask_user_question` で具体的立場＋**自社名**を確認する。**④は buyer/supplier の語で迷わせず取引の向きで訊く**:「自社が対象から**仕入れる**→buyer(対象=供給元)／自社が対象に**売る**→supplier(対象=顧客)」。**単一企業母集団（タイプ C）でも 5 立場すべて有効**（self/neutral だけと決めつけない）
- [ ] **立場が `competitor` / `buyer` / `supplier`（関係性立場）に確定したら続けて自社名を尋ねる（必須）**: 未取得時は追加の `ask_user_question` で「本分析を行う『自社』（対象企業の {競合／取引先・買い手／サプライヤー} として提言を導く主体）の社名は？」を確認し `narrative_stance.own_company` に保存。**Phase B の Web 調査テーマに「自社（{own_company}）の事業・技術・特許ポジション」を必ず含める**。ユーザーが「伏せる／一般的な視点でよい」なら `own_company` は空文字にし 1 行報告して従来どおり進める

**C-4. `reports/_phase_a_decisions.json` への保存**
- [ ] 確定内容を以下のフィールドで保存: `population_type` / **`narrative_stance`**（`code`=self/competitor/buyer/supplier/neutral / `label` / `subject_company` / `own_company`（competitor/buyer/supplier で分析を行う自社名。self は subject_company と同一、neutral・伏せる場合は空）/ `first_person_allowed`（self のみ true）/ `reasoning` / `confirmed_by_user`）/ `query_intent_summary` / `sub_questions` / `query_logic_structure` / `intent_logic_divergences` / `data_level2_warnings` / `forbidden_expressions` / `nebula_strategy` / `user_notes`（詳細: `population_type_metrics.md` §4-3、`narrative_stance` は `terminology.md` §6-2-B）

🛑 **PHASE A STOP-GATE D (NEBULA 戦略判定) — 必須（全ケースで実施）**:
- [ ] `data/nebula_*.json` の存在確認
- [ ] 存在すれば `nebula_strategy.selected_mode = "execute"` を自動決定
- [ ] 存在しない場合、`ask_user_question` で 2 択提示:
  - **🌐 Web 補完モード**: Phase B で 4 カテゴリ必須カバー（市場規模 / 政策・規制 / 学術動向 / 主要企業動向）→ 「外部環境分析（Web 調査）」章を設置、各主張に `#footnote[...]` で出所明記
  - **📘 省略モード**: NEBULA 章なし + 「本分析の範囲と限界」章で「特許情報のみ対象」と注記、学術-特許クロス分析も省略
- [ ] 確定結果を `_phase_a_decisions.json` の `nebula_strategy` に保存

→ **完了条件**: terminology.md §1-6 読了 / population_meta 4 フィールド確認 / patents.csv 統計把握 / 全 JSON 主要数値抽出 / AI インサイト 3 件以上読了 / 4 つの Phase A STOP-GATE（A / query_intent / SQ / B / C / D）完了 / `_phase_a_decisions.json` 永続化 / データセット全体像メモをユーザーに提示

### Phase A-2: レポートタイトルの決定

🛑 **STOP-GATE**: 以下を全て実行するまで Phase B へ進むな
- [ ] Mission Objective とデータ特性を踏まえ、タイトル+サブタイトルの **3案** を生成する
  - **タイトル**: **オーソドックス**（標準的・保守的）な体言止め。**20 文字以内**の目安
    - ✅ OK: 「CNF 特許動向分析 2026」「水素貯蔵技術の競合ポジション分析」「次世代半導体製造技術ランドスケープ」
    - ❌ NG: 「独断 — CNF の未来」等の扇情的・文学的タイトル／「CNF はどこへ向かうのか？」等の問いかけ型
    - 指針: 「{技術分野 / 対象企業} の {分析種別}」の単純な組み合わせが基本。クリエイティブなコピーは不要
  - **サブタイトル**: 30 文字以内。具体的な件数・期間・分析軸を含める
- [ ] `ask_user_question` ツール（Codex TUI）で 3案を提示し、ユーザーに選択してもらう（テンプレ: `.codex/skills/apollo-capcom/prompts/phase_a2_titles.md` 参照）
- [ ] ユーザーが選択した案（または「Other」で指定された案）を採用
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
- [ ] `capcom_schema/analysis/common_framework.md` を読了 → 4層分析モデルと数値根拠の書式を把握
- [ ] `capcom_schema/analysis/data_notes.md` を読了 → 特許/NPL 非対称性と Web 調査ルールを把握
- [ ] `capcom_schema/analysis/cross_module.md` を読了 → 13種のクロスパターンから3つ以上を選定
- [ ] `ask_user_question` ツールで「採用するクロスパターン3つ(例: P1/P4/P13)」をユーザーに提示・確認（テンプレ: `prompts/phase_b_cross.md`）
- [ ] ユーザー応答を待つ

🛑 **STOP-GATE 2 (Web調査の意思確認)**: Phase B 本体作業前に必須

- [ ] **`narrative_stance` を確認 — 立場が `competitor` / `buyer` / `supplier`（関係性立場）かつ `own_company` が非空なら（下記モード問わず必須）**: 提示テーマに「**自社（{own_company}）の事業・主要製品・技術/特許ポジション・市場での立ち位置**」を必ず1件含める。対象企業（`subject_company`）との対比材料であり、提言を「自社は X が強く／Y が手薄 → …」と具体化する土台。buyer/supplier の依存・交渉力は母集団がドメイン（A/A'/B）なら HHI で、**単一企業（タイプ C）では HHI 無意味なので Web 調査（市場シェア・取引構造）で**読む（特許≠市場）。出所は脚注（サイト名・URL・取得日）を付し、最低1章で自社 vs 対象企業の対比に使う（`data_notes.md` §3、`terminology.md` §6-2-B）
- [ ] **`reports/_phase_a_decisions.json` の `nebula_strategy.selected_mode` を確認**し、モード別に対応:

**モード `execute`（NEBULA 実行済み）**:
- [ ] Mission Objective から導出された Web 調査テーマ 3-5 件を提示
- [ ] `ask_user_question` で「実施する / しない / テーマ修正」3 択 + Other を提示（テンプレ: `prompts/phase_b_webresearch.md`）

**モード `web_compensation`（NEBULA 未実行・Web 補完）**:
- [ ] Web 調査は **スキップ不可**（Phase A STOP-GATE D でユーザーが補完を選択済み）
- [ ] **4 カテゴリすべて**をカバーするテーマを起草:
  1. **市場規模**: 業界全体の市場規模・成長予測
  2. **政策・規制**: 政策・規制動向・標準化活動
  3. **学術動向**: 学術論文引用動向・キーパーソン
  4. **主要企業動向**: 主要出願人の事業戦略・M&A・プレスリリース
- [ ] `ask_user_question` で 4 カテゴリ分のテーマ（カテゴリごと 1-3 件）を提示、ユーザー確認
- [ ] 4 カテゴリが 1 つでも欠ける場合は警告して再確認（Phase D gate Check 13 で FAIL 対象）

**モード `omit`（NEBULA 未実行・省略）**:
- [ ] 通常通り任意 Web 調査として進行（3-5 件提示、3 択）
- [ ] 「外部環境分析」章は作らないが、任意 Web 調査は可

- [ ] ユーザー応答を待つ。AI 自己判断禁止

詳細: `analysis/population_type_metrics.md` §4-3（nebula_strategy フィールド仕様）

**Phase A の情報を参照せずに Phase B を進めてはならない。**

1. 上記3ファイルを読む（必読）
2. Evidence全件から優先順位を付ける（Mission Objectiveへの直結度で1-3のランク付け）
3. 優先度の高い5-8件を1件ずつ順次読む
4. 各Evidenceを読む際に: AIインサイトとの照合 / `capcom_schema/analysis/map_reading.md` の該当セクション読解 / 代表特許の抽出 / スナップショット画像パス記録
5. **代表特許の具体的確認**: `data/patents.csv` をpandasで条件検索し、代表特許のタイトル・出願人・公開番号を**最低15件**取得する
6. `capcom_schema/analysis/cross_module.md` の基本原則を読み、最低3パターン（P1-P13から）を選択・実行する
7. クロス分析で得られた洞察を記録する

→ **完了条件**: Evidence 5件以上精読済み / AIインサイト照合メモ作成済み / 代表特許15件以上取得済み / クロス分析3パターン以上の仮説→検証→結論を完了済み
→ **データ特性の注意**: `capcom_schema/analysis/data_notes.md`（特許とNPLの非対称性、ギャップ分析の注意）
→ **Web調査ルール**: `capcom_schema/analysis/data_notes.md` セクション3

---

### Phase C: モジュール別deep dive ⚠ スキップ禁止

🛑 **STOP-GATE (リファレンス読了 + 計画確認)**: 以下を全て実行するまで deep_dive の執筆を始めるな
- [ ] `capcom_schema/analysis/deep_dive_guide.md` を読了 → 各 Step の必須セクション数と最低行数を把握
- [ ] （予約）各 deep_dive の「統合的戦略インサイト」節の執筆**直前**に `capcom_schema/analysis/structured_techniques.md` §1 を読む（ACH＝対立仮説の検討。deep_dive 側は推奨・結論章では必須）
- [ ] `ask_user_question` ツールで「各モジュールの Step 数・最低行数の理解(例: Saturn V 13セクション/250行)を一覧で提示し、これで進めて良いか」をユーザーに確認（テンプレ: `prompts/phase_c_plan.md`）
- [ ] ユーザー応答を待つ

exemplars を参照し、全モジュールのdeep_dive.typを生成する。Phase DはPhase Cの出力ファイルを前提とする。

1. **`capcom_schema/analysis/deep_dive_guide.md` を読む** → 各Stepの必須セクション数と最低行数を把握
2. 各モジュールのexemplarを読む → deep_dive.typを生成（exemplar は `capcom_schema/exemplars/`）
3. 全deep_diveにミクロ分析A（代表特許15件以上）+ B（出願人5社以上、各5行以上）を含める
4. Step 0: NEBULA → Step 1: Saturn V → Step 2: Explorer → Step 3: MEGA → Step 4: ATLAS → Step 5: CORE → Step 6: CREW の順で処理
5. **Phase C 完了ゲート (必須実行)**: 以下のスクリプトを実行し、exit code が 0 でない場合は Phase D 開始禁止。不足モジュールを補強してから再実行する。

   ```bash
   bash capcom_schema/scripts/phase_c_gate.sh
   ```

   このスクリプトは各 deep_dive ファイルの存在と最低行数を客観的に判定する。**「実質的にOK」等の AI の質的判断による上書きは禁止**(`## 0. 絶対遵守ゲートルール` 第3項)。

→ **完了条件**: deep_dive 4ファイル以上（Saturn V + Explorer + MEGA + ATLAS）、各最低行数を満たす
→ **詳細手順**: `capcom_schema/analysis/deep_dive_guide.md`（Step 0-6の必須セクション・最低行数・ミクロ分析ルール全て記載）

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
- [ ] `capcom_schema/analysis/report_structure.md` を読了 → 章構成と deep_dive コピールールを把握
- [ ] `capcom_schema/analysis/quality_checklist.md` を読了 → 定量チェックコマンドとチェック項目を把握
- [ ] `capcom_schema/analysis/structured_techniques.md` を読了 → ACH・リンチピン・三つの環（結論・提言章で必須。`phase_d_gate.sh` Check 30/31/33/34 が検査）
- [ ] `ask_user_question` ツールで「report.typ の章構成(例: 10章)と品質チェック項目数(例: 15項目)で進めて良いか」をユーザーに確認（テンプレ: `prompts/phase_d_plan.md`）
- [ ] ユーザー応答を待つ

全deep_diveを統合し、report.typを生成する。品質チェックリスト確認。

**前提条件**: `reports/` に最低4つの `*_deep_dive.typ` が存在すること（4つ未満ならPhase Cに戻る）。

1. `ls reports/*_deep_dive.typ` でファイル存在を確認する
2. `capcom_schema/analysis/patent_citation.md` セクション2-3を読む（引用書式の確認）
3. Phase Cで生成した全deep_diveファイルを読む
4. `report.typ` を生成する（→ `capcom_schema/analysis/report_structure.md` セクション1の構造に従う）
5. **deep_diveの全文コピー**: 要約・圧縮・省略は一切禁止（→ `capcom_schema/analysis/report_structure.md` セクション2）
6. **品質検証ゲート (必須実行)**: 以下のスクリプトを実行し、結果をユーザーに報告する。exit code が 0 でない場合、不合格項目を修正してから再実行する。

   ```bash
   bash capcom_schema/scripts/phase_d_gate.sh
   ```

   このスクリプトは `capcom_schema/analysis/quality_checklist.md` の section 1 にある定量チェックコマンドを全て自動実行する。**「自前のチェックで代替」は禁止**(再現性のないチェックは無効)。

→ **完了条件**: report.typが品質基準を満たす
→ **レポート構造**: `capcom_schema/analysis/report_structure.md`（全体構造・deep_diveコピールール・結論章ガイド・付録テンプレート）
→ **品質検証**: `capcom_schema/analysis/quality_checklist.md`（定量チェックコマンド・全チェック項目・推奨項目）

---

## 3. モジュール一覧

| モジュール | JSON ファイル | 概要 | スキーマ |
|-----------|-------------|------|---------|
| ATLAS | atlas_statistics.json | 時系列推移、ランキング、ライフサイクル分析 | `capcom_schema/references/atlas_schema.md` |
| CORE | core_classification.json | ルールベース特許分類 | `capcom_schema/references/core_schema.md` |
| Saturn V | saturnv_clusters.json, saturnv_drilldown.json | AIクラスタリング (TELESCOPE/PROBE) | `capcom_schema/references/saturnv_schema.md` |
| MEGA | mega_momentum.json, mega_drilldown.json | 動態分析 (CAGR x 活動量 4象限) | `capcom_schema/references/mega_schema.md` |
| Explorer | explorer_global_network.json, explorer_trend.json, explorer_dominance.json | キーワード共起ネットワーク | `capcom_schema/references/explorer_schema.md` |
| CREW | crew_network.json | 発明者/出願人ネットワーク (要約版) | `capcom_schema/references/crew_schema.md` |
| EAGLE | eagle_clusters.json | 探索的ランドスケープ (手動クラスタリング) | `capcom_schema/references/eagle_schema.md` |
| NEBULA | nebula_hype_cycle.json, nebula_macro_events.json | 非特許文献統合・環境分析 | `capcom_schema/references/nebula_schema.md` |
| VOYAGER | voyager/mission.json, evidence/, context.json | 戦略レポート用データパッケージ | `capcom_schema/references/voyager_schema.md` |
| (共通) | *_wordcloud.json | 各モジュールのワードクラウド単語頻度 | `capcom_schema/references/wordcloud_schema.md` |

**スキーマ参照ルール**: `capcom_schema/references/` のスキーマファイルは、そのモジュールのJSONを実際に読む直前に参照する。全スキーマの一括読み込みは禁止。

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
3. **クロス検証**: 複数モジュールのデータを組み合わせて結論を補強する。最低3パターン実施（→ `capcom_schema/analysis/cross_module.md`）
4. **事実と推論の分離**: 4層分析モデルを適用（→ `capcom_schema/analysis/common_framework.md`）
5. **可視化参照（全章必須）**: 全ての章に最低1つの `#snapshot-figure()` を含める
6. **AIインサイト活用**: `prompts/` のAIインサイトを必ず参照し、深い読み取りをレポートに反映する
7. **データソーストレーサビリティ**: 全ての数値に具体的なモジュール名を含むマーカーを付与する
8. **Evidence網羅性**: Evidence総数の半数以上を分析に活用する
9. **Web調査（推奨）**: 外部情報を積極的に収集する（→ `capcom_schema/analysis/data_notes.md` セクション3）

## 5.5 データ特性に関する注意事項

→ **詳細**: `capcom_schema/analysis/data_notes.md`（特許とNPLの非対称性、ギャップ分析の注意、Web調査ルール）

## 5.6 分析ガイド (analysis/) と AIインサイト (prompts/)

`capcom_schema/references/` = データの「読み方」（辞書）、`capcom_schema/analysis/` = 「考え方・書き方」（分析手法）、`prompts/` = 「マップからの読み取り結果」（AIインサイト）。

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
| `query_logic_reading.md` | 母集団検索式の読解（7 DB 別構文 + 意図整合性検査 + データ逆読み） | Phase A（STOP-GATE A/B/C で必読） |
| `population_type_metrics.md` | 母集団 5 タイプ分類と指標解釈ルール（タイプ B/C/D の市場・業界表現禁止） | Phase A STOP-GATE C、Phase C/D 執筆時 |
| `terminology.md` | 用語統一ルール（最優先・内部識別子禁止・スコープ限定・サブクエスチョン化） | Phase A STEP 0、Phase D |
| `executive_summary_guide.md` | 経営層向け要約版（別冊）執筆ガイド（別冊フラグ ON 時のみ） | Phase A（確認）、Phase D（生成時） |

### exemplars/ ファイル一覧

| ファイル | 内容 | 使用フェーズ |
|---------|------|------------|
| `capcom_schema/exemplars/nebula_exemplar.typ` | NEBULA環境分析のお手本 | Phase C Step 0 |
| `capcom_schema/exemplars/saturnv_exemplar.typ` | Saturn V / EAGLE分析のお手本 | Phase C Step 1 |
| `capcom_schema/exemplars/explorer_exemplar.typ` | Explorer分析のお手本 | Phase C Step 2 |
| `capcom_schema/exemplars/mega_exemplar.typ` | MEGA PULSE分析のお手本 | Phase C Step 3 |
| `capcom_schema/exemplars/atlas_exemplar.typ` | ATLAS統計分析のお手本（権利化率分析 §10 含む） | Phase C Step 4 |
| `capcom_schema/exemplars/core_exemplar.typ` | CORE ルールベース分類分析のお手本 | Phase C Step 5 |
| `capcom_schema/exemplars/crew_exemplar.typ` | CREW 人的ネットワーク分析のお手本（指標別の解釈） | Phase C Step 6 |

> **お手本の使い方**: exemplar は「どう書くか」を具体例で示す。**exemplarを読まずにdeep_diveを書き始めてはならない。**（全7モジュールに exemplar あり）

### 段階的読み込みルール

**capcom_schema/analysis/**:
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
- `point-lead[...]` — **要点ストリップ**（番号セクション見出し直後に置く結論先出し1〜2行＝走査層。散文の上に重ねる。散文の代替ではない）
- `hl[...]` — インライン強調（数値以外のキーワードを選択的に。1セクション数語まで。多用禁止）
- `snapshot-figure("パス", caption: "説明")` — スナップショット画像
- `styled-table(columns: ..., header: (...), ..body)` — BCG風テーブル
- `conclusion-box("タイトル")[本文]` — 主要結論ハイライト
- `recommendation-card("高", "タイトル", "説明", timeframe: "短期")` — 優先度付き推奨
- `action-items("アクション1", "アクション2", ...)` — ToDoリスト

> 📐 **読みやすさ（走査層）— 詳細は `analysis/deep_dive_guide.md`「読みやすさ（走査層）」**: 地の文の塊を減らすため、①**各番号セクションの冒頭に `#point-lead[...]` を1個**置き結論を1〜2行で先出しする（散文は下にそのまま書く＝薄くしない）②件数・%・倍などの**数値＋単位はテンプレートが自動で太字強調**するので**手動で数字を太字化しない**③段落余白・見出しバーも自動。要点だけ書いて散文を削るのは Check 1（文字数）で不合格。

**注意**: `report_style.typ` のフォント設定を変更しないこと。`#set text(font: ...)` を report.typ に直接書かないこと。画像パスは `reports/` からの相対パス。typst compile に `--root ".."` を付けること。旧API（`#setup-page()` / `#cover-page(...)` 等）は廃止済み。

### python-pptx PPT
> ⚠️ **PPTX は `capcom_schema/templates/slides_spec.md` が唯一の正**。本節は要約に過ぎない。矛盾したら slides_spec を採用すること。
1. **🔑 ヘルパー関数は `apollo_slides.py` を import して使う（コピーしない）**: 生成スクリプトの冒頭で `import sys; sys.path.insert(0, "capcom_schema/templates"); from apollo_slides import *` し、`_apply_font` / `add_title_shape` / `add_sub_message` / `add_kpi_slide` / `add_cards_slide` / `add_matrix_2x2_slide` / `add_arrow_flow_slide` / `add_donut_slide` / `add_issue_tree_slide` / `add_process_slide` / `add_table_slide` / `add_bottom_bar_and_footer` 等を呼ぶ（`slides_spec.md` から写経しない）。**自前で pptx のフォント・色・レイアウトを書き起こさない**。フォント（**Noto Sans JP**）・多段ウェイト（見出し=Black / サブメッセージ=Medium / 本文=Regular / 出典=Light）・上下中央寄せ・箱の充填・タイトル下線・ボトムバーは**すべてこれらヘルパーに内蔵**されている。自前実装するとこの品質が失われる（＝過去に Codex で発生した「単一ウェイト・平板」の原因）
2. **🔑 デッキは完成レポート `reports/report.typ` を土台に作る** — evidence の寄せ集めでなく、各章の主張→根拠（数値）→示唆を凝縮し章順に沿わせる（slides_spec §0.9）
3. `capcom_schema/templates/slides_spec.md` を **Section 0〜6 まで設計ガイドとして熟読**（いつ・どのヘルパーを・どんな主張骨格で使うか。とくに §0.9「レポートを土台にする」「数値一貫性」「出所」「過剰修辞禁止」）→ `apollo_template.pptx` を `reports/` にコピー → `slide_layouts[6]`（Blank）で生成
4. **出所**: 分析モジュール名や **`reports/report.typ` 等のファイルパスを出所に書かない**。特許データ由来は「本分析の特許データセット」、事業/市場ファクトは Web 実出所（付録C）。タイトルに波ダッシュ「～」副題を使わない。過剰修辞（越境者・狼煙 等）禁止
5. **構成比率**: チャート+注釈 50%以上 / 同一スライドタイプ 3 枚連続禁止 / 空白を作らない（§0.8）。出力: `reports/presentation.pptx`、推奨 25〜35 枚
6. **完了後に必ず `bash capcom_schema/scripts/phase_d_gate.sh`** を実行し Check 16（PPTX 機械チェック）の WARN も可能な限り解消する
7. **別冊（経営層向け要約版）を生成する場合は `analysis/executive_summary_guide.md` に従い 8〜12 ページ**（要点ごとに 8〜12 行＋数値 callout）。表紙＋2 段落だけの薄い別冊は品質不合格

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

---

## 9. Codex CLI 固有の運用

### 9.1 スキル配置と呼び出し

**配置場所**: 本ファイルは `session_YYYYMMDD_HHMMSS/.codex/skills/apollo-capcom/SKILL.md` に置かれる。プロジェクトスコープのスキルとして、セッションディレクトリを cwd にして Codex を起動した時のみ有効。

**呼び出し**:
- 明示起動: Codex チャットで `$apollo-capcom` とタイプ
- メニュー起動: `/skills` → `apollo-capcom` を選択
- 暗黙起動: Mission Objective を含むユーザー発話から Codex が自動選択する場合あり（description に基づく）

### 9.2 AGENTS.md との関係

`session_*/AGENTS.md` は **プロジェクト全体のルール** を定義します。本SKILLはAGENTS.mdで指定された「cwd 規約」「bash gate 必須」「サブエージェント禁止」「capcom_schema/ の共有資産を参照」を前提に書かれています。AGENTS.md を削除・変更すると本SKILLの動作が不安定になります。

階層優先度: `~/.codex/AGENTS.md` < `session_*/AGENTS.md` < 本SKILL.md

### 9.3 対話モード（TUI）必須 ＋ 部分自動化

本スキルは**判断を要するユーザー応答待ちゲートを多数**持ちます（各 Phase A STOP-GATE＋タイトル決定＋**各フェーズ境界のセッション・チェックポイント**）。Codex の `ask_user_question`（対話質問）は **TUI（対話モード）でしか使えない**（非対話 `codex exec` での対話質問は [openai/codex#10384/#11536](https://github.com/openai/codex/issues/11536) として未実装）ため、**フルパイプラインは必ず `codex` で TUI 起動してください**。

❌ `codex exec "レポートを書いて"` — 非対話モードでは `ask_user_question` が利用不可（最初の判断ゲートで停止）
✅ `codex` → TUI でチャット `$apollo-capcom` → 各ゲートでユーザー応答

**部分自動化（A/B 対話 → C/D 非対話再開）**: 2026-06 時点で **`codex exec resume --last` / `<SESSION_ID>`**（追従プロンプト可）が使えるようになりました。Phase A/B を TUI で通過して判断を確定し、`reports/_carryover.md`（必要なら `voyager/mission.json`）に固定すれば、Phase C/D は `codex exec resume` で非対話実行できます（C/D の合否は `phase_c_gate.sh`/`phase_d_gate.sh` が客観判定するため自動化可能。非対話で回す場合チェックポイントは「一気に最後まで」扱いで省略）。**完全な end-to-end 非対話化は exec 内の対話質問が未実装のため不可**。詳細は `exec_mode_addendum.md` 参照。

### 9.4 ツール呼び出しマッピング

Codex CLI で本スキルが使うツールは以下：

| 用途 | Codex ツール | 旧 Claude Code 相当 |
|---|---|---|
| ファイル読み | `Read` (ファイル) | `Read` |
| ファイル検索 | `Grep` / `Glob` | 同名 |
| コマンド実行 | `Bash`（bash gate 実行用） | `Bash` |
| ユーザー質問 | `ask_user_question`（TUIのみ） | `AskUserQuestion` |
| エディット | `Edit` / `Write` | 同名 |
| コンテキスト圧縮 | `/compact` | `/compact`（共通） |

**prompts/ 配下のテンプレ**: `.codex/skills/apollo-capcom/prompts/phase_a2_titles.md` 等は、`ask_user_question` に渡す質問文と選択肢を構造化したマークダウンです。ゲート通過時に該当テンプレを参照し、必要項目を埋めてから `ask_user_question` を呼び出してください。

### 9.5 サブエージェント禁止の再確認

Codex CLI には 2026年4月時点で Claude Code の Agent tool のような汎用サブエージェント機構は **存在しません**。本項は念のための防衛規定であり、仮に将来 Codex が追加した場合でも本スキル内では起動しない、という意味です。

### 9.6 `/compact` の使い方

Codex の `/compact` は Claude Code と同一機能。Phase C 途中でコンテキストが逼迫した場合：

1. 現在の Phase/Step を明示してユーザーに `/compact` 実行を依頼
2. `/compact` 実行後、`voyager/mission.json` と現在作業中のモジュールのみ再読み込み
3. 共有資産（`capcom_schema/analysis/`）は必要なセクションのみ `grep` で部分読み込み

ゲート通過の事実（どの Phase まで完了したか）は `reports/` 配下の成果物ファイルから復元可能です。
