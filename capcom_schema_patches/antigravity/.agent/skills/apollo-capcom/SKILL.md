---
name: apollo-capcom
description: >
  APOLLO特許分析プラットフォームのCAPCOMセッションデータを
  解釈し、Artifact駆動（task.md / implementation_plan.md / walkthrough.md）で
  戦略レポートを生成する Antigravity IDE 用スキル。
  Review Policy = "Request Review" 推奨。
---

> **このファイルは要約版。各フェーズの開始前に指定されたリファレンスファイルを必ず読むこと。**
> **Antigravity IDE 専用版**。Claude Code 用の `capcom_schema/SKILL.md` を Antigravity の **Artifact-first パラダイム** に翻案しています。
> 共有資産（`analysis/`, `references/`, `exemplars/`, `templates/`, `scripts/`）は既存の `capcom_schema/` 配下をそのまま参照します。

## 進行様式の分岐（自律生成 / 対話型 KATHERINE）— 最初に確認

作業開始時に **`voyager/context.json` の `report_mode` を必ず確認**する:

- **`"autonomous"`（既定・未指定含む）**: 本ファイルの手順（自律生成モード）でそのまま進行する。
- **`"interactive"`（またはユーザーが対話型を明示した場合）**: **`capcom_schema/interactive/SKILL_INTERACTIVE.md` を進行の正本として読み**、`capcom_schema/interactive/dialogue_points.md`（対話ポイント CP-1〜8）を併読して**対話型レポート作成モード（KATHERINE）**で進行する。起動点は `.agent/workflows/06_interactive.md`。品質ゲート・成果物形式・トークン効率制約は自律生成モードと完全に同一（本ファイルの §0・トークン効率制約・各 Phase 完了条件はそのまま適用）。

**対話型の Antigravity 翻案**（詳細は `GEMINI.md`「レポート生成の進行様式」）:
- SKILL_INTERACTIVE.md の `AskUserQuestion` は **Artifact Review 承認**に読み替える（✅確定 = チェックボックス `[x]` / ✏️修正・🔄別案 = Google Docs 式コメント → 修正 → 再レビュー / 🤖おまかせ = 第4のチェックボックス）
- 確定必須ブロックには `<!-- ANTIGRAVITY_REVIEW_REQUIRED -->`、応答不要の情報提示（CP-7 突合サマリ等）には `<!-- ANTIGRAVITY_INFO_ONLY -->` マーカーを使う（INFO_ONLY では承認待ちにしない）
- CP-4/6/7 の対話は **`dialogue_review.md`**（雛形: `artifacts_templates/dialogue_review.md.tmpl`）で運用する。CP-1/2/3/5/8 は `implementation_plan.md` の対応欄（4部構成欄・結論候補節・WARN トリアージ表）を使う
- **対話型では Review Policy = "Request Review" が必須**。他設定を検出したら Phase 0 で停止し、ユーザーに変更を依頼する（`review_policy_recommendation.md`）
- ⚠️ 対話型は Claude Code で検証済み・**Antigravity では実機検証未了のベストエフォート対応**（進行様式は同一、確定手段のみ読み替え）

## 0. 絶対遵守ゲートルール (最優先)

**以下は他の全ルール(トークン効率制約含む)に優先する。例外なく適用する。**

1. **全ゲートは省略不可**: 「ユーザーが短く指示した」「効率上スキップしたい」等の理由でゲートを省略してはならない
2. **Artifact Review 必須**: 「ユーザーに確認」と書かれた箇所では、該当 Artifact（`implementation_plan.md` のセクション or `task.md` のチェックボックス）を更新したあと、**Antigravity の Artifact Review でユーザー承認を待つ**。AI 自己判断で次 Phase に進まない
3. **不合格時は強制ループ**: Phase 完了条件を満たさない場合（特に bash gate FAIL 時）、必ず該当 Phase に戻る。「実質的にOK」「内容は保持」等の質的判断で量的基準(行数・件数)を上書きしない
4. **指示の長さで手順を変えない**: ユーザー指示が「レポートを書いて」のように短くても、本 SKILL.md の全手順に従う
5. **「省略します」と宣言する前に立ち止まる**: 何かを省略する判断をした瞬間、ユーザーに `implementation_plan.md` のコメント or チャットで省略の可否を確認

6. **水増し（同一文の反復）禁止 — 量より固有性が合否を決める**: 「最低◯◯行/◯件」は深さの目安であり、合否は内容の固有性で決まる。**同一文・同一構文の反復、回転する名詞だけ変えた定型文の量産、「○○観点 1, 2, 3…」式の連番見出しで行数・件数を稼ぐことは禁止＝`phase_d_gate.sh` Check 19 で自動不合格**。**反復でなくても、データ（本母集団）を見なくても言える自明な一般論で字数を稼ぐのも水増しに含む**（例:「特許は権利文書である」「定型語は技術的意味を持たない」）。行数が不足する時は、文を繰り返す代わりに ①新しい代表特許（固有の公開番号）②新しい数値根拠 ③別のクロスパターン ④Web調査の裏付け を足す。各段落は本母集団固有の事実（固有の数値・公開番号・クラスタ名・出願人名のいずれか）を最低1つ含めること（理由: 反復・自明な一般論は読者に無価値で、機械ゲートで弾かれる）

7. **本文のスクリプト生成・ゲート回避の禁止**: レポート本文・`deep_dive.typ`・`report.typ` を Python 等のスクリプトでテンプレート生成してはならない（`reports/generate_*.py` のような本文生成スクリプトは `phase_d_gate.sh` Check 19a で自動不合格）。各文は特許群の固有事実に基づく分析として直接書く。「最低行数を満たすための補助文・つなぎ文」を入れてはならない（行数不足＝分析不足。固有の事実を足すか、その章を短く確定する）。**`phase_d_gate.sh` を読んで反復検出を『回避』する目的で、接続詞・語順・文体だけを変えて内容の重複を温存することは禁止**。ゲートは実在する欠陥（重複）を検出している。正しい対処は重複文の削除と固有内容への置換であって、検出のすり抜けではない。**また「最低◯◯行」を満たすために1文ずつ改行して行数を稼ぐのも水増し**＝ゲートは行数(`wc -l`)でなく**非空白文字数**で判定する（Typst では行内改行は描画上スペースで見た目不変・行数は改行で増やせるが文字は増やせない）。「第一の…である。第二の…である。」式の一文一行の羅列でなく、複数文で論証（主張→根拠→示唆）を組む段落を書くこと

8. **工程ナレーション節・後続章への申し送りの禁止**: 「後続分析への接続」「次章への申し送り」のような、**他章でやることのToDoを並べただけのメタ節・段落を作らない＝意味のない水増し**（完成レポートでは各モジュール章が既に存在するため無価値）。「Explorer分析では〜を確認する」「後続のCORE分類で確認する必要がある」式の**前向きの申し送り**も禁止。各章は自章の分析と結論で閉じ、章間連携は『クロスモジュール統合分析』章で行う。他章への言及は「〜で確認された〈事実〉」の**過去形・根拠引用**に限る。`phase_d_gate.sh` **Check 8e** で自動検出 FAIL（→ `analysis/deep_dive_guide.md`「工程ナレーション節を作らない」・`analysis/terminology.md` §1-4）

9. **STOP-GATE はコンテキスト限界でも死守（捏造・先送り厳禁）**: STOP-GATE（Antigravity では該当 Artifact を更新し `<!-- ANTIGRAVITY_REVIEW_REQUIRED -->` を置いて **Artifact Review でユーザー承認を待つ**動作）を**実際に発行する前**に次フェーズへ進んではならない。**STOP-GATE に到達する前にコンテキスト/クォータが限界に近づいたら**: ① `reports/_carryover.md`（＋ brain）に現在地・確定値を保存 → ② ユーザーに「コンテキストが厳しいので一旦タスクを区切ります／`/compact` します。再開後に必ず STOP-GATE（母集団タイプ・分析の立場・別冊・タイトル・重点）を Artifact Review で確認します」と**告げてから**区切る → ③ 再開後、**最初に該当 Artifact を更新して Artifact Review を出す**。**Artifact Review を実際に発行していないのに「ユーザーが承認した／承認が得られなかった」と仮定して進むのは厳禁**（存在しない承認の捏造＝重大違反。`<!-- ANTIGRAVITY_REVIEW_REQUIRED -->` を置いて実際にユーザーが応答するまでフェーズを進めない）。そもそも Phase A のデータ精読で枯渇させないため、統計は **ARTIFACT GATE C の C-1 ワンショットスクリプトで1回だけ算出**し、CSV の試行錯誤・カラム名の探り当て・再読み込みをしない（今までの枯渇の主因）。

このメタルールは下記「トークン効率に関する制約」よりも上位。両者が衝突する場合、本ルールが勝つ。

## トークン効率に関する制約（ツァーリ・ボンバ対策）

**以下のルールはレポートの品質とトークン効率を両立するために厳守すること。**

1. **サブエージェント禁止**: Antigravity には Manager + Browser subagent 等あるが、**本スキル実行中は起動しない**。全処理をメインコンテキスト内で完結させる（`brain` メモリは活用可）
2. **ファイル読み込み最小化**: 一度読んだ内容は Artifact に記録して参照し、再読み込みしない
3. **バッチ処理**: 複数のdeep diveをまとめて処理する
4. **Phase別スキーマ参照**: `capcom_schema/references/` の個別スキーマは対象モジュールのみ読む

### 🚨 ゲートとの優先順位

**トークン効率制約は品質ゲートを犠牲にする理由にはならない。** ゲートが優先(`## 0. 絶対遵守ゲートルール` 参照)。

両者が衝突する場合、**ゲート優先**。トークンが足りない場合：
- Antigravity の `brain` メモリに「Phase A完了時点の要点」「Phase B結論」等を段階的に保存
- 不足分は再度 `capcom_schema/` から該当セクションのみ部分読み込み

## ⚠️ 大型レポートのフェーズ分割（コンテキスト枯渇・クォータ対策）

CAPCOM のレポート生成は巨大なタスク（patents.csv＋多数のJSON＋prompts読込 → 800行超の report.typ＋6〜7本の deep_dive＋PPTX）で、1スレッド/1タスクで一気に通すとコンテキスト窓を超過しやすい。本フローは**各フェーズの成果物をディスクに書く設計**なので、タスクを分けても続きから再開できる。

**分割実行のしかた**:
1. `.agents/workflows/` の **フェーズ別ワークフロー（00_master / 01_phase_a / … / 05_phase_d）を1タスク=1フェーズで実行**する（Phase C は1モジュールずつ）。各フェーズ完了で Artifact をレビューし、次タスクへ
2. 新タスクでは、まず `ls reports/` で完了状況を確認してから続きを実行する:
   - `reports/_phase_a_decisions.json` あり → Phase A 完了
   - `reports/<module>_deep_dive.typ` あり → そのモジュールの Phase C 完了
   - `reports/report.typ` あり → Phase D 統合済み
3. 各ゲート（`phase_c_gate.sh` / `phase_d_gate.sh`）はディスク状態を見るので、再開後も同じ判定が効く
4. **クォータ/レート制限**（無償枠は週次リフレッシュで長時間ロックアウトの報告あり）に備え、長時間タスクは中断・再開に強い分割を徹底する

**記憶の引き継ぎ（`reports/_carryover.md`）**: 成果物 `*.typ` と `_phase_a_decisions.json` だけでは Phase B の仮説検証過程・**Web調査の出所(URL/取得日)**・判断理由が新タスクで失われる。Antigravity は brain/Artifact（`implementation_plan.md` の § Confirmed Cross Patterns / § Confirmed Web Research 等）がネイティブ記憶だが、**3ツール共通の跨タスク正本は `reports/_carryover.md`**（矛盾時はこちら優先）。
- 無ければ `capcom_schema/templates/carryover_template.md` を `reports/_carryover.md` にコピー。新タスク冒頭で通読 → Artifact/brain を再ロードして主記憶を再構築
- 各 Artifact 確定（クロス確定・Web確定・walkthrough ゲート結果）の直後に `_carryover.md` の該当節へ**要約をミラー**（Artifact にある詳細はポインタのみ・二重の全文転記はしない）。**Web は1件ヒットごとに即 WEB出所台帳へ**
- **レポート本文へは転載しない**（内部作業メモ）。詳細は `capcom_schema/SKILL.md` §フェーズ間引き継ぎ日誌

**やってはいけない**: 全フェーズを1タスクで一気に通そうとする ／ 本文生成スクリプトを書いて巨大な文字列をコンテキストに抱える（§0 第7項で禁止）。

### 🔄 セッション・チェックポイント（各フェーズ境界で必須・枯渇の予防）

「完走できそうにない」と感じてから分割するのでは遅い（枯渇・クォータ超過は予兆なく起きる）。そこで **各フェーズのキリのいい境界で、ルーチンとしてタスク（セッション）切替を提案して一旦停止する**。対象境界:

- **Phase A 完了**（`reports/_phase_a_decisions.json` 保存後）
- **Phase B 完了**（エビデンス＋クロス確定後）
- **Phase C の各モジュール完了ごと**（deep_dive 1本ごと＝最も枯渇しやすい区切り）
- **Phase D 着手前**（`report.typ` 統合は重いので、その前で一度切る）

各境界で必ず以下を順に実行する:

1. 当該フェーズ/モジュールの **ゲート・Artifact 完了条件を満たしたことを確認**（`phase_c_gate.sh` 等）
2. **`reports/_carryover.md` を更新**（STATUS=現在地・RESUME=次にやること・直近の固有事実・Web出所台帳。Artifact 確定の要約もミラー）
3. **ユーザーにチェックポイントを提示して応答を待つ**（テキスト出力だけで満足せず、ユーザーの選択を取得するまで次に進まない）:

   > ✅ **Phase X（／モジュール M）完了・ゲート通過。`reports/_carryover.md` 更新済み。**
   > ここは安全な区切りです。コンテキスト枯渇・クォータ超過を避けるため、**新しいタスク（セッション）への切り替えを推奨します**。
   > 新タスクでは `reports/_carryover.md` ＋ Artifact/brain を読んで **次（Phase Y ／ 次モジュール）から自動再開**します。
   > - 🔄 **新タスクに切替（推奨）**: いまのタスクを閉じ、新規タスクで `ls reports/` ＋ `reports/_carryover.md` を読んで再開
   > - ▶️ **このまま続行**: コンテキストにまだ余裕がある場合のみ（次の境界で再びチェックポイントを出す）
   > - 🗜️ **`/compact` で続行**: 同タスクのまま圧縮（軽い選択肢）

4. ユーザーが **切替** を選んだら、現タスクはここで終了してよい（成果物・Artifact・`_carryover.md` はディスク/brain に残る）。**続行/`/compact`** を選んだらそのまま次へ進み、**次の境界で再びチェックポイントを出す**。

**重要**: このチェックポイントは「枯渇しそうな時だけ」ではなく **各境界で必ず出す**（予防が目的）。ユーザーが「最後まで一気に続けて」と明示した場合のみ、以降のチェックポイントを省略してよい。

# APOLLO CAPCOM Skills (Antigravity版)

## 1. 概要

**APOLLO** は Streamlit ベースの特許分析プラットフォーム。9つのモジュールが特許データを多角的に分析する。

**CAPCOM** (Capsule Communicator) は APOLLO と AI coding agent を繋ぐ通信モジュール。分析結果をファイル出力し、Antigravity IDE で開いたセッションフォルダ上でレポート生成を行う。

### セッションフォルダ構造

```
session_YYYYMMDD_HHMMSS/
├── capcom_schema/         # 共有資産（分析手法・スキーマ・品質ゲート）
├── data/                  # patents.csv + 各モジュールJSON
├── voyager/               # Mission Objective + Evidence
├── snapshots/             # スナップショット画像(PNG)
├── prompts/               # AIインサイト(Markdown)
├── reports/               # ★レポート出力先
├── .agent/                # Antigravity スキル配置（本ファイル等）
│   ├── skills/apollo-capcom/SKILL.md
│   └── workflows/*.md     # Phase別起動点
├── artifacts_templates/   # 本スキルで使う Artifact 雛形（対話型用 dialogue_review.md.tmpl 含む）
├── task.md                # Artifact: 4フェーズチェックリスト（本スキル起動時に生成）
├── implementation_plan.md # Artifact: 承認対象セクション群（同上）
├── walkthrough.md         # Artifact: ゲート結果記録（Phase C/D 完了時）
├── dialogue_review.md     # Artifact: 対話型 KATHERINE のみ（CP-4/6/7 のローリング対話）
├── GEMINI.md              # Antigravity最優先ルール
├── AGENTS.md              # fallback / Codex互換
└── metadata.json
```

## 2. 本スキル起動時の初動

レポート生成依頼を受けたら、以下の順序で **必ず最初に Artifact 3ファイルを生成**：

1. `artifacts_templates/task.md.tmpl` を `task.md` にコピー
2. `artifacts_templates/implementation_plan.md.tmpl` を `implementation_plan.md` にコピー
3. `artifacts_templates/walkthrough.md.tmpl` を `walkthrough.md` にコピー
4. 3ファイルを Antigravity に **Artifact として登録**（Review Policy = "Request Review" 時、ユーザー承認待ちになる）

これ以降、各 Phase の進行に伴って Artifact を更新し、**ユーザー承認（Artifact への ✅ or コメント）を経てから次 Phase に進む**。

### Review Policy の確認

ユーザー側で Antigravity の Review Policy が **"Request Review"** または **"Agent Decides"** になっているか確認する。"Always Proceed" だと本スキルのゲートが機能しないため、その場合はユーザーに設定変更を依頼する（`review_policy_recommendation.md` 参照）。

## 3. 利用モード

### コンテキスト管理の原則（全モード共通）

1. **patents.csvは絶対に全量読み込みしない**: `head -5` でカラム構成を確認し、必要な分析の都度pandasで条件検索する
2. **JSONは必要なモジュールのみ読む**: 全JSONの一括読み込み禁止
3. **references/スキーマは対象モジュールのみ読む**: 全スキーマの一括読み込み禁止
4. **analysis/ガイドは段階的に読む**: まず `capcom_schema/analysis/common_framework.md` のみ

### 自由分析モード
`data/` 配下の CSV/JSON をユーザーの質問に応じて読み取り、回答する。Artifact 生成は不要（会話内のやり取り）。

### レポート生成モード
本スキルの中核。VOYAGER Export 後に利用。`voyager/mission.json` の Mission Objective に基づく正式レポートを作成する。以下の4フェーズで進行。

---

## 3.5 環境準備（依存インストール・最初に1回）

レポート生成は **patents.csv の解析に `pandas`、スライド生成に `python-pptx` / `Pillow`** を使う。これらはセッションフォルダ直下の **`requirements-session.txt`** に列挙済み。**Phase A のデータ精読に入る前に、IDE のターミナルで依存を必ず確認・導入すること**（未導入のまま `import pandas` / `import pptx` すると `ModuleNotFoundError`（例: `No module named 'pandas'`）で止まる）:

```bash
# セッションフォルダ直下で実行（揃っていればスキップ、無ければ一括導入）
python3 -c "import pandas, pptx, PIL" 2>/dev/null && echo "依存OK" || pip install -r requirements-session.txt
```

- `pip` が見つからなければ `python3 -m pip install -r requirements-session.txt`。書き込み権限エラーなら末尾に `--user` を付す。
- 仮想環境を使うなら、セッションフォルダで `python3 -m venv .venv && source .venv/bin/activate` の後にインストールし、以降の `python3` も同じシェルで実行する。
- ネットワーク制限等で `pip install` が通らない場合は、依存が無いまま分析を始めず、ユーザーに「セッションフォルダで `pip install -r requirements-session.txt` を実行してから再開してください」と伝えて一旦停止する。

## 4. レポート生成 4フェーズ手順（Artifact駆動）

### Phase A: ミッション理解 + データ精読

voyager/mission.json を読み、data/以下のJSONとpatents.csvを把握する。**Phase A は複数の Artifact Review STOP-GATE で構成される**（本家 Claude Code 版と機能的に等価。Antigravity では `AskUserQuestion` に相当する動作を **Artifact Review（`implementation_plan.md` セクション更新 + ユーザー承認待ち）** で実現する）。

終了時に `implementation_plan.md` の以下セクションが埋まっている必要あり:
- § Mission Objective
- § Dataset Context
- § Evidence Inventory
- § Key AI Insights
- § Population Meta（4 フィールド）
- § query_logic Reading（STOP-GATE A 結果、指定時）
- § query_intent 3-Point Summary（指定時）
- § Sub-Questions（指定時、内部メモ）
- § Intent-Logic Divergences（STOP-GATE B 結果、両方指定時）
- § Data Level 2 Reverse-Read（STOP-GATE C 結果）
- § Population Type（A/A'/B/C/D）
- § Narrative Stance（self/competitor/buyer/supplier/neutral、STOP-GATE C-2' 結果）
- § NEBULA Strategy（STOP-GATE D 結果）
- § Executive Summary Edition Decision

**全ステップは省略不可。**

🛑 **STEP 0 (最優先)**: 用語統一ルールの読了と母集団メタ情報の確認
- [ ] `analysis/terminology.md` を**最初に**読む（§1-6 すべて: 内部識別子の露出禁止 / Mission Objective ベタ貼り禁止 / 母集団メタ §5 / スコープ限定ルール §6 / サブクエスチョン化 §5-A-2）
- [ ] `voyager/context.json` の `population_meta` 4 フィールドを `implementation_plan.md` § Population Meta に転記:
  - `query_intent` / `query_logic` / `coverage_years` / `database_name`
  - **未指定の `database_name` は「提供された特許データセット」と汎用表記**（J-PlatPat 等を勝手に補わない）

🛑 **ARTIFACT GATE (経営層向け要約版〈別冊〉の生成確認)**:
- [ ] `implementation_plan.md` § Executive Summary Edition Decision に 3 択を記入:
  ```markdown
  - [ ] ✅ 両方生成（本編 + 別冊）
  - [ ] 📘 本編のみ
  - [ ] ❓ 相談したい
  ```
- [ ] `<!-- ANTIGRAVITY_REVIEW_REQUIRED -->` を直前に配置し、**Antigravity Artifact Review でユーザー承認**を待機
- [ ] 選択結果を作業メモに固定。「両方生成」選択時 → **別冊生成フラグ = ON**、Phase D で `reports/report_executive.typ` を生成

詳細ガイド: `analysis/executive_summary_guide.md`

🛑 **ARTIFACT GATE A (query_logic 構造化読解) — `query_logic` が指定されている場合のみ必須**:
検索式を付録 D にコピペするだけで済ませるのは禁止。4 ステップ:
- [ ] `analysis/query_logic_reading.md` を読了（7 DB 構文: J-PlatPat / JP-NET / Patentfield / Shareresearch / BizCruncher / PatentSQUARE / PatSnap）
- [ ] **Step 1-3** を `implementation_plan.md` § query_logic Reading に記入（DB 識別 → 構文分解 → 意図推定）
- [ ] **Step 4**: `<!-- ANTIGRAVITY_REVIEW_REQUIRED -->` を配置、**Artifact Review で「この読解で合っているか」をユーザー確認**
- [ ] ユーザー承認後、§ query_logic Reading に「✅ Confirmed」を追記

🛑 **ARTIFACT GATE (`query_intent` 3 点整理) — `query_intent` が指定されている場合のみ必須**:
- [ ] `implementation_plan.md` § query_intent 3-Point Summary に記入:
  ```markdown
  - 分析目的 (1 行): ...
  - 母集団の輪郭 (2-3 行): ...
  - 分析の視座 (1-2 行): ...
  ```
- [ ] `<!-- ANTIGRAVITY_REVIEW_REQUIRED -->` を配置、**Artifact Review でユーザー合意**を待機
- [ ] **ベタ貼り禁止**: 原文のままレポートに書かず、Phase B 以降で「分析の視座」として内在化
- [ ] **設計意図を無視した汎用分析は品質不合格**

🛑 **ARTIFACT GATE (サブクエスチョン化) — `query_intent` が指定されている場合のみ必須**:
- [ ] `implementation_plan.md` § Sub-Questions に 3-5 個の観点を箇条書きで起草（各観点にキーワード 1-3 個を付記）。**確定した立場（`narrative_stance`）の観点で SQ の抜けを点検**（self=自社の弱点・空白／competitor=対象企業の隙・参入余地／neutral=強み・リスク・投資妙味）。`query_intent` を最優先（分析者は通常その立場で母集団を設計済み）
- [ ] `<!-- ANTIGRAVITY_REVIEW_REQUIRED -->` を配置、**Artifact Review でユーザー確認**
- [ ] 確定後、`reports/_phase_a_decisions.json` の `sub_questions` に保存
- [ ] **⚠️ 絶対制約**: サブクエスチョンは**内部メモ専用**。レポート本文に「Q1 / A1 / SQ1 / 問い 1」等の記号・形式は禁止。本文は通常の宣言調で書く（詳細: `terminology.md` §5-A-2）

🛑 **ARTIFACT GATE B (意図 ↔ 論理 整合性検査) — `query_intent` と `query_logic` が両方指定されている場合のみ必須**:
- [ ] `analysis/query_logic_reading.md` §4 の **8 項目**（技術領域 / 用途 / 対象期間 / 地域 / 出願人絞り込み / 除外条件 / 公報種別 / 分類階層）で対比
- [ ] 乖離を 3 段階に分類、`implementation_plan.md` § Intent-Logic Divergences に記入:
  - 🔴 Critical / 🟡 Warning / 🔵 Info
  - 各乖離に **具体的な改善提案**を添える（例: 「末尾に `* NOT (A23*/IC)` を追加すると意図に沿う」）
- [ ] `<!-- ANTIGRAVITY_REVIEW_REQUIRED -->` を配置、**Artifact Review で対処方針を確認**（[A] 修正して再抽出 / [B] このまま進めて「範囲と限界」章で明記 / [C] 無視 / ✅ 乖離なし）
- [ ] Critical 検出でも進行可能（ユーザー判断尊重）

1. `voyager/mission.json` を読む（Mission Objective + Evidence 一覧）→ `implementation_plan.md` § Mission Objective に転記
2. `voyager/context.json` でデータセットのメタ情報と `population_meta` / `capcom_tools` / `report_directives`（`image_slide_instruction`＝画像・スライドのユーザー指示）を確認 → § Dataset Context に転記
3. `evidence_list` の全件を走査し、§ Evidence Inventory に一覧表を作成
4. `snapshots/` のファイル一覧を取得し、Evidence と紐付け
5. `data/patents.csv` を読む: `head -5` + pandas で出願人上位 10 社・クラスタ別・年別件数把握 → § Dataset Context に記録
6. `data/` 以下の全 JSON ファイルから主要数値を抽出 → メモ
7. `prompts/` の AI インサイトを **主要モジュール（Saturn V/MEGA/ATLAS/Explorer/CREW/NEBULA/CORE）各1件以上、かつ全体で最低8件** 読み（インサイトが少なければ全件読む。読了数が少ないと deep_dive が表面的になる）、要点を § Key AI Insights に記録
8. `task.md` の Phase A チェックボックスを更新

コンテキスト管理: `saturn_drill_insight.md`（最大 220KB）や `crew_network_insight.md`（最大 400KB）は全量読み込み禁止。対象箇所のみ `grep` で部分読み込みすること。

🛑 **ARTIFACT GATE C (データ側からの母集団実態確認 + 母集団タイプ判定) — 必須（全ケースで実施）**:

**C-1. データ Level 2 逆読み**

**⚠️ patents.csv の実カラム（試行錯誤＝コンテキスト枯渇を避けるため最初に把握する）**: `data/patents.csv` は APOLLO 処理済みで、カラムは **処理済み**（`applicant_main`=主出願人 / `inventor_main` / `year`=出願年 / `ipc_main_group` / `cluster` / `cluster_label` / `umap_x`,`umap_y` / `core_技術分類`,`core_課題分類`,`core_解決手段分類`）と **原データ**（`発明名称`〈先頭に BOM あり〉/ `要約` / `出願番号` / `公開番号`）の**混在**。⚠️ **`applicant_main` / `inventor_main` / `ipc_main_group` は `"['キオクシア', '東芝']"` のような Python リストの文字列**なので、集計には `ast.literal_eval` での展開（explode）が必須（しないと共同出願を1社と誤カウントする）。`cluster` は整数、`cluster_label` は `'[3] 半導体記憶, メモリセル, 半導体'` 形式の文字列。**`voyager/context.json` の `column_mapping` は元 CSV 名（`applicant`=出願人 等）で、patents.csv の実カラムとは一致しない**ので照合に使わない。**ステータス（権利状況）列は patents.csv に無い** — 権利化率は `prompts/atlas_*_insight.md` のステータス内訳から読む。

**ワンショット統計スクリプト**（実データで検証済み・出願人HHI 等が正しく出る。BOM 対応 `encoding="utf-8-sig"`、リスト文字列は展開、出力は `.to_string()` で1ブロックに収める。**heredoc の多重実行・カラム名の探り当て・Unicode 正規化の試行は禁止＝今までの枯渇の主因**。結果は § Data Level 2 Reverse-Read に転記）:
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

- [ ] `analysis/query_logic_reading.md` §5 の **Level 2 項目**を算出し、`implementation_plan.md` § Data Level 2 Reverse-Read に記入:
  - 総件数・対象期間・使用 DB / 上位 10 出願人・シェア / 主要 IPC/FI 上位 10 / 出願年分布 / 出願人集中度 HHI / 国・地域分布
- [ ] **自動偏り警告**: 上位 1 社 30% 超 / 上位 1 IPC 40% 超 / 直近 2 年 50% 超集中 / HHI > 0.25 / 特定国 95% 超 を検出

**C-2. 母集団タイプ判定**
- [ ] `analysis/population_type_metrics.md` を読了し、5 タイプから候補を推定して § Population Type に記入:
  - **A 業界全体** / **A' 技術領域** / **B 競合限定** / **C 単一企業** / **D 特定製品・技術テーマ**
  - 判定目安: 上位 1 社 > 90% → C / 上位 5 社で 95% 超 → B / 上位 10 社 40-70% → A' / 上位 10 社 < 40% → A / 複合的絞り込み + 上位 10 社 > 70% → D
- [ ] タイプ C では出願人 HHI 算出無意味（HHI=1.0）、タイプ B/C/D では「市場集中」「業界シェア」等の **市場・業界解釈は禁止**（`population_type_metrics.md` §3）

**C-2'. 分析の立場（叙述スタンス）判定 — 母集団タイプとは独立に必ず判定**

母集団タイプ（データの構成）と「**誰の意思決定のためのレポートか**（＝提言・主張を語る立場）」は**別物**。母集団が単一企業（タイプ C）でも読者＝依頼主は対象企業自身とは限らず、競合・投資家・アナリストのこともある。**`population_type` が C だからといって対象企業を自動的に「当社」と呼んではならない**（取り違えると、対象企業を勝手に「当社」と書く／中立であるべき評価が当事者寄りになる、といった誤りが生じる）。
- [ ] `query_intent` / Mission Objective から **分析の立場** を 5 分類で推定し、`implementation_plan.md` § Narrative Stance に記入:
  - **self**（自社視点・当事者本人）: 対象企業を「当社」と呼ぶ／一人称可。手がかり: `query_intent` に「自社」「当社」「我々の」
  - **competitor**（競合視点・水平）: 対象企業を企業名で三人称（「キオクシアは」）。一人称は読み手＝競合自身を指す時のみ。手がかり: 「競合」「ベンチマーク」「対抗」
  - **buyer**（**自社＝買い手／対象＝供給元**。自社が対象から仕入れる。例: 自社=Apple・対象=キオクシア）: 対象を三人称。手がかり: 「調達」「サプライヤー選定」「供給元」「仕入れ」
  - **supplier**（**自社＝供給元／対象＝顧客**。自社が対象に納入する。例: 自社=東京エレクトロン・対象=キオクシア）: 対象を三人称。手がかり: 「販売先」「納入先」「顧客の技術動向」
  - ※**コードは常に『自社の役割』**（buyer=自社が買い手／supplier=自社が供給元。対象は相手方）。「buyer=対象が買い手」ではない。判別は取引の向き（自社が対象から買う→buyer／自社が対象に売る→supplier）
  - **neutral**（中立・投資家・アナリスト）: 対象企業を企業名で三人称／一人称「当社」不可。手がかり: 「投資判断」「評価」「調査」や立場の記述なし（**既定**）
- [ ] **手がかりが弱ければ `neutral` を仮置きし、C-3 の Artifact Review で必ずユーザーに確認**（勝手に self にしない）
- [ ] 確定した立場は提言・主張・エグゼクティブサマリー・別冊の**全セクションで一貫**させる。**呼称だけでなく分析の力点・提言のロジックも立場に合わせる**（`self`=自社の打ち手／`competitor`=競合の対抗・参入／`buyer`=調達戦略・依存リスク／`supplier`=供給戦略・内製化リスク／`neutral`=第三者の評価・予測。同じ事実でも読み方と打ち手が変わる＝呼称を三人称にしただけの「べき論」にしない）。`self` 以外では対象企業を三人称で呼び「当社/弊社/我が社」を使わない（詳細は `terminology.md` の「分析の立場」節 §6-2-B）
- [ ] **立場が `competitor` / `buyer` / `supplier`（関係性立場・自社 ≠ 対象）なら「自社（分析を行う側）」を特定（必須）**: 対象企業（`subject_company`）だけでなく分析を行う**自社名**を C-3 で尋ね `implementation_plan.md` § Narrative Stance と `narrative_stance.own_company` に記録。自社は Phase B で **Web 調査**し（事業・製品・技術/特許ポジション・市場での立ち位置）、対象企業と対比する（提言を一般論でなく「自社は X が強く／Y が手薄 → Z で差別化・参入。buyer=調達戦略／supplier=供給戦略」に具体化）。`buyer`/`supplier` の依存・交渉力は、母集団がドメイン（タイプ A/A'/B）なら出願人 HHI・上位集中で、**単一企業（タイプ C。例: キオクシア）では HHI 無意味なので Web 調査（市場シェア・取引構造）で**読む（**特許 ≠ 市場**）。`self`=`subject_company` が自社／`neutral`=自社なし（空）（詳細: `data_notes.md` §3、`terminology.md` §6-2-B）

**C-3. Artifact Review**
- [ ] `<!-- ANTIGRAVITY_REVIEW_REQUIRED -->` を配置、**Artifact Review でデータ実態 + タイプ推定 + 分析の立場推定（self/competitor/buyer/supplier/neutral とその根拠）を確認**（✅ この実態・タイプ・立場で進める / ✏️ タイプ変更 / 👤 立場が違う（自社／競合／取引先・買い手／サプライヤー／中立を指定）/ 💬 偏りあり・範囲と限界に明記 / 🔙 再抽出）。※関係性立場（competitor/buyer/supplier）なら分析を行う『自社』の社名も確認する旨を明示
- [ ] **⚠️ 立場の選択肢を提示する時は（`task.md`/`implementation_plan.md` のレビュー選択肢でも）必須**: **competitor / buyer / supplier を『その他』に畳まない**。「①中立(推奨) ②自社視点 ③競合視点 ④取引先・買い手／サプライヤー視点」を明示し、③④選択時に＋**自社名**を続けて確認する。**④は buyer/supplier の語で迷わせず取引の向きで訊く**:「自社が対象から**仕入れる**→buyer(対象=供給元)／自社が対象に**売る**→supplier(対象=顧客)」。**単一企業母集団（タイプ C）でも 5 立場すべて有効**（self/neutral だけと決めつけない）
- [ ] **立場が `competitor` / `buyer` / `supplier`（関係性立場）に確定したら続けて自社名を尋ねる（必須）**: 未取得時は追加の Artifact Review で「本分析を行う『自社』（対象企業の {競合／取引先・買い手／サプライヤー} として提言を導く主体）の社名は？」を確認し `narrative_stance.own_company` に保存。**Phase B の § Web Research Themes に「自社（{own_company}）の事業・技術・特許ポジション」を必ず含める**。ユーザーが「伏せる／一般的な視点でよい」なら `own_company` は空文字にし 1 行報告して従来どおり進める

**C-4. `reports/_phase_a_decisions.json` への保存**
- [ ] 確定内容を以下のフィールドで保存: `population_type` / **`narrative_stance`**（`code`=self/competitor/buyer/supplier/neutral / `label` / `subject_company` / `own_company`（competitor/buyer/supplier で分析を行う自社名。self は subject_company と同一、neutral・伏せる場合は空）/ `first_person_allowed`（self のみ true）/ `reasoning` / `confirmed_by_user`）/ `query_intent_summary` / `sub_questions` / `query_logic_structure` / `intent_logic_divergences` / `data_level2_warnings` / `forbidden_expressions` / `nebula_strategy` / `user_notes`（詳細: `population_type_metrics.md` §4-3、`narrative_stance` は `terminology.md` §6-2-B）

🛑 **ARTIFACT GATE D (NEBULA 戦略判定) — 必須（全ケースで実施）**:
- [ ] `data/nebula_*.json` の存在確認
- [ ] 存在すれば `nebula_strategy.selected_mode = "execute"` を自動決定
- [ ] 存在しない場合、`implementation_plan.md` § NEBULA Strategy に 2 択を記入:
  - **🌐 Web 補完モード**: Phase B で 4 カテゴリ必須カバー（市場規模 / 政策・規制 / 学術動向 / 主要企業動向）→ 「外部環境分析（Web 調査）」章を設置、各主張に `#footnote[...]` で出所明記
  - **📘 省略モード**: NEBULA 章なし + 「本分析の範囲と限界」章で「特許情報のみ対象」と注記、学術-特許クロス分析も省略
- [ ] `<!-- ANTIGRAVITY_REVIEW_REQUIRED -->` を配置、**Artifact Review でモード選択**を待機
- [ ] 確定結果を `_phase_a_decisions.json` の `nebula_strategy` に保存

→ **完了条件**: `implementation_plan.md` の Phase A 関連セクションすべて完成（Mission / Dataset / Evidence / Insights / Population Meta / query_logic Reading / query_intent 3-Point / Sub-Questions / Divergences / Level 2 / Population Type / Narrative Stance / NEBULA Strategy / Executive Summary Decision） / `task.md` Phase A 全チェック / `reports/_phase_a_decisions.json` 永続化

### Phase A-2: レポートタイトルの決定 🛑 ARTIFACT GATE

🛑 **STOP-GATE**: 以下を全て実行するまで Phase B へ進むな

1. Mission Objective とデータ特性を踏まえ、タイトル+サブタイトルの **3案** を生成
   - **タイトル**: **オーソドックス**（標準的・保守的）な体言止め。**20 文字以内**の目安
     - ✅ OK: 「CNF 特許動向分析 2026」「全固体電池の競合ポジション分析」「次世代半導体製造技術ランドスケープ」
     - ❌ NG: 「独断 — 電池の未来」等の扇情的・文学的タイトル／「電池はどこへ向かうのか？」等の問いかけ型
     - 指針: 「{技術分野 / 対象企業} の {分析種別}」の単純な組み合わせが基本。クリエイティブなコピーは不要
   - **サブタイトル**: 30 文字以内。具体的な件数・期間・分析軸を含める
2. 3案を `implementation_plan.md` § Title Candidates に記入（チェックボックス付き）
3. Antigravity Artifact Review で **ユーザーが1案に ✅ を付ける or コメントで指示するまで待機**
4. ユーザー確定後、`implementation_plan.md` § Confirmed Title に転記 + `voyager/mission.json` に `confirmed_title` フィールドで保存
5. `task.md` Phase A-2 チェックボックスを更新

> `<!-- ANTIGRAVITY_REVIEW_REQUIRED -->` コメントを § Title Candidates の直前に埋め込むと、Antigravity が自発的に Review 要求を出しやすい

**AI 側で勝手にタイトルを決定するのは禁止**(提示だけで満足してはならない)。

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

### Phase B: エビデンス精読 + クロスモジュール分析 🛑 ARTIFACT GATE x2

🛑 **STOP-GATE 1 (リファレンス読了 + クロスパターン確認)**
- [ ] `capcom_schema/analysis/common_framework.md` を読了 → 4層分析モデル把握
- [ ] `capcom_schema/analysis/data_notes.md` を読了 → 特許/NPL非対称性・Web調査ルール把握
- [ ] `capcom_schema/analysis/cross_module.md` を読了 → 13種クロスパターン把握
- [ ] `implementation_plan.md` § Cross-Module Pattern Selection に13パターン + Agent推奨5つ（★）を記載
- [ ] **Artifact Review: ユーザーが5パターン以上選定**（✅ or Other でカスタム指定）
- [ ] ユーザー選定後、§ Confirmed Cross Patterns に転記

🛑 **STOP-GATE 2 (Web調査の意思確認)**

- [ ] **`narrative_stance` を確認 — 立場が `competitor` / `buyer` / `supplier`（関係性立場）かつ `own_company` が非空なら（下記モード問わず必須）**: § Web Research Themes に「**自社（{own_company}）の事業・主要製品・技術/特許ポジション・市場での立ち位置**」を必ず1件含める。対象企業（`subject_company`）との対比材料であり、提言を「自社は X が強く／Y が手薄 → …」と具体化する土台。buyer/supplier の依存・交渉力は母集団がドメイン（A/A'/B）なら HHI で、**単一企業（タイプ C）では HHI 無意味なので Web 調査（市場シェア・取引構造）で**読む（特許≠市場）。出所は脚注（サイト名・URL・取得日）を付し、最低1章で自社 vs 対象企業の対比に使う（`data_notes.md` §3、`terminology.md` §6-2-B）
- [ ] **`reports/_phase_a_decisions.json` の `nebula_strategy.selected_mode` を確認**し、モード別に対応:

**モード `execute`（NEBULA 実行済み）**:
- [ ] Mission Objective から導出した Web 調査テーマ 3-5 件を `implementation_plan.md` § Web Research Themes に記載
- [ ] `task.md` § Phase B Gates の Web Research チェックボックス（「実施/スキップ/修正」）を提示
- [ ] **Artifact Review: ユーザーが1つ選択**

**モード `web_compensation`（NEBULA 未実行・Web 補完）**:
- [ ] Web 調査は **スキップ不可**（Phase A ARTIFACT GATE D でユーザーが補完を選択済み）
- [ ] **4 カテゴリすべて**をカバーするテーマを § Web Research Themes に記載:
  1. **市場規模**: 業界全体の市場規模・成長予測
  2. **政策・規制**: 政策・規制動向・標準化活動
  3. **学術動向**: 学術論文引用動向・キーパーソン
  4. **主要企業動向**: 主要出願人の事業戦略・M&A・プレスリリース
- [ ] **Artifact Review: 4 カテゴリをカバーするテーマでユーザー承認**を待機
- [ ] 4 カテゴリが 1 つでも欠ける場合は警告して再確認（Phase D gate Check 13 で FAIL 対象）

**モード `omit`（NEBULA 未実行・省略）**:
- [ ] 通常通り任意 Web 調査として進行（3-5 件提示、3 択）
- [ ] 「外部環境分析」章は作らないが、任意 Web 調査は可

- [ ] ユーザー選択後、§ Confirmed Web Research に転記。「省略します」等の AI 自己判断は禁止

詳細: `analysis/population_type_metrics.md` §4-3（nebula_strategy フィールド仕様）

**Phase A の情報を参照せずに Phase B を進めてはならない。**

1. 上記3ファイルを読む（必読）
2. Evidence全件から優先順位を付ける（Mission Objective への直結度で 1-3 のランク付け）→ `implementation_plan.md` § Evidence Inventory の優先度列を更新
3. 優先度の高い5-8件を1件ずつ順次読む
4. 各Evidenceを読む際に: AIインサイトとの照合 / `capcom_schema/analysis/map_reading.md` の該当セクション読解 / 代表特許の抽出 / スナップショット画像パス記録
5. **代表特許の具体的確認**: `data/patents.csv` を pandas で条件検索し、代表特許のタイトル・出願人・公開番号を **最低15件** 取得
6. `capcom_schema/analysis/cross_module.md` の基本原則を読み、選定した5パターン以上を実行
7. クロス分析の洞察を `implementation_plan.md` § Phase B Output Summary に記録

→ **完了条件**: Evidence 5件以上精読済み / AIインサイト照合メモ作成済み / 代表特許15件以上取得済み / クロス分析5パターン以上の仮説→検証→結論を完了済み / `task.md` Phase B 全チェック

---

### Phase C: モジュール別deep dive ⚠ スキップ禁止 🛑 ARTIFACT + SCRIPTED GATE

🛑 **STOP-GATE (リファレンス読了 + 計画確認)**
- [ ] `capcom_schema/analysis/deep_dive_guide.md` を読了 → 各 Step の必須セクション数・最低行数把握
- [ ] （予約）各 deep_dive の「統合的戦略インサイト」節の執筆**直前**に `capcom_schema/analysis/structured_techniques.md` §1 を読む（ACH＝対立仮説の検討。deep_dive 側は推奨・結論章では必須）
- [ ] `implementation_plan.md` § Deep Dive Plan にテーブル形式で記載（Step / モジュール / 最低行数 / 必須セクション数）
- [ ] **Artifact Review: ユーザーが Deep Dive Plan を承認**（コメント or ✅）

exemplars を参照し、全モジュールのdeep_dive.typを生成する。Phase DはPhase Cの出力ファイルを前提とする。

1. **`capcom_schema/analysis/deep_dive_guide.md` を読む** → 各Stepの必須セクション数と最低行数把握
2. **代表特許の決定的選定（Phase C の最初に1回だけ・v9 必須）**: 代表特許は「決まった手順」で選ぶ（**自由選択の禁止**。モデルが自由に選ぶ限り、結論に都合の良い特許を選べてしまう＝つまみ食い）。選定スクリプトを実行する:

   ```bash
   python3 capcom_schema/scripts/select_representatives.py
   ```

   `reports/representative_patents.json` が生成される（Saturn V=SBERT重心近傍〔patiroha算出済み〕、MEGA=象限内で出願年昇順→番号昇順、Explorer/CREW=中心性上位ノード対応。全て決定的でタイブレークまで固定）。**ミクロ分析A で引用する特許番号は、この JSON に載った番号だけ**とする（`phase_d_gate.sh` Check 35 が、ミクロ分析節にリスト外の番号があれば警告する）。JSON の `cite_as` / `title` / `applicant` を使い、技術的意義と戦略的文脈を1-2文添えて引用する。プレースホルダ番号（`特開2023-XXXXXX` 等）・捏造番号は自動不合格（詳細: `analysis/deep_dive_guide.md` ミクロ分析A）
3. 各モジュールの exemplar を読む → deep_dive.typを生成（exemplar は `capcom_schema/exemplars/`）
4. 全deep_diveにミクロ分析A（代表特許15件以上・**引用は `reports/representative_patents.json` の番号のみ**）+ B（出願人5社以上、各5行以上）を含める
5. **走査層（v9）**: 各番号セクション冒頭に `#point-lead[...]` を1個、各章末に `#chapter-summary[...]` を置く（散文の代替にしない。gate Check 25/26）
6. Step 0: NEBULA → Step 1: Saturn V → Step 2: Explorer → Step 3: MEGA → Step 4: ATLAS → Step 5: CORE → Step 6: CREW の順で処理
7. **Phase C 完了ゲート (必須実行)**: 以下のスクリプトを実行し、exit code が 0 でない場合は Phase D 開始禁止

   ```bash
   bash capcom_schema/scripts/phase_c_gate.sh
   ```

8. **スクリプトの stdout/stderr を `walkthrough.md` § Phase C Gate Result に全文転記**（加工・要約禁止）
9. `task.md` Phase C チェックボックスを更新

**「実質的にOK」等の AI の質的判断による上書きは禁止**(`## 0. 絶対遵守ゲートルール` 第3項)。

→ **完了条件**: `reports/representative_patents.json` 生成済み（決定的選定）/ deep_dive 4ファイル以上（Saturn V + Explorer + MEGA + ATLAS）、各最低行数を満たす / `phase_c_gate.sh` exit 0 / `walkthrough.md` 転記済み

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

### Phase D: 統合レポート + 品質検証 🛑 ARTIFACT + SCRIPTED GATE

🛑 **STOP-GATE (リファレンス読了 + 構成確認)**
- [ ] `capcom_schema/analysis/report_structure.md` を読了 → 章構成・deep_dive コピールール把握
- [ ] `capcom_schema/analysis/quality_checklist.md` を読了 → 定量チェックコマンド・全チェック項目把握
- [ ] `capcom_schema/analysis/structured_techniques.md` を読了 → ACH・リンチピン・三つの環（結論・提言章で必須。`phase_d_gate.sh` Check 30/31/33/34 が検査）
- [ ] `implementation_plan.md` § Report Structure & Quality Plan を完成
- [ ] **Artifact Review: ユーザーが Report Plan を承認**

全 deep_dive を統合し、report.typ を生成する。

**前提条件**: `reports/` に最低4つの `*_deep_dive.typ` が存在すること（4つ未満なら Phase C に戻る）。

1. `ls reports/*_deep_dive.typ` でファイル存在を確認
2. `capcom_schema/analysis/patent_citation.md` セクション2-3を読む（引用書式の確認）
3. Phase C で生成した全 deep_dive ファイルを読む
4. `report.typ` を生成する（→ `capcom_schema/analysis/report_structure.md` セクション1の構造）
5. **deep_dive の全文コピー**: 要約・圧縮・省略は一切禁止（→ `capcom_schema/analysis/report_structure.md` セクション2）
   - Phase D で新規に書く章（クロスモジュール統合分析・結論等）にも**走査層**（各番号セクション冒頭の `#point-lead` + 各章末の `#chapter-summary`）を適用する（gate Check 25/26）
6. **品質検証ゲート (必須実行)**:

   ```bash
   bash capcom_schema/scripts/phase_d_gate.sh
   ```

7. **スクリプト出力を `walkthrough.md` § Phase D Gate Result に全文転記**
8. FAIL 時は `task.md` の Phase D チェックを **入れずに** Phase C または Phase D 該当項目に戻る
9. 成功時は PDF 出力: `typst compile --root ".." reports/report.typ reports/report.pdf`
10. `walkthrough.md` § Final Deliverables に出力ファイル一覧記録

**「自前のチェックで代替」は禁止**(再現性のないチェックは無効)。

→ **完了条件**: `phase_d_gate.sh` exit 0 / PDF 出力成功 / `walkthrough.md` 完成

---

## 5. モジュール一覧

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

## 6. patents.csv 仕様

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

## 7. 分析の基本原則

1. **数値根拠**: 全ての主張に具体的な数値を含める
2. **特許引用**: 代表特許を具体的に引用する（番号、タイトル、出願人）
3. **クロス検証**: 最低5パターン実施（→ `capcom_schema/analysis/cross_module.md`）
4. **事実と推論の分離**: 4層分析モデルを適用（→ `capcom_schema/analysis/common_framework.md`）
5. **可視化参照（全章必須）**: 全ての章に最低1つの `#snapshot-figure()` を含める
6. **AIインサイト活用**: `prompts/` のAIインサイトを必ず参照
7. **データソーストレーサビリティ**: 全ての数値にモジュール名マーカーを付与
8. **Evidence網羅性**: Evidence総数の半数以上を分析に活用
9. **Web調査（推奨）**: 出所（URL・サイト名・取得日）を必ず明記

## 8. レポート出力

### Typst PDF
1. `capcom_schema/templates/report_style.typ` を `reports/` にコピー
2. `report.typ` を生成（`#show: apollo-report.with(...)` で開始）
3. スナップショット画像は `#snapshot-figure("../snapshots/xxx.png", caption: "説明")` で挿入
4. `typst compile --root ".." reports/report.typ reports/report.pdf`

### 利用可能な関数
- `exec-summary[...]` — エグゼクティブサマリー
- `kpi-dashboard(cols: 3, kpi-card(...), ...)` — KPIダッシュボード
- `kpi-card("ラベル", "値", note: "補足")` — KPIカード（**ドル記号禁止**）
- `evidence-box(番号, "タイトル")[...]` — Evidenceボックス
- `insight-box[...]` — Key Insightボックス
- `point-lead[...]` — **要点ストリップ**（番号セクション見出し直後に置く結論先出し1〜2行＝走査層。散文の上に重ねる。散文の代替ではない）
- `chapter-summary[...]` — **本章のまとめ**（各章末に置く走査層。まとめの後に本文を続けない。gate Check 25/26）
- `hl[...]` — インライン強調（数値以外のキーワードを選択的に。1セクション数語まで。多用禁止）
- `snapshot-figure("パス", caption: "説明")` — スナップショット画像
- `styled-table(columns: ..., header: (...), ..body)` — BCG風テーブル
- `conclusion-box("タイトル")[本文]` — 主要結論
- `recommendation-card("高", "タイトル", "説明", timeframe: "短期")` — 優先度付き推奨

> 📐 **読みやすさ（走査層）— 詳細は `analysis/deep_dive_guide.md`「読みやすさ（走査層）」**: ①**各番号セクション冒頭に `#point-lead[...]` を1個**置き結論を1〜2行で先出し（散文は下にそのまま＝薄くしない）②**各章末に `#chapter-summary[...]`**（本章のまとめ。散文の代替にしない。gate Check 25/26）③**数値＋単位はテンプレートが自動で太字強調**＝**手動で数字を太字化しない**④余白・見出しバーも自動。要点だけ書いて散文を削るのは Check 1（文字数）で不合格。

**注意**: `report_style.typ` のフォント設定を変更しないこと。画像パスは `reports/` からの相対パス。`--root ".."` 必須。

### python-pptx PPT
> ⚠️ **PPTX は `capcom_schema/templates/slides_spec.md` が唯一の正**（本節は要約）。Section 0〜6 を設計ガイド（いつ・どのヘルパーを・どんな主張骨格で使うか）として熟読し、矛盾したら slides_spec を採用。
- **🔑 ヘルパー関数は `apollo_slides.py` を import して使う（コピーしない）**: 生成スクリプトの冒頭で `import sys; sys.path.insert(0, "capcom_schema/templates"); from apollo_slides import *` し、`add_title_shape` / `add_sub_message` / `add_kpi_slide` / `add_matrix_2x2_slide` / `add_arrow_flow_slide` / `add_donut_slide` / `add_issue_tree_slide` 等を呼ぶ（`slides_spec.md` から写経しない）。**自前で pptx のフォント・色・レイアウトを書き起こさない**。フォント（**Noto Sans JP**）・多段ウェイト（見出し=Black / サブメッセージ=Medium / 本文=Regular / 出典=Light）・上下中央寄せ・箱の充填はヘルパー内蔵（自前実装は単一ウェイト・平板の原因）
- **🔑 デッキは完成レポート `reports/report.typ` を土台に**（evidence の寄せ集め禁止）。各章の主張→根拠→示唆を凝縮し章順に沿わせる（§0.9）
- **出所**: モジュール名や `report.typ` 等のファイルパスを出所にしない（特許データ由来＝データセット名／事業ファクト＝Web 実出所）。タイトルに「～」副題を使わない・過剰修辞禁止
- チャート+注釈 50%以上 / 同タイプ 3 枚連続禁止 / 推奨 25〜35 枚 / 出力 `reports/presentation.pptx`
- 完了後 `bash capcom_schema/scripts/phase_d_gate.sh`（Check 16 PPTX 機械チェック）を実行。別冊は `executive_summary_guide.md` に従い **8〜12 ページ**（薄い別冊は不合格）

---

## 9. Antigravity IDE 固有の運用

### 9.1 Artifact-first パラダイム

Antigravity は Claude Code/Codex と異なり **Artifact-first** です：
- ユーザー確認ゲートは `ask_user_question` 相当ツールではなく、**Artifact ファイルへの編集・コメント・承認** で実現
- 対応する Artifact: `task.md`, `implementation_plan.md`, `walkthrough.md`

各 Phase の STOP-GATE は以下のマッピング：

| Phase Gate | Artifact 操作 |
|---|---|
| Phase A-2: タイトル3案 | `implementation_plan.md` § Title Candidates にチェックボックス付き3案 → ユーザーが ✅ |
| Phase B-1: クロスパターン5つ以上 | `implementation_plan.md` § Cross-Module Pattern Selection の13パターンから5つ以上選定（Phase B 完了条件・gate Check 4 の最低数と同じ） |
| Phase B-2: Web調査可否 | `task.md` § Phase B Gates の Web Research 3択チェックボックス |
| Phase C: Deep Dive Plan | `implementation_plan.md` § Deep Dive Plan にテーブル → ユーザー承認 |
| Phase C: 完了ゲート | `bash phase_c_gate.sh` + `walkthrough.md` § Phase C Gate Result に全文転記 |
| Phase D: Report Plan | `implementation_plan.md` § Report Structure & Quality Plan → 承認 |
| Phase D: 品質ゲート | `bash phase_d_gate.sh` + `walkthrough.md` § Phase D Gate Result に全文転記 |

### 9.2 Review Policy 推奨設定

Antigravity の設定パネルで `apollo-capcom` skill に対して **"Request Review"** を設定することを推奨します。これにより：
- Artifact への重要な変更（Title Candidates の確定、Cross Patterns の選定等）でユーザー承認待ちが自動発動
- ユーザーは Google Docs 式コメントで修正指示を残せる
- Agent は承認されるまで次 Phase に進まない

設定手順は `review_policy_recommendation.md` を参照してください。

### 9.3 `.agent/workflows/` からの起動

本スキルは全 Phase を一気通貫で実行しますが、特定 Phase だけを再実行したい場合は `.agent/workflows/` 配下の個別ワークフローから起動できます：

- `.agent/workflows/00_capcom_master.md` — Phase A → D を順次実行（本スキルの通常起動）
- `.agent/workflows/01_phase_a_data_intake.md` — Phase A のみ
- `.agent/workflows/02_phase_a2_title_selection.md` — Phase A-2 のみ
- `.agent/workflows/03_phase_b_evidence_cross.md` — Phase B のみ
- `.agent/workflows/04_phase_c_deep_dive.md` — Phase C のみ
- `.agent/workflows/05_phase_d_integration.md` — Phase D のみ
- `.agent/workflows/06_interactive.md` — 対話型レポート作成モード（KATHERINE）の起動点（`report_mode=interactive` 時）

各 Phase 個別起動時は、前 Phase の Artifact（特に `implementation_plan.md`）が完成していることを前提とします。

### 9.4 サブエージェント禁止

Antigravity には Manager + Browser subagent 等の機構がありますが、**本スキル実行中はサブエージェントを起動しません**。トークン効率化のため、全処理をメインコンテキスト内で完結させます。

例外: Web調査時に Browser subagent を使いたい場合は、Phase B STOP-GATE 2 でユーザーが「実施する」を選択した後、該当フェーズ内で限定的に使用可（Gemini Pro Manager と Gemma 4 subagent の組み合わせが推奨）。ただし分析本体（patents.csv 読解、deep_dive 執筆等）は必ずメインコンテキストで実行。

### 9.5 `brain/` メモリ活用

Antigravity の `.gemini/antigravity/brain/` メモリは本スキルでも活用可：
- Phase A 完了時点の要点（データ統計・主要数値）を保存 → Phase B 以降で参照
- Phase B のクロス分析結論を保存 → Phase C/D で活用
- ユーザー固有の好み（タイトルの文体、出典書式等）を保存 → 将来セッションで再利用

ただし、**brain の内容は capcom_schema/ の共有資産を上書きしません**。共有資産は常に Single Source of Truth として尊重します。

### 9.6 ユーザー指示の解釈ルール

| ユーザーが言ったこと | 正しい解釈 | 誤った解釈(禁止) |
|---|---|---|
| 「レポートを書いて」 | SKILL.md の全フェーズに従う | 急いでいる→省略OK |
| 「早く」「すぐに」 | Artifact を素早く生成(Gateは守る) | ゲート省略OK |
| 「簡単でいい」 | 各セクションの記述量を短く | ゲート省略OK |
| 「適当に」 | デフォルト設定で進める | Artifact Review スキップOK |
| 「次へ」「進めて」 | 当該ステップが完了済みなら次へ | 未完了でも次へ進む |

**省略を許可するのは、ユーザーが明示的に「Phase B は飛ばして」「Web 調査いらない」等と言った時のみ。** AI 側の推測で省略してはならない(`## 0. 絶対遵守ゲートルール` 第5項)。
