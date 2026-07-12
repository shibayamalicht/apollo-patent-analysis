# APOLLO CAPCOM プロジェクトルール（Antigravity IDE 最優先）

このフォルダは APOLLO v9 の CAPCOM セッション（`session_YYYYMMDD_HHMMSS/`）です。Antigravity IDE の **Artifact-first パラダイム** に沿って戦略レポートを生成します。

本 GEMINI.md は Antigravity での **最優先ルール** です。`AGENTS.md` は GEMINI.md 非対応の派生エディタ用 fallback であり、本ファイルと内容が重複する部分は本ファイルが優先します。

---

## 🎯 このフォルダで必ず行うこと

### スキル起動と Artifact 生成

1. **レポート生成依頼を受けたら必ず `apollo-capcom` スキルを起動**
   - チャットで「apollo-capcom スキルで…」と指示
   - または `.agent/workflows/00_capcom_master.md` を直接実行
2. **起動直後に以下3 Artifact を必ず生成**
   - `task.md` ← `artifacts_templates/task.md.tmpl` をコピー
   - `implementation_plan.md` ← `artifacts_templates/implementation_plan.md.tmpl` をコピー
   - `walkthrough.md` ← `artifacts_templates/walkthrough.md.tmpl` をコピー
3. **3 Artifact を Antigravity に登録**（Review Policy = "Request Review" で動作）
4. **フェーズ間引き継ぎ日誌 `reports/_carryover.md` を生成**（`capcom_schema/templates/carryover_template.md` をコピー）

> **Artifact と `_carryover.md` の役割分担**: Artifact（`task.md`/`implementation_plan.md`/`walkthrough.md`）は**人間レビュー・作業ビュー**、`reports/_carryover.md` は**3ツール共通の跨タスク正本（再開ポインタ＋Web出所台帳）**。タスク分割・`/compact` でネイティブ記憶が飛んでも `_carryover.md` から復元する。Artifact 確定（クロス確定・Web確定・ゲート結果）の直後に `_carryover.md` の該当節へ要約をミラー（詳細は二重転記せずポインタのみ）。矛盾時は `_carryover.md` を正とする。`_carryover.md` はレポート本文へ転記しない（内部メモ）。

> **🔄 セッション・チェックポイント（各フェーズ境界で必須・枯渇/クォータの予防）**: 各フェーズの区切り（**Phase A完了・Phase B完了・Phase Cの各モジュール完了ごと・Phase D着手前**）で必ず、ゲート/Artifact 通過＋`_carryover.md` 更新の後に「**新タスク（セッション）に切り替えますか？**」と提案して一旦停止する（続行/切替/`/compact` をユーザーが選ぶ）。「枯渇しそうな時だけ」ではなく**予防として各境界で必ず出す**。切替時は新タスクが `_carryover.md`＋Artifact から自動再開。詳細はスキル本体 `### 🔄 セッション・チェックポイント`。

### Review Policy 必須

Antigravity の **Review Policy を必ず "Request Review" または "Agent Decides" に設定** してください。"Always Proceed" では本スキルのゲートが機能しません。**対話型レポート作成モード（KATHERINE）では "Request Review" が必須**です（下記「レポート生成の進行様式」参照）。

詳細は [`review_policy_recommendation.md`](review_policy_recommendation.md) を参照。

### レポート生成の進行様式（自律生成 / 対話型 KATHERINE）

`voyager/context.json` の `report_mode` を必ず確認する:

- **`"autonomous"`（既定・未指定含む）**: 従来どおり本パッチのスキル手順（自律生成モード）で進行する。
- **`"interactive"`（またはユーザーが対話型を明示した場合）**: **`capcom_schema/interactive/SKILL_INTERACTIVE.md` を進行の正本として読み**、`capcom_schema/interactive/dialogue_points.md`（対話ポイント CP-1〜8）を併読して対話型レポート作成モード（KATHERINE）で進行する。品質ゲート・成果物形式・トークン効率制約は自律生成モードと完全に同一。起動点は `.agent/workflows/06_interactive.md`。

**対話型の Antigravity Artifact 翻案（確定手段の読替）** — SKILL_INTERACTIVE.md の `AskUserQuestion` は Antigravity では次のとおり Artifact 操作に読み替える:

| 対話型の選択肢 | Antigravity での実現方法 |
|---|---|
| ✅ この案で確定 | 該当ブロックの**チェックボックスを `[x]`** にする |
| ✏️ 修正して確定 / 🔄 別案・差し替え | **Google Docs 式コメント**で指示 → Agent が修正 → **再度 Artifact Review** に通す |
| 🤖 おまかせ | **第4のチェックボックス**として各確定ブロックに常設する（選択時は判断ログに `[対話→委任]` と記録） |

**マーカー規約（2種）**:
- `<!-- ANTIGRAVITY_REVIEW_REQUIRED -->` — **確定必須ブロック**。ユーザーの応答（✅ / コメント）を得るまで次へ進まない（従来どおり）
- `<!-- ANTIGRAVITY_INFO_ONLY -->` — **応答不要の情報提示ブロック**（新設）。CP-7 の委任時・ライト深度時の「突合サマリ」等、過程の透明化のために提示するが承認は求めない。Agent はこのマーカーのブロックで承認待ちにしない

CP-4（マップ読解）/ CP-6（仮説と検証）/ CP-7（統合インサイト）の対話は、ローリング Artifact **`dialogue_review.md`**（雛形: `artifacts_templates/dialogue_review.md.tmpl`）で行う。CP-1/2/3/5/8 は `implementation_plan.md` の該当セクション（4部構成欄・結論候補節・WARN トリアージ表）を使う。

**対話型では Review Policy = "Request Review" が必須**。他の設定（"Agent Decides" / "Always Proceed"）を検出したら **Phase 0 で停止**し、ユーザーに設定変更を依頼する（対話ポイントの確定が Artifact Review 承認で成立する前提のため。`review_policy_recommendation.md` 参照）。

### ゲート遵守

以下のゲートは **絶対に省略不可**（SKILL.md §0 絶対遵守ゲートルール。**ゲートの構成・個数の正本はスキル本体 `.agent/skills/apollo-capcom/SKILL.md` の各 Phase 節**。Phase A にも別冊確認〜NEBULA 戦略まで最大 7 の Artifact Review ゲートがあり、対話型ではさらに対話ポイント CP-1〜8 の確定が加わる）：

| Phase | Gate 種別 | 承認媒体（代表例） |
|---|---|---|
| A | Artifact Review（最大 7） | `implementation_plan.md` § Executive Summary 〜 § NEBULA Strategy |
| A-2 | Artifact Review | `implementation_plan.md` § Title Candidates |
| B-1 | Artifact Review | `implementation_plan.md` § Cross-Module Pattern Selection |
| B-2 | Artifact Review | `task.md` § Phase B Gates (Web Research) |
| C | Artifact Review + Scripted | `implementation_plan.md` § Deep Dive Plan → `bash phase_c_gate.sh` |
| D | Artifact Review + Scripted | `implementation_plan.md` § Report Plan → `bash phase_d_gate.sh` |
| （対話型のみ） | Artifact Review | CP-1〜8（`implementation_plan.md` 4部構成欄 / `dialogue_review.md`） |

### cwd 規約

**常に `session_*/` ルートを cwd として作業する。** `capcom_schema/...`, `data/...`, `reports/...` 等の相対パスがこの前提で書かれています。

### bash Gate 必須

Phase C / D の完了時には必ず以下のスクリプトを実行し、結果を `walkthrough.md` に **全文転記**（加工禁止）：

```bash
# Phase C 完了時
bash capcom_schema/scripts/phase_c_gate.sh
# → stdout/stderr を walkthrough.md § Phase C Gate Result にコピペ

# Phase D 完了時
bash capcom_schema/scripts/phase_d_gate.sh
# → stdout/stderr を walkthrough.md § Phase D Gate Result にコピペ
```

**AI の主観判断で「実質的にOK」と進むのは禁止**（SKILL.md §0 第3項）。

---

## 🚫 禁止事項

- **スキル未起動でレポート着手**: `apollo-capcom` を起動せずに `data/` を解析してレポート本体を書き始めるのは禁止
- **Artifact Review 省略**: Artifact Review ゲート（構成の正本はスキル本体の各 Phase 節。対話型では CP-1〜8 の確定も含む）を Agent 側で自発的に「通過」扱いにするのは禁止
- **サブエージェント起動**: Manager + Browser subagent は本スキル実行中は **起動しない**（Web調査時の Browser subagent のみ Phase B 限定で許可）
- **patents.csv 全量表示**: `print(df)` / `cat data/patents.csv` 等
- **deep_dive の圧縮**: Phase D で report.typ に deep_dive をコピーする際、要約・省略は禁止（全文コピー）
- **水増し（コピペ反復）**: 同一文・同一構文の反復、回転する名詞だけ変えた定型文の量産、「○○観点 1, 2, 3…」式の連番見出しで行数・件数を稼ぐこと。**`phase_d_gate.sh` Check 19 で自動不合格**。行数が足りない時は文を繰り返さず、新しい代表特許（固有の公開番号）・数値根拠・別のクロスパターン・Web裏付けを足す。各段落は前段落と異なる固有の事実を最低1つ含めること
- **本文のスクリプト生成**: `deep_dive.typ` / `report.typ` を Python 等のスクリプトでテンプレート生成すること（`reports/generate_*.py` は **Check 19a で自動不合格**）。各文は固有の分析として直接書く。「最低行数を満たすための補助文・つなぎ文」も禁止
- **ゲート回避（specification gaming）**: `phase_d_gate.sh` を読んで反復検出を**すり抜ける目的**で接続詞・語順・文体だけ変えて内容の重複を温存すること。対処は重複の削除と固有内容への置換（末尾22字の重複も Check 19 で検出）
- **bash Gate スキップ**: `phase_c_gate.sh` / `phase_d_gate.sh` の実行を省略するのは絶対禁止

---

## 📁 フォルダ構成

```
session_YYYYMMDD_HHMMSS/               ← cwd
├── capcom_schema/                     # 共有資産（読み取り専用、変更禁止）
│   ├── SKILL.md                       # Claude Code用（Antigravity では参照のみ）
│   ├── analysis/                      # 分析手法ガイド（9ファイル）
│   ├── references/                    # モジュール別スキーマ（10ファイル）
│   ├── exemplars/                     # deep_dive 執筆見本（7 Typst）
│   ├── templates/                     # Typst / PPT テンプレート
│   └── scripts/                       # bash 品質ゲート
├── data/                              # patents.csv + モジュールJSON
├── voyager/                           # Mission Objective + Evidence
├── snapshots/                         # 可視化PNG
├── prompts/                           # AIインサイト（Markdown）
├── reports/                           # ★レポート出力先
├── .agent/                            # Antigravity スキル配置
│   ├── skills/apollo-capcom/SKILL.md  ← 本スキルの本体
│   └── workflows/                     # Phase別起動点（00-05.md + 06_interactive.md〈対話型〉）
├── artifacts_templates/               # Artifact 雛形（起動時にコピー。対話型用 dialogue_review.md.tmpl 含む）
├── task.md                            ← ★ Artifact（4フェーズチェックリスト）
├── implementation_plan.md             ← ★ Artifact（承認対象セクション）
├── walkthrough.md                     ← ★ Artifact（ゲート結果記録）
├── dialogue_review.md                 ← ★ Artifact（対話型 KATHERINE のみ。CP-4/6/7 のローリング対話）
├── GEMINI.md                          ← 本ファイル
├── AGENTS.md                          # fallback
└── metadata.json
```

---

## 🔧 Antigravity 固有の機能活用

### Artifact の 4 パターン

1. **task.md**: 生きたチェックリスト。Agent が進行中に `[x]` を埋める
2. **implementation_plan.md**: 承認対象。ユーザーが ✅ / コメントで指示
3. **walkthrough.md**: 完了証跡。bash gate 結果を **改ざんせずに** 転記
4. **(補助) brain/**: `.gemini/antigravity/brain/` に要点を永続化（セッションをまたいだ知識保持）

対話型（KATHERINE）ではこれに **dialogue_review.md**（CP-4/6/7 のローリング対話 Artifact。雛形: `artifacts_templates/dialogue_review.md.tmpl`）が加わる。

### Google Docs 式コメント

ユーザーは Artifact の任意箇所を選択してコメントで指示できます：
- 「この案2のサブタイトルをもう少し定量的に」
- 「P13 ではなく P8 に変更」
- 「この章は○○の観点を追加してほしい」

Agent はコメントを検出して該当セクションを修正し、再度 Artifact Review に通す。

### 部分再実行

Phase C の特定モジュールだけ再生成したい場合、`.agent/workflows/04_phase_c_deep_dive.md` を呼び出して Step 指定で再実行可能。

---

## 🔖 ファイルの役割分担

- **GEMINI.md** （本ファイル）: Antigravity 向けのプロジェクト全体ルール（最優先）
- **AGENTS.md**: GEMINI.md 非対応の派生エディタ用 fallback（内容は GEMINI.md と重複）
- **`.agent/skills/apollo-capcom/SKILL.md`**: 具体的な4フェーズ手順書
- **`.agent/workflows/*.md`**: Phase別起動点（`00_capcom_master.md` がマスター、`06_interactive.md` が対話型 KATHERINE の起動点）
- **`artifacts_templates/*.tmpl`**: スキル起動時に Artifact として複製される雛形（対話型用 `dialogue_review.md.tmpl` 含む）
- **`capcom_schema/interactive/SKILL_INTERACTIVE.md` + `dialogue_points.md`**: 対話型モードの進行正本（共有資産・読み取り専用）
- **`review_policy_recommendation.md`**: Review Policy の推奨設定ガイド

本ファイル（GEMINI.md）と SKILL.md / workflows が衝突した場合、**SKILL.md の具体手順が優先**します（本ファイルはより一般的なルール）。

---

## 📚 参照

- スキル本体: `.agent/skills/apollo-capcom/SKILL.md`（全 Phase 手順）
- 分析手法: `capcom_schema/analysis/`（9ファイル）
- データスキーマ: `capcom_schema/references/`（10ファイル）
- Typst テンプレート: `capcom_schema/templates/`
- 品質ゲート: `capcom_schema/scripts/phase_c_gate.sh`, `phase_d_gate.sh`

本ファイルを削除するとスキル実行の前提条件（Review Policy, Artifact, cwd規約, サブエージェント禁止）が崩壊します。**削除禁止**。
