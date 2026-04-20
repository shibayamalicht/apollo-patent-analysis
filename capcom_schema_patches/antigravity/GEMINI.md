# APOLLO CAPCOM プロジェクトルール（Antigravity IDE 最優先）

このフォルダは APOLLO v7 の CAPCOM セッション（`session_YYYYMMDD_HHMMSS/`）です。Antigravity IDE の **Artifact-first パラダイム** に沿って戦略レポートを生成します。

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

### Review Policy 必須

Antigravity の **Review Policy を必ず "Request Review" または "Agent Decides" に設定** してください。"Always Proceed" では本スキルのゲートが機能しません。

詳細は [`review_policy_recommendation.md`](review_policy_recommendation.md) を参照。

### ゲート遵守

以下のゲートは **絶対に省略不可**（SKILL.md §0 絶対遵守ゲートルール）：

| Phase | Gate 種別 | 承認媒体 |
|---|---|---|
| A-2 | Artifact Review | `implementation_plan.md` § Title Candidates |
| B-1 | Artifact Review | `implementation_plan.md` § Cross-Module Pattern Selection |
| B-2 | Artifact Review | `task.md` § Phase B Gates (Web Research) |
| C | Artifact Review + Scripted | `implementation_plan.md` § Deep Dive Plan → `bash phase_c_gate.sh` |
| D | Artifact Review + Scripted | `implementation_plan.md` § Report Plan → `bash phase_d_gate.sh` |

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
- **Artifact Review 省略**: 5箇所の Artifact Review ゲートを Agent 側で自発的に「通過」扱いにするのは禁止
- **サブエージェント起動**: Manager + Browser subagent は本スキル実行中は **起動しない**（Web調査時の Browser subagent のみ Phase B 限定で許可）
- **patents.csv 全量表示**: `print(df)` / `cat data/patents.csv` 等
- **deep_dive の圧縮**: Phase D で report.typ に deep_dive をコピーする際、要約・省略は禁止（全文コピー）
- **bash Gate スキップ**: `phase_c_gate.sh` / `phase_d_gate.sh` の実行を省略するのは絶対禁止

---

## 📁 フォルダ構成

```
session_YYYYMMDD_HHMMSS/               ← cwd
├── capcom_schema/                     # 共有資産（読み取り専用、変更禁止）
│   ├── SKILL.md                       # Claude Code用（Antigravity では参照のみ）
│   ├── analysis/                      # 分析手法ガイド（9ファイル）
│   ├── references/                    # モジュール別スキーマ（10ファイル）
│   ├── exemplars/                     # deep_dive 執筆見本（5 Typst）
│   ├── templates/                     # Typst / PPT テンプレート
│   └── scripts/                       # bash 品質ゲート
├── data/                              # patents.csv + モジュールJSON
├── voyager/                           # Mission Objective + Evidence
├── snapshots/                         # 可視化PNG
├── prompts/                           # AIインサイト（Markdown）
├── reports/                           # ★レポート出力先
├── .agent/                            # Antigravity スキル配置
│   ├── skills/apollo-capcom/SKILL.md  ← 本スキルの本体
│   └── workflows/                     # Phase別起動点（00-05.md）
├── artifacts_templates/               # Artifact 雛形（起動時にコピー）
├── task.md                            ← ★ Artifact（4フェーズチェックリスト）
├── implementation_plan.md             ← ★ Artifact（承認対象セクション）
├── walkthrough.md                     ← ★ Artifact（ゲート結果記録）
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
- **`.agent/workflows/*.md`**: Phase別起動点（`00_capcom_master.md` がマスター）
- **`artifacts_templates/*.tmpl`**: スキル起動時に Artifact として複製される雛形
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
