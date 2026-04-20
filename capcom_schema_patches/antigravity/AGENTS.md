# APOLLO CAPCOM プロジェクトルール（Codex CLI / Antigravity 共通）

このフォルダは APOLLO v7 の CAPCOM セッション（`session_YYYYMMDD_HHMMSS/`）です。特許分析データから **戦略レポート** を生成する作業ディレクトリです。

このファイルは Codex CLI（`AGENTS.md` として解釈）と Antigravity IDE（`GEMINI.md` の fallback として解釈）の両方で同じ役割を果たします。Antigravity 固有の追加ルールは `GEMINI.md` を参照してください。

---

## 🎯 このフォルダで必ず行うこと

1. **レポート生成依頼を受けたら必ず `apollo-capcom` スキルを起動する**
   - Codex: `$apollo-capcom` または `/skills` → `apollo-capcom`
   - Antigravity: チャットで「apollo-capcom スキルで…」と指示
   - スキル未起動のままレポートを書き始めることは**禁止**
2. **常に本フォルダ（`session_*/`）をcwd として作業する**
   - 相対パス `capcom_schema/...`, `data/...`, `reports/...` 等がこの cwd 前提で書かれている
3. **`capcom_schema/scripts/phase_c_gate.sh` と `phase_d_gate.sh` を省略不可**
   - Phase C 完了時: `bash capcom_schema/scripts/phase_c_gate.sh` → exit 0 必須
   - Phase D 完了時: `bash capcom_schema/scripts/phase_d_gate.sh` → exit 0 必須
   - **AI の主観判断で「実質的にOK」と飛ばすのは禁止**
4. **`data/patents.csv` は絶対に全量読み込まない**
   - `head -5` でカラム構成確認 → pandas でフィルタして `.head()` 制限
   - `print(df)` / `cat patents.csv` は禁止
5. **ユーザー確認ゲートを省略しない**
   - Phase A-2（タイトル3案）/ Phase B 前（クロスパターン+Web調査）/ Phase C 前 / Phase D 前
   - Codex: `ask_user_question` ツールで取得
   - Antigravity: `implementation_plan.md` / `task.md` のユーザーレビュー待ち

---

## 🚫 禁止事項

- **スキル未起動でレポート着手**: `apollo-capcom` を起動せずに `data/` を解析してレポート本体を書き始めるのは禁止
- **ゲート省略**: bash gate スクリプトの実行を飛ばす / ユーザー確認を省略する
- **サブエージェント起動**: Codex は組込なし、Antigravity は `apollo-capcom` では禁止。トークン効率化のため全処理をメインコンテキストで完結
- **patents.csv 全量表示**: `print(df)` / `cat data/patents.csv` 等
- **deep_dive の圧縮**: Phase D で report.typ に deep_dive をコピーする際、要約・省略は禁止（全文コピー）

---

## 📁 フォルダ構成

```
session_YYYYMMDD_HHMMSS/               ← cwd
├── capcom_schema/                     # 共有資産（3ツール共通、読み取り専用）
│   ├── SKILL.md                       # Claude Code 用（本フォルダでは Codex/Antigravity が優先）
│   ├── analysis/                      # 分析手法ガイド（9ファイル）
│   ├── references/                    # モジュール別スキーマ（10ファイル）
│   ├── exemplars/                     # deep_dive 執筆見本（5 Typst）
│   ├── templates/                     # Typst / PPT テンプレート
│   └── scripts/                       # bash 品質ゲート
├── data/                              # patents.csv + 各モジュールJSON
├── voyager/                           # Mission Objective + Evidence
├── snapshots/                         # 可視化PNG
├── prompts/                           # AIインサイト（Markdown）
├── reports/                           # ★レポート出力先
├── .codex/                            # Codex 適用時のみ（skills/apollo-capcom/）
├── .agent/                            # Antigravity 適用時のみ
├── AGENTS.md                          ← 本ファイル
├── GEMINI.md                          # Antigravity 適用時のみ
└── metadata.json
```

---

## 🔧 ツール別の注意

### Codex CLI
- **対話モード必須**: `codex` で TUI 起動。`codex exec` では `ask_user_question` が利用不可のため本スキルの STOP-GATE が機能しない
- **`/compact`** でコンテキスト圧縮可能（Phase C の長文生成中に使用）
- **スキル優先**: プロジェクトスコープのスキル `.codex/skills/apollo-capcom/SKILL.md` が本 AGENTS.md より詳細なため、両者が衝突した場合スキル側を優先

### Antigravity IDE
- **Review Policy = "Request Review"** を推奨（5箇所の STOP-GATE で必須）
- **Artifact 駆動**: `task.md` / `implementation_plan.md` / `walkthrough.md` がユーザー承認の媒体
- **GEMINI.md** が Antigravity 固有のルールを定義（本 AGENTS.md は fallback）

---

## 📜 参照ドキュメント

- スキル本体（Claude Code 版）: `capcom_schema/SKILL.md`
- スキル本体（Codex 版）: `.codex/skills/apollo-capcom/SKILL.md`（Codex パッチ適用時）
- スキル本体（Antigravity 版）: `.agent/skills/apollo-capcom/SKILL.md`（Antigravity パッチ適用時）
- 分析手法ガイド: `capcom_schema/analysis/` 9ファイル
- 品質ゲート: `capcom_schema/scripts/phase_c_gate.sh`, `phase_d_gate.sh`

---

## 🔖 本ファイルの役割

`session_*/AGENTS.md` は **プロジェクト全体のルール**。`.codex/skills/apollo-capcom/SKILL.md` や `.agent/skills/apollo-capcom/SKILL.md` は **具体的な作業手順**。両者は役割分担しており、スキル実行時に両方を読み込んで動作する前提です。

本ファイルを削除するとスキルの起動条件（cwd 規約、gate 必須性）が曖昧になり、品質が低下します。**削除禁止**。
