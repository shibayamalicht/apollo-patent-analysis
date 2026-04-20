# APOLLO CAPCOM — Codex CLI パッチ

**Codex CLI（OpenAI）でも Claude Code と同等の品質で CAPCOM レポート生成を実行できるようにするパッチ。**

---

## 🎯 このパッチで追加される機能

- **`apollo-capcom` Skill**: セッションフォルダでレポート生成を依頼すると自動起動
- **`ask_user_question` ベースのゲート**: Phase A-2 / B / C / D の各ユーザー確認
- **共通 bash ゲート利用**: `capcom_schema/scripts/phase_c_gate.sh` と `phase_d_gate.sh` で品質を客観判定
- **`AGENTS.md`**: Codex の階層ルールに則ったプロジェクトルール配置

---

## 🚀 使い方

### 前提
- APOLLO で CAPCOM セッション実行済み
- ZIP をダウンロードしてローカルで展開済み（例: `~/Downloads/session_20260416_143022/`）
- Codex CLI インストール済み（`codex --version` で確認）

### パッチ適用（1コマンド）

```bash
bash /path/to/apollo_v7/capcom_schema_patches/codex/apply_patch.sh ~/Downloads/session_20260416_143022
```

### Codex CLI 起動 → レポート生成

```bash
cd ~/Downloads/session_20260416_143022
codex                                  # ⚠️ TUIモード必須（codex exec は使えない）
```

Codex のチャットで：
```
$apollo-capcom レポートを書いてください
```

または：
```
/skills
# → apollo-capcom を選択
```

---

## 📋 ワークフロー

本パッチ適用後、Codex は以下のフロー（SKILL.md §0 に基づく絶対遵守ゲート）でレポートを生成します：

1. **Phase A**: ミッション理解 + データ精読
2. **Phase A-2**: タイトル3案提示 → `ask_user_question` でユーザー選択
3. **Phase B**: エビデンス精読 + クロス分析
   - STOP-GATE 1: クロスパターン3つを `ask_user_question` で承認
   - STOP-GATE 2: Web調査可否を `ask_user_question` で承認
4. **Phase C**: モジュール別 deep_dive 生成
   - 完了時: `bash capcom_schema/scripts/phase_c_gate.sh` で機械的判定
5. **Phase D**: report.typ 統合生成
   - 完了時: `bash capcom_schema/scripts/phase_d_gate.sh` で品質判定

---

## 📁 このパッチで追加されるファイル

```
session_YYYYMMDD_HHMMSS/
├── .codex/
│   └── skills/apollo-capcom/
│       ├── SKILL.md                    # Codex版スキル本体（約460行）
│       └── prompts/
│           ├── phase_a2_titles.md      # タイトル3案用テンプレ
│           ├── phase_b_cross.md        # クロス選定用テンプレ
│           ├── phase_b_webresearch.md  # Web調査可否テンプレ
│           ├── phase_c_plan.md         # Deep Dive計画テンプレ
│           └── phase_d_plan.md         # Report構造テンプレ
├── AGENTS.md                           # Codex階層ルール
└── exec_mode_addendum.md               # codex exec非対話モード注意書き
```

**既存の `capcom_schema/` には一切変更を加えません。**

---

## ⚠️ 重要な制約

### 対話モード（TUI）必須

本スキルは **`ask_user_question` ツール** を6箇所以上使います。これは **Codex の TUI モードでのみ動作** します：

- ✅ `codex` → TUI 起動 → チャットで `$apollo-capcom`
- ❌ `codex exec "レポート書いて"` → `ask_user_question` が使えないため Phase A-2 で停止

非対話モード（CI/CD等）での実行は初版では非推奨です。将来拡張の詳細は [`exec_mode_addendum.md`](exec_mode_addendum.md) を参照してください。

### サブエージェント起動禁止

トークン効率化のため、本スキル実行中はサブエージェント（他の coding agent の呼び出し）を起動しません。Codex CLI には 2026年4月時点で組込サブエージェント機構はありませんが、念のための防衛規定です。

### cwd 規約

本スキルは **常にセッションフォルダ（`session_*/`）を cwd として実行** します。`capcom_schema/analysis/...` 等の相対パスがこの cwd 前提で書かれているため、違うディレクトリから実行すると失敗します。

---

## 🔧 Windows の場合

PowerShell で同等の操作：

```powershell
$patchDir = "C:\path\to\apollo_v7\capcom_schema_patches\codex"
$sessionDir = "C:\Users\you\Downloads\session_20260416_143022"

# .codex/ をコピー
Copy-Item -Recurse "$patchDir\.codex" $sessionDir

# AGENTS.md
Copy-Item "$patchDir\AGENTS.md" $sessionDir

# exec_mode_addendum.md
Copy-Item "$patchDir\exec_mode_addendum.md" $sessionDir
```

---

## 🧪 動作確認

適用後、以下を確認してください：

```bash
cd ~/Downloads/session_20260416_143022

# 1. パッチファイルが存在する
ls -la .codex/skills/apollo-capcom/SKILL.md
ls -la AGENTS.md

# 2. 既存資産が不変
ls -la capcom_schema/SKILL.md       # 元のまま
ls -la capcom_schema/analysis/      # 9ファイル

# 3. bash gate が動作する（Phase C/D 実行前は期待通り FAIL）
bash capcom_schema/scripts/phase_c_gate.sh
# → ❌ MISSING: reports/nebula_deep_dive.typ ... (Phase C 未実行なので FAIL 想定)

# 4. Codex 起動
codex
```

---

## 🧩 Antigravity と併用したい場合

Codex パッチと Antigravity パッチは **同時に適用可能** です（`.codex/` と `.agent/` は名前空間が別）：

```bash
bash /path/to/capcom_schema_patches/codex/apply_patch.sh ~/Downloads/session_xxx
bash /path/to/capcom_schema_patches/antigravity/apply_patch.sh ~/Downloads/session_xxx
```

`AGENTS.md` は両方のパッチで共通内容なので衝突しません（後から適用した方のコピーが残りますが、中身は同じ）。

---

## 🐛 よくある問題

### Q. `$apollo-capcom` と打っても反応しない

**A**. Codex の Skill 検出には `.codex/skills/<name>/SKILL.md` の配置と、Codex を `session_*/` を cwd として起動することが必要。以下を確認：
```bash
cd ~/Downloads/session_20260416_143022
ls .codex/skills/apollo-capcom/SKILL.md
codex
```

### Q. `ask_user_question` が使えないと言われる

**A**. `codex exec` で起動していませんか？ `codex`（TUI）で起動してください。

### Q. `phase_c_gate.sh` が FAIL する

**A**. 正常動作です。Phase C で生成すべき `reports/*_deep_dive.typ` がまだ無い状態です。スキルの指示に従って不足モジュールを生成し、再度ゲートを実行してください。

---

## 📚 参考リンク

- [Codex CLI 公式ドキュメント](https://developers.openai.com/codex/cli)
- [Codex Agent Skills](https://developers.openai.com/codex/skills)
- [Codex AGENTS.md 仕様](https://developers.openai.com/codex/guides/agents-md)
- 親 README: [`../README.md`](../README.md)
- Antigravity 用パッチ: [`../antigravity/`](../antigravity/)
