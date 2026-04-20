# APOLLO CAPCOM — Antigravity IDE パッチ

**Google Antigravity IDE で、Artifact 駆動パラダイムに沿って CAPCOM レポート生成を実行できるようにするパッチ。**

---

## 🎯 このパッチで追加される機能

- **`apollo-capcom` Skill**: Antigravity のチャットでスキル名を言及すると起動
- **Artifact 駆動ゲート**: `task.md` / `implementation_plan.md` / `walkthrough.md` による5箇所の承認ポイント
- **`.agent/workflows/`**: Phase別の個別起動点（マスター + Phase 1-5）
- **共通 bash ゲート利用**: `capcom_schema/scripts/phase_c_gate.sh` と `phase_d_gate.sh` で品質を客観判定
- **`GEMINI.md`**: Antigravity 最優先ルール
- **Review Policy 推奨設定ガイド**: `review_policy_recommendation.md`

---

## 🚀 使い方

### 前提
- APOLLO で CAPCOM セッション実行済み
- ZIP をダウンロードしてローカルで展開済み（例: `~/Downloads/session_20260416_143022/`）
- Antigravity IDE インストール済み（v1.20.0 以上推奨）

### パッチ適用（1コマンド）

```bash
bash /path/to/apollo_v7/capcom_schema_patches/antigravity/apply_patch.sh ~/Downloads/session_20260416_143022
```

### Review Policy 設定

Antigravity IDE で `session_*/` を開いた後、**必ず Review Policy を `"Request Review"` に設定** してください。これが設定されていないと本スキルのゲートが機能しません。

詳細は [`review_policy_recommendation.md`](review_policy_recommendation.md) を参照。

### スキル起動

Antigravity IDE のチャットで：
```
apollo-capcom スキルでレポート生成してください
```

または、`.agent/workflows/00_capcom_master.md` を Workflows UI から直接起動。

---

## 📋 Artifact 駆動ワークフロー

本パッチ適用後、Antigravity は以下の Artifact を活用してレポート生成を進めます：

### task.md
4フェーズの完全なチェックリスト。Agent が進行中に `[x]` を埋め、ユーザー判断項目（例: Web調査実施/スキップ）はユーザーがチェック。

### implementation_plan.md
**ユーザー承認対象の Artifact**。以下のセクションで Artifact Review が発動：
1. **§ Title Candidates** — Phase A-2 でタイトル3案を提示、ユーザーが ✅ で選定
2. **§ Cross-Module Pattern Selection** — Phase B で13パターンから3つ選定
3. **§ Deep Dive Plan** — Phase C の実行計画を承認
4. **§ Report Structure & Quality Plan** — Phase D の構成と品質基準を承認

### walkthrough.md
Phase C/D の bash gate 実行結果を **そのまま転記** する完了証跡。AI が改ざん・要約してはいけない。

---

## 📁 このパッチで追加されるファイル

```
session_YYYYMMDD_HHMMSS/
├── .agent/
│   ├── skills/apollo-capcom/
│   │   └── SKILL.md                    # Antigravity版スキル本体（Artifact駆動）
│   └── workflows/
│       ├── 00_capcom_master.md         # Phase A→D 一気通貫
│       ├── 01_phase_a_data_intake.md   # Phase A のみ
│       ├── 02_phase_a2_title_selection.md
│       ├── 03_phase_b_evidence_cross.md
│       ├── 04_phase_c_deep_dive.md
│       └── 05_phase_d_integration.md
├── artifacts_templates/
│   ├── task.md.tmpl                    # チェックリスト雛形
│   ├── implementation_plan.md.tmpl     # 承認対象セクション雛形
│   └── walkthrough.md.tmpl             # ゲート結果記録雛形
├── GEMINI.md                           # Antigravity最優先ルール
├── AGENTS.md                           # fallback / Codex互換
└── review_policy_recommendation.md     # Review Policy 設定ガイド
```

**既存の `capcom_schema/` には一切変更を加えません。**

---

## ⚠️ 重要な制約

### Review Policy 設定必須

Review Policy が **"Always Proceed"** だと、本スキルの5箇所のゲートが全てスキップされ、AI が全ての判断（タイトル決定、クロスパターン選定、Web調査可否）を勝手に行ってしまいます。これは品質保証の崩壊を意味します。

必ず **"Request Review"** に設定してください。

### サブエージェント制約

Antigravity には Manager + Browser subagent 等の機構がありますが、**本スキルの分析本体ではサブエージェントを起動しません**（トークン効率のため）。

ただし、**Phase B で Web調査を実施する場合** は、Browser subagent を限定的に使用できます。Gemini Pro Manager + Gemma 4 subagent の組み合わせが推奨される使用例です。

### cwd 規約

本スキルは **常にセッションフォルダ（`session_*/`）を cwd として実行** します。Antigravity IDE でフォルダを開く際、必ず `session_*/` ルートを開いてください。

---

## 🔧 Windows の場合

PowerShell で同等の操作：

```powershell
$patchDir = "C:\path\to\apollo_v7\capcom_schema_patches\antigravity"
$sessionDir = "C:\Users\you\Downloads\session_20260416_143022"

# .agent/ をコピー
Copy-Item -Recurse "$patchDir\.agent" $sessionDir

# artifacts_templates/ をコピー
Copy-Item -Recurse "$patchDir\artifacts_templates" $sessionDir

# ルート配置ファイル
Copy-Item "$patchDir\GEMINI.md" $sessionDir
Copy-Item "$patchDir\AGENTS.md" $sessionDir
Copy-Item "$patchDir\review_policy_recommendation.md" $sessionDir
```

---

## 🧪 動作確認

適用後、以下を確認してください：

```bash
cd ~/Downloads/session_20260416_143022

# 1. パッチファイルが存在する
ls -la .agent/skills/apollo-capcom/SKILL.md
ls -la .agent/workflows/  # 6ファイル
ls -la artifacts_templates/  # 3ファイル
ls -la GEMINI.md AGENTS.md review_policy_recommendation.md

# 2. 既存資産が不変
ls -la capcom_schema/SKILL.md       # 元のまま
ls -la capcom_schema/analysis/      # 9ファイル

# 3. bash gate が動作する（Phase C/D 実行前は期待通り FAIL）
bash capcom_schema/scripts/phase_c_gate.sh
```

Antigravity IDE を起動してフォルダを開いたら：
1. Agent Manager → Review Policy = "Request Review" になっているか確認
2. チャットで `apollo-capcom` と入力 → スキルが検出されることを確認
3. 「レポート生成してください」で起動 → `task.md` / `implementation_plan.md` / `walkthrough.md` が自動生成されることを確認

---

## 🧩 Codex CLI と併用したい場合

Codex パッチと Antigravity パッチは **同時に適用可能** です：

```bash
bash /path/to/capcom_schema_patches/codex/apply_patch.sh ~/Downloads/session_xxx
bash /path/to/capcom_schema_patches/antigravity/apply_patch.sh ~/Downloads/session_xxx
```

`AGENTS.md` は両パッチで **共通内容** なので、どちらを後から適用してもファイルの中身は同じです（ただし上書き確認プロンプトが出ます）。

---

## 🐛 よくある問題

### Q. `apollo-capcom` スキルが検出されない

**A**. 以下を確認:
- Antigravity IDE の v1.20.0 以上を使用しているか
- フォルダとして `session_*/` ルートを開いているか（サブフォルダだけ開くと検出されない）
- `.agent/skills/apollo-capcom/SKILL.md` のファイル権限 (読み取り可能か)

### Q. Artifact Review が発動しない

**A**. Review Policy を確認。"Always Proceed" になっている場合は変更してください。詳細は [`review_policy_recommendation.md`](review_policy_recommendation.md)。

### Q. `phase_c_gate.sh` が見つからない

**A**. `session_*/capcom_schema/scripts/phase_c_gate.sh` にあるはず。CAPCOM ZIP 展開が正しく完了しているか確認。

### Q. Artifact のチェックボックスを ✅ にしたのに Agent が次に進まない

**A**. Antigravity の Artifact 同期遅延の可能性。`Cmd/Ctrl+S` で明示保存、あるいは Artifact タブをリロードしてみてください。

### Q. Browser subagent を Phase B で使いたい

**A**. 対応可能。Phase B STOP-GATE 2 で「Web調査を実施する」をユーザーが選択した後、Agent に「Browser subagent で市場調査を実施してください」と指示してください。`03_phase_b_evidence_cross.md` のステップ15 にその旨記載あり。

---

## 📚 参考リンク

- [Antigravity 公式ドキュメント](https://antigravity.google/docs)
- [Implementation Plan spec](https://antigravity.google/docs/implementation-plan)
- 親 README: [`../README.md`](../README.md)
- Codex 用パッチ: [`../codex/`](../codex/)
