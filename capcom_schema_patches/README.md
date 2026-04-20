# APOLLO CAPCOM マルチツール対応パッチ集

**APOLLO v7 の CAPCOM 機能を Claude Code 以外の AI coding agent でも快適に使うためのパッチパッケージ。**

APOLLO の `capcom_schema/SKILL.md` は Claude Code 固有のツール（`AskUserQuestion`, `/compact`, Agent tool 禁止ルール等）を前提に書かれています。本ディレクトリは、既存の `capcom_schema/` を **一切書き換えず**、各ツール用のファイルをセッションディレクトリに **オーバーレイ（上書き追加）** することで、Codex CLI / Antigravity IDE でも同等の品質で CAPCOM レポート生成を実行できるようにします。

---

## 🎯 対応ツール（2026年4月時点）

| ツール | ディレクトリ | 配布状態 |
|---|---|---|
| **Claude Code**（Anthropic） | *(既存の `capcom_schema/` がそのまま使える)* | ✅ デフォルト |
| **Codex CLI**（OpenAI） | [`codex/`](codex/) | ✅ 利用可能 |
| **Antigravity IDE**（Google） | [`antigravity/`](antigravity/) | ✅ 利用可能 |
| Cursor / Gemini CLI / Windsurf 等 | *(将来追加予定)* | ⏳ 計画中 |

3ツールは 2026年初頭から **Open Agent Skills Standard (OAS)** を共有しており、SKILL.md フォーマット自体は互換です。本パッチは差分のみを実装します。

---

## 💡 設計思想：オーバーレイ方式

```
session_YYYYMMDD_HHMMSS/          # CAPCOM ZIP 展開後のワーキングフォルダ
├── capcom_schema/                # 🔒 既存、不変（Claude Code用の完全版）
│   ├── SKILL.md
│   ├── analysis/
│   ├── references/
│   ├── exemplars/
│   ├── templates/
│   └── scripts/                  # ← phase_c_gate.sh / phase_d_gate.sh は 3ツール共通で使う
├── data/  voyager/  snapshots/  prompts/  reports/  metadata.json
│
├── .codex/                       # ← Codex パッチ適用時に追加（.codex/skills/apollo-capcom/）
├── .agent/                       # ← Antigravity パッチ適用時に追加（.agent/skills/apollo-capcom/ と .agent/workflows/）
├── AGENTS.md                     # ← Codex & Antigravity 共通ルール
└── GEMINI.md                     # ← Antigravity 固有ルール（Artifact運用）
```

### なぜオーバーレイ？

- **既存パス記述が全て無変更で動く**: SKILL.md 内の `capcom_schema/analysis/...` 等がそのまま有効
- **Windows 対応**: シンボリックリンク不要、純粋なファイルコピー
- **衝突なし**: `.codex/` と `.agent/` は名前空間が分離しているので同時適用可能
- **既存の Claude Code 運用が壊れない**: `capcom_schema/` には触らない

---

## 🚀 クイックスタート

### ステップ1: CAPCOM ZIP を展開

APOLLO の CAPCOM ページから ZIP をダウンロードし、ローカルで展開：
```bash
unzip session_YYYYMMDD_HHMMSS.zip
cd session_YYYYMMDD_HHMMSS/
```

### ステップ2: 利用するツール用のパッチを適用

**Claude Code 利用者** → **追加作業なし**。そのまま `claude` を起動してレポート生成を依頼するだけ。

**Codex CLI 利用者**
```bash
# リポジトリから patches を取得（初回のみ）
git clone https://github.com/<owner>/apollo.git /tmp/apollo
# パッチ適用
bash /tmp/apollo/apollo_v7/capcom_schema_patches/codex/apply_patch.sh .
# Codex起動
codex
# チャットで: $apollo-capcom  または  /skills から apollo-capcom を選択
```

**Antigravity IDE 利用者**
```bash
git clone https://github.com/<owner>/apollo.git /tmp/apollo
bash /tmp/apollo/apollo_v7/capcom_schema_patches/antigravity/apply_patch.sh .
# Antigravity IDE で session_YYYYMMDD_HHMMSS/ フォルダを開く
# Review Policy を "Request Review" に設定（初回のみ、review_policy_recommendation.md 参照）
# チャットで「apollo-capcom スキルでレポート生成」と依頼
```

### Windows の場合

PowerShell で同等の操作：
```powershell
# Codex
Copy-Item -Recurse /path/to/apollo/apollo_v7/capcom_schema_patches/codex/.codex .
Copy-Item /path/to/apollo/apollo_v7/capcom_schema_patches/codex/AGENTS.md .

# Antigravity
Copy-Item -Recurse /path/to/apollo/apollo_v7/capcom_schema_patches/antigravity/.agent .
Copy-Item /path/to/apollo/apollo_v7/capcom_schema_patches/antigravity/GEMINI.md .
Copy-Item /path/to/apollo/apollo_v7/capcom_schema_patches/antigravity/AGENTS.md .
```

### 同時適用（Codex と Antigravity を併用）

両方のパッチを同時に適用しても衝突しません。`.codex/` と `.agent/` は名前空間が別、`AGENTS.md` は両ツール共通内容で記述されています。

---

## 🔍 3ツール実装比較表

| 機能 | Claude Code | Codex CLI | Antigravity |
|---|---|---|---|
| **Skill配置** | `.claude/skills/` | `.codex/skills/apollo-capcom/` | `.agent/skills/apollo-capcom/` |
| **Skill呼び出し** | スキル自動起動 | `$apollo-capcom` or `/skills` | チャットで名前言及 |
| **ユーザー確認ゲート** | `AskUserQuestion` ツール | `ask_user_question` ツール（TUI）<br>※ `codex exec` では非対応 | Artifact レビュー（`task.md` / `implementation_plan.md` の承認） |
| **プロジェクトルール** | `CLAUDE.md` | `AGENTS.md`（階層的） | `GEMINI.md` + `AGENTS.md`（fallback） |
| **コンテキスト圧縮** | `/compact` | `/compact` | 自動 + `.gemini/antigravity/brain/` |
| **サブエージェント起動** | 禁止（既存ルール） | 組込なし | 禁止（本スキル） |
| **bash ゲート** | `phase_c_gate.sh` / `phase_d_gate.sh` を `Bash` ツールで実行 | 同左 | 同左 |
| **対話モード必須** | Yes | **Yes**（`codex exec` 不可） | 推奨（`Review Policy = "Request Review"`） |

**品質保証の要**: すべてのツールで `phase_c_gate.sh` / `phase_d_gate.sh` を共通利用。これが AI の主観判断を排した客観的な品質ゲートとなります。

---

## 📁 本ディレクトリの構造

```
apollo_v7/capcom_schema_patches/
├── README.md                         # ← 本ファイル
│
├── codex/                            # Codex CLI 用パッチ
│   ├── README.md                     # Codex適用手順
│   ├── apply_patch.sh                # overlay適用スクリプト
│   ├── AGENTS.md                     # session dirに配置される
│   ├── exec_mode_addendum.md         # codex exec 非対話モード注意書き
│   └── .codex/
│       └── skills/apollo-capcom/
│           ├── SKILL.md
│           └── prompts/              # ask_user_question 用テンプレ
│
└── antigravity/                      # Antigravity IDE 用パッチ
    ├── README.md
    ├── apply_patch.sh
    ├── GEMINI.md                     # Antigravity最優先ルール
    ├── AGENTS.md                     # fallback / Codex互換
    ├── review_policy_recommendation.md
    ├── .agent/
    │   ├── skills/apollo-capcom/SKILL.md
    │   └── workflows/                # Phase別起動点（.md × 6）
    └── artifacts_templates/          # task.md / implementation_plan.md / walkthrough.md 雛形
```

---

## 🔒 既存ファイルの不変性保証

このパッチを適用しても、以下のファイルは **一切変更されません**：
- `apollo_v7/capcom_schema/` 配下の全ファイル
- `apollo_v7/capcom.py`（Streamlit ZIP export 処理）
- `apollo_v7/pages/10_📡_CAPCOM.py`（CAPCOM UI）
- CAPCOM ZIP の内容自体

つまり、**Claude Code 利用者に対する影響はゼロ** です。

---

## 🛠 開発者向け: 新規ツール対応の追加方法

Cursor / Gemini CLI 等への対応を追加したい場合：

1. `apollo_v7/capcom_schema_patches/<tool_name>/` ディレクトリを作成
2. そのツールの Skill 配置規約に合わせて `.<tool>/skills/apollo-capcom/SKILL.md` を配置
3. 既存の Codex 版または Antigravity 版の SKILL.md を雛形にし、ツール固有の差分（ユーザー確認ツール名、配置パス等）を書き換え
4. `apply_patch.sh` と `README.md` を追加
5. 本ファイル（`apollo_v7/capcom_schema_patches/README.md`）の対応ツール表を更新

OAS（Open Agent Skills Standard）準拠のツールであれば、SKILL.md 本体は 95% 流用できるはずです。

---

## 📄 ライセンス

本パッチパッケージは APOLLO 本体と同じ MIT License です。

---

© 2025-2026 しばやま
