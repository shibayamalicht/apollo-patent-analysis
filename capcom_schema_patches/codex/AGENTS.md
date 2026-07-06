# APOLLO CAPCOM プロジェクトルール（Codex CLI / Antigravity 共通）

このフォルダは APOLLO v9 の CAPCOM セッション（`session_YYYYMMDD_HHMMSS/`）です。特許分析データから **戦略レポート** を生成する作業ディレクトリです。

このファイルは Codex CLI（`AGENTS.md` として解釈）と Antigravity IDE（`GEMINI.md` の fallback として解釈）の両方で同じ役割を果たします。Antigravity 固有の追加ルールは `GEMINI.md` を参照してください。

---

## 🎯 このフォルダで必ず行うこと

0. **最初に依存をインストールする（1回だけ）**: patents.csv 解析の `pandas`・スライドの `python-pptx`/`Pillow` を入れる。未導入だと `ModuleNotFoundError`（例: `No module named 'pandas'`）で止まる。セッションフォルダ直下で:
   - `python3 -c "import pandas, pptx, PIL" 2>/dev/null && echo "依存OK" || pip install -r requirements-session.txt`
   - `pip` 不在なら `python3 -m pip install -r requirements-session.txt`。詳細は `apollo-capcom` スキルの「環境準備」節
1. **レポート生成依頼を受けたら必ず `apollo-capcom` スキルを起動する**
   - Codex: `$apollo-capcom` または `/skills` → `apollo-capcom`
   - Antigravity: チャットで「apollo-capcom スキルで…」と指示
   - スキル未起動のままレポートを書き始めることは**禁止**
   - **スキルが `/skills` 一覧に出ない場合**（ツールごとの自動探索パスの差異による）: `.agents/skills/apollo-capcom/SKILL.md`（無ければ `.codex/skills/apollo-capcom/SKILL.md` / `.agent/skills/apollo-capcom/SKILL.md`）を **直接 Read してから** 着手する。スキル本体を読まずに着手することは禁止
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
- **水増し（コピペ反復）**: 同一文・同一構文の反復、回転する名詞だけ変えた定型文の量産、「○○観点 1, 2, 3…」式の連番見出しで行数・件数を稼ぐこと。**`phase_d_gate.sh` Check 19 で自動不合格**になる。行数が足りない時は文を繰り返さず、新しい代表特許（固有の公開番号）・新しい数値根拠・別のクロスパターン・Web調査の裏付けを足す。各段落は前段落と異なる固有の事実を最低1つ含めること
- **本文のスクリプト生成**: `deep_dive.typ` / `report.typ` を Python 等のスクリプトでテンプレート生成すること（`reports/generate_*.py` は **Check 19a で自動不合格**）。各文は固有の分析として直接書く。「最低行数を満たすための補助文・つなぎ文」も禁止
- **ゲート回避（specification gaming）**: `phase_d_gate.sh` を読んで反復検出を**すり抜ける目的**で接続詞・語順・文体だけ変えて内容の重複を温存すること。ゲートは実在する欠陥を検出している。対処は重複の削除と固有内容への置換であって、検出回避ではない（末尾22字の重複も Check 19 で検出する）

---

## 📁 フォルダ構成

```
session_YYYYMMDD_HHMMSS/               ← cwd
├── capcom_schema/                     # 共有資産（3ツール共通、読み取り専用）
│   ├── SKILL.md                       # Claude Code 用（本フォルダでは Codex/Antigravity が優先）
│   ├── analysis/                      # 分析手法ガイド（9ファイル）
│   ├── references/                    # モジュール別スキーマ（10ファイル）
│   ├── exemplars/                     # deep_dive 執筆見本（7 Typst）
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
- **config.toml 推奨設定（深く考え、簡潔に書く）**: 深い分析フェーズでは `~/.codex/config.toml` に `model_reasoning_effort = "high"`（推論を深く）と `model_verbosity = "low"`〜`"medium"`（出力は簡潔に）を設定する。**reasoning_effort（思考の深さ）と verbosity（出力の量）は別レバー**であり、「深く考えるが冗長に書かない＝水増ししない」状態に寄せる。**IDE のモデル選択では「高」を選ぶ＝「非常に高い」(xhigh) はコンテキストを最も消費し、大型レポートで枯渇しやすい**
- **大型レポートはフェーズ分割で（コンテキスト枯渇対策）**: 1スレッドで全フェーズを通すと "ran out of context" になりやすい。**1スレッド=1フェーズ（Phase C は1モジュールずつ）に分け、区切りで `/compact`**。成果物はディスクに残るので、新スレッドで `ls reports/` と **`reports/_carryover.md`（フェーズ間引き継ぎ日誌）** を読んで続きから再開できる（詳細はスキル本体 `## 大型レポートのフェーズ分割`）
- **フェーズ間引き継ぎ日誌の更新**: 分割実行で分析の記憶（仮説検証・Web出所・判断理由）を失わないため、各フェーズ完了時・`/compact` 直前に `reports/_carryover.md` へ追記する。**Web調査は1件ごとに即 出所(URL/取得日)を台帳へ**。本日誌はレポート本文へ転載しない（内部メモ）
- **🔄 セッション・チェックポイント（各フェーズ境界で必須）**: 「枯渇しそう」と感じてから分割するのでは遅い。**各フェーズの区切り（Phase A完了・Phase B完了・Phase Cの各モジュール完了ごと・Phase D着手前）で必ず、ゲート通過＋`_carryover.md` 更新の後に「新セッションに切り替えますか？」と提案して一旦停止**する（続行/切替/`/compact` をユーザーが選ぶ）。切替時は新スレッドが `_carryover.md` から自動再開。詳細はスキル本体 `### 🔄 セッション・チェックポイント`
- **スキル優先**: プロジェクトスコープのスキル（自動探索 `.agents/skills/apollo-capcom/SKILL.md`、無ければ `.codex/skills/apollo-capcom/SKILL.md`）が本 AGENTS.md より詳細なため、両者が衝突した場合スキル側を優先

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
