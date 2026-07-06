# `codex exec` 非対話モード運用メモ（対話モード推奨＋部分自動化）

> **方針（確定）**: `apollo-capcom` のレポート生成は本質的に対話的なので、**フルパイプラインは TUI（対話）モードで実行**する。
> 非対話（`codex exec`）だけで全工程を回すことは**サポートしない**（判断ゲートが TUI 必須のため）。
> ただし 2026-06 時点で `codex exec resume` が使えるようになり、**A/B を対話で通過してから C/D を非対話で再開**する
> 部分自動化は現実的になった（下記）。

## 概要

Codex CLI には2モードがある：

- **TUI モード**: `codex` で起動。対話型。`ask_user_question`（対話質問・選択ポップアップ）が**利用可能**。
- **非対話モード**: `codex exec "<prompt>"` で起動。スクリプト/CI 向け。進捗は stderr、最終メッセージのみ stdout。`ask_user_question` は**利用不可**（後述）。

本スキルは **判断を要するユーザー確認ゲートを多数**持つ（各 Phase A STOP-GATE〔A / query_intent / SQ / B / C / D〕＋ Phase A-2 タイトル決定 ＋ **各フェーズ境界のセッション・チェックポイント**〔Phase A/B 完了・Phase C の各モジュール完了・Phase D 着手前〕）。これらはタイトル選択・クロスパターン選択・Web 調査の可否・別冊生成の可否など**人間の判断**を必要とし、`ask_user_question` でしか取得できない。

### 非対話モードで対話質問が使えるか（2026-06 時点）

**使えない。** `codex exec` で Plan モード相当の対話質問（`request_user_input` / 選択ポップアップ）を出す要望は [openai/codex#11536（Continue on Ask Question Tool）](https://github.com/openai/codex/issues/11536) として挙がっているが、[#10384](https://github.com/openai/codex/issues/10384) の重複として close されており**未実装**。プロンプトで「質問して」と促しても、Plan モードのポップアップ/選択 UX は再現できない。→ **判断ゲートは TUI 必須**。

---

## 推奨される使い方（フルパイプライン＝TUI）

```bash
cd session_YYYYMMDD_HHMMSS/
codex                                    # TUI起動
> $apollo-capcom レポートを書いてください
# 各ゲート（タイトル決定・Web調査・別冊可否・各フェーズ境界のチェックポイント）で
# ask_user_question が対話的に動作する
```

非対話で全工程を走らせると、最初の判断ゲート（例: Phase A-2 タイトル決定）で `ask_user_question` が呼べず：
- 「自前でタイトルを決めて進行」＝ **SKILL.md §0 第2項違反**
- 「決定不可のため停止」＝ exit して**完了条件を満たせない**

どちらにせよフルパイプラインは非対話では完走できない。

---

## 部分自動化パターン（A/B を対話 → C/D を `codex exec resume` で非対話）

2026-06 時点で **`codex exec resume`** が利用可能（[公式: Non-interactive mode](https://developers.openai.com/codex/noninteractive)）。前回 exec セッションの**トランスクリプト・plan・承認を保持したまま**、追従プロンプトを与えて再開できる：

```bash
# 直近の exec セッションを再開（追従プロンプト付き）
codex exec resume --last "Phase C を Saturn V から続けてください"
# セッションIDを指定して再開
codex exec resume <SESSION_ID> "Phase D の統合に進んでください"
```

これと本スキルの**再開インフラ**（`reports/_carryover.md`＋`ls reports/` でディスク状態から現在地を復元）を組み合わせると、判断ゲートのない後半を非対話で回せる：

1. **Phase A / B を TUI（対話）で通過**し、タイトル・クロスパターン・Web 調査可否・別冊可否などの**判断を確定**する。
2. 確定情報を **`reports/_carryover.md`**（および必要に応じて `voyager/mission.json` の `confirmed_title` / `selected_cross_patterns` 等）に固定する。
3. **Phase C / D を非対話で実行**する（`codex exec resume` ＋ `_carryover.md` から再開）。Phase C/D の合否は `phase_c_gate.sh` / `phase_d_gate.sh` が**bash で客観判定**するため、人間の判断ゲートを挟まず自動化できる。
   - ただし **セッション・チェックポイント**（Phase C 各モジュール後・Phase D 着手前）は本来 TUI で停止提案するもの。非対話で回す場合は「ユーザーが C/D の一括自動実行を明示的に選んだ」前提とし、チェックポイントは省略してよい（SKILL.md のチェックポイント節「一気に最後まで」を選んだ扱い）。

> 注: フルパイプラインの**前半（A/B）は引き続き TUI 必須**。完全な end-to-end 非対話化には、`codex exec` 側で対話質問（#10384/#11536）が実装される必要がある。

---

## 将来拡張: USER_INPUT_NEEDED マーカー方式（設計メモ・一部は実現済み）

初版で構想した「非対話ゲートを state ファイルで解決する」方式は、**一部が現行インフラで実現済み**：

- **中断/再開**: 初版時点（2026-04）の Codex CLI には再開フラグが無かったが、現在は **`codex exec resume --last` / `<SESSION_ID>`** が提供されている（上記）。
- **state ファイル / 現在地復元**: 本スキルの **`reports/_carryover.md`＋ディスク成果物（`*_deep_dive.typ` / `_phase_a_decisions.json` / `report.typ`）** が事実上の state ファイルとして機能し、新セッション/再開時に現在地を復元する。
- **冪等性 / 部分実行**: ゲートがディスク状態を見るため、完了済みフェーズはスキップして続きから再開できる（部分実行が成立）。

**残る欠落**は「非対話モードでの判断ゲート（対話質問）」だけ。これが [#10384 / #11536](https://github.com/openai/codex/issues/11536) で実装されれば、A/B も含む完全な非対話化が視野に入る。それまでは「A/B は対話・C/D は非対話再開」の部分自動化が現実解。

### 完全非対話化が実装された場合の設計（参考・未使用）

スキル起動時に非対話を検出し、STOP-GATE 到達時に機械可読マーカーを stdout に出して exit、ユーザーが state ファイル/環境変数で応答 → `codex exec resume` で再開、という流れ：

```
===== USER_INPUT_NEEDED =====
key: phase_a2_title_selection
generated_options:
  1. タイトル案1 / サブタイトル案1
  2. タイトル案2 / サブタイトル案2
state_file: reports/_carryover.md（または .capcom_state.json）
===== END =====
```

未解決の課題（参考）: エラー回復時の state 更新方針、回答が来ない場合のタイムアウト。

---

## 参考

- Codex CLI 公式: [Non-interactive mode](https://developers.openai.com/codex/noninteractive)（`codex exec` / `codex exec resume --last`・`<SESSION_ID>`・`--json`）
- Codex CLI 公式: [Command line options](https://developers.openai.com/codex/cli/reference)
- GitHub Issue: [openai/codex#11536 Continue on Ask Question Tool](https://github.com/openai/codex/issues/11536)（#10384 の重複として close・**未実装**＝非対話の対話質問は不可）
