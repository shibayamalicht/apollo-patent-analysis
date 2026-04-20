# `codex exec` 非対話モード運用メモ（初版では非推奨）

## 概要

Codex CLI には以下の2モードがあります：

- **TUI モード**: `codex` で起動。対話型。`ask_user_question` 利用可能。
- **非対話モード**: `codex exec "<prompt>"` で起動。一回限り実行。`ask_user_question` **利用不可**。

本 `apollo-capcom` スキルは **ユーザー確認ゲートを6箇所以上** 持つため、**非対話モードでの利用は初版では非推奨** です。

---

## 推奨される使い方

```bash
cd session_YYYYMMDD_HHMMSS/
codex                                    # TUI起動
> $apollo-capcom レポートを書いてください
# 各ゲートで ask_user_question が対話的に動作
```

---

## 非対話モードで動かすと何が起きるか

```bash
codex exec "$apollo-capcom レポートを書いてください"
```

Phase A-2 STOP-GATE に到達した時点で `ask_user_question` が呼べず：
- AI が「自前でタイトルを決めて進行」するのは **SKILL.md §0 第2項違反**
- AI が「タイトル決定不可のため停止します」と出力して exit
- どちらにせよ **SKILL.md の完了条件を満たせない**

---

## 将来拡張: USER_INPUT_NEEDED マーカー方式（設計メモのみ、未実装）

将来版で非対話モード対応を実装する場合、以下の設計を検討：

### 基本フロー

1. スキル起動時に `$CODEX_MODE` や stdin TTY 有無を確認
2. 非対話モード検出時、STOP-GATE に到達すると以下を stdout に出力し exit 0：

```
===== USER_INPUT_NEEDED =====
key: phase_a2_title_selection
prompt_template: .codex/skills/apollo-capcom/prompts/phase_a2_titles.md
generated_options:
  1. タイトル案1 / サブタイトル案1
  2. タイトル案2 / サブタイトル案2
  3. タイトル案3 / サブタイトル案3
state_file: .capcom_state.json
===== END =====
```

3. ユーザーは以下のいずれかで応答を記録：

**(a) 環境変数方式**
```bash
export APOLLO_CAPCOM_TITLE_CHOICE=2
codex exec "$apollo-capcom レポートを書いてください" --resume
```

**(b) state ファイル方式**
```json
// .capcom_state.json
{
  "phase_a2_title_selection": 2,
  "phase_b_cross_patterns": ["P1", "P4", "P13"],
  "phase_b_web_research": "proceed",
  "phase_c_plan_approved": true,
  "phase_d_plan_approved": true
}
```

4. スキル再起動時に state ファイルを読み、既に応答済みゲートはスキップして次のゲートに進む

### 未解決の課題

- **冪等性**: 再実行時に Phase A の分析結果をキャッシュするか、毎回再実行するか
- **部分実行**: Phase C だけを再実行する等の部分起動
- **エラー回復**: Phase D ゲート FAIL 時の state 更新方針
- **タイムアウト**: 非対話モードで回答が永久に来ない場合の扱い

### 実装の前提条件

- Codex 側で `codex exec --resume` のような中断/再開フラグが提供されるか、CLI 引数で state ファイルを指定できる仕組みが必要
- 現在の Codex CLI（2026-04時点）はこの仕組みを持たないため、本設計は将来版想定

---

## 当面の推奨

**初版では TUI モードのみ対応** します。CI/CD などで非対話実行が必要な場合は：

1. 事前に人間が TUI モードで Phase A-2 / Phase B ゲートを通過させ、タイトルとクロスパターンを確定させる
2. 確定情報を `voyager/mission.json` に追記する（`confirmed_title`, `selected_cross_patterns` 等）
3. 非対話モードで Phase C/D のみ自動実行する（これらのゲートは bash スクリプトで客観判定されるため自動化可能）

---

## 参考

- Codex CLI 公式: [codex exec command reference](https://developers.openai.com/codex/cli/reference)
- GitHub Issue: [openai/codex#11536 Continue on Ask Question Tool](https://github.com/openai/codex/issues/11536)
