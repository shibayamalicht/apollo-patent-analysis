# KATHERINE 対話型モード — Codex CLI 補遺（`exec_mode_addendum.md` と同格）

> **位置づけ**: 対話型レポート作成モード（KATHERINE）を Codex CLI で動かすための薄い補遺。
> **進行の正本は `capcom_schema/interactive/SKILL_INTERACTIVE.md`**（対話ポイントの実施定義は
> `capcom_schema/interactive/dialogue_points.md`）、フェーズ・STOP-GATE・完了条件の正本は
> `.codex/skills/apollo-capcom/SKILL.md`（Codex 版）である。
> **本ファイルには数値基準（文字数・件数・行数等）を一切複製しない**（二重管理禁止。
> 基準は常に SKILL.md と `capcom_schema/analysis/` 正本を参照する）。
>
> ⚠️ **対話型モードは Claude Code で検証済み。Codex CLI での実機検証は未了（ベストエフォート対応）。**
> 読替の正本は SKILL_INTERACTIVE.md §2 の対話ツール読替表（`AskUserQuestion` → `ask_user_question`、
> TUI・対話モード必須）。本補遺はその読替を Codex の実運用に落とすための注記に徹する。

---

## 1. §0 絶対遵守ゲートルールの項番対応表（本家 7 項 ⇄ Codex 版 9 項）

SKILL_INTERACTIVE.md §0-I は「SKILL.md §0 の全項は対話型でもそのまま適用される。**項数はツール版の
SKILL で異なりうるため、項番ではなく内容で参照する**」と定めている。Codex 版 SKILL.md §0 は、
本家 `capcom_schema/SKILL.md` §0 の第 6 項（水増し・反復・すり抜けリライトの禁止）を Codex での
実失敗事例に合わせて 3 項に分割しているため項数が異なるが、内容は同一である:

| 本家 SKILL.md §0（7 項） | Codex 版 SKILL.md §0（9 項） |
|---|---|
| 1. 全ゲートは省略不可 | 1. 同左 |
| 2. ユーザー応答待ち必須（`AskUserQuestion`） | 2. 同左（`ask_user_question`・TUI 必須に読替） |
| 3. 不合格時は強制ループ | 3. 同左 |
| 4. 指示の長さで手順を変えない | 4. 同左 |
| 5. 「省略します」と宣言する前に立ち止まる | 5. 同左 |
| 6. 水増し・反復・すり抜けリライトの禁止 | 6. 水増し禁止 ＋ 7. 本文スクリプト生成・ゲート回避の禁止 ＋ 8. 工程ナレーション節・申し送りの禁止 |
| 7. STOP-GATE はコンテキスト限界でも死守 | 9. 同左 |

SKILL_INTERACTIVE.md は項番非依存の内容参照で書かれているため、この項数差は対話型の進行に
影響しない。§0-I の追補 3 項（I-1 根拠なし提案の禁止 / I-2 応答なし確定の禁止 / I-3 迎合の禁止）も
Codex でそのまま適用する。

## 2. CP 確定の `ask_user_question` 流儀

- **基本 4 択をそのまま使う**: 「✅ この案で確定 / ✏️ 修正して確定 / 🔄 別案・差し替え / 🤖 おまかせ」
  （設計規則の正本: `dialogue_points.md` §確定の選択肢設計。既存 STOP-GATE に CP を重ねる場合は
  SKILL.md 側で定義された選択肢を優先する規則も同正本のまま）。
- **選択肢に「Other（その他）」を自前で入れない**: Codex の `ask_user_question` は自由記述の
  Other 相当が自動付与される。自前で足すと選択肢枠を浪費し、自動付与分と重複する。
- **1 確定 = 1 呼び出しを既定とする**: `multiSelect` を使った CP-6（複数パターンの一括判定）・
  CP-8（複数結論の一括採否）のまとめ確定は **Codex 実機で未検証**のため、既定では 1 つの確定に
  つき 1 回 `ask_user_question` を呼ぶ。`dialogue_points.md` が認める**提示のバッチ**（複数パターンを
  まとめて 1 回で提示）は可。その場合も確定は「一括確定 / 個別に見る」を含む単一選択で取る。
- **抱き合わせ禁止**（既存 STOP-GATE の確認と CP の進め方・委任の確認を 1 質問にまとめない）は
  `dialogue_points.md` §共通規則のまま適用する。

## 3. セッション・チェックポイントと「1 スレッド = 1 フェーズ」標準の対応

- KATHERINE では「1 スレッド = 1 フェーズ」分割が**標準**（SKILL_INTERACTIVE.md §7。自律生成
  モードでは推奨）。これは Codex 版 SKILL.md の「🔄 セッション・チェックポイント」（各フェーズ
  境界で切替提案・一旦停止）とそのまま重なる。対話型では各境界のチェックポイントで
  **「🔄 新セッションに切替」を既定の推奨**として提示する（対話の往復が増えるぶん、自律生成
  モードより枯渇が早いため）。
- 中断・再開の手順は SKILL.md の固定手順（`ls reports/` → `reports/_carryover.md` →
  `reports/_phase_a_decisions.json` → 着手フェーズのガイド）と同一。**判断ログの `[対話]`
  エントリと「対話の深さ」の確定値も `_carryover.md` から復元**する（SKILL_INTERACTIVE.md §7）。

## 4. 「対話フロント＋自動バック」運用（任意・Phase A/B を TUI → Phase C/D を `codex exec resume` で無人実行）

`exec_mode_addendum.md` の部分自動化（A/B 対話 → C/D 非対話）を KATHERINE と組み合わせる場合の
手順。非対話実行中は `ask_user_question` が使えないため、**Phase C/D で発動する対話ポイント
（CP-7 / CP-8）の委任を Phase B 終了時の TUI で明示的に確定してから**でなければ、この運用に
入ってはならない（委任なしで無人実行に入ると、I-2〈応答なし確定の禁止〉違反になる）。

1. Phase A〜B を TUI で対話進行し、CP-1〜CP-6（発動分）を確定する。
2. **Phase B 終了時の TUI で、次の 2 つの委任を「明示の選択肢」として確定する**
   （`dialogue_points.md` §確定の選択肢設計の一括委任規則: 委任の既定単位は確定 1 件分であり、
   複数章・フェーズ一括の委任は**選択肢にそのスコープを明示して分析者が選んだ場合のみ**成立する）:
   - **CP-7**: 「Phase C 全章（全モジュール）の統合インサイトを一括で AI に委任する」旨を明示した選択肢
   - **CP-8**: 「結論の確定と WARN トリアージを AI に委任する」旨を明示した選択肢
   - 確定したら判断ログに `[対話→委任]`（スコープ「CP-7 全章委任」「CP-8 委任」）を記録する。
   - この委任確認は Phase B のセッション・チェックポイントや Phase C 計画確認と**別の質問**として
     取る（§共通規則の抱き合わせ禁止）。
3. 委任の確定値を `reports/_carryover.md` に固定してから、`codex exec resume --last`
   （または `<SESSION_ID>`）で Phase C/D を非対話実行する（再開の詳細は `exec_mode_addendum.md`）。
4. **委任時も過程の透明化の義務は残る**（`dialogue_points.md` CP-7 / CP-8 のおまかせ規定）:
   - CP-7 全章委任時も、**各モジュール完了ごとの「突合サマリ」**（使用データ・突合の組み合わせ
     〈特許×特許／特許×NPL／NPL×NPL〉・一致点/矛盾点）は省略不可。非対話実行中は TUI 提示の
     代替として **`reports/_carryover.md` へ追記して残す**。
   - CP-8(b) を委任した場合、**「WARN を AI 判断で処理した」旨＋各 WARN の直す/残すの判断と理由**を
     `_carryover.md` の判断ログに記録し、**最終報告（`codex exec` の最終メッセージ）に明記**する。
   - FAIL は委任の有無にかかわらず AI 主導の強制修正ループ（分析者の同意でも緩和不可）。
5. SKILL_INTERACTIVE.md §8 の完了条件（発動 CP の判断ログ記録・内部ガイドファイル名 0 件の
   grep 確認・対話サマリの申し送り追記）は、無人実行区間にも同一に適用される。

この運用を選ばない場合（全フェーズ TUI）は本節は不要。フルパイプラインの TUI 実行が引き続き基本形。

## 5. 実機検証ステータス

**Codex CLI での KATHERINE 対話型モードの通し実行は未検証（2026-07 時点・ベストエフォート）。**
検証済みなのは Claude Code のみ。Codex 固有の未検証点: `ask_user_question` の選択肢数上限・
Other 自動付与の挙動・`multiSelect` の可否・`codex exec resume` と委任確定の組み合わせ。
実機で挙動差を確認した場合は、本補遺と SKILL_INTERACTIVE.md §2 の読替表（正本）の双方を
点検・更新すること。
