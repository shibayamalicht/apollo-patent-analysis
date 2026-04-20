# Antigravity Review Policy 推奨設定

APOLLO CAPCOM レポートを Antigravity IDE で生成する際の **Review Policy** 推奨設定ガイドです。

---

## 🎯 なぜ Review Policy の設定が重要か

APOLLO CAPCOM は品質保証のため **5箇所のユーザー承認ゲート** を持ちます：

| Phase | Gate | 承認対象 |
|---|---|---|
| A-2 | タイトル選定 | `implementation_plan.md` § Title Candidates |
| B-1 | クロスパターン選定 | `implementation_plan.md` § Cross-Module Pattern Selection |
| B-2 | Web調査可否 | `task.md` § Phase B Gates (Web Research) |
| C | Deep Dive Plan | `implementation_plan.md` § Deep Dive Plan |
| D | Report Plan | `implementation_plan.md` § Report Structure & Quality Plan |

Antigravity の Review Policy が **"Always Proceed"** になっていると、Agent が Artifact を更新してもユーザーに承認待ちが表示されず、AI が自動で全フェーズを進めてしまいます。これは SKILL.md §0 絶対遵守ゲートルール第2項（ユーザー応答待ち必須）に違反します。

---

## ✅ 推奨設定

### オプション A: 全 Phase で "Request Review" （最も安全、推奨）

**設定手順**:
1. Antigravity IDE の設定パネル → Agent Manager → Review Policy
2. **"Request Review"** を選択
3. （推奨）本 skill `apollo-capcom` に対する個別オーバーライドも "Request Review" に設定

**メリット**:
- 5箇所全ての Artifact Review でユーザー承認待ちが自動発動
- ユーザーは Google Docs 式コメントで修正指示可能
- SKILL.md のゲートが確実に機能

**デメリット**:
- 自動進行しないため、ユーザーが明示的に承認する必要がある
- 承認作業のコストが発生（ただし5箇所のみ、本来必要な判断）

### オプション B: "Agent Decides" （速度重視、ただし Agent が適切に判断する前提）

**設定手順**: Review Policy を **"Agent Decides"** に設定

**メリット**:
- Agent が重要判断で自発的にユーザー承認を要求
- 速度と品質のバランス

**デメリット**:
- Agent の「重要判断」の自発性に依存
- 本スキルの SKILL.md には `<!-- ANTIGRAVITY_REVIEW_REQUIRED -->` コメントを埋め込んで誘導していますが、 Agent がこれを見落とす可能性あり
- ゲートがスキップされると品質が大きく低下するリスク

### オプション C: "Always Proceed" （**非推奨**、本スキルでは動作しない）

- 全 Artifact がユーザー承認なしで進行
- **SKILL.md ゲートが機能せず、Agent が自動でタイトル決定・クロスパターン選定・Web調査可否判断を全て行ってしまう**
- 品質保証が崩壊するため、**本スキルでは使わないでください**

---

## 🔧 Antigravity UI での設定手順

### 設定画面までの導線

1. Antigravity IDE を起動
2. 上部メニュー: `Settings` → `Agent Manager` → `Review Policy`
3. または コマンドパレット（`Ctrl/Cmd+Shift+P`）→ 「Review Policy」で検索

### プロジェクト単位での設定

本 CAPCOM セッションでのみ "Request Review" にしたい場合：
1. 設定画面で「Project-specific override」を有効化
2. 現在のワークスペース（`session_*/`）を選択
3. Review Policy を "Request Review" に設定

### スキル単位での設定（推奨）

`apollo-capcom` スキルに対してのみ "Request Review" を適用：
1. 設定画面で「Skill-specific overrides」を開く
2. `apollo-capcom` を選択
3. Review Policy を "Request Review" に設定

この方法なら、他のスキル（例: コード生成用の軽量スキル）は "Always Proceed" のまま使えます。

---

## 🛠 Agent 側の Review 誘導メカニズム

本スキルは Artifact Review の発動を促すため、以下の仕組みを埋め込んでいます：

### 1. `<!-- ANTIGRAVITY_REVIEW_REQUIRED -->` マーカー

各 STOP-GATE セクションの直前に HTML コメントマーカーを埋め込み、Agent が自発的に Review を要求するよう誘導します。

例（`implementation_plan.md` 内）:
```markdown
## 🛑 Phase A-2 Gate: Title Candidates

<!-- ANTIGRAVITY_REVIEW_REQUIRED -->

> **ユーザー承認対象**。以下3案から1つを選んで ✅ を付けるか、Other でカスタム指定してください。

- [ ] **案1**: ...
```

### 2. 明示的な Artifact 更新

Agent は `implementation_plan.md` を編集するたびに Antigravity の Artifact システムに変更通知を送ります。Review Policy = "Request Review" 下では、この通知に対してユーザー承認待ちが自動発動します。

### 3. チェックボックス依存

`task.md` の各 Phase チェックボックスは、ユーザー判断が必要なものとAgent自動のものを明確に分けています：

- **ユーザー判断項目**（Agent は `[x]` を付けない）:
  - 「Web調査を実施する/スキップする/修正したい」のいずれか
- **Agent自動項目**:
  - 「patents.csv 統計把握」「AI insights 3件読了」等

Agent はユーザー判断項目を自発的にチェックしてはいけません。

---

## 📊 Review Policy 別の動作比較

| 設定 | タイトル選定 | クロス選定 | Web調査 | Deep Dive Plan | Report Plan | 品質 |
|---|---|---|---|---|---|---|
| **Request Review** | ✅ 承認待ち | ✅ 承認待ち | ✅ 承認待ち | ✅ 承認待ち | ✅ 承認待ち | ⭐⭐⭐⭐⭐ |
| **Agent Decides** | ⚠️ Agent次第 | ⚠️ Agent次第 | ⚠️ Agent次第 | ⚠️ Agent次第 | ⚠️ Agent次第 | ⭐⭐⭐ |
| **Always Proceed** | ❌ AI自動決定 | ❌ AI自動決定 | ❌ AI自動決定 | ❌ AI自動決定 | ❌ AI自動決定 | ⭐ |

---

## 🧪 設定確認方法

本スキル起動後、Agent が最初に以下を報告することを確認してください：

```
✅ Review Policy 確認完了: Request Review
   → 5箇所の Artifact Review GATE でユーザー承認待ちが自動発動します
```

もし以下のような警告が出たら、設定変更が必要：

```
⚠️ Review Policy が "Always Proceed" です。
   本スキルのゲートが機能しません。Review Policy を "Request Review" または
   "Agent Decides" に変更してください。
   設定方法は review_policy_recommendation.md 参照。
```

---

## 🐛 トラブルシューティング

### Q. Review Policy を "Request Review" にしたのに承認待ちが発動しない

**A**. 以下を確認:
1. Antigravity のバージョンが v1.20.0 以上か（旧バージョンはバグあり）
2. `apollo-capcom` スキルに skill-specific override が別設定されていないか
3. `implementation_plan.md` を編集する際、Agent が Artifact として登録しているか（コメント `<!-- ANTIGRAVITY_ARTIFACT -->` が先頭にあるか）

### Q. 承認待ちが出すぎて進まない

**A**. 本スキルの承認ポイントは5箇所のみ（全 Phase で）。それ以上に発動している場合、Agent が余計な Artifact を作っている可能性。チャットで「Phase X の承認だけにして」と指示してください。

### Q. 承認した Artifact がAgentに伝わらない

**A**. Antigravity の Artifact システムが時々同期遅延します。`Ctrl/Cmd+S` で保存を明示的に行うと同期が促進されます。

---

## 📚 参考リンク

- [Antigravity 公式: Review Policy](https://antigravity.google/docs/agent/review-policy)
- [Antigravity 公式: Artifacts](https://antigravity.google/docs/implementation-plan)
- 本パッチの親 README: [`../README.md`](../README.md)
- スキル本体: [`.agent/skills/apollo-capcom/SKILL.md`](.agent/skills/apollo-capcom/SKILL.md)
