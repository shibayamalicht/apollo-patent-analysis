#!/usr/bin/env bash
# APOLLO CAPCOM Codex パッチ適用スクリプト
# 使い方:
#   bash apply_patch.sh <session_dir>
#   例) bash /path/to/apollo_v7/capcom_schema_patches/codex/apply_patch.sh ~/Downloads/session_20260416_143022
#
# 本スクリプトは以下のファイルを <session_dir> に追加します:
#   - .codex/skills/apollo-capcom/SKILL.md
#   - .codex/skills/apollo-capcom/prompts/*.md (5ファイル)
#   - AGENTS.md
#   - exec_mode_addendum.md
#
# 既存の capcom_schema/ には一切手を加えません。

set -o pipefail

# パッチ元ディレクトリ（このスクリプトが置かれているディレクトリ）
PATCH_DIR="$(cd "$(dirname "$0")" && pwd)"

# 引数チェック
if [ $# -lt 1 ]; then
  echo "使い方: bash $0 <session_dir>"
  echo "  <session_dir>: 展開済みの CAPCOM セッションフォルダ (session_YYYYMMDD_HHMMSS)"
  exit 1
fi

SESSION_DIR="$1"

if [ ! -d "$SESSION_DIR" ]; then
  echo "❌ エラー: $SESSION_DIR はディレクトリではありません"
  exit 1
fi

if [ ! -d "$SESSION_DIR/capcom_schema" ]; then
  echo "❌ エラー: $SESSION_DIR/capcom_schema が見つかりません"
  echo "   CAPCOM ZIP を展開したフォルダを指定してください"
  exit 1
fi

if [ ! -f "$SESSION_DIR/capcom_schema/SKILL.md" ]; then
  echo "❌ エラー: $SESSION_DIR/capcom_schema/SKILL.md が見つかりません"
  echo "   セッションフォルダの構造が正しくありません"
  exit 1
fi

echo "=== APOLLO CAPCOM Codex パッチ適用 ==="
echo "パッチ元: $PATCH_DIR"
echo "適用先:   $SESSION_DIR"
echo ""

# 1. .codex/skills/apollo-capcom/ をコピー
echo "1/3: .codex/skills/apollo-capcom/ をコピー..."
mkdir -p "$SESSION_DIR/.codex/skills/apollo-capcom/prompts"
cp "$PATCH_DIR/.codex/skills/apollo-capcom/SKILL.md" \
   "$SESSION_DIR/.codex/skills/apollo-capcom/SKILL.md"
cp "$PATCH_DIR/.codex/skills/apollo-capcom/prompts/"*.md \
   "$SESSION_DIR/.codex/skills/apollo-capcom/prompts/"
echo "   ✅ SKILL.md + prompts/ 5ファイル"

# 2. AGENTS.md を配置（既存があれば上書き確認）
echo ""
echo "2/3: AGENTS.md を配置..."
if [ -f "$SESSION_DIR/AGENTS.md" ]; then
  echo "   ⚠️  $SESSION_DIR/AGENTS.md は既に存在します"
  echo "   既存ファイルを上書きしますか? [y/N]"
  read -r ans
  case "$ans" in
    y|Y|yes|Yes|YES)
      cp "$PATCH_DIR/AGENTS.md" "$SESSION_DIR/AGENTS.md"
      echo "   ✅ 上書きしました"
      ;;
    *)
      echo "   ⏭️  スキップしました（既存 AGENTS.md を保持）"
      ;;
  esac
else
  cp "$PATCH_DIR/AGENTS.md" "$SESSION_DIR/AGENTS.md"
  echo "   ✅ 配置完了"
fi

# 3. exec_mode_addendum.md を配置
echo ""
echo "3/3: exec_mode_addendum.md を配置..."
cp "$PATCH_DIR/exec_mode_addendum.md" "$SESSION_DIR/exec_mode_addendum.md"
echo "   ✅ 配置完了"

# 検証
echo ""
echo "=== 検証 ==="
added_files=(
  ".codex/skills/apollo-capcom/SKILL.md"
  ".codex/skills/apollo-capcom/prompts/phase_a2_titles.md"
  ".codex/skills/apollo-capcom/prompts/phase_b_cross.md"
  ".codex/skills/apollo-capcom/prompts/phase_b_webresearch.md"
  ".codex/skills/apollo-capcom/prompts/phase_c_plan.md"
  ".codex/skills/apollo-capcom/prompts/phase_d_plan.md"
  "AGENTS.md"
  "exec_mode_addendum.md"
)

missing=0
for f in "${added_files[@]}"; do
  if [ -f "$SESSION_DIR/$f" ]; then
    echo "  ✅ $f"
  else
    echo "  ❌ $f (missing)"
    missing=1
  fi
done

echo ""
if [ $missing -eq 0 ]; then
  echo "🎉 パッチ適用完了！"
  echo ""
  echo "次のステップ:"
  echo "  cd \"$SESSION_DIR\""
  echo "  codex                              # Codex CLI (TUIモード) を起動"
  echo "  > \$apollo-capcom レポートを書いてください"
  echo ""
  echo "⚠️  注意: 本スキルは対話モード必須です。codex exec では動きません (exec_mode_addendum.md 参照)"
  exit 0
else
  echo "🛑 一部ファイルが欠落しています。パッチ元ディレクトリを確認してください。"
  exit 1
fi
