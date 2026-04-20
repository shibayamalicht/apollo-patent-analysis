#!/usr/bin/env bash
# APOLLO CAPCOM Antigravity パッチ適用スクリプト
# 使い方:
#   bash apply_patch.sh <session_dir>
#   例) bash /path/to/apollo_v7/capcom_schema_patches/antigravity/apply_patch.sh ~/Downloads/session_20260416_143022
#
# 本スクリプトは以下のファイルを <session_dir> に追加します:
#   - .agent/skills/apollo-capcom/SKILL.md
#   - .agent/workflows/*.md (6ファイル)
#   - artifacts_templates/*.tmpl (3ファイル)
#   - GEMINI.md
#   - AGENTS.md
#   - review_policy_recommendation.md
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

echo "=== APOLLO CAPCOM Antigravity パッチ適用 ==="
echo "パッチ元: $PATCH_DIR"
echo "適用先:   $SESSION_DIR"
echo ""

# 1. .agent/skills/apollo-capcom/ をコピー
echo "1/5: .agent/skills/apollo-capcom/ をコピー..."
mkdir -p "$SESSION_DIR/.agent/skills/apollo-capcom"
cp "$PATCH_DIR/.agent/skills/apollo-capcom/SKILL.md" \
   "$SESSION_DIR/.agent/skills/apollo-capcom/SKILL.md"
echo "   ✅ SKILL.md"

# 2. .agent/workflows/ をコピー
echo ""
echo "2/5: .agent/workflows/ をコピー..."
mkdir -p "$SESSION_DIR/.agent/workflows"
cp "$PATCH_DIR/.agent/workflows/"*.md "$SESSION_DIR/.agent/workflows/"
echo "   ✅ 6ファイル (00_capcom_master.md, 01-05_phase_*.md)"

# 3. artifacts_templates/ をコピー
echo ""
echo "3/5: artifacts_templates/ をコピー..."
mkdir -p "$SESSION_DIR/artifacts_templates"
cp "$PATCH_DIR/artifacts_templates/"*.tmpl "$SESSION_DIR/artifacts_templates/"
echo "   ✅ 3ファイル (task / implementation_plan / walkthrough の雛形)"

# 4. GEMINI.md と AGENTS.md を配置
echo ""
echo "4/5: GEMINI.md / AGENTS.md を配置..."

for f in GEMINI.md AGENTS.md; do
  if [ -f "$SESSION_DIR/$f" ]; then
    echo "   ⚠️  $SESSION_DIR/$f は既に存在します"
    echo "   既存ファイルを上書きしますか? [y/N]"
    read -r ans
    case "$ans" in
      y|Y|yes|Yes|YES)
        cp "$PATCH_DIR/$f" "$SESSION_DIR/$f"
        echo "   ✅ $f 上書き完了"
        ;;
      *)
        echo "   ⏭️  $f スキップ（既存ファイル保持）"
        ;;
    esac
  else
    cp "$PATCH_DIR/$f" "$SESSION_DIR/$f"
    echo "   ✅ $f 配置完了"
  fi
done

# 5. review_policy_recommendation.md を配置
echo ""
echo "5/5: review_policy_recommendation.md を配置..."
cp "$PATCH_DIR/review_policy_recommendation.md" "$SESSION_DIR/review_policy_recommendation.md"
echo "   ✅ 配置完了"

# 検証
echo ""
echo "=== 検証 ==="
added_files=(
  ".agent/skills/apollo-capcom/SKILL.md"
  ".agent/workflows/00_capcom_master.md"
  ".agent/workflows/01_phase_a_data_intake.md"
  ".agent/workflows/02_phase_a2_title_selection.md"
  ".agent/workflows/03_phase_b_evidence_cross.md"
  ".agent/workflows/04_phase_c_deep_dive.md"
  ".agent/workflows/05_phase_d_integration.md"
  "artifacts_templates/task.md.tmpl"
  "artifacts_templates/implementation_plan.md.tmpl"
  "artifacts_templates/walkthrough.md.tmpl"
  "GEMINI.md"
  "AGENTS.md"
  "review_policy_recommendation.md"
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
  echo "  1. Antigravity IDE を起動"
  echo "  2. ファイル → フォルダを開く → \"$SESSION_DIR\" を選択"
  echo "  3. Settings → Agent Manager → Review Policy を 'Request Review' に設定"
  echo "     (詳細: $SESSION_DIR/review_policy_recommendation.md)"
  echo "  4. チャットで「apollo-capcom スキルでレポート生成」と依頼"
  echo ""
  echo "ℹ️  注意: Review Policy = 'Always Proceed' だと本スキルのゲートが機能しません"
  exit 0
else
  echo "🛑 一部ファイルが欠落しています。パッチ元ディレクトリを確認してください。"
  exit 1
fi
