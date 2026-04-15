#!/usr/bin/env python3
"""
APOLLO CAPCOM テンプレートPPTX生成スクリプト

テーマカラー・デフォルトフォントを設定したテンプレートPPTXを生成する。
python-pptxではスライドレイアウトの新規作成は不可のため、
テーマ/色/フォント設定のみをテンプレートに含め、
実際のレイアウトはgenerate_ppt.pyでプログラマティックに構築する。

使い方:
    python create_ppt_template.py

出力:
    capcom_schema/templates/apollo_template.pptx
"""

import os
import sys
from lxml import etree

# python-pptxインポート
try:
    from pptx import Presentation
    from pptx.util import Inches
except ImportError:
    print("エラー: python-pptx がインストールされていません")
    print("  pip install python-pptx")
    sys.exit(1)


# === カラーパレット定義 ===
COLORS = {
    "navy":        "1B2A4A",
    "blue":        "2E5090",
    "accent":      "3B7DD8",
    "dark_gray":   "333333",
    "medium_gray": "666666",
    "light_gray":  "F5F5F5",
    "white":       "FFFFFF",
    "red_accent":  "D64545",
    "green_accent": "2E8B57",
    "amber":       "D4A017",
}

# === 日本語フォント設定 ===
FONT_JAPANESE = "Hiragino Sans"  # macOS標準 → Windows: メイリオ, Linux: IPAゴシック にフォールバック

# XML名前空間
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
NSMAP = {"a": A_NS}


def get_theme_part(prs):
    """スライドマスターからテーマパーツを取得する"""
    sm = prs.slide_masters[0]
    for rel in sm.part.rels.values():
        if "theme" in rel.reltype:
            return rel.target_part
    return None


def modify_theme_xml(theme_blob):
    """テーマXMLのblob（bytes）を解析・変更して返す"""
    root = etree.fromstring(theme_blob)

    # テーマ名を変更
    root.set("name", "APOLLO CAPCOM")

    # === カラースキーム設定 ===
    clr_scheme = root.find(f".//{{{A_NS}}}clrScheme")
    if clr_scheme is not None:
        clr_scheme.set("name", "APOLLO CAPCOM")

        # テーマカラーマッピング（Office スロット → APOLLO カラー）
        color_mapping = {
            "dk1":     COLORS["navy"],        # Dark 1 → Navy（主要テキスト色）
            "dk2":     COLORS["dark_gray"],    # Dark 2 → Dark Gray
            "lt1":     COLORS["white"],        # Light 1 → White
            "lt2":     COLORS["light_gray"],   # Light 2 → Light Gray
            "accent1": COLORS["accent"],       # Accent 1 → Accent Blue
            "accent2": COLORS["blue"],         # Accent 2 → Blue
            "accent3": COLORS["green_accent"], # Accent 3 → Green
            "accent4": COLORS["amber"],        # Accent 4 → Amber
            "accent5": COLORS["red_accent"],   # Accent 5 → Red
            "accent6": COLORS["medium_gray"],  # Accent 6 → Medium Gray
            "hlink":   COLORS["accent"],       # Hyperlink → Accent
            "folHlink": COLORS["medium_gray"], # Followed Hyperlink → Medium Gray
        }

        for slot_name, hex_color in color_mapping.items():
            slot = clr_scheme.find(f"{{{A_NS}}}{slot_name}")
            if slot is not None:
                # 既存の子要素を削除
                for child in list(slot):
                    slot.remove(child)
                # srgbClr要素を追加
                srgb = etree.SubElement(slot, f"{{{A_NS}}}srgbClr")
                srgb.set("val", hex_color)

        print("  テーマカラー設定完了")
    else:
        print("  警告: clrScheme要素が見つかりません")

    # === フォントスキーム設定 ===
    font_scheme = root.find(f".//{{{A_NS}}}fontScheme")
    if font_scheme is not None:
        font_scheme.set("name", "APOLLO CAPCOM")

        for font_type in ["majorFont", "minorFont"]:
            font_elem = font_scheme.find(f"{{{A_NS}}}{font_type}")
            if font_elem is None:
                continue

            # Latin フォント
            latin = font_elem.find(f"{{{A_NS}}}latin")
            if latin is not None:
                latin.set("typeface", FONT_JAPANESE)

            # East Asian フォント
            ea = font_elem.find(f"{{{A_NS}}}ea")
            if ea is not None:
                ea.set("typeface", FONT_JAPANESE)
            else:
                ea = etree.SubElement(font_elem, f"{{{A_NS}}}ea")
                ea.set("typeface", FONT_JAPANESE)

            # Jpan スクリプト用フォント
            found_jpan = False
            for font_child in font_elem.findall(f"{{{A_NS}}}font"):
                if font_child.get("script") == "Jpan":
                    font_child.set("typeface", FONT_JAPANESE)
                    found_jpan = True
                    break
            if not found_jpan:
                jpan_font = etree.SubElement(font_elem, f"{{{A_NS}}}font")
                jpan_font.set("script", "Jpan")
                jpan_font.set("typeface", FONT_JAPANESE)

        print("  テーマフォント設定完了")
    else:
        print("  警告: fontScheme要素が見つかりません")

    # 変更後のXMLをbytesとして返す
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def create_template():
    """テンプレートPPTXを生成する"""
    print("APOLLO CAPCOM テンプレートPPTX生成中...")

    # 新規プレゼンテーション作成
    prs = Presentation()

    # スライドサイズ: 16:9 ワイドスクリーン
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    print("  スライドサイズ: 13.333 x 7.5 インチ (16:9)")

    # テーマパーツを取得
    theme_part = get_theme_part(prs)
    if theme_part is None:
        print("エラー: テーマパーツが見つかりません")
        sys.exit(1)

    # テーマXMLを変更
    modified_xml = modify_theme_xml(theme_part.blob)

    # 変更後のXMLをテーマパーツに書き戻す
    theme_part._blob = modified_xml
    print("  テーマXML書き戻し完了")

    # 出力先
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "apollo_template.pptx")
    prs.save(output_path)

    print(f"\n生成完了: {output_path}")
    print(f"ファイルサイズ: {os.path.getsize(output_path):,} bytes")

    # 検証: 再読み込みして確認
    print("\n検証中...")
    prs2 = Presentation(output_path)
    print(f"  スライドサイズ: {prs2.slide_width} x {prs2.slide_height}")
    print(f"  スライドレイアウト数: {len(prs2.slide_layouts)}")
    print(f"  ブランクレイアウト(index 6): {prs2.slide_layouts[6].name}")

    # テーマの検証
    theme_part2 = get_theme_part(prs2)
    if theme_part2:
        root = etree.fromstring(theme_part2.blob)
        theme_name = root.get("name", "不明")
        clr = root.find(f".//{{{A_NS}}}clrScheme")
        clr_name = clr.get("name", "不明") if clr is not None else "不明"
        font = root.find(f".//{{{A_NS}}}fontScheme")
        font_name = font.get("name", "不明") if font is not None else "不明"
        print(f"  テーマ名: {theme_name}")
        print(f"  カラースキーム: {clr_name}")
        print(f"  フォントスキーム: {font_name}")

    print("\n✅ テンプレート生成・検証完了")
    return output_path


if __name__ == "__main__":
    create_template()
