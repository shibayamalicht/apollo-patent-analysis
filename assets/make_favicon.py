# APOLLO ファビコン生成スクリプト
# 軌道マーク: 円盤（月／地球）を軌道リングが横切り、探査機が1点。
# ブラウザタブの16-32px表示でも潰れないよう、太い線と高コントラストにしている。
# 再生成: python3 make_favicon.py
import math

from PIL import Image, ImageDraw

S = 512
SS = 4  # アンチエイリアス用のスーパーサンプリング倍率
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)

DISC = (20, 27, 35, 255)       # #141B23 グラファイトの円盤
IGNITION = (226, 88, 42, 255)  # #E2582A イグニッション
BG = (247, 249, 251, 255)      # #F7F9FB 背景（タブ上で輪郭を保つ）

cx = cy = S / 2
d.ellipse([0, 0, S, S], fill=BG)


def orbit_layer(front_only):
    """軌道リングを描いたレイヤを返す。front_only=True で下半分（円盤の手前）だけ。"""
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    rx, ry, w = S * 0.435, S * 0.187, S * 0.058
    box = [cx - rx, cy - ry, cx + rx, cy + ry]
    if front_only:
        ld.arc(box, 0, 180, fill=IGNITION, width=int(w))
    else:
        ld.ellipse(box, outline=IGNITION, width=int(w))
    return layer.rotate(28, resample=Image.BICUBIC, center=(cx, cy))


img.alpha_composite(orbit_layer(False))
r = S * 0.225
d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=DISC)
img.alpha_composite(orbit_layer(True))

# 探査機（軌道上の1点）
ang = math.radians(-50)
px = cx + S * 0.435 * math.cos(ang)
py = cy + S * 0.187 * math.sin(ang)
prx, pry = px - cx, py - cy
rot = math.radians(-28)
px = cx + prx * math.cos(rot) - pry * math.sin(rot)
py = cy + prx * math.sin(rot) + pry * math.cos(rot)
pr = S * 0.055
d.ellipse([px - pr, py - pr, px + pr, py + pr], fill=IGNITION)

img.resize((128, 128), Image.LANCZOS).save("favicon_128.png")  # サイドバー表示用
img.resize((64, 64), Image.LANCZOS).save("favicon.png")
img.resize((32, 32), Image.LANCZOS).save("favicon_32.png")
print("saved favicon_128.png / favicon.png / favicon_32.png")
