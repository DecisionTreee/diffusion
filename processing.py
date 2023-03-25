import os
from tqdm import tqdm
from PIL import Image

path = os.listdir("raw/test")
for i in tqdm(range(len(path))):
    img = Image.open(f"raw/test/{path[i]}").convert("RGB")
    _max = max(img.width, img.height)

    mask = Image.new("RGB", (_max, _max), (0, 0, 0, 0))

    x = (_max - img.width) // 2
    y = (_max - img.height) // 2
    mask.paste(img, (x, y))

    mask.save(f"raw/output/{i}.png", "PNG")
