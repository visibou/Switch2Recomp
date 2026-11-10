"""A run's screenshots on one sheet: python tools/sheet.py <run folder> [columns] -> <run folder>/sheet.png"""
import glob, os, sys
from PIL import Image
run = sys.argv[1]; cols = int(sys.argv[2]) if len(sys.argv) > 2 else 3
fs = sorted(glob.glob(os.path.join(run, 'shot_screenshot_*.png'))) + glob.glob(os.path.join(run, 'shot_final.png'))
w, h = 640, 360
rows = (len(fs) + cols - 1) // cols
s = Image.new('RGB', (w * cols, h * rows))
for i, f in enumerate(fs):
    s.paste(Image.open(f).convert('RGB').resize((w, h), Image.BILINEAR), ((i % cols) * w, (i // cols) * h))
s.save(os.path.join(run, 'sheet.png'))
print(len(fs), 'shots')
