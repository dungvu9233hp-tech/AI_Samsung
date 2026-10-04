"""Tạo cấu trúc thư mục dự án. Chạy: python init_structure.py"""
import os
DIRS = ["src", "training", "eval/results/errors", "eval/cache", "tools", "models", "assets",
        "data/raw", "data/own", "data/processed", "logs", "docs"]
for d in DIRS:
    os.makedirs(d, exist_ok=True)
    if d.startswith(("data", "logs", "models", "eval/cache", "eval/results")):
        open(os.path.join(d, ".gitkeep"), "a").close()
for f in ["src/__init__.py"]:
    open(f, "a").close()
print("Đã tạo cấu trúc thư mục.")
