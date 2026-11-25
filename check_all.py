"""在仓库根目录运行：python check_all.py。"""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
CHECKS = [
    "checks/test_foundations.py",
    "checks/test_vision.py",
    "Residual Network/ResNet-34 .py",
    "checks/test_sequence.py",
    "checks/test_generative.py",
    "checks/test_modern.py",
    "checks/test_reinforcement.py",
    "checks/test_expansion_001_035.py",
    "checks/test_expansion_036_060.py",
    "checks/test_expansion_061_078.py",
    "checks/test_expansion_079_100.py",
    "checks/test_catalog.py",
]


def main():
    for relative_path in CHECKS:
        script = ROOT / relative_path
        print(f"\n=== {relative_path} ===", flush=True)
        subprocess.run([sys.executable, str(script)], cwd=script.parent, check=True)
    print("\n所有源码检查通过。")


if __name__ == "__main__":
    main()
