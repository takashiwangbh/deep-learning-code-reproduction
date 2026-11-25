"""在仓库根目录运行：python Deep-Learning-Sourcebook/check_all.py。"""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
CHECKS = [
    "foundations/test_foundations.py",
    "vision/test_models.py",
    "../Residual Network/ResNet-34 .py",
    "sequence/test_sequence.py",
    "generative/test_models.py",
    "modern/smoke_test.py",
    "reinforcement/smoke_test.py",
]


def main():
    for relative_path in CHECKS:
        script = ROOT / relative_path
        print(f"\n=== {relative_path} ===", flush=True)
        subprocess.run([sys.executable, str(script)], cwd=script.parent, check=True)
    print("\n所有源码检查通过。")


if __name__ == "__main__":
    main()
