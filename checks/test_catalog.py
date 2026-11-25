"""Keep the 100-item index synchronized with local, readable project source."""

import ast
from pathlib import Path
import re
import unittest
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\]\(([^)]+)\)")
ENTRY = re.compile(r"^- (\d{3}) \[[^]]+\]\(([^)]+/README\.md)\)$", re.MULTILINE)
HEADINGS = ["1. 出处、提出者与意义", "2. 源码导航与直观解释", "3. 数学解释"]


class CatalogTests(unittest.TestCase):
    def test_one_hundred_complete_numbered_entries(self):
        entries = ENTRY.findall((ROOT / "README.md").read_text())
        self.assertEqual([number for number, _ in entries],
                         [f"{number:03d}" for number in range(1, 101)])
        self.assertEqual(len({path for _, path in entries}), 100)
        for number, link in entries:
            path = ROOT / unquote(link)
            self.assertTrue(path.is_file(), f"{number}: missing {path}")
            self.assertTrue(any(path.parent.glob("*.py")), f"{number}: no local source")

    def test_every_project_has_source_and_three_sections(self):
        projects = [folder for folder in ROOT.iterdir()
                    if folder.is_dir() and not folder.name.startswith(".")
                    and folder.name != "checks"]
        for folder in projects:
            with self.subTest(project=folder.name):
                readme = folder / "README.md"
                self.assertTrue(readme.is_file())
                headings = re.findall(r"^## (.+)$", readme.read_text(), re.MULTILINE)
                self.assertEqual(headings, HEADINGS)
                sources = list(folder.glob("*.py"))
                self.assertTrue(sources, "Model source must be local")
                for source in sources:
                    tree = ast.parse(source.read_text(), filename=str(source))
                    self.assertTrue(any(isinstance(node, (ast.ClassDef, ast.FunctionDef))
                                        for node in tree.body), f"{source} only imports code")
                    forbidden = {"torchvision", "transformers", "diffusers", "timm"}
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ImportFrom) and node.module:
                            self.assertNotIn(node.module.split(".")[0], forbidden, str(source))
                        elif isinstance(node, ast.Import):
                            for alias in node.names:
                                self.assertNotIn(alias.name.split(".")[0], forbidden, str(source))

    def test_all_local_readme_links(self):
        readmes = [ROOT / "README.md", *(folder / "README.md" for folder in ROOT.iterdir()
                                               if folder.is_dir() and (folder / "README.md").is_file())]
        for readme in readmes:
            for link in LINK.findall(readme.read_text()):
                if link.startswith(("https://", "http://", "mailto:", "#")):
                    continue
                target = readme.parent / unquote(link.split("#", 1)[0])
                with self.subTest(readme=readme.name, link=link):
                    self.assertTrue(target.exists(), f"Broken link in {readme}: {link}")


if __name__ == "__main__":
    unittest.main()
