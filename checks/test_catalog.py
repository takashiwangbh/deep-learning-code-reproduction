"""Keep the category index synchronized with every local project folder."""

import ast
from pathlib import Path
import re
import unittest
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\]\(([^)]+)\)")
PROJECT_LINK = re.compile(r"^- \[[^]]+\]\(([^)]+)/\)$", re.MULTILINE)
HEADINGS = ["1. 出处、提出者与意义", "2. 源码导航与直观解释", "3. 数学解释"]


class CatalogTests(unittest.TestCase):
    def test_every_project_has_one_folder_link(self):
        home = (ROOT / "README.md").read_text()
        links = [unquote(link) for link in PROJECT_LINK.findall(home)]
        self.assertEqual(len(links), len(LINK.findall(home)),
                         "Home navigation links should point to folders, not README files")
        self.assertEqual(len(links), len(set(links)), "Duplicate folder link")
        folders = {folder.name for folder in ROOT.iterdir()
                   if folder.is_dir() and not folder.name.startswith(".")
                   and folder.name != "checks"}
        self.assertGreaterEqual(len(folders), 100)
        self.assertEqual(set(links), folders, "Missing or extra home navigation links")
        self.assertNotRegex(home, re.compile(r"^##\s*\d|^-\s*\d{3}\b", re.MULTILINE))
        self.assertNotIn("补充项目", home)
        for link in links:
            project = ROOT / link
            self.assertTrue((project / "README.md").is_file(), f"Missing project README: {link}")
            self.assertTrue(any(project.glob("*.py")), f"Missing local source: {link}")

    def test_every_project_has_source_and_three_sections(self):
        projects = [folder for folder in ROOT.iterdir()
                    if folder.is_dir() and not folder.name.startswith(".")
                    and folder.name != "checks"]
        for folder in projects:
            with self.subTest(project=folder.name):
                readme = folder / "README.md"
                self.assertTrue(readme.is_file())
                content = readme.read_text()
                headings = re.findall(r"^## (.+)$", content, re.MULTILINE)
                self.assertEqual(headings, HEADINGS)
                origin_section = content.split("## 2.", 1)[0]
                self.assertRegex(origin_section, r"https?://", "Project origin needs a source link")
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
