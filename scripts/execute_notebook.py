"""Execute every cell with the current Python, saving only real outputs."""
import json
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parent.parent
kernel_root = ROOT / "tmp" / "kernels"
kernel_dir = kernel_root / "rice-python"
kernel_dir.mkdir(parents=True, exist_ok=True)
(kernel_dir / "kernel.json").write_text(json.dumps({
    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "Rice project Python", "language": "python",
}), encoding="utf-8")
os.environ["IPYTHONDIR"] = str(ROOT / "tmp" / "ipython")
path = ROOT / "notebooks" / "rice_leaf_classification.ipynb"
notebook = nbformat.read(path, as_version=4)
manager = KernelManager(kernel_name="rice-python", kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]))
client = NotebookClient(notebook, km=manager, timeout=1800, resources={"metadata": {"path": str(ROOT)}})
client.execute()
nbformat.write(notebook, path)
print("Notebook executed successfully; all cells completed with real outputs.")
