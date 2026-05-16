from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

NB_PATH = Path(__file__).resolve().parent.parent / "src" / "notebooks" / "notebook.ipynb"


def main():
    nb = nbformat.read(NB_PATH, as_version=4)

    client = NotebookClient(
        nb,
        timeout=14400,
        kernel_name="python3",
        resources={"metadata": {"path": str(NB_PATH.parent)}},
    )
    client.allow_errors = False
    client.create_kernel_manager()
    client.start_new_kernel()
    client.start_new_kernel_client()

    try:
        total = len(nb.cells)
        for idx, cell in enumerate(nb.cells):
            kind = cell.cell_type
            label = f"[cell {idx}/{total - 1}] ({kind})"
            src_preview = (cell.get("source") or "").splitlines()
            preview = src_preview[0][:80] if src_preview else "<empty>"
            print(f"{label} {preview}", flush=True)
            if kind != "code":
                continue
            start = time.time()
            try:
                client.execute_cell(cell, idx)
            except CellExecutionError as exc:
                elapsed = time.time() - start
                print(f"  ERROR after {elapsed:.1f}s: {exc.ename}: {exc.evalue}", flush=True)
                # save what we have so far so the notebook reflects partial state
                nbformat.write(nb, NB_PATH)
                raise
            except Exception:
                elapsed = time.time() - start
                print(f"  KERNEL DIED after {elapsed:.1f}s", flush=True)
                traceback.print_exc()
                nbformat.write(nb, NB_PATH)
                raise
            elapsed = time.time() - start
            print(f"  done in {elapsed:.1f}s", flush=True)
            # periodic save
            if idx % 3 == 0:
                nbformat.write(nb, NB_PATH)
    finally:
        nbformat.write(nb, NB_PATH)
        try:
            client.cleanup_kernel()
        except Exception:
            pass

    print("[run_notebook] all cells executed", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
