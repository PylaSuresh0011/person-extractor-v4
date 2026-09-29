"""
gui.py
======
A small window for the full model: paste file or folder paths (one per
line), press Run, see the names. Same pipeline as cli.py --input-list;
everything stays on this machine.

    .venv\\Scripts\\python gui.py        (or double-click run_gui.bat)

Each run is also saved to output/gui/results_<time>.csv and .txt.
"""

from __future__ import annotations

import copy
import csv
import os
import queue
import sys
import threading
from datetime import datetime
from pathlib import Path

# pythonw has no console: give stray writes somewhere to go.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import BASE_DIR, DEFAULT_CONFIG  # noqa: E402
from src.candidate.boundary_refiner import refine_combined_counts  # noqa: E402
from src.io.dispatcher import ReaderDispatcher  # noqa: E402

SUPPORTED = set(ReaderDispatcher().supported_extensions())
OUTPUT_DIR = BASE_DIR / "output" / "gui"


def parse_paths(text: str) -> tuple[list[Path], list[str]]:
    """Files to run from pasted text: one path per line, quotes allowed
    (Explorer's "Copy as path"), blank and # lines ignored. A folder
    expands to every supported file in it. Returns (files, problems)."""
    files: list[Path] = []
    problems: list[str] = []
    for raw in text.splitlines():
        line = raw.strip().strip('"').strip("'").strip()
        if not line or line.startswith("#"):
            continue
        path = Path(line)
        if path.is_dir():
            found = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
            if not found:
                problems.append(f"No supported files in folder: {line}")
            files.extend(found)
        elif path.is_file():
            if path.suffix.lower() in SUPPORTED:
                files.append(path)
            else:
                problems.append(f"Unsupported file type: {line}")
        else:
            problems.append(f"Not found: {line}")
    seen: set[Path] = set()
    return [f for f in files if not (f in seen or seen.add(f))], problems


def combine(results: list[tuple[str, object]]) -> list[dict]:
    """Names across files, as cli.py run_batch combines them: occurrences
    summed, "accepted" if accepted in any file. Most frequent first."""
    combined: dict[str, dict] = {}
    for source, result in results:
        for decision, persons in (("accepted", result.persons), ("review", result.review_persons)):
            for p in persons:
                e = combined.setdefault(p.normalized_text, {"display": p.display_text, "occurrences": 0,
                                                            "files": [], "decision": "review"})
                e["occurrences"] += p.occurrence_count
                if source not in e["files"]:
                    e["files"].append(source)
                if decision == "accepted":
                    e["decision"] = "accepted"
    return sorted(refine_combined_counts(combined).values(), key=lambda e: (-e["occurrences"], e["display"]))


def save(rows: list[dict], files: list[Path], failed: list[str], out_dir: Path = OUTPUT_DIR) -> Path:
    """Writes results_<time>.csv and .txt; returns the .csv path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "total_occurrences", "decision", "source_files"])
        for e in rows:
            w.writerow([e["display"], e["occurrences"], e["decision"], "; ".join(e["files"])])
    width = max([len(e["display"]) for e in rows] + [4])
    lines = [f"PERSON_EXTRACTOR_V4 - {len(files)} file(s), {len(rows)} name(s) "
             f"({sum(e['decision'] == 'accepted' for e in rows)} accepted, "
             f"{sum(e['decision'] == 'review' for e in rows)} review)", "", "Files:"]
    lines += [f"  {p}" for p in files] + [f"  FAILED: {x}" for x in failed] + [""]
    lines.append(f"{'NAME':<{width}}  {'COUNT':>5}  DECISION  FILES")
    lines += [f"{e['display']:<{width}}  {e['occurrences']:>5}  {e['decision']:<8}  {'; '.join(e['files'])}" for e in rows]
    stem.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return stem.with_suffix(".csv")


class App:
    def __init__(self, root) -> None:
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.root = tk, root
        self.events: queue.Queue = queue.Queue()
        self.pipelines: dict[bool, object] = {}
        self.last_csv: Path | None = None

        root.title("Person Extractor")
        root.geometry("900x620")
        root.minsize(620, 420)
        frame = ttk.Frame(root, padding=12)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Paste file or folder paths, one per line:").pack(anchor="w")
        self.paths = tk.Text(frame, height=7, wrap="none", font=("Consolas", 10), undo=True)
        self.paths.pack(fill="x", pady=(4, 8))

        bar = ttk.Frame(frame)
        bar.pack(fill="x")
        self.run_btn = ttk.Button(bar, text="Run", command=self.start)
        self.run_btn.pack(side="left")
        self.fast = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="Fast (skip the LLM reviewer)", variable=self.fast).pack(side="left", padx=12)
        self.show_review = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Show review names", variable=self.show_review,
                        command=self.refresh).pack(side="left")
        self.open_btn = ttk.Button(bar, text="Open results folder", command=self.open_folder, state="disabled")
        self.open_btn.pack(side="right")

        self.progress = ttk.Progressbar(frame, mode="determinate")
        self.progress.pack(fill="x", pady=(8, 4))
        self.status = ttk.Label(frame, text="Ready.", justify="left")
        self.status.pack(anchor="w", fill="x")
        frame.bind("<Configure>", lambda e: self.status.config(wraplength=max(200, e.width - 24)))

        table = ttk.Frame(frame)
        table.pack(fill="both", expand=True, pady=(8, 0))
        cols = ("name", "decision", "count", "files")
        self.tree = ttk.Treeview(table, columns=cols, show="headings")
        for col, title, width, anchor in (("name", "Name", 240, "w"), ("decision", "Decision", 90, "w"),
                                          ("count", "Count", 70, "e"), ("files", "Files", 420, "w")):
            self.tree.heading(col, text=title)
            self.tree.column(col, width=width, anchor=anchor, stretch=(col in ("name", "files")))
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.tree.tag_configure("review", foreground="#6f6d67")
        self.rows: list[dict] = []
        root.after(100, self.poll)

    def start(self) -> None:
        files, problems = parse_paths(self.paths.get("1.0", "end"))
        if not files:
            self.status.config(text="Nothing to run. " + ("  ".join(problems) if problems else "Paste at least one path."))
            return
        self.run_btn.config(state="disabled")
        self.progress.config(maximum=len(files), value=0)
        note = f" ({len(problems)} path(s) skipped: {problems[0]}...)" if problems else ""
        self.status.config(text=f"Loading the model...{note}")
        threading.Thread(target=self.work, args=(files, self.fast.get()), daemon=True).start()

    def work(self, files: list[Path], fast: bool) -> None:
        try:
            from src.pipeline.orchestrator import Pipeline
            if fast not in self.pipelines:
                cfg = copy.deepcopy(DEFAULT_CONFIG)
                if fast:
                    cfg["llm_reviewer"]["enabled"] = False
                self.pipelines[fast] = Pipeline(config=cfg, base_dir=BASE_DIR)
            pipeline = self.pipelines[fast]
            results, failed = [], []
            for i, path in enumerate(files, 1):
                self.events.put(("progress", i - 1, f"Processing {i}/{len(files)}: {path.name}"))
                result = pipeline.run(str(path))
                if result.success:
                    results.append((path.name, result))
                else:
                    failed.append(f"{path.name}: {result.error}")
            rows = combine(results)
            csv_path = save(rows, files, failed)
            llm = "off (Fast mode)" if fast else pipeline.extractor.provenance.get("llm_reviewer", "off")
            self.events.put(("done", rows, csv_path, len(files), failed, llm))
        except Exception as exc:  # noqa: BLE001 - shown in the window, never a silent crash
            self.events.put(("error", str(exc)))

    def poll(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == "progress":
                    self.progress.config(value=event[1])
                    self.status.config(text=event[2])
                elif event[0] == "done":
                    _, self.rows, self.last_csv, n, failed, llm = event
                    self.progress.config(value=n)
                    acc = sum(e["decision"] == "accepted" for e in self.rows)
                    text = (f"Done: {n} file(s), {len(self.rows)} name(s) - {acc} accepted, "
                            f"{len(self.rows) - acc} review.  LLM reviewer: {llm}.\n"
                            f"Saved: {self.last_csv} (and .txt)")
                    if failed:
                        text += f"  Failed: {len(failed)} ({failed[0]})"
                    self.status.config(text=text)
                    self.run_btn.config(state="normal")
                    self.open_btn.config(state="normal")
                    self.refresh()
                elif event[0] == "error":
                    self.status.config(text=f"Error: {event[1]}")
                    self.run_btn.config(state="normal")
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def refresh(self) -> None:
        self.tree.delete(*self.tree.get_children())
        for e in self.rows:
            if e["decision"] == "review" and not self.show_review.get():
                continue
            self.tree.insert("", "end", values=(e["display"], e["decision"], e["occurrences"], "; ".join(e["files"])),
                             tags=(e["decision"],))

    def open_folder(self) -> None:
        if self.last_csv:
            os.startfile(self.last_csv.parent)  # noqa: S606 - opens the local results folder


def main() -> int:
    import tkinter as tk
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
