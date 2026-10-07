"""
gui.py
======
A small window for the full model: paste file or folder paths (one per
line), press Run, see the names. Same pipeline as cli.py --input-list;
everything stays on this machine.

    .venv\\Scripts\\python gui.py        (or double-click run_gui.bat)

Every run is stored on this machine - see save_run(): its own folder under
output/gui/runs/ (names, a readable report with date, start and end time
and time taken, and each file's seconds), plus a line in
output/gui/run_history.csv.
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
from src.candidate.alias_resolution import place, resolve_aliases  # noqa: E402
from src.candidate.boundary_refiner import refine_combined_counts  # noqa: E402
from src.decision.feature_score import entity_reason  # noqa: E402
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


def _add_score_and_reason(entry: dict, person) -> None:
    """Combined row: the highest feature score over files, and the reason
    from the first file the name was found in (src/decision/feature_score.py)."""
    score = getattr(person, "score", None)
    if score is not None and (entry.get("score") is None or score > entry["score"]):
        entry["score"] = score
    if not entry.get("entity_reason") and getattr(person, "mentions", None):
        entry["entity_reason"] = entity_reason(person.mentions)


def combine(results: list[tuple[str, object]], merge_aliases: bool = False) -> list[dict]:
    """Names across files, as cli.py run_batch combines them: occurrences
    summed, "accepted" if accepted in any file. Most frequent first. Each
    row lists the other forms of its person under "aliases"; merge_aliases
    makes them one row (src/candidate/alias_resolution.py)."""
    combined: dict[str, dict] = {}
    for source, result in results:
        for decision, persons in (("accepted", result.persons), ("review", result.review_persons)):
            for p in persons:
                e = combined.setdefault(p.normalized_text, {"display": p.display_text, "occurrences": 0,
                                                            "files": [], "decision": "review"})
                e["occurrences"] += p.occurrence_count
                e.setdefault("places", set()).update(place(source, loc) for loc in getattr(p, "locations", ()))
                _add_score_and_reason(e, p)
                if source not in e["files"]:
                    e["files"].append(source)
                if decision == "accepted":
                    e["decision"] = "accepted"
    # "Kalam", "Abdul Kalam", "Dr. Kalam" are one person: each row lists the
    # others under aliases (src/candidate/alias_resolution.py).
    explicit = [pair for _, result in results for pair in getattr(result, "aliases", ())]
    return resolve_aliases(list(refine_combined_counts(combined).values()), explicit, merge=merge_aliases)


HISTORY_COLUMNS = ["run_folder", "date", "start_time", "end_time", "time_taken_seconds", "time_taken",
                   "files", "files_failed", "names", "accepted", "review", "mode", "llm_reviewer", "input_paths"]
# Run data + extraction data in one row per name - each run's results.csv,
# and every run appended to output/gui/all_runs_with_names.csv.
RUN_COLUMNS = ["run_folder", "date", "start_time", "end_time", "time_taken_seconds", "time_taken", "mode",
               "llm_reviewer", "files"]
NAME_COLUMNS = ["name", "total_occurrences", "decision", "source_files", "aliases", "score", "entity_reason"]
ALL_RUNS_FILE = "all_runs_with_names.csv"


def _run_name_rows(run_values: list, rows: list[dict]) -> list[list]:
    """One row per name, each carrying its run's data; a run with no names
    still gets one row (blank name), so every run appears."""
    if not rows:
        return [run_values + [""] * len(NAME_COLUMNS)]
    return [run_values + [e["display"], e["occurrences"], e["decision"], "; ".join(e["files"]),
                          "; ".join(e.get("aliases", [])), _score_text(e.get("score")), e.get("entity_reason", "")]
            for e in rows]


def _score_text(score) -> str:
    return "" if score is None else f"{score:.2f}"


def _start_new_if_columns_changed(path: Path, header: list[str]) -> None:
    """An all-runs file written before columns were added keeps its rows
    under its own header: it is moved aside (same folder, dated name), and
    a new file starts."""
    if not path.exists():
        return
    with path.open(encoding="utf-8", newline="") as f:
        first = next(csv.reader(f), None)
    if first != header:
        path.rename(path.with_name(f"{path.stem}_before_{datetime.now():%Y-%m-%d_%H-%M-%S}{path.suffix}"))


def format_duration(seconds: float) -> str:
    """12.4s / 3m 07s / 1h 02m 05s"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def save_run(rows: list[dict], file_rows: list[dict], started: datetime, finished: datetime,
             mode: str, llm: str, inputs: list[str], out_dir: Path = OUTPUT_DIR) -> Path:
    """Stores one run on this machine and returns its folder:

      out_dir/runs/<date>_<start time>/results.csv  - the names, each row with the run's data
                                       results.txt  - readable report, with date, start, end, time taken
                                       files.csv    - each file: status, names, seconds
      out_dir/run_history.csv                       - one line per run, appended
      out_dir/all_runs_with_names.csv               - every run's names with its run data, appended

    file_rows: {"path", "status", "accepted", "review", "seconds"} per file."""
    run_dir = out_dir / "runs" / started.strftime("%Y-%m-%d_%H-%M-%S")
    n = 2
    while run_dir.exists():
        run_dir, n = run_dir.with_name(f"{started.strftime('%Y-%m-%d_%H-%M-%S')}_{n}"), n + 1
    run_dir.mkdir(parents=True)
    seconds = (finished - started).total_seconds()
    accepted = sum(e["decision"] == "accepted" for e in rows)
    failed = sum(not r["status"].startswith(("ok", "skipped")) for r in file_rows)

    run_values = [run_dir.name, f"{started:%Y-%m-%d}", f"{started:%H:%M:%S}", f"{finished:%H:%M:%S}",
                  f"{seconds:.1f}", format_duration(seconds), mode, llm, len(file_rows)]
    name_rows = _run_name_rows(run_values, rows)
    with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(RUN_COLUMNS + NAME_COLUMNS)
        w.writerows(name_rows)
    all_runs = out_dir / ALL_RUNS_FILE
    _start_new_if_columns_changed(all_runs, RUN_COLUMNS + NAME_COLUMNS)
    is_new_all = not all_runs.exists()
    with all_runs.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new_all:
            w.writerow(RUN_COLUMNS + NAME_COLUMNS)
        w.writerows(name_rows)
    with (run_dir / "files.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "status", "accepted", "review", "seconds"])
        for r in file_rows:
            w.writerow([r["path"], r["status"], r["accepted"], r["review"], f"{r['seconds']:.2f}"])

    width = max([len(e["display"]) for e in rows] + [4])
    lines = ["PERSON_EXTRACTOR_V4 - extraction run", "",
             f"  Date        : {started:%Y-%m-%d}",
             f"  Started     : {started:%H:%M:%S}",
             f"  Finished    : {finished:%H:%M:%S}" + ("" if finished.date() == started.date() else f" ({finished:%Y-%m-%d})"),
             f"  Time taken  : {format_duration(seconds)}",
             f"  Mode        : {mode}   (LLM reviewer: {llm})",
             f"  Files       : {len(file_rows)} ({failed} failed)",
             f"  Names       : {len(rows)} ({accepted} accepted, {len(rows) - accepted} review)", "",
             "FILES", f"  {'SECONDS':>8}  {'ACC':>4}  {'REV':>4}  STATUS / FILE"]
    lines += [f"  {r['seconds']:>8.2f}  {r['accepted']:>4}  {r['review']:>4}  {r['status']}  {r['path']}" for r in file_rows]
    lines += ["", "NAMES", f"  {'NAME':<{width}}  {'COUNT':>5}  DECISION  FILES  (ALSO AS)"]
    lines += [f"  {e['display']:<{width}}  {e['occurrences']:>5}  {e['decision']:<8}  {'; '.join(e['files'])}"
              + (f"  (also as: {'; '.join(e['aliases'])})" if e.get("aliases") else "") for e in rows]
    (run_dir / "results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    history = out_dir / "run_history.csv"
    is_new = not history.exists()
    with history.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(HISTORY_COLUMNS)
        w.writerow([run_dir.name, f"{started:%Y-%m-%d}", f"{started:%H:%M:%S}", f"{finished:%H:%M:%S}",
                    f"{seconds:.1f}", format_duration(seconds), len(file_rows), failed, len(rows), accepted,
                    len(rows) - accepted, mode, llm, "; ".join(inputs)])
    return run_dir


class App:
    def __init__(self, root) -> None:
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.root = tk, root
        self.events: queue.Queue = queue.Queue()
        self.pipelines: dict[bool, object] = {}
        self.last_run: Path | None = None

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
        ttk.Button(bar, text="Run history", command=self.open_history).pack(side="right")
        self.open_btn = ttk.Button(bar, text="Open results folder", command=self.open_folder, state="disabled")
        self.open_btn.pack(side="right", padx=(0, 8))

        self.progress = ttk.Progressbar(frame, mode="determinate")
        self.progress.pack(fill="x", pady=(8, 4))
        self.status = ttk.Label(frame, text="Ready.", justify="left")
        self.status.pack(anchor="w", fill="x")
        frame.bind("<Configure>", lambda e: self.status.config(wraplength=max(200, e.width - 24)))

        table = ttk.Frame(frame)
        table.pack(fill="both", expand=True, pady=(8, 0))
        cols = ("name", "decision", "count", "aliases", "files")
        self.tree = ttk.Treeview(table, columns=cols, show="headings")
        for col, title, width, anchor in (("name", "Name", 240, "w"), ("decision", "Decision", 90, "w"),
                                          ("count", "Count", 70, "e"), ("aliases", "Also as", 200, "w"),
                                          ("files", "Files", 320, "w")):
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
        text = self.paths.get("1.0", "end")
        files, problems = parse_paths(text)
        # The paths as pasted (a folder stays one line) go to the history;
        # every file they expanded to goes to the run's files.csv.
        self.inputs = [ln.strip().strip('"').strip("'").strip() for ln in text.splitlines()
                       if ln.strip() and not ln.strip().startswith("#")]
        if not files:
            self.status.config(text="Nothing to run. " + ("  ".join(problems) if problems else "Paste at least one path."))
            return
        self.run_btn.config(state="disabled")
        self.progress.config(maximum=len(files), value=0)
        note = f" ({len(problems)} path(s) skipped: {problems[0]}...)" if problems else ""
        self.status.config(text=f"Loading the model...{note}")
        threading.Thread(target=self.work, args=(files, self.fast.get()), daemon=True).start()

    def work(self, files: list[Path], fast: bool) -> None:
        # Keeps the PC awake and this run at full speed while the window is
        # in the background (src/utils/keep_awake.py) - on this thread,
        # which is the one Windows ties the no-sleep request to.
        from src.utils.keep_awake import keep_awake
        with keep_awake():
            self._work(files, fast)

    def _work(self, files: list[Path], fast: bool) -> None:
        try:
            from src.pipeline.orchestrator import Pipeline
            if fast not in self.pipelines:
                cfg = copy.deepcopy(DEFAULT_CONFIG)
                if fast:
                    cfg["llm_reviewer"]["enabled"] = False
                self.pipelines[fast] = Pipeline(config=cfg, base_dir=BASE_DIR)
            pipeline = self.pipelines[fast]
            started = datetime.now()
            results, file_rows = [], []
            for i, path in enumerate(files, 1):
                elapsed = format_duration((datetime.now() - started).total_seconds())
                self.events.put(("progress", i - 1, f"Started {started:%H:%M:%S} - {elapsed} so far. "
                                                    f"Processing {i}/{len(files)}: {path.name}"))
                t0 = datetime.now()
                result = pipeline.run(str(path))
                secs = (datetime.now() - t0).total_seconds()
                skipped = result.model_info.get("skipped_reason") if result.success else None
                status = (f"failed: {result.error}" if not result.success
                          else f"skipped: {skipped}" if skipped else "ok")
                file_rows.append({"path": str(path), "status": status, "seconds": secs,
                                  "accepted": len(result.persons), "review": len(result.review_persons)})
                if result.success:
                    results.append((path.name, result))
            rows = combine(results)
            finished = datetime.now()
            llm = "off (Fast mode)" if fast else pipeline.extractor.provenance.get("llm_reviewer", "off")
            run_dir = save_run(rows, file_rows, started, finished, "fast" if fast else "full", llm,
                               getattr(self, "inputs", [str(p) for p in files]))
            failed = [r for r in file_rows if r["status"].startswith("failed")]
            self.events.put(("done", rows, run_dir, len(files), failed, llm, started, finished))
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
                    _, self.rows, self.last_run, n, failed, llm, started, finished = event
                    self.progress.config(value=n)
                    acc = sum(e["decision"] == "accepted" for e in self.rows)
                    took = format_duration((finished - started).total_seconds())
                    text = (f"Done: {n} file(s), {len(self.rows)} name(s) - {acc} accepted, "
                            f"{len(self.rows) - acc} review.  LLM reviewer: {llm}.\n"
                            f"{started:%Y-%m-%d}  started {started:%H:%M:%S}, finished {finished:%H:%M:%S}, "
                            f"took {took}.\nSaved: {self.last_run}")
                    if failed:
                        text += f"\nFailed: {len(failed)} ({Path(failed[0]['path']).name}: {failed[0]['status']})"
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
            self.tree.insert("", "end", values=(e["display"], e["decision"], e["occurrences"],
                                                "; ".join(e.get("aliases", [])), "; ".join(e["files"])),
                             tags=(e["decision"],))

    def open_folder(self) -> None:
        if self.last_run:
            os.startfile(self.last_run)  # noqa: S606 - opens this run's local folder

    def open_history(self) -> None:
        history = OUTPUT_DIR / "run_history.csv"
        if history.exists():
            os.startfile(history)  # noqa: S606 - opens the local history in Excel / the default app
        else:
            self.status.config(text="No runs stored yet - the history starts with your first run.")


def main() -> int:
    import tkinter as tk
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
