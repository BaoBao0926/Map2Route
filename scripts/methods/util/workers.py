"""Small, deterministic episode-level worker pool shared by method runners."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from typing import Callable, Iterator, TypeVar


Result = TypeVar("Result")


def normalize_worker_count(workers: int) -> int:
    """Validate the user-facing ``--workers`` value."""
    worker_count = int(workers)
    if worker_count < 1:
        raise ValueError("--workers must be at least 1.")
    return worker_count


def run_instruction_jobs(
    instruction_files: list[Path],
    *,
    workers: int,
    run_one: Callable[[int, Path], Result],
) -> Iterator[tuple[int, Result]]:
    """Run independent episodes, yielding ``(input_index, result)``.

    A thread pool is deliberately used here: episode execution includes API calls,
    disk I/O, image writing, and NumPy/SciPy metric work.  Each caller keeps
    summary writes on its main thread, so concurrent workers never race on a
    method's ``summary.json``.
    """
    worker_count = normalize_worker_count(workers)
    indexed_files = list(enumerate(instruction_files, start=1))
    if worker_count == 1:
        for index, instruction_file in indexed_files:
            yield index, run_one(index, instruction_file)
        return

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures: dict[Future[Result], int] = {
            executor.submit(run_one, index, instruction_file): index
            for index, instruction_file in indexed_files
        }
        for future in as_completed(futures):
            yield futures[future], future.result()


def run_instruction_jobs_grouped(
    instruction_files: list[Path],
    *,
    workers: int,
    group_key: Callable[[Path], str],
    run_one: Callable[[int, Path], Result],
) -> Iterator[tuple[int, Result]]:
    """Run groups in parallel while serialising jobs within each group.

    This suits work which temporarily loads large group-scoped state, such as a
    map metric cache.  It avoids submitting many same-map jobs ahead of every
    other map, which would otherwise occupy the entire pool while blocked on a
    per-map lock.
    """
    worker_count = normalize_worker_count(workers)
    grouped: dict[str, list[tuple[int, Path]]] = {}
    for index, instruction_file in enumerate(instruction_files, start=1):
        grouped.setdefault(group_key(instruction_file), []).append((index, instruction_file))

    if worker_count == 1:
        for jobs in grouped.values():
            for index, instruction_file in jobs:
                yield index, run_one(index, instruction_file)
        return

    # Emit each completed episode immediately, rather than waiting for every
    # instruction in a map group.  This keeps long metric batches observable
    # and lets callers write useful progress checkpoints.
    events: Queue[tuple[str, object]] = Queue()

    def run_group(jobs: list[tuple[int, Path]]) -> None:
        try:
            for index, instruction_file in jobs:
                events.put(("result", (index, run_one(index, instruction_file))))
        except BaseException as exc:
            events.put(("error", exc))
        finally:
            events.put(("complete", None))

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures: dict[Future[None], str] = {
            executor.submit(run_group, jobs): key
            for key, jobs in grouped.items()
        }
        unfinished_groups = len(futures)
        while unfinished_groups:
            event, payload = events.get()
            if event == "result":
                index, result = payload  # type: ignore[misc]
                yield index, result
            elif event == "error":
                raise payload  # type: ignore[misc]
            else:
                unfinished_groups -= 1
        for future in futures:
            future.result()
