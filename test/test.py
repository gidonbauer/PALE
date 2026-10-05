#! /usr/bin/python3

import shlex
import subprocess
import sys
import os
import resource
import time
import tempfile
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Union, List, Dict, Tuple, Set, TextIO
from dataclasses import dataclass


BIN_DIR = "bin/test"

_MAXRSS_SCALE = 1 if sys.platform == "darwin" else 1024

@dataclass
class TestResult:
    ret: int
    stdout: str
    stderr: str
    runtime: float    # wall seconds
    cpu_time: float   # user + system seconds
    max_rss: int      # peak resident set, bytes

Input = Union[int, str, None]
NameInput = Tuple[str, Input]
Results = Dict[NameInput, TestResult]


@dataclass
class TestCase:
    name: str
    input: Union[List[int], List[str], None]
    parallel: bool

    def __hash__(self):
        inp = tuple(self.input) if self.input is not None else None
        return hash((self.name, inp, self.parallel))


ALL_TESTS = [
    TestCase("Taylor-Green-MG",             [8, 16, 64],                                    False),
    TestCase("Taylor-Green-FFT",            [8, 16, 64],                                    False),
    TestCase("Channel-MG",                  [16, 32, 64],                                   False),
    TestCase("Channel-FFT",                 [16, 32, 64],                                   False),
    TestCase("Polar-Couette",               [8, 16, 32],                                    False),
    TestCase("Polar-Channel",               [8, 16, 32, 64],                                False),
    TestCase("Advection-Cartesian",         [16, 32, 64, 128],                              False),
    TestCase("Advection-Polar",             [16, 32, 64, 128],                              False),
    TestCase("Multigrid",                   [32, 64, 128, 512, 1024],                       False),
    TestCase("Multigrid-Spherical",         [16, 32, 64, 128, 256],                         False),
    TestCase("Hill-Vortex",                 [16, 32, 64],                                   False),
    TestCase("ALE-Polar-Conservation",      [16, 32, 64, 128],                              False),
    TestCase("ALE-Spherical-Conservation",  [16, 32, 64, 128],                              False),
    TestCase("ALE-Scalar-Source",           ["Cartesian", "Polar", "Symmetric-Spherical"],  False),
    TestCase("Scalar-Source",               ["Cartesian", "Polar", "Symmetric-Spherical"],  False),
    TestCase("GCL",                         ["Cartesian", "Polar", "Symmetric-Spherical"],  False),
    TestCase("Scriven-2D",                  [16, 32, 64],                                   False),
    TestCase("Scriven-3D",                  [16, 32, 64],                                   False),
    TestCase("IB-Channel-MG-IE",            [16, 32, 64],                                   False),
    TestCase("IB-Channel-FFT-IE",           [16, 32, 64],                                   False),
    TestCase("IB-Channel-MG-SA",            [16, 32, 64],                                   False),
    TestCase("IB-Channel-FFT-SA",           [16, 32, 64],                                   False),
    TestCase("IB-Polar-Channel-IE",         [8, 16, 32, 64],                                False),
    TestCase("IB-Polar-Channel-SA",         [8, 16, 32, 64],                                False),
    TestCase("Iterator",                    None,                                           True),
    TestCase("Boundary",                    None,                                           False),
]


def run_cmd(cmd, echo=False):
    if echo:
        print(f"[CMD] {' '.join(map(shlex.quote, cmd))}")
    ret = subprocess.run(cmd, capture_output=True)
    return ret.returncode, ret.stdout.decode("utf-8"), ret.stderr.decode("utf-8")


def build_tests(cases: List[TestCase], jobs: int = 1, force_parallel: bool = False, verbose: bool = False) -> bool:
    parallel_flag    = "PARALLEL=1"
    serial_targets   = [f"{BIN_DIR}/{case.name}" for case in cases if not case.parallel and not force_parallel]
    parallel_targets = [f"{BIN_DIR}/{case.name}" for case in cases if case.parallel or force_parallel]

    if len(serial_targets) > 0:
        print(f"[INFO] Build {len(serial_targets)} serial target(s) with -j{jobs}")
        ret, stdout, stderr = run_cmd(["make", "-B", f"-j{jobs}", *serial_targets], echo=verbose)
        if ret != 0:
            print("Build failed", file=sys.stderr)
            print(stderr, file=sys.stderr)
            return False

    if len(parallel_targets) > 0:
        print(f"[INFO] Build {len(parallel_targets)} parallel target(s) with -j{jobs}")
        ret, stdout, stderr = run_cmd(["make", parallel_flag, "-B", f"-j{jobs}", *parallel_targets], echo=verbose)
        if ret != 0:
            print("Build failed", file=sys.stderr)
            print(stderr, file=sys.stderr)
            return False

    return True


def run_name(name: str, inp: Input):
    return name if inp is None else f"{name}-{inp}"


_print_lock = threading.Lock()
def log(msg: str) -> None:
    with _print_lock:
        sys.stdout.write(msg + "\n")
        sys.stdout.flush()


def run_test(cmd: List[str], name: str, echo: bool = False) -> TestResult:
    log(f"[INFO] Run {name}...")

    if echo:
        print(f"[CMD] {' '.join(map(shlex.quote, cmd))}")

    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        start = time.perf_counter()
        env = dict(os.environ)
        proc = subprocess.Popen(cmd, stdout=out, stderr=err, env=env)
        _pid, status, ru = os.wait4(proc.pid, 0)
        runtime = time.perf_counter() - start

        proc.returncode = os.waitstatus_to_exitcode(status)

        out.seek(0)
        err.seek(0)
        stdout = out.read().decode("utf-8")
        stderr = err.read().decode("utf-8")

    return TestResult(
        ret=proc.returncode,
        stdout=stdout,
        stderr=stderr,
        runtime=runtime,
        cpu_time=ru.ru_utime + ru.ru_stime,
        max_rss=ru.ru_maxrss * _MAXRSS_SCALE,
    )


def run_tests(cases: List[TestCase], jobs: int = 1, verbose: bool = False) -> Results:
    work: List[Tuple[NameInput, List[str]]] = []
    for test in cases:
        exe = f"{BIN_DIR}/{test.name}"
        if test.input is None:
            work.append(((test.name, None), [exe]))
        else:
            for n in test.input:
                work.append(((test.name, n), [exe, f"{n}"]))

    if jobs == 1:
        return {key: run_test(cmd, run_name(*key), echo=verbose) for key, cmd in work}

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(run_test, cmd, run_name(*key), echo=verbose) for key, cmd in work]
        return {key: fut.result() for (key, _), fut in zip(work, futures)}


_UNITS = ("B", "kB", "MB", "GB", "TB", "PB", "EB")
def format_bytes(n: int, precision: int = 1, base: int = 1024) -> str:
    if n < 0:
        return "-" + format_bytes(-n, precision, base)

    size = float(n)
    unit = _UNITS[0]
    for i, unit in enumerate(_UNITS):
        if i == len(_UNITS) - 1 or round(size, precision) < base:
            break
        size /= base

    if unit == "B":
        return f"{int(size)}{unit}"

    text = f"{size:.{precision}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return f"{text}{unit}"


GREEN = "\033[32m"
RED   = "\033[31m"
RESET = "\033[0m"
STATUS_WIDTH = 4
MIN_DOTS = 30        # minimum dots next to the longest name
METRICS_INDENT = 6   # spaces before the metrics line, after the leading space

def render_results(results: Results, show_only_failed: bool, file: TextIO = sys.stdout) -> None:
    color = file.isatty()
    WALL_WIDTH = 10
    MEM_WIDTH  = 10

    groups: Dict[str, List[Tuple[Input, TestResult]]] = {}
    for (name, inp), res in results.items():
        groups.setdefault(name, []).append((inp, res))

    num_passed = sum(res.ret == 0 for res in results.values())
    total_wall = sum(res.runtime for res in results.values())

    # (name, inputs, ok, wall, mem, failed details)
    rows: List[Tuple[str, str, bool, str, str, List[str]]] = []
    for name, runs in groups.items():
        ok = all(res.ret == 0 for _, res in runs)
        if show_only_failed and ok: continue
        inputs = " ".join("-" if inp is None else str(inp) for inp, _ in runs)
        wall   = f"{sum(res.runtime for _, res in runs):.2f}s"
        mem    = format_bytes(max(res.max_rss for _, res in runs))
        inp_width = max(len("-" if inp is None else str(inp)) for inp, _ in runs)
        details = [
            f"    x {'-' if inp is None else str(inp):<{inp_width}}     "
            f"wall={res.runtime:.2f}s, "
            f"cpu={res.cpu_time:.2f}s, "
            f"mem={format_bytes(res.max_rss)}"
            for inp, res in runs if res.ret != 0
        ]
        rows.append((name, inputs, ok, wall, mem, details))

    summary = f"Passed {num_passed}/{len(results)} cases in {len(groups)} tests"
    if color:
        color_begin = GREEN if num_passed == len(results) else RED
        summary_colored = summary.replace(
            f"{num_passed}/{len(results)}", f"{color_begin}{num_passed}/{len(results)}{RESET}", 1
        )
    else:
        summary_colored = summary

    name_width = max([len("Test")] + [len(r[0]) for r in rows]) + 2
    inp_width  = max([len("Inputs")] + [len(r[1]) for r in rows]) + 2
    inp_width  = max(inp_width, len(summary) + 2 - name_width - STATUS_WIDTH)
    left_width = name_width + inp_width + STATUS_WIDTH
    inner      = left_width + WALL_WIDTH + MEM_WIDTH

    print(f" {'=' * inner}", file=file)
    if rows:
        print(f" {'Test':<{name_width}}{'Inputs':<{inp_width}}{'':{STATUS_WIDTH}}"
              f"{'Wall':>{WALL_WIDTH}}{'Mem':>{MEM_WIDTH}}", file=file)
        print(f" {'-' * inner}", file=file)

        for name, inputs, ok, wall, mem, details in rows:
            status = "PASS" if ok else "FAIL"
            if color:
                status = f"{GREEN if ok else RED}{status}{RESET}"
            print(f" {name:<{name_width}}{inputs:<{inp_width}}{status}"
                  f"{wall:>{WALL_WIDTH}}{mem:>{MEM_WIDTH}}", file=file)
            for detail in details:
                print(f" {detail}", file=file)

        print(f" {'-' * inner}", file=file)
    padding = " " * (left_width - len(summary))
    print(f" {summary_colored}{padding}{f'{total_wall:.2f}s':>{WALL_WIDTH}}{'':{MEM_WIDTH}}", file=file)
    print(f" {'=' * inner}", file=file)


LOG_PATH = "test/logs/"
def log_paths(name: str, inp: Input) -> Tuple[str, str]:
    rname = run_name(name, inp)
    stdout_path = f"{LOG_PATH}/{rname}.stdout".replace("//", "/")
    stderr_path = f"{LOG_PATH}/{rname}.stderr".replace("//", "/")
    return stdout_path, stderr_path


def write_logs(results: Results) -> bool:
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)

    try:
        for (name, inp), res in results.items():
            stdout_path, stderr_path = log_paths(name, inp)
            with open(stdout_path, "w") as f:
                print(res.stdout, file=f)
            with open(stderr_path, "w") as f:
                print(res.stderr, file=f)
        return True
    except IOError as err:
        print(f"{err}", file=sys.stderr)
        return False



def dump_failed(results: Results):
    for (name, inp), res in results.items():
        if res.ret == 0:
            continue
        stdout_path, stderr_path = log_paths(name, inp)
        print(f" {run_name(name, inp)} failed:")
        print(f"   stdout: {stdout_path}")
        print(f"   stderr: {stderr_path}")


def find_matching_tests(pattern: str) -> Set[TestCase]:
    matches = set()
    for test in ALL_TESTS:
        if pattern in test.name:
            matches.add(test)
    return matches


def parse_args(argv: Union[List[str], None] = None) -> Tuple[argparse.Namespace, List[TestCase]]:
    p = argparse.ArgumentParser(description="Build and run the test suite.")
    p.add_argument(
        "-j", "--jobs",
        type=int, nargs="?", const=os.cpu_count() or 1, default=1, metavar="N",
        help=f"run N build/test jobs in parallel (default: 1; bare -j uses all {os.cpu_count() or 1} cores)",
    )
    p.add_argument(
        "-p", "--parallel",
        action="store_true",
        help="build all the test case with PARALLEL=1 (default: False)",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="print executed commands (default: False)",
    )
    p.add_argument(
        "--show-only-failed",
        action="store_true",
        help="print only failed test cases (default: False)"
    )
    p.add_argument(
            "case_names",
            nargs="*",
            help="test cases to run, can match on substrings (default: all cases)"
    )
    args = p.parse_args(argv)
    if args.jobs < 1:
        p.error("-j must be at least 1")
    if len(args.case_names) == 0:
        return args, ALL_TESTS
    else:
        cases = set()
        for case_name in args.case_names:
            matches = find_matching_tests(case_name)
            if len(matches) == 0:
                p.error(f"Invalid case name or pattern `{case_name}`, test names are {', '.join([test.name for test in ALL_TESTS])}")
            cases.update(matches)
        return args, list(cases)


def main():
    args, cases = parse_args()
    if not build_tests(cases, jobs=args.jobs, force_parallel=args.parallel, verbose=args.verbose):
        sys.exit(1)
    results = run_tests(cases, jobs=args.jobs, verbose=args.verbose)
    render_results(results, args.show_only_failed)
    if not write_logs(results):
        sys.exit(1)
    dump_failed(results)


if __name__ == "__main__":
    main()
