import csv
import os
import re
import statistics
import subprocess
import time

BINARY_PATH = "bin/{name}"
IMPLEMENTATIONS = ["sequential", "thrust", "cuda", "cuda_shared_memory"]
NUM_SAMPLES = 100
MAX_ITERATIONS = 200
THRESHOLD = "1e-5"
SEED = "8675309"

FILENAME_PATTERN = re.compile(r"^.*-n(\d+)-d(\d+)-c(\d+)\.txt$")
TRANSFER_PATTERN = re.compile(
    r"KMEANS_TRANSFER_TIMING h2d_ms=([\d.eE+-]+) "
    r"d2h_ms=([\d.eE+-]+) total_ms=([\d.eE+-]+)"
)


def main():
    valid_inputs = sorted(
        f for f in os.listdir("tests/")
        if ("answer" not in f) and ("test" not in f)
    )

    raw_rows = []
    for file_no, input_file in enumerate(valid_inputs):
        match = FILENAME_PATTERN.fullmatch(input_file)
        if not match:
            continue

        num_points, dims, num_clusters = map(int, match.groups())
        input_path = os.path.join("tests", input_file)
        print(f"Benchmarking {input_file} ({NUM_SAMPLES} samples per implementation)")

        for sample in range(1, NUM_SAMPLES + 1):
            # Rotate the order slightly between samples to reduce order bias.
            sample_order = IMPLEMENTATIONS[(sample - 1) % len(IMPLEMENTATIONS):] + \
                IMPLEMENTATIONS[:(sample - 1) % len(IMPLEMENTATIONS)]

            for name in sample_order:
                binary_path = BINARY_PATH.format(name=name)
                start_time = time.perf_counter()
                result = subprocess.run(
                    [binary_path, "-k", str(num_clusters),
                     "-d", str(dims), "-i", input_path,
                     "-m", str(MAX_ITERATIONS), "-t", THRESHOLD,
                     "-s", SEED, "-c"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    check=True,
                )
                process_wall_ms = (time.perf_counter() - start_time) * 1000.0

                output_lines = result.stdout.splitlines()
                if not output_lines:
                    raise RuntimeError(f"{name} produced no output for {input_file}")
                iterations_text, iteration_time_text = output_lines[0].split(",", 1)
                iterations = int(iterations_text.strip())
                time_per_iteration_ms = float(iteration_time_text.strip())

                h2d_ms = ""
                d2h_ms = ""
                function_total_ms = ""
                if name in ("cuda", "cuda_shared_memory"):
                    timing_matches = TRANSFER_PATTERN.findall(result.stderr)
                    if not timing_matches:
                        raise RuntimeError(
                            f"Missing transfer timing from {name} on {input_file}. "
                            f"stderr was:\n{result.stderr}"
                        )
                    h2d_text, d2h_text, total_text = timing_matches[-1]
                    h2d_ms, d2h_ms, function_total_ms = map(
                        float, (h2d_text, d2h_text, total_text)
                    )

                raw_rows.append({
                    "input": input_file,
                    "n": num_points,
                    "dims": dims,
                    "clusters": num_clusters,
                    "implementation": name,
                    "sample": sample,
                    "iterations": iterations,
                    "ms_per_iteration": time_per_iteration_ms,
                    "reported_loop_ms": iterations * time_per_iteration_ms,
                    "process_wall_ms": process_wall_ms,
                    "h2d_ms": h2d_ms,
                    "d2h_ms": d2h_ms,
                    "function_total_ms": function_total_ms,
                })

                print(
                    f"  {file_no + 1}, sample {sample}, {name}: "
                    f"{iterations} iterations, {time_per_iteration_ms:.6f} ms/iteration"
                )

    raw_fields = [
        "input", "n", "dims", "clusters", "implementation", "sample",
        "iterations", "ms_per_iteration", "reported_loop_ms", "process_wall_ms",
        "h2d_ms", "d2h_ms", "function_total_ms",
    ]
    with open("benchmark_raw.csv", "w", newline="") as raw_file:
        writer = csv.DictWriter(raw_file, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(raw_rows)

    summary_rows = []
    for input_file in valid_inputs:
        for name in IMPLEMENTATIONS:
            runs = [
                row for row in raw_rows
                if row["input"] == input_file and row["implementation"] == name
            ]
            if not runs:
                continue

            mean_loop_ms = statistics.mean(row["reported_loop_ms"] for row in runs)
            summary = {
                "input": input_file,
                "implementation": name,
                "samples": len(runs),
                "mean_iterations": statistics.mean(row["iterations"] for row in runs),
                "mean_ms_per_iteration": statistics.mean(row["ms_per_iteration"] for row in runs),
                "median_ms_per_iteration": statistics.median(row["ms_per_iteration"] for row in runs),
                "mean_reported_loop_ms": mean_loop_ms,
                "mean_process_wall_ms": statistics.mean(row["process_wall_ms"] for row in runs),
                "mean_h2d_ms": "",
                "mean_d2h_ms": "",
                "mean_function_total_ms": "",
                "transfer_fraction_pct": "",
                "speedup_vs_sequential": "",
            }

            if name in ("cuda", "cuda_shared_memory"):
                mean_h2d = statistics.mean(row["h2d_ms"] for row in runs)
                mean_d2h = statistics.mean(row["d2h_ms"] for row in runs)
                mean_total = statistics.mean(row["function_total_ms"] for row in runs)
                summary["mean_h2d_ms"] = mean_h2d
                summary["mean_d2h_ms"] = mean_d2h
                summary["mean_function_total_ms"] = mean_total
                summary["transfer_fraction_pct"] = (
                    100.0 * (mean_h2d + mean_d2h) / mean_total if mean_total else 0.0
                )
            summary_rows.append(summary)

    for summary in summary_rows:
        sequential_runs = [
            row for row in summary_rows
            if row["input"] == summary["input"]
            and row["implementation"] == "sequential"
        ]
        if sequential_runs and summary["mean_reported_loop_ms"]:
            summary["speedup_vs_sequential"] = (
                sequential_runs[0]["mean_reported_loop_ms"]
                / summary["mean_reported_loop_ms"]
            )

    summary_fields = [
        "input", "implementation", "samples", "mean_iterations",
        "mean_ms_per_iteration", "median_ms_per_iteration",
        "mean_reported_loop_ms", "mean_process_wall_ms", "mean_h2d_ms",
        "mean_d2h_ms", "mean_function_total_ms", "transfer_fraction_pct",
        "speedup_vs_sequential",
    ]
    with open("benchmark_summary.csv", "w", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("Raw measurements written to benchmark_raw.csv")
    print("Summary metrics written to benchmark_summary.csv")


if __name__ == "__main__":
    main()
