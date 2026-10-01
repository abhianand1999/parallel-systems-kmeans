import os
import re 
import subprocess
from argparse import ArgumentParser

BINARY_PATH = "bin/{name}"
FLOAT_THRESHOLD = 1e-3


def main(name): 
    valid_inputs = [f for f in os.listdir('tests/') if (('answer' not in f) and ('test' not in f))]
    filename_pattern = re.compile(r"^.*-n(\d+)-d(\d+)-c(\d+)\.txt$")
    for file_no, input_file in enumerate(valid_inputs): 
        match = filename_pattern.fullmatch(input_file)
        num_points, dims, num_clusters = map(int, match.groups())
        input_path = os.path.join("tests", input_file)
        output_path = os.path.join("tests", f"{os.path.splitext(input_file)[0]}-test.txt")
        with open(output_path, "w") as output_file:
            subprocess.run(
                [BINARY_PATH.format(name=name), "-k", str(num_clusters),
                 "-d", str(dims), "-i", input_path, "-c"],
                stdout=output_file,
                check=True,
            )
        answer_path = os.path.join("tests", f"{os.path.splitext(input_file)[0]}-answer.txt")
        with open(output_path, "rb") as output_file, open(answer_path, "rb") as answer_file:
            output_lines = output_file.readlines()[1:]
            answer_lines = answer_file.readlines()

        if len(output_lines) != len(answer_lines):
            raise AssertionError(
                f"{input_file}: output has {len(output_lines)} clusters; "
                f"expected {len(answer_lines)}"
            )

        for output_line, answer_line in zip(output_lines, answer_lines):
            cluster_id, *output_values = output_line.split()
            _, *answer_values = answer_line.split()
            diffs = [(float(f1) - float(f2)) ** 2 for f1, f2 in zip(output_values, answer_values)]
            error = sum(diffs) ** .5
            if (error > FLOAT_THRESHOLD): 
                raise AssertionError(f"Cluster ID {cluster_id} differs from answers by more than threshold.")
        print(f"{file_no + 1}: Comparisons for {input_file} succeeded.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, default="sequential")
    args = parser.parse_args()
    main(name=args.name)
