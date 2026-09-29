import os
import re 
import subprocess
from tqdm import tqdm

BINARY_PATH = "bin/{name}"
FLOAT_THRESHOLD = 1e-5



def main(): 
    valid_inputs = [f for f in os.listdir('tests/') if (('answer' not in f) and ('test' not in f))]
    filename_pattern = re.compile(r"^.*-n(\d+)-d(\d+)-c(\d+)\.txt$")
    for input_file in valid_inputs: 
        match = filename_pattern.fullmatch(input_file)
        num_points, dims, num_clusters = map(int, match.groups())
        input_path = os.path.join("tests", input_file)
        output_path = os.path.join("tests", f"{os.path.splitext(input_file)[0]}-test.txt")
        with open(output_path, "w") as output_file:
            subprocess.run(
                [BINARY_PATH.format(name="sequential"), "-k", str(num_clusters),
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
            if len(output_values) != len(answer_values):
                raise AssertionError(
                    f"{input_file}: cluster {cluster_id.decode()} has "
                    f"{len(output_values)} values; expected {len(answer_values)}"
                )

            for index, (actual, expected) in enumerate(tqdm(
                zip(output_values, answer_values),
                total=len(answer_values),
                desc=f"{input_file} cluster {cluster_id.decode()}",
            )):
                if abs(float(actual) - float(expected)) > FLOAT_THRESHOLD:
                    raise AssertionError(
                        f"{input_file}: cluster {cluster_id.decode()} value {index} differs: "
                        f"{actual.decode()} != {expected.decode()}"
                    )

if __name__ == "__main__":
    main()
