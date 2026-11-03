"""Compare the final centerline results with the isentropic reference states."""

import argparse
import math
import re
from pathlib import Path

from DIRS.dirs import RESULTS_DIR
from MISC.calc_ref_states import calc_ref_states

QUANTITIES = {
    "velocity": 2,
    "massflow": 6,
    "rho": 1,
    "p": 3,
}
RESULT_FILE_PATTERN = re.compile(
    r"centerline_(velocity|massflow|rho|p)_tr(\d+\.\d+)_ma(\d+\.\d+)\.txt"
)


def read_final_values(path: Path) -> tuple[list[float], list[float]]:
    """Read the x-coordinate header and the last non-empty result row."""
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    if len(rows) < 2:
        raise ValueError(f"{path} must contain an x-coordinate header and result row")

    try:
        x_values = [float(value) for value in rows[0]]
        final_values = [float(value) for value in rows[-1]]
    except ValueError as error:
        raise ValueError(f"{path} contains a non-numeric value") from error

    if not x_values or len(x_values) != len(final_values):
        raise ValueError(
            f"{path} has {len(x_values)} x-coordinates but "
            f"{len(final_values)} values in its final row"
        )
    if not all(math.isfinite(value) for value in x_values + final_values):
        raise ValueError(f"{path} contains a non-finite value")

    return x_values, final_values


def relative_l1_error(values: list[float], reference: list[float]) -> float:
    """Return ||values - reference||_1 / ||reference||_1."""
    if len(values) != len(reference):
        raise ValueError("Values and reference must have the same length")
    reference_norm = sum(abs(value) for value in reference)
    if reference_norm == 0.0:
        raise ValueError("Cannot calculate relative error for a zero reference")
    return sum(
        abs(value - expected) for value, expected in zip(values, reference)
    ) / reference_norm


def pointwise_relative_error(value: float, reference: float) -> float:
    """Return |value - reference| / |reference| for one location."""
    if reference == 0.0:
        raise ValueError("Cannot calculate relative error for a zero reference")
    return abs(value - reference) / abs(reference)


def calculate_errors(
    results_dir: Path,
) -> list[tuple[float, float, str, float, list[tuple[float, float]]]]:
    """Calculate aggregate and per-location errors for centerline result files."""
    errors = []
    for path in sorted(results_dir.glob("centerline_*_tr*_ma*.txt")):
        match = RESULT_FILE_PATTERN.fullmatch(path.name)
        if match is None:
            continue

        quantity, throat_ratio_text, mach_text = match.groups()
        throat_ratio = float(throat_ratio_text)
        mach_number = float(mach_text)
        x_values, final_values = read_final_values(path)
        reference_values = [
            calc_ref_states(throat_ratio, mach_number, x)[QUANTITIES[quantity]]
            for x in x_values
        ]
        location_errors = [
            (x, pointwise_relative_error(value, reference))
            for x, value, reference in zip(x_values, final_values, reference_values)
        ]
        errors.append(
            (
                throat_ratio,
                mach_number,
                quantity,
                relative_l1_error(final_values, reference_values),
                location_errors,
            )
        )

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_dir",
        nargs="?",
        type=Path,
        default=RESULTS_DIR,
        help=f"directory containing centerline result files (default: {RESULTS_DIR})",
    )
    args = parser.parse_args()

    errors = calculate_errors(args.results_dir)
    if not errors:
        raise FileNotFoundError(
            f"No centerline result files found in {args.results_dir}"
        )

    print("Aggregate relative L1 errors")
    print("throat_ratio  exit_mach  quantity  relative_L1_error")
    for throat_ratio, mach_number, quantity, error, _ in errors:
        print(
            f"{throat_ratio:12.2f}  {mach_number:9.2f}  " f"{quantity:8s}  {error:.6%}"
        )

    print("\nPer-location relative errors")
    print("throat_ratio  exit_mach  quantity      x  relative_error")
    for throat_ratio, mach_number, quantity, _, location_errors in errors:
        for x, error in location_errors:
            print(
                f"{throat_ratio:12.2f}  {mach_number:9.2f}  "
                f"{quantity:8s}  {x:5.2f}  {error:.6%}"
            )


if __name__ == "__main__":
    main()
