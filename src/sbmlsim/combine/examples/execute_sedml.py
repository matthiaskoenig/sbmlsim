"""Execute SED-ML."""
from pathlib import Path

from sbmlsim.combine.sedml.runner import execute_sedml


examples_dir = Path(__file__).parent / "l1v4"
# examples_dir = Path(__file__).parent / "l1v4_l1v3"

l1v4_l1v3_files = ["repeated-stochastic-runs/repeated-stochastic-runs.sedml"]

l1v4_files = [
    # plotting
    "algorithm_parameters.sedml",
    # "axis.sedml",
    # "axis_grids.sedml",
    # "axis_minmax.sedml",
    # "axis_minmax_smaller.sedml",
    # "axis_minormax.sedml",
    # "concentration_amount.sedml",  # FIXME
    # "concentration_amount_b.sedml"  # FIXME
    # "curve_types.sedml",
    # "curve_types_errors.sedml",
    # "heat_map_ls.sedml",
    # "markertype.sedml",
    # "linetype.sedml",
    # "right_yaxis.sedml",
    # "line_overlap_order.sedml",
    # "repressilator_figure.xml",
    # "repressilator.xml",
    # "repressilator_urn.xml",
    # "test_file_1.sedml",
    # "test_line_fill.sedml",
    # "stacked_bar.sedml",
    # "test_3hbarstacked.sedml",
    # "test_bar.sedml",
    # "test_bar3stacked.sedml",
    # "test_file.sedml",
    # "test_hbar_stacked.sedml",
    # "test_shaded_area.sedml",
    # "test_shaded_area_overlap_order.sedml",
    # "test_base_styles.sedml",
]

if __name__ == "__main__":
    # ----------------------
    # L1V4 Plotting
    # ----------------------
    working_dir = examples_dir
    for sedml_file in l1v4_files:
        # for sedml_file in l1v4_l1v3_files:
        execute_sedml(
            path=examples_dir / sedml_file,
            working_dir=examples_dir / "sbmlsim",
            output_path=examples_dir / "sbmlsim",
        )

    # ----------------------
    # L1V4 Parameter Fitting
    # ----------------------
    # working_dir = base_path / "l1v4_parameter_fitting"
    # for name, sedml_file in [
    #     # "Elowitz_Nature2000.xml",
    # ]:
    #     execute_sedml(
    #         working_dir=working_dir,
    #         path=working_dir / sedml_file
    #     )
