"""Extract OpenFAST linear-mode shapes from a .lin file via command-line arguments.

This is the cleaned counterpart to the legacy
``Modal_OutPut_Final_02.py`` script. The legacy script hardcodes its
input path (``outputs_lin_No_SubD/parametric/case_1.1.lin``) and writes
its output (``Extracted_Mode_Shapes.xlsx``) to whatever directory it is
launched from. That hardcoding makes the legacy script unusable as a
reproducible runner because users cannot vary the input file or control
the output location without editing the source code.

This wrapper exposes the legacy extraction logic through a proper
command-line interface. The numerical work is unchanged: the wrapper
imports the ``analyze_modes`` function from the legacy script and calls
it directly, so the eigendecomposition of the OpenFAST state matrix,
the tower and blade state-index identification, the projection of
eigenvectors through the output matrix to physical mode shapes, and the
two-sheet Excel export (``Tower_Modes`` and ``Blade_Modes``) all happen
exactly as they do in the legacy script. The only new logic is path
handling: the ``--lin-file`` argument selects the input, and
``--output`` selects the destination of the Excel file via a temporary
``chdir`` because ``analyze_modes`` writes its output to the current
working directory.

Typical use:

    python scripts/openfast/extract_linear_modes.py \\
        --lin-file outputs_lin/parametric/case_1.1.lin \\
        --output results/tables/Extracted_Mode_Shapes.xlsx

If ``--output`` is omitted, the file is written to
``results/tables/Extracted_Mode_Shapes.xlsx`` so that downstream MAC
validation runs find it without further configuration.
"""
from pathlib import Path
import argparse
import importlib.util
import os
import sys


# Path to the legacy script that owns the actual extraction logic. This
# repository keeps that script under legacy/ for traceability rather than
# moving its body into src/eftwin/, because the function depends on
# specific pyFAST/openfast_toolbox API details that are easier to maintain
# in one place rather than refactoring across the package.
LEGACY_SCRIPT = Path(__file__).resolve().parents[2] / "legacy" / "openfast_original" / "Modal_OutPut_Final_02.py"


def _load_legacy_module():
    """Import Modal_OutPut_Final_02.py as a Python module without running it.

    The legacy file has an ``if __name__ == '__main__'`` guard at the
    bottom that uses a hardcoded path; importing the file rather than
    running it avoids that guard while still giving us access to the
    ``analyze_modes`` function defined inside.
    """
    if not LEGACY_SCRIPT.exists():
        raise FileNotFoundError(
            f"Legacy linearisation script not found at {LEGACY_SCRIPT}. "
            "If you have moved or renamed the legacy directory, update "
            "the LEGACY_SCRIPT constant in this file."
        )
    spec = importlib.util.spec_from_file_location("modal_output_final_02", LEGACY_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(
        description="Extract OpenFAST linear-mode shapes from a .lin file."
    )
    parser.add_argument(
        "--lin-file",
        required=True,
        type=Path,
        help="Path to the OpenFAST linearisation output (.lin) to process.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tables/Extracted_Mode_Shapes.xlsx"),
        help="Destination path for the Excel file containing tower and blade modes.",
    )
    parser.add_argument(
        "--freq-min",
        type=float,
        default=0.0,
        help="Lower bound of the frequency band reported (Hz).",
    )
    parser.add_argument(
        "--freq-max",
        type=float,
        default=5.0,
        help="Upper bound of the frequency band reported (Hz).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Suppress the per-mode tower-shape plots that the legacy "
             "function produces interactively.",
    )
    args = parser.parse_args()

    # Validate the input path before we import the legacy module so that
    # users get a clear error message rather than a deep traceback if
    # they pointed --lin-file at a non-existent location.
    lin_file = args.lin_file.resolve()
    if not lin_file.exists():
        raise FileNotFoundError(f"Linearisation file not found: {lin_file}")

    # The legacy ``analyze_modes`` function writes to a fixed filename
    # (Extracted_Mode_Shapes.xlsx) in the current working directory, so
    # to redirect the output we briefly change directory to the user's
    # chosen output folder. We restore the original cwd in a try/finally
    # so that an exception during extraction does not leave the process
    # in a confused state.
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    legacy = _load_legacy_module()

    # If the user wants to suppress plots, we monkey-patch matplotlib.show
    # to a no-op for the duration of the call. The legacy function uses
    # plt.show() at several places; this is the least invasive way to
    # get a non-interactive run.
    if args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.show = lambda *a, **k: None

    original_cwd = Path.cwd()
    try:
        os.chdir(output_path.parent)
        legacy.analyze_modes(
            str(lin_file),
            freq_min=args.freq_min,
            freq_max=args.freq_max,
            plot_tower=not args.no_plot,
            plot_blade=not args.no_plot,
        )
    finally:
        os.chdir(original_cwd)

    # The legacy function always writes Extracted_Mode_Shapes.xlsx;
    # if the user asked for a different filename, rename it now.
    written = output_path.parent / "Extracted_Mode_Shapes.xlsx"
    if written != output_path:
        if not written.exists():
            raise RuntimeError(
                f"Expected legacy output at {written} but the file was not created. "
                "Inspect the legacy function output above for errors."
            )
        written.rename(output_path)

    print(f"Wrote linear-mode shapes to {output_path}")


if __name__ == "__main__":
    sys.exit(main())
