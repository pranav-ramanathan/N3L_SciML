# generate_training_data.py
import os
import time
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from no_three_in_line import generate_training_file
import torch

app = typer.Typer(add_completion=False)

@app.command()
def main(
    n: int = typer.Option(10, help="Grid size (n×n)"),
    batch_size: int = typer.Option(20_000, help="Number of boards to generate"),
    save_path: str = typer.Option("data/n_10.h5", help="Output dataset file path (.h5 recommended)"),
    verbose: bool = typer.Option(True, help="Print progress with Rich"),
    # Advanced options (mirrors no_three_in_line.generate_training_file):
    save_triplets_mode: str = typer.Option(
        "rowcol1",
        help='How to save triplets: "rowcol1" for 1-based (row,col) (Julia-friendly) or "xy0" for 0-based (x,y).',
    ),
    aggressive_blocking: bool = typer.Option(
        True,
        help="Cheap heuristic: block whole row/col/diag when they already have ≥2 points.",
    ),
    post_add_full_update: bool = typer.Option(
        False,
        help="After each successful add, run full forbidden-square refresh (slower, stronger pruning).",
    ),
    seed: int = typer.Option(42, help="Random seed (forwarded to generator; may be used in future)"),
):
    """
    Generate no-three-in-line training data for SciML / UDE training.

    Notes:
      • The underlying writer uses HDF5. Using a '.h5' extension is recommended,
        even if you choose a different file name here.
      • With save_triplets_mode='rowcol1', Julia can consume /triplets directly:
          (i,j) are 1-based (row, col), matching `reshape(x, n, n)[i, j]`.
    """
    console = Console()

    # Device probe (informational)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device_guess = "mps"
    elif torch.cuda.is_available():
        device_guess = "cuda"
    else:
        device_guess = "cpu"

    if verbose:
        tbl = Table(title="No-Three-In-Line Dataset Generation", show_lines=True)
        tbl.add_column("Param", no_wrap=True)
        tbl.add_column("Value")
        tbl.add_row("n", str(n))
        tbl.add_row("batch_size", f"{batch_size:,}")
        tbl.add_row("save_path", save_path)
        tbl.add_row("detected_device", device_guess)
        tbl.add_row("save_triplets_mode", save_triplets_mode)
        tbl.add_row("aggressive_blocking", str(aggressive_blocking))
        tbl.add_row("post_add_full_update", str(post_add_full_update))
        tbl.add_row("seed", str(seed))
        # Extension hint
        ext = os.path.splitext(save_path)[1].lower()
        if ext not in (".h5", ".hdf5"):
            tbl.add_row("⚠︎ note", "Writer is HDF5; consider using .h5")
        console.print(tbl)

    # Ensure directory exists before generator tries to save
    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    start_total = time.time()
    try:
        with console.status(
            "[bold green]Generating dataset... (can take a while for large batches)",
            spinner="dots",
        ) as status:
            t0 = time.time()
            path = generate_training_file(
                n=n,
                batch_size=batch_size,
                save_path=save_path,
                seed=seed,
                save_triplets_mode=save_triplets_mode,
                post_add_full_update=post_add_full_update,
                aggressive_blocking=aggressive_blocking,
            )
            gen_time = time.time() - t0
            status.update(status="[bold green]Finalizing...")

        # File size
        try:
            size_bytes = os.path.getsize(save_path)
            size_str = f"{size_bytes/1e6:.1f} MB"
        except Exception:
            size_str = "unknown"

        if verbose:
            console.print(
                Panel.fit(
                    f"Saved dataset to [bold]{path}[/bold]\n"
                    f"Generation time: {gen_time:.2f}s\n"
                    f"Total time: {time.time() - start_total:.2f}s\n"
                    f"File size: {size_str}",
                    title="Success",
                    border_style="green",
                )
            )
        else:
            console.print(f"Saved to {path}")

    except Exception as e:
        console.print(Panel.fit(str(e), title="Error", border_style="red"))
        raise

if __name__ == "__main__":
    app()
