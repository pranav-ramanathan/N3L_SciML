import time
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from no_three_in_line import generate_training_file
import torch

def main(
    n: int = typer.Option(5, help="Grid size (n×n)"),
    batch_size: int = typer.Option(10_000, help="Number of boards to generate"),
    save_path: str = typer.Option("training_data.jld2", help="Output .jld2 file path"),
    verbose: bool = typer.Option(True, help="Print detailed progress with Rich"),
):
    """Generate no-three-in-line training data for SciML UDE training."""
    console = Console()

    device_guess = "mps" if getattr(torch, "mps", None) and torch.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        tbl = Table(title="No-Three-In-Line Dataset Generation", show_lines=True)
        tbl.add_column("Param")
        tbl.add_column("Value")
        tbl.add_row("n", str(n))
        tbl.add_row("batch_size", str(batch_size))
        tbl.add_row("save_path", save_path)
        tbl.add_row("detected_device", device_guess)
        console.print(tbl)

    start_total = time.time()
    try:
        with console.status("[bold green]Generating dataset... This may take a while for large batches", spinner="dots") as status:
            t0 = time.time()
            path = generate_training_file(n, batch_size, save_path)
            gen_time = time.time() - t0
            status.update(status="[bold green]Finalizing and saving file...")

        size_info = "unknown"
        try:
            import os
            size_info = f"{os.path.getsize(save_path):,} bytes"
        except Exception:
            pass

        if verbose:
            console.print(Panel.fit(
                f"Saved dataset to [bold]{save_path}[/bold]\n"
                f"Generation time: {gen_time:.2f}s\n"
                f"Total time: {time.time() - start_total:.2f}s\n"
                f"File size: {size_info}",
                title="Success",
                border_style="green"
            ))
        else:
            console.print(f"Saved to {save_path}")

    except Exception as e:
        console.print(Panel.fit(str(e), title="Error", border_style="red"))
        raise

if __name__ == "__main__":
    typer.run(main)