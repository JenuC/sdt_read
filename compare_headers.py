import sdtfile
import numpy as np
from rich import print
from rich.console import Console
from rich.table import Table
import sys


def compare_records(rec1, rec2, name1="Data1", name2="Data2"):
    """
    Compares two NumPy records and displays the differences and similarities in a table.
    Handles cases where dtypes are not exactly the same by comparing common fields.
    """
    table = Table(title=f"Comparison of {name1} and {name2}")
    table.add_column("Field", style="white")
    table.add_column(name1, style="cyan")
    table.add_column(name2, style="magenta")
    table.add_column("Status", style="green")

    # Get the common fields between the two records
    common_fields = set(rec1.dtype.names) & set(rec2.dtype.names)

    if not common_fields:
        print(f"[red]No common fields found between {name1} and {name2}[/red]")
        return

    diffs = {}
    sames = {}
    for field in common_fields:
        val1 = str(rec1[field])
        val2 = str(rec2[field])
        if np.array_equal(rec1[field], rec2[field]):
            sames[field] = (val1, val2)
        else:
            diffs[field] = (val1, val2)

    # Add similar records first
    for field, (val1, val2) in sames.items():
        table.add_row(field, val1, val2, "[green]Same[/green]")

    # Add different records after
    for field, (val1, val2) in diffs.items():
        table.add_row(field, val1, val2, "[red]Different[/red]")

    console.print(table)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        fn1 = sys.argv[1]
        fn2 = sys.argv[2]
    else:
        fn1 = r"C:\dev\test_data\Images FLIM i3S\coumarin6Image-03202025-1307-001\LifetimeData_Cycle00001_000001.sdt"
        fn2 = r"C:\dev\test_data\enterocytes_FLIM_sdt.sdt"
    console = Console()

data1 = sdtfile.SdtFile(fn1)
data2 = sdtfile.SdtFile(fn2)
img1 = data1.data
print(f"[cyan]Data1: {len(img1)} blocks, first block shape: {img1[0].shape}[/cyan]")
img2 = data2.data
print(
    f"[magenta]Data2: {len(img2)} blocks, first block shape: {img2[0].shape}[/magenta]"
)

print("[bold]Header Comparison[/bold]")
compare_records(data1.header, data2.header, "Data1 Header", "Data2 Header")


print("[bold]Block Header Comparison[/bold]")
compare_records(
    data1.block_headers[0],
    data2.block_headers[0],
    "Data1 Block Header",
    "Data2 Block Header",
)

print("[bold]Measure Info Comparison[/bold]")
compare_records(
    data1.measure_info[0],
    data2.measure_info[0],
    "Data1 Measure Info",
    "Data2 Measure Info",
)
