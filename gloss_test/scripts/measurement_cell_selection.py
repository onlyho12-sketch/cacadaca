"""Pure parser for sparse representative grid-cell measurements."""


def parse_measurement_cells(value, grid):
    """Parse semicolon-separated 1-based cells, preserving their order."""
    if value is None:
        return None
    cells = []
    for token in value.split(";"):
        parts = token.strip().split(",")
        if len(parts) != 2:
            raise ValueError(f"invalid measurement cell: {token!r}")
        cell = (int(parts[0]), int(parts[1]))
        if not (1 <= cell[0] <= grid and 1 <= cell[1] <= grid):
            raise ValueError(f"measurement cell outside {grid}x{grid}: {cell}")
        if cell not in cells:
            cells.append(cell)
    if not cells:
        raise ValueError("at least one measurement cell is required")
    center = (grid // 2 + 1, grid // 2 + 1)
    if center not in cells:
        cells.append(center)
    return cells
