"""Plain backtracking Sudoku solver used by server.py's /solve endpoint."""


def _peers(r, c):
    peers = set()
    for k in range(9):
        peers.add((r, k))
        peers.add((k, c))
    br, bc = (r // 3) * 3, (c // 3) * 3
    for dr in range(3):
        for dc in range(3):
            peers.add((br + dr, bc + dc))
    peers.discard((r, c))
    return peers


_PEERS = {(r, c): _peers(r, c) for r in range(9) for c in range(9)}


def find_conflicts(grid):
    """grid: 9x9 list of ints (0 = blank). Returns sorted [(r, c), ...] of cells
    that duplicate a given digit in their row, column, or box."""
    conflicts = set()
    for r in range(9):
        for c in range(9):
            v = grid[r][c]
            if v == 0:
                continue
            for pr, pc in _PEERS[(r, c)]:
                if grid[pr][pc] == v:
                    conflicts.add((r, c))
                    conflicts.add((pr, pc))
    return sorted(conflicts)


def solve(grid):
    """Solves `grid` (9x9 list of ints, 0 = blank) in place via backtracking.
    Returns True if a solution was found."""
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                used = {grid[pr][pc] for pr, pc in _PEERS[(r, c)]}
                for v in range(1, 10):
                    if v not in used:
                        grid[r][c] = v
                        if solve(grid):
                            return True
                        grid[r][c] = 0
                return False
    return True
