import numpy as np
from itertools import product

IDX_CHARMAP = '0123456789ABCDEFGHIJKL'
ALLOW_WARP = True
DEBUG = True
N = 20

ENTRY_POINT = (1, 0)
ENTRY_DIRECTION = (0, 1)

START = 3
GOAL = 2
PATH = 1
BLOCK = 0
UNSET = -1

BLOCK_CHAR = "█"
PATH_CHAR = "░"
UNSET_CHAR = " "
GOAL_CHAR = "$"

BFS_steps_queue = []

max_len = 0
max_len_coords = ENTRY_POINT


def update_max_len_coords(x, y, length):
    global max_len, max_len_coords
    if length > max_len:
        max_len_coords = (x, y)
        max_len = length


def val_to_char(val):
    if val == PATH:
        return PATH_CHAR
    if val == BLOCK:
        return BLOCK_CHAR
    if val == GOAL:
        return "$"
    return " "


def draw(grid):
    print('\n ' + IDX_CHARMAP)
    for i, row in enumerate(grid.tolist()):
        print(IDX_CHARMAP[i], end="")
        print("".join(map(val_to_char, row)))
    print()


def is_legal_step(grid, x, y, dx, dy):
    try:
        return grid[y, x] == UNSET and grid[y + dy, x + dx] in (UNSET, BLOCK)
    except IndexError:
        return False


def do_step(grid, x, y, length):
    assert grid[y, x] == UNSET, f'({y}, {x}) == {val_to_char(grid[y, x])}'
    grid[y, x] = PATH
    update_max_len_coords(x, y, length)
    if DEBUG:
        draw(grid)


def get_random_fork_config(
        keep_straight_options=(0, 1),
        fork_left_options=(0, 1),
        fork_right_options=(0, 1)
):
    options = np.array(list(product(keep_straight_options, fork_left_options, fork_right_options)))
    options = [option for option in options if any(option)]
    if not options:
        return (0, 0, 0)
    n_forks = np.sum(options, axis=1, dtype=float)
    probabilities_raw = 1 / n_forks
    probabilities_normalized = probabilities_raw / np.sum(probabilities_raw)
    idx = np.random.choice(range(len(options)), p=probabilities_normalized)
    return options[idx]


def try_block(grid, x, y, sx, sy):
    try:
        assert grid[y, x] in (UNSET, BLOCK), f'cannot block ({y}, {x}) == {val_to_char(grid[y, x])} [{sy},{sx}]'
        grid[y, x] = BLOCK
    except IndexError:
        pass
    except AssertionError:
        pass  # TODO fix


def fork(grid, x, y, dx, dy, length):
    dx0, dy0 = dx, dy
    ldx, ldy = dy, -dx
    rdx, rdy = -dy, dx
    keep_straight, fork_left, fork_right = get_random_fork_config(
        keep_straight_options=(0, 1) if is_legal_step(grid, x + dx0, y + dy0, dx0, dy0) else (0,),
        fork_left_options=(0, 1) if is_legal_step(grid, x + ldx, y + ldy, ldx, ldy) else (0,),
        fork_right_options=(0, 1) if is_legal_step(grid, x + rdx, y + rdy, rdx, rdy) else (0,),
    )

    def try_block_diagonals():
        if keep_straight and fork_left:
            try_block(grid, x + dx + ldx, y + dy + ldy, x, y)
        if keep_straight and fork_right:
            try_block(grid, x + dx + rdx, y + dy + rdy, x, y)

    try_block_diagonals()
    for fork_in_direction, dx, dy in [(keep_straight, dx0, dy0), (fork_left, ldx, ldy), (fork_right, rdx, rdy)]:
        if DEBUG:
            print((x + dx, y + dy), PATH_CHAR if fork_in_direction else BLOCK_CHAR)
        if fork_in_direction:
            # add_step(grid,x+dx,y+dy,dx,dy,length+1)
            BFS_steps_queue.append((x + dx, y + dy, dx, dy, length + 1))
        else:
            try_block(grid, x + dx, y + dy, x, y)


def add_step(grid, x, y, dx, dy, length):
    if not is_legal_step(grid, x, y, dx, dy):
        return
    do_step(grid, x, y, length)
    fork(grid, x, y, dx, dy, length)


def init():
    x, y = ENTRY_POINT
    dx, dy = ENTRY_DIRECTION
    grid = np.zeros((N + 2, N + 2))
    grid[:] = UNSET
    if not ALLOW_WARP:
        grid[0, :] = grid[:, 0] = BLOCK
        grid[-1, :] = grid[:, -1] = BLOCK
    grid[y, x] = UNSET

    return grid, x, y, dx, dy


def post_process(grid):
    x, y = max_len_coords
    grid[y, x] = GOAL
    grid[grid == UNSET] = PATH


def main():
    grid, x, y, dx, dy = init()
    BFS_steps_queue.append((x, y, dx, dy, 0))
    while BFS_steps_queue:
        x, y, dx, dy, length = BFS_steps_queue.pop(0)
        add_step(grid, x, y, dx, dy, length)
    post_process(grid)
    draw(grid)


main()
