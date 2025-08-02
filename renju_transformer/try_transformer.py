import matplotlib.pyplot as plt
import torch
from renju_transformer.train_transformer import RenjuPositionTransformer, RPTConfig
from general_utils import position_from_putpos


def plot_renju_board(moves, name="renjuboard", save=False):
    """
    Plots a Renju board (15x15) and shows the given moves.

    Args:
        moves: List of (row, col) tuples, 0-indexed. Black moves first.
    """
    size = 15
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw grid
    for i in range(size):
        ax.plot([0, size - 1], [i, i], color="black")
        ax.plot([i, i], [0, size - 1], color="black")

    # Star points (hoshi)
    hoshi_points = [(3, 3), (3, 11), (7, 7), (11, 3), (11, 11)]
    for y, x in hoshi_points:
        ax.plot(x, y, "ko", markersize=4)

    # Plot moves
    for idx, (row, col) in enumerate(moves):
        color = "black" if idx % 2 == 0 else "white"
        edge = "black"
        ax.add_patch(plt.Circle((col, row), 0.4, color=color, ec=edge, zorder=2))
        ax.text(
            col,
            row,
            str(idx + 1),
            color="red" if color == "white" else "white",
            fontsize=8,
            ha="center",
            va="center",
            zorder=3,
        )

        # Add coordinates
    col_labels = [chr(ord("A") + i) for i in range(size)]
    row_labels = list(reversed([str(i + 1) for i in range(size)]))

    for i in range(size):
        # Top and bottom column labels
        ax.text(i, -1, col_labels[i], ha="center", va="center", fontsize=10)
        ax.text(i, size, col_labels[i], ha="center", va="center", fontsize=10)

        # Left and right row labels
        ax.text(-1, i, row_labels[i], ha="center", va="center", fontsize=10)
        ax.text(size, i, row_labels[i], ha="center", va="center", fontsize=10)

    ax.set_xlim(-2, size + 1)
    ax.set_ylim(-2, size + 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_title("Renju Board")
    plt.grid(False)

    if save:
        plt.savefig(f"plots/{name}.png")
    else:
        plt.show()


def make_pos_tensor(moves):
    position = torch.zeros((1, 15, 15))
    for i, m in enumerate(moves):
        position[0, m[0], m[1]] = 2 * (1 - i % 2) - 1
    return position


name = "playok_100k"
epoch = 8
batch = 300
putpos = "g7j10l13"

start_moves = position_from_putpos(putpos)

config = RPTConfig.load(f"configs/config_{name}.json")
model = RenjuPositionTransformer(config, device="cpu")

H = model.conf.H
W = model.conf.W
pad_token_id = model.conf.pad_token_id
start_token_id = model.conf.start_token_id

dc = torch.load(f"./checkpoints/{name}_{epoch}_{batch}.pt")
model.load_state_dict(dc)

moves = model.generate_beam_search(start_moves, n=10, B=16)

# for i in range(len(moves)):
#     plot_renju_board(moves[:i+1], f'after_{i+1}')

for i in range(len(moves)):
    plot_renju_board(moves[i], f"start_pos_{i}")
