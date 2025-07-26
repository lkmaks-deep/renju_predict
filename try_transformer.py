import matplotlib.pyplot as plt
import torch
from torch import nn
from transformer import RenjuPositionTransformer


def plot_renju_board(moves, name='renjuboard'):
    """
    Plots a Renju board (15x15) and shows the given moves.

    Args:
        moves: List of (row, col) tuples, 0-indexed. Black moves first.
    """
    size = 15
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw grid
    for i in range(size):
        ax.plot([0, size - 1], [i, i], color='black')
        ax.plot([i, i], [0, size - 1], color='black')

    # Star points (hoshi)
    hoshi_points = [(3, 3), (3, 11), (7, 7), (11, 3), (11, 11)]
    for y, x in hoshi_points:
        ax.plot(x, y, 'ko', markersize=4)

    # Plot moves
    for idx, (row, col) in enumerate(moves):
        color = 'black' if idx % 2 == 0 else 'white'
        edge = 'black'
        ax.add_patch(plt.Circle((col, row), 0.4, color=color, ec=edge, zorder=2))
        ax.text(col, row, str(idx + 1), color='red' if color == 'white' else 'white',
                fontsize=8, ha='center', va='center', zorder=3)

    ax.set_xlim(-1, size)
    ax.set_ylim(-1, size)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()  # (0,0) should be top-left like a Renju board
    ax.set_title('Renju Board')
    plt.grid(False)
    # plt.savefig(f'plots/{name}.png')\
    plt.show()


def make_pos_tensor(moves):
    position = torch.zeros((1, 15, 15))
    for i, m in enumerate(moves):
        position[0, m[0], m[1]] = 2 * (1 - i % 2) - 1
    return position


moves = [(7, 7), (6, 7), (9, 7)]
color = 1
plot_renju_board(moves, 'start_pos')


H, W = 15, 15
pad_token_id = H * W
start_token_id = pad_token_id + 1

model = RenjuPositionTransformer(
    vocab_size=start_token_id + 1,
    start_token_id=start_token_id,
    pad_token_id=pad_token_id,
)

dc = torch.load('./checkpoints_transformer/fixed_loss_9_600.pt')
model.load_state_dict(dc)


moves = model.generate([(7, 7), (6, 7), (9, 7)], 15)

# for i in range(len(moves)):
#     plot_renju_board(moves[:i+1], f'after_{i+1}')

plot_renju_board(moves, 'start_pos')