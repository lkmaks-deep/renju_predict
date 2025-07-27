import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np



def exponential_smoothing(series, alpha):
    """
    Apply exponential smoothing to a 1D tensor.
    """
    result = torch.zeros_like(torch.tensor(series))
    result[0] = series[0]
    for t in range(1, len(series)):
        result[t] = alpha * series[t] + (1 - alpha) * result[t - 1]
    return result


class RenjuPositionsDataset(torch.utils.data.Dataset):
    def __init__(self, filename='positions.txt', from_line=0, to_line=100_000):
        super().__init__()
        self.positions = []
        self.pref_lengths = [0]
        for li, line in enumerate(open(filename, 'r')):
            if len(line.strip()) == 0:
                continue
            if from_line <= li and li <= to_line:
                moves = line.strip().split(';')
                moves = [(int(m.split(',')[0]), int(m.split(',')[1])) for m in moves]
                self.positions.append(moves)
                self.pref_lengths.append(self.pref_lengths[-1] + len(moves))

    def __len__(self):
        return self.pref_lengths[-1]

    def __getitem__(self, i):
        pos_id = 0
        while self.pref_lengths[pos_id] <= i:
            pos_id += 1

        i -= self.pref_lengths[pos_id]
        pos_id -= 1
        moves = self.positions[pos_id][:i]

        position = torch.zeros((15, 15))
        for i, m in enumerate(moves):
            position[m[0], m[1]] = 2 * (1 - i % 2) - 1

        return position, i % 2, self.positions[pos_id][i][0], self.positions[pos_id][i][1]


# model = nn.Sequential(nn.Conv2d(2, 4, 3, padding=1),
#                       nn.ReLU(),
#                       nn.Conv2d(4, 8, 3, padding=1),
#                       nn.ReLU(),
#                       nn.Conv2d(8, 16, 3, padding=1),
#                       nn.ReLU(),
#                       nn.Conv2d(16, 32, 3, padding=1),
#                       nn.ReLU(),
#                       nn.Conv2d(32, 64, 3, padding=1),
#                       nn.ReLU(),
#                       nn.Conv2d(64, 2, 1, padding=0),
#                       )


batch_size = 64
train_dataset = RenjuPositionsDataset(from_line=0, to_line=100_000)
val_dataset = RenjuPositionsDataset(from_line=100_001, to_line=100_000 + 1024)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

h, w = 15, 15
n_in = h * w
model = nn.Sequential(nn.Flatten(),
                      nn.Linear(n_in, n_in * 2),
                      nn.BatchNorm1d(n_in * 2),
                      nn.ReLU(),
                      nn.Linear(n_in * 2, n_in * 4),
                      nn.BatchNorm1d(n_in * 4),
                      nn.ReLU(),
                      nn.Linear(n_in * 4, n_in * 4),
                      nn.BatchNorm1d(n_in * 4),
                      nn.ReLU(),
                      nn.Linear(n_in * 4, n_in * 4),
                      nn.BatchNorm1d(n_in * 4),
                      nn.ReLU(),
                      nn.Linear(n_in * 4, n_in * 2),
                      )


# model.load_state_dict(torch.load('./checkpoints/model_0_14756.pt'))

opt = torch.optim.Adam(model.parameters(), lr=1e-3)


losses = []
val_losses = []

N_iters = len(train_dataset) // batch_size
print(N_iters, 'iterations in one epoch')

for epoch in range(2, 100):
    sum_loss = 0
    for n, (pos, color, x, y) in enumerate(train_dataloader):
        logits = model(pos)
        labels = torch.tensor([x[i] * h + y[i] for i in range(pos.shape[0])])

        loss = 0
        for i in range(pos.shape[0]):
            start = n_in * color[i]
            loss += nn.CrossEntropyLoss()(logits[i][start:start+n_in], labels[i])
        loss /= pos.shape[0]

        loss.backward()
        opt.step()
        opt.zero_grad()

        sum_loss += loss.item()

        losses.append(loss.item())

        if n % 100 == 0:
            sum_val_loss = 0
            for m, (pos, color, x, y) in enumerate(val_dataloader):
                logits = model(pos)
                labels = torch.tensor([x[i] * h + y[i] for i in range(pos.shape[0])])

                val_loss = 0
                for i in range(pos.shape[0]):
                    start = n_in * color[i]
                    val_loss += nn.CrossEntropyLoss()(logits[i][start:start + n_in], labels[i])
                val_loss /= pos.shape[0]

                sum_val_loss += val_loss.item()

            res_val_loss = sum_val_loss / m

        val_losses.append(res_val_loss)
        plt.plot(exponential_smoothing(losses, alpha=0.9))
        plt.plot(exponential_smoothing(val_losses, alpha=0.9))
        plt.show()

        print(f'EPOCH {epoch}:', losses[-1], val_losses[-1])
        # -log(p) = loss
        # p = e^-loss
        print(np.exp(-val_losses[-1]), 1 / 225)

        if n % (N_iters // 100) == 0:
            torch.save(model.state_dict(), f'./checkpoints/model_{epoch}_{n}.pt')

