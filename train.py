import torch
from torch import nn
import matplotlib.pyplot as plt

def gen_data(from_line=0, to_line=1000):
    for li, line in enumerate(open('positions.txt', 'r')):
        if from_line <= li and li <= to_line:
            moves = line.split(';')
            moves = [(int(m.split(',')[0]), int(m.split(',')[1])) for m in moves]
            position = torch.zeros((1, 2, 15, 15))
            for i, m in enumerate(moves):
                yield position, [i % 2], [m[0]], [m[1]]
                position[0, i % 2, m[0], m[1]] = 1

            if li == to_line:
                break


model = nn.Sequential(nn.Conv2d(2, 4, 3, padding=1),
                      nn.ReLU(),
                      nn.Conv2d(4, 8, 3, padding=1),
                      nn.ReLU(),
                      nn.Conv2d(8, 16, 3, padding=1),
                      nn.ReLU(),
                      nn.Conv2d(16, 32, 3, padding=1),
                      nn.ReLU(),
                      nn.Conv2d(32, 64, 3, padding=1),
                      nn.ReLU(),
                      nn.Conv2d(64, 2, 1, padding=0),
                      )

opt = torch.optim.Adam(model.parameters(), lr=1e-3)


losses = []
val_losses = []
N_train = 100_000
N_val = 100

for epoch in range(100):
    sum_loss = 0
    for n, (pos, color, x, y) in enumerate(gen_data(0, N_train)):
        logits = model(pos)

        loss = 0
        for i in range(pos.shape[0]):
            loss += -nn.LogSoftmax()(logits[i,color[i],:,:].flatten())[15 * x[i] + y[i]]

        loss.backward()
        opt.step()
        opt.zero_grad()

        sum_loss += loss.item()

        losses.append(sum_loss / (n + 1))

        if n % 100 == 0:
            val_loss = 0
            for m, (pos, color, x, y) in enumerate(gen_data(N_train, N_val + N_train)):
                logits = model(pos)

                for i in range(pos.shape[0]):
                    val_loss += -nn.LogSoftmax()(logits[i, color[i], :, :].flatten())[15 * x[i] + y[i]]

            val_loss = val_loss.item() / m

        val_losses.append(val_loss)
        plt.plot(losses)
        plt.plot(val_losses)
        plt.show()

        print(losses[-1], val_losses[-1])

    torch.save(model.state_dict(), f'./checkpoints/model_{epoch}.pt')
