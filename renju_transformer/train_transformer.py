import torch
import wandb
from time import time
from transformer import RenjuPositionTransformer, RPTConfig
from utils import pad_tokens, PerplexityLoss, transform_to_tokens


class RenjuPositionsDatasetFullPositions(torch.utils.data.Dataset):
    def __init__(
        self, filename="positions.txt", from_line=0, to_line=100_000, transform=True
    ):
        super().__init__()
        self.transform = transform
        self.positions = []
        for li, line in enumerate(open(filename, "r")):
            if len(line.strip()) == 0:
                continue
            if from_line <= li and li < to_line:
                moves = line.strip().split(";")
                moves = [(int(m.split(",")[0]), int(m.split(",")[1])) for m in moves]
                self.positions.append(moves)

    def __len__(self):
        return len(self.positions)

    def rotate_move(self, move, n_times=0):
        for _ in range(n_times):
            move = move[1], -(move[0] - 7) + 7
        return move

    def mirror_move(self, move, x=True):
        if x:
            return move[0], 14 - move[1]
        else:
            return 14 - move[0], move[1]

    def mirror(self, position, x=True):
        return [self.mirror_move(move) for move in position]

    def rotate(self, position, n_times=0):
        return [self.rotate_move(move, n_times) for move in position]

    def __getitem__(self, i):
        import random

        position = self.positions[i]
        if self.transform:
            position = self.rotate(position, random.randint(0, 3))
            if random.randint(0, 1) == 0:
                position = self.mirror(position, x=True)
            if random.randint(0, 1) == 0:
                position = self.mirror(position, x=False)
        return position


def train(
    conf,
    batch_size=64,
    eval_every=100,
    name="renju",
    dataset_file="positions.txt",
    train_examples=100_000,
    val_examples=1_000,
    transform=True,
):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print("USING DEVICE:", device)

    dataset = RenjuPositionsDatasetFullPositions(
        filename=dataset_file,
        from_line=0,
        to_line=train_examples + val_examples,
        transform=transform,
    )

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_examples, val_examples]
    )

    def collate(batch):
        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate
    )

    model = RenjuPositionTransformer(
        conf=conf,
        device=device,
    ).to(device)
    model.save_config(f"configs/config_{name}.json")

    optimizer = torch.optim.Adam(model.parameters())

    loss_fn = PerplexityLoss

    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="deeporigin",
        # Set the wandb project where this run will be logged.
        project="MaximLavrikPersonalProects",
        # Track hyperparameters and run metadata.
        name=name,
        config=conf.to_dict().update(
            dict(
                batch_size=batch_size,
                eval_every=eval_every,
                dataset_file=dataset_file,
                train_examples=train_examples,
                val_examples=val_examples,
            )
        ),
    )

    ts = []

    print("Train batches: ", len(train_dataloader))
    print("Val batches: ", len(val_dataloader))
    print(f"Eval every {eval_every} train batches")

    for epoch in range(100):
        for batch_idx, data in enumerate(train_dataloader):
            data = [transform_to_tokens(lst) for lst in data]
            data = pad_tokens(data, pad_token_id=conf.pad_token_id)
            data = torch.tensor(data, dtype=torch.long).to(device)

            x = model(data)
            loss = loss_fn(x, data)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch_idx % eval_every == 0:
                torch.save(
                    model.state_dict(),
                    f"./checkpoints/{name}_{epoch}_{batch_idx}.pt",
                )

                sum_val_loss = 0
                for val_batch_idx, data in enumerate(val_dataloader):
                    data = [transform_to_tokens(lst) for lst in data]
                    data = pad_tokens(data, pad_token_id=pad_token_id)
                    data = torch.tensor(data, dtype=torch.long).to(device)

                    x = model(data.clone())
                    loss = loss_fn(x, data)

                    sum_val_loss += loss.item()

                sum_val_loss = sum_val_loss / len(val_dataloader)
                print(f"{name}_{epoch}_{batch_idx}", sum_val_loss)

                ts.append(time())
                if len(ts) >= 2:
                    print(
                        f"Time for {eval_every} batches (unless its last batches): {ts[-1] - ts[-2]:.2f}s"
                    )

            run.log({"loss": loss.item(), "val_loss": sum_val_loss})


if __name__ == "__main__":
    name = "playok_100k"

    H, W = 15, 15
    vocab_size = H * W + 2
    start_token_id = H * W
    pad_token_id = H * W + 1

    train_examples = 100_000
    val_examples = 3_000
    eval_every = 100

    d_model = 256
    n_heads = 4
    n_layers = 1
    dim_ffn = d_model * 4
    batch_size = 64

    conf = RPTConfig(
        H=H,
        W=W,
        vocab_size=vocab_size,
        start_token_id=start_token_id,
        pad_token_id=pad_token_id,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dim_ffn=dim_ffn,
    )

    train_kwargs = dict(
        name=name,
        batch_size=batch_size,
        eval_every=eval_every,
        dataset_file="../scrape_gomoku/results/gomoku_positions.txt",
        train_examples=train_examples,
        val_examples=val_examples,
        transform=True,
    )

    train(conf, **train_kwargs)
