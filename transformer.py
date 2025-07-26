import torch
from torch import nn
import wandb
from time import time


class RenjuPositionTransformer(nn.Module):
    def __init__(self, vocab_size, start_token_id, pad_token_id, emb_dim=128, n_heads=4, n_layers=4, dim_ffn=128*4, device='cpu'):
        super().__init__()
        self.vocab_size = vocab_size
        self.start_token_id = start_token_id
        self.pad_token_id = pad_token_id

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.decoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, dim_feedforward=dim_ffn)
        self.decoder = nn.TransformerEncoder(self.decoder_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, vocab_size)

        self.device = device

    def forward(self, x):
        """
        x: [batch_size, seq_len]
        return: [seq_len, batch_size, vocab_size]
        """
        x = torch.cat([torch.full((x.shape[0], 1), self.start_token_id, dtype=torch.long).to(self.device), x], dim=1)
        tokens = x.clone()
        # (N, T)
        x = self.embedding(x) # (N, T, E)
        N, T, E = x.shape

        # positional embeddings (only color)
        for i in range(T):
            x[:,i,0] = i % 2

        mask = nn.Transformer.generate_square_subsequent_mask(T).to(self.device)
        x = x.transpose(0, 1) # (T, N, E)
        x = self.decoder(x, mask=mask, src_key_padding_mask=(tokens == self.pad_token_id))
        x = self.ln(x)
        x = self.head(x)
        return x

    def generate(self, moves, n=10, W=15):
        tokens = transform_to_tokens(moves, W)
        position = torch.LongTensor(tokens).view(1, -1)
        for i in range(n):
            logits = self.forward(position)[-1,0].detach()
            probs = nn.functional.softmax(logits, dim=-1)

            for token in position[0]:
                probs[token] = 0
            probs /= torch.sum(probs)

            token = torch.multinomial(probs, 1)
            position = torch.cat([ position, torch.LongTensor([[token]]).to(self.device)], dim=1)

        result = [(p // 15, p % 15) for p in position[0]]
        return result


def PerplexityLoss(logits, true_tokens, pad_token_id=15*15):
    """
    Args:
        logits: [seq_len, batch_size, vocab_size]
        true_tokens: [batch_size, seq_len]

    Returns: loss

    """
    true_tokens = true_tokens.transpose(0, 1)
    mask = (true_tokens != pad_token_id)
    flat_logits = logits[:-1,:,:].view(-1, logits.size(-1))
    flat_tokens = true_tokens.reshape(-1)
    flat_mask = mask.reshape(-1)
    log_probs = nn.functional.log_softmax(flat_logits, dim=-1) * flat_mask.view(-1, 1)
    return torch.mean(-log_probs[torch.arange(flat_tokens.shape[0]),flat_tokens])


class RenjuPositionsDatasetFullPositions(torch.utils.data.Dataset):
    def __init__(self, filename='positions.txt', from_line=0, to_line=100_000):
        super().__init__()
        self.positions = []
        for li, line in enumerate(open(filename, 'r')):
            if len(line.strip()) == 0:
                continue
            if from_line <= li and li <= to_line:
                moves = line.strip().split(';')
                moves = [(int(m.split(',')[0]), int(m.split(',')[1])) for m in moves]
                self.positions.append(moves)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, i):
        return self.positions[i]


def collate(batch):
    return batch

def transform_to_tokens(lst, W=15):
    res = []
    for tup in lst:
        res.append(tup[0] * W + tup[1])
    return res

def pad_tokens(arrs_list, pad_token_id=15*15):
    max_len = max(len(lst) for lst in arrs_list)
    res = []
    for arr in arrs_list:
        res.append(arr + [pad_token_id] * (max_len - len(arr)))
    return res





def train(name='renju'):
    device = torch.device('mps' if torch.mps.is_available() else 'cpu')
    print('USING DEVICE:', device)
    batch_size = 64
    val_batches = 50
    eval_every = 100

    H, W = 15, 15
    pad_token_id = H * W
    start_token_id = pad_token_id + 1

    train_dataset = RenjuPositionsDatasetFullPositions(from_line=0, to_line=100_000)
    val_dataset = RenjuPositionsDatasetFullPositions(from_line=100_001, to_line=100_000 + batch_size * val_batches)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate
    )

    model = RenjuPositionTransformer(vocab_size=start_token_id + 1,
                                     start_token_id=start_token_id,
                                     pad_token_id=pad_token_id,
                                     device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    loss_fn = PerplexityLoss

    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="deeporigin",
        # Set the wandb project where this run will be logged.
        project="MaximLavrikPersonalProects",
        # Track hyperparameters and run metadata.
        name=name,
        config=dict(batch_size=batch_size),
    )

    ts = []

    for epoch in range(100):
        for batch_idx, data in enumerate(train_dataloader):
            data = [transform_to_tokens(lst) for lst in data]
            data = pad_tokens(data, pad_token_id=pad_token_id)
            data = torch.tensor(data, dtype=torch.long).to(device)

            x = model(data)
            loss = loss_fn(x, data)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch_idx % eval_every == 0:
                torch.save(model.state_dict(), f"./checkpoints_transformer/{name}_{epoch}_{batch_idx}.pt")
                sum_val_loss = 0
                for val_batch_idx, data in enumerate(val_dataloader):
                    data = [transform_to_tokens(lst) for lst in data]
                    data = pad_tokens(data, pad_token_id=pad_token_id)
                    data = torch.tensor(data, dtype=torch.long).to(device)

                    x = model(data.clone())
                    loss = loss_fn(x, data)

                    sum_val_loss += loss.item()

                sum_val_loss = sum_val_loss / len(val_dataloader)
                print(f'{name}_{epoch}_{batch_idx}', sum_val_loss)

                ts.append(time())
                if len(ts) >= 2:
                    print(f'Time per {eval_every} batches: {ts[-1] - ts[-2]:.2f}s')


            run.log({"loss": loss.item(), 'val_loss': sum_val_loss})


if __name__ == '__main__':
    train('fixed_loss')