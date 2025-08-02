import json
import torch
from utils import transform_to_tokens
from torch import nn


class RPTConfig:
    def __init__(
        self,
        H=15,
        W=15,
        vocab_size=225 + 2,
        start_token_id=225,
        pad_token_id=226,
        d_model=128,
        n_heads=4,
        n_layers=4,
        dim_ffn=2048,
        **kwargs,
    ):
        self.H = H
        self.W = W
        self.vocab_size = vocab_size
        self.start_token_id = start_token_id
        self.pad_token_id = pad_token_id

        self.d_model = d_model
        self.nhead = n_heads
        self.n_layers = n_layers
        self.dim_ffn = dim_ffn

    def to_dict(self):
        return dict(
            H=self.H,
            W=self.W,
            vocab_size=self.vocab_size,
            start_token_id=self.start_token_id,
            pad_token_id=self.pad_token_id,
            d_model=self.d_model,
            nhead=self.nhead,
            n_layers=self.n_layers,
            dim_ffn=self.dim_ffn,
        )

    def save(self, path):
        json.dump(
            self.to_dict(),
            open(path, "w"),
            indent=2,
        )

    @staticmethod
    def load(path):
        dc = json.load(open(path, "r"))
        return RPTConfig(**dc)


class RenjuPositionTransformer(nn.Module):
    def __init__(
        self,
        conf,
        device="cpu",
    ):
        super().__init__()

        self.conf = conf

        self.H = conf.H
        self.W = conf.W
        self.vocab_size = conf.vocab_size
        self.start_token_id = conf.start_token_id
        self.pad_token_id = conf.pad_token_id

        self.d_model = conf.d_model
        self.nhead = conf.nhead
        self.n_layers = conf.n_layers
        self.dim_ffn = conf.dim_ffn

        self.device = device

        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_ffn
        )
        self.decoder = nn.TransformerEncoder(
            self.decoder_layer, num_layers=self.n_layers
        )
        self.ln = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, x):
        """
        x: [batch_size, seq_len]
        return: [seq_len, batch_size, vocab_size]
        """
        x = torch.cat(
            [
                torch.full((x.shape[0], 1), self.start_token_id, dtype=torch.long).to(
                    self.device
                ),
                x,
            ],
            dim=1,
        )
        tokens = x.clone()
        # (N, T)
        x = self.embedding(x)  # (N, T, E)
        N, T, E = x.shape

        # positional embeddings (only color)
        for i in range(T):
            x[:, i, 0] = i % 2

        mask = nn.Transformer.generate_square_subsequent_mask(T).to(self.device)
        x = x.transpose(0, 1)  # (T, N, E)
        x = self.decoder(
            x, mask=mask, src_key_padding_mask=(tokens == self.pad_token_id)
        )
        x = self.ln(x)
        x = self.head(x)
        return x

    def generate(self, moves, n=10):
        tokens = transform_to_tokens(moves, self.W)
        position = torch.LongTensor(tokens).view(1, -1)
        for i in range(n):
            logits = self.forward(position)[-1, 0].detach()
            probs = nn.functional.softmax(logits, dim=-1)

            for token in position[0]:
                probs[token] = 0
            probs /= torch.sum(probs)

            token = torch.multinomial(probs, 1)
            position = torch.cat(
                [position, torch.LongTensor([[token]]).to(self.device)], dim=1
            )

        result = [(p // 15, p % 15) for p in position[0]]
        return result

    def generate_beam_search(self, moves, n=10, B=64):
        tokens = transform_to_tokens(moves, self.conf.W)
        positions = torch.LongTensor(tokens).view(1, -1)
        pos_log_probs = [0]
        for i in range(n):
            logits = self.forward(positions)[-1,:,:].detach()
            log_probs = nn.functional.log_softmax(logits, dim=-1)

            all_options = []
            for pid in range(log_probs.shape[0]):
                top_tids = []
                for tid in range(log_probs.shape[1]):
                    if tid not in positions[pid]:
                        top_tids.append((log_probs[pid, tid], tid))
                top_tids = sorted(top_tids, key=lambda x: x[0], reverse=True)
                for val, tid in top_tids[:B]:
                    all_options.append((pid, tid, pos_log_probs[pid] + val))

            all_options.sort(key=lambda x: x[2], reverse=True)

            new_positions = []
            for pid, tid, _ in all_options[:B]:
                new_positions.append(torch.cat([positions[pid], torch.LongTensor([tid])], dim=0))

            positions = torch.stack(new_positions, dim=0)
            pos_log_probs = [x[2] for x in all_options[:B]]

        result = [[(p // self.conf.W, p % self.conf.W) for p in positions[i]] for i in range(positions.shape[0])]
        return result

    def save_config(self, path):
        self.conf.save(path)
