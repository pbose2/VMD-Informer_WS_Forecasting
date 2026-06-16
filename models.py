import math
import torch
import torch.nn as nn


class GRUForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=3, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.head(h_n[-1])


class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])


class BiLSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        return self.head(out)


class GRUEncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=3, dropout=0.2):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.encoder = nn.GRU(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0
        )
        self.decoder = nn.GRU(
            input_size  = hidden_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        batch = x.size(0)
        _, h_n = self.encoder(x)
        dec_input = torch.zeros(batch, 1, self.hidden_size).to(x.device)
        h = h_n
        outputs = []
        for _ in range(self.output_size):
            out, h = self.decoder(dec_input, h)
            outputs.append(self.head(out.squeeze(1)))
            dec_input = out
        return torch.cat(outputs, dim=-1)


class AttentionForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 num_heads=4, output_size=3, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0
        )
        self.attention = nn.MultiheadAttention(
            embed_dim   = hidden_size,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        enc_out, _ = self.gru(x)
        query = enc_out[:, -1:, :]
        attn_out, _ = self.attention(query=query, key=enc_out, value=enc_out)
        context = self.norm(attn_out.squeeze(1) + query.squeeze(1))
        return self.head(context)


class TransformerForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 num_heads=4, output_size=3, dropout=0.2, max_len=144):
        super().__init__()
        self.input_proj  = nn.Linear(input_size, hidden_size)
        self.pos_encoding = nn.Embedding(max_len, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = hidden_size,
            nhead           = num_heads,
            dim_feedforward = hidden_size * 4,
            dropout         = dropout,
            batch_first     = True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        _, seq_len, _ = x.shape
        x = self.input_proj(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pos_encoding(positions)
        enc_out = self.encoder(x)
        return self.head(enc_out[:, -1, :])


class CNNLSTM(nn.Module):
    def __init__(self, input_dim, cnn_channels, kernel_size, lstm_hidden,
                 lstm_layers, output_dim, dropout):
        super().__init__()
        self.conv1   = nn.Conv1d(input_dim, cnn_channels, kernel_size=kernel_size)
        self.relu    = nn.ReLU()
        self.pool    = nn.MaxPool1d(kernel_size=2)
        self.lstm    = nn.LSTM(
            input_size  = cnn_channels,
            hidden_size = lstm_hidden,
            num_layers  = lstm_layers,
            batch_first = True,
            dropout     = dropout if lstm_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(lstm_hidden, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.conv1(x)))
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        return self.fc(self.dropout(lstm_out[:, -1, :]))


class GRUEncoderDecoderAttn(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=3, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.encoder = nn.GRU(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0
        )
        self.decoder = nn.GRU(
            input_size  = hidden_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0
        )
        self.W_q          = nn.Linear(hidden_size, hidden_size)
        self.W_k          = nn.Linear(hidden_size, hidden_size)
        self.v            = nn.Linear(hidden_size, 1)
        self.context_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        batch = x.size(0)
        enc_out, h_n = self.encoder(x)
        dec_input = torch.zeros(batch, 1, self.hidden_size).to(x.device)
        h = h_n
        outputs = []
        for _ in range(self.output_size):
            dec_out, h = self.decoder(dec_input, h)
            scores = self.v(torch.tanh(
                self.W_q(dec_out) + self.W_k(enc_out)
            )).squeeze(-1).unsqueeze(1)
            context  = torch.bmm(torch.softmax(scores, dim=-1), enc_out)
            combined = self.context_proj(torch.cat([dec_out, context], dim=-1))
            outputs.append(self.head(combined).squeeze(-1))
            dec_input = combined
        return torch.cat(outputs, dim=-1)


class ProbSparseAttention(nn.Module):
    """ProbSparse self-attention — only top-u queries are computed fully."""
    def __init__(self, embed_dim, num_heads, dropout=0.0, factor=5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.factor    = factor
        self.scale     = self.head_dim ** -0.5
        self.W_q  = nn.Linear(embed_dim, embed_dim)
        self.W_k  = nn.Linear(embed_dim, embed_dim)
        self.W_v  = nn.Linear(embed_dim, embed_dim)
        self.out  = nn.Linear(embed_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def _prob_sparse_scores(self, Q, K):
        B, H, L_Q, D = Q.shape
        _, _, L_K, _ = K.shape
        u        = min(self.factor * int(math.log(L_Q) + 1), L_Q)
        u_sample = min(self.factor * int(math.log(L_K) + 1), L_K)
        idx      = torch.randint(L_K, (u_sample,), device=Q.device)
        K_sample = K[:, :, idx, :]
        Q_K      = torch.matmul(Q, K_sample.transpose(-2, -1))
        M        = Q_K.max(-1).values - Q_K.mean(-1)
        return M.topk(u, dim=-1).indices, u

    def forward(self, Q, K, V):
        B, L_Q, _ = Q.shape
        _, L_K, _ = K.shape
        Q = self.W_q(Q).view(B, L_Q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(K).view(B, L_K, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(V).view(B, L_K, self.num_heads, self.head_dim).transpose(1, 2)
        M_top, u = self._prob_sparse_scores(Q, K)
        Q_reduce = Q[
            torch.arange(B)[:, None, None],
            torch.arange(self.num_heads)[None, :, None],
            M_top
        ]
        scores   = torch.matmul(Q_reduce, K.transpose(-2, -1)) * self.scale
        attn     = self.drop(torch.softmax(scores, dim=-1))
        V_reduce = torch.matmul(attn, V)
        out = V.mean(dim=-2, keepdim=True).expand(B, self.num_heads, L_Q, self.head_dim).clone()
        out[
            torch.arange(B)[:, None, None],
            torch.arange(self.num_heads)[None, :, None],
            M_top
        ] = V_reduce
        return self.out(out.transpose(1, 2).contiguous().view(B, L_Q, -1))


class ConvDistilling(nn.Module):
    """Halves sequence length via conv + max-pool."""
    def __init__(self, hidden_size):
        super().__init__()
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(hidden_size)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x.transpose(1, 2))
        x = torch.relu(self.norm(x))
        return self.pool(x).transpose(1, 2)


class InformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1, factor=5):
        super().__init__()
        self.attn  = ProbSparseAttention(hidden_size, num_heads, dropout, factor)
        self.ffn   = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.drop(self.attn(x, x, x)))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x


class Informer(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3,
                 num_heads=4, output_size=3, dropout=0.1,
                 factor=5, max_len=144, distil=True):
        super().__init__()
        self.distil      = distil
        self.input_proj  = nn.Linear(input_size, hidden_size)
        self.pos_enc     = nn.Embedding(max_len, hidden_size)
        self.encoder_layers = nn.ModuleList([
            InformerEncoderLayer(hidden_size, num_heads, dropout, factor)
            for _ in range(num_layers)
        ])
        self.distil_layers = nn.ModuleList([
            ConvDistilling(hidden_size) for _ in range(num_layers - 1)
        ]) if distil else None
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        _, T, _ = x.shape
        x   = self.input_proj(x)
        x   = x + self.pos_enc(torch.arange(T, device=x.device).unsqueeze(0))
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            if self.distil and i < len(self.distil_layers):
                x = self.distil_layers[i](x)
        return self.head(self.norm(x)[:, -1, :])
