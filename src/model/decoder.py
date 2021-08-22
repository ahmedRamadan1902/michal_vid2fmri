import torch
import torch.nn as nn

from model.layer import norm_linear

class StatsModel(nn.Module):
    def __init__(self):
        super(StatsModel, self).__init__()

    def forward(self, seq):
        embed_min = torch.min(seq, dim=1)[0]
        embed_max = torch.max(seq, dim=1)[0]
        embed_std, embed_avg = torch.std_mean(seq, unbiased=False, dim=1)
        return torch.cat([seq[:, -1, :], embed_min, embed_max, embed_avg, embed_std], dim=1)


class HierarchicalRnnModel(nn.Module):
    def __init__(self, input_size, rnn_size, dropout_rate=0.4):
        super(HierarchicalRnnModel, self).__init__()

        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.rnn_1 = nn.GRU(input_size=input_size, hidden_size=rnn_size, num_layers=1, batch_first=True,)
        self.rnn_2 = nn.GRU(input_size=input_size, hidden_size=rnn_size, num_layers=2, batch_first=True,)
        self.rnn_3 = nn.GRU(input_size=input_size, hidden_size=rnn_size, num_layers=4, batch_first=True,)
    
    def forward(self, input):
        RNN_out_1, _ = self.rnn_1(self.dropout(input))
        RNN_out_2, _ = self.rnn_2(self.dropout(input))
        RNN_out_3, _ = self.rnn_3(self.dropout(input))

        RNN_out = torch.cat([RNN_out_1, 
                             RNN_out_2, 
                             RNN_out_3,
                             ], dim=2)

        RNN_out = RNN_out[:, -1, :]
        RNN_out = self.act(RNN_out)
        return RNN_out


class Vid2FMRIModel(nn.Module):
    def __init__(self, encoder, output_size=256, rnn_features=True, dropout_rate=0.2):
        super(Vid2FMRIModel, self).__init__()

        self.encoder = encoder
        embed_size = encoder.embed_size
        self.rnn_features = rnn_features

        fc_size = embed_size*5 + 1
        self.stats = StatsModel()
        
        if self.rnn_features:
            rnn_size = embed_size//2
            fc_size += rnn_size*3
            self.rnn = HierarchicalRnnModel(embed_size, rnn_size)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = norm_linear(fc_size, output_size)

    def forward(self, input, fps):
        embeds_seq = self.encoder(input)
        f_stats = self.stats(embeds_seq)
        
        if self.rnn_features:
            f_rnn = self.rnn(embeds_seq)
            f = torch.cat([f_stats, f_rnn, fps.unsqueeze(1)], dim=1)
        else:
            f = torch.cat([f_stats, fps.unsqueeze(1)], dim=1)

        f = self.dropout(f)
        out = self.fc(f)
        return out