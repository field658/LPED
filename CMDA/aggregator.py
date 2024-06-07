from typing import Tuple

import torch

from cmda.utils import list_cat
from cmda.config import Config


class Encoder(torch.torch.nn.Module):
    def __init__(self, embedding_m):
        super().__init__()
        self.embedding = torch.nn.Embedding(*embedding_m.shape, padding_idx=0)  # TODO remove hardcoded padding index
        self.embedding.weight.data.copy_(embedding_m)
        self.embedding.weight.require_grad = False
        self.delim_desc_start = [torch.tensor([2]).to(Config.get(Config.get("device")))]  # TODO remove hardcoded index
        self.delim_desc_end = [torch.tensor([3]).to(Config.get("device"))]  # TODO remove hardcoded index

    @staticmethod
    def pad(*args, **kwargs):
     
        res = tuple(torch.nn.utils.rnn.pad_sequence(x, batch_first=True) for x in args)
        return res[0] if len(res) == 1 else res

    def forward(self, name: torch.tensor, description: torch.tensor):
   
        raise NotImplementedError


class CNNEncoder(Encoder):
    def __init__(self, embedding_m: torch.tensor, output_d: int, n_filters: int = 3,
                 filter_sizes: Tuple[int, ...] = (2, 3, 5)):
        super().__init__(embedding_m)
 
        self.conv = torch.nn.ModuleList([torch.nn.Conv2d(1, output_d, (filter_sizes[k],
                                                                       self.embedding.weight.size(-1))).to(
                                                                            Config.get("device")
                                                                            )
                                         for k in range(n_filters)])
    
    def forward(self, name: torch.tensor, description: torch.tensor) -> torch.tensor:
        seq = self.pad(list_cat((name, description), dim=-1))
        enc_seq = self.embedding(seq).unsqueeze(1)  # [B, 1, W, D]

        out = [k(enc_seq).squeeze(3) for k in self.conv]
        out = [torch.nn.functional.F.max_pool1d(k, k.size(2)).squeeze(2) for k in out]
        out = torch.cat(out, 1) if len(out) > 1 else out[0]
     
        return out


class LSTMEncoder(Encoder):
    def __init__(self, embedding_m: torch.tensor, output_d: int):
        super().__init__(embedding_m)
        self.bidirectional = Config.get("LSTMBidirectional")
        self.hidden_d = output_d // 2 if self.bidirectional else output_d  # TODO make configurable

        self.rnn = torch.nn.LSTM(
            input_size=self.embedding.weight.size(-1),
            hidden_size=self.hidden_d,
            num_layers=1,   
            dropout=0.3,   
            batch_first=True,
            bidirectional=self.bidirectional   
        )
 

    def forward(self, name: torch.tensor, description: torch.tensor):
        delim_desc_start = self.delim_desc_start * len(name)   
        delim_desc_end = self.delim_desc_end * len(name)   

         seq = list_cat((delim_desc_start, name, delim_desc_start, description, delim_desc_start), dim=-1)   
        seq = self.pad(seq)
 
        enc_seq = self.embedding(seq)   
        output, (_, _) = self.rnn(enc_seq)
        weighted_states = output[:, -1, :].squeeze()   
        return weighted_states


class AvgEncoder(Encoder):
    def __init__(self, embedding_m: torch.tensor):
        super().__init__(embedding_m)
        self.dropout = None
        if Config.get("AverageWordDropout"):
            self.dropout = torch.nn.Dropout2d(p=Config.get("AverageWordDropout"))

    def forward(self, name: torch.tensor, description: torch.tensor):
      
        name, description = self.pad(name, description)   
        seq = torch.cat((name, description), dim=-1)   
        enc_seq = self.embedding(seq)   
        if self.dropout:
            enc_seq = self.dropout(enc_seq)
   
        if self.training:
            return enc_seq.mean(dim=1)
        else:
            return torch.t(torch.t(enc_seq.sum(dim=1)).div((seq != 0).sum(dim=1).float())) # [B, D]
