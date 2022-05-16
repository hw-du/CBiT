from torch import nn as nn

from models.bert_modules.embedding import BERTEmbedding
from models.bert_modules.transformer import TransformerBlock
from utils import fix_random_seed_as


class BERT(nn.Module):
    def __init__(self, args, fixed=False):
        super().__init__()


        fix_random_seed_as(args.model_init_seed)



        max_len = args.bert_max_len
        num_items = args.num_items

        n_layers = args.bert_num_blocks

        heads = args.bert_num_heads
        vocab_size = num_items + 2

        hidden = args.bert_hidden_units
        self.hidden = hidden

        dropout = args.bert_dropout


        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=self.hidden, max_len=max_len, dropout=dropout)


        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, heads, hidden * 4, dropout) for _ in range(n_layers)])
        ######fixed model
        if fixed:
            for param in self.parameters():
                param.requires_grad=False


    def forward(self, x):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        x = self.embedding(x)
        layer_output = []
        layer_output.append(x)


        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
            layer_output.append(x)

        return layer_output
    def fix_model(self):
        for param in self.parameters():
            param.requires_grad = False
    def init_weights(self):
        pass
