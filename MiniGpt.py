#Transformer model 
import torch.nn as nn 
from torch.nn import functional as F
import torch 

#hyperparamters
seed = 1337
PATH = "Weights.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
train_perc = 0.9
batch_size = 64
learning_rate = 3e-4
n_steps = 5000
eval_iter = 200
eval_inter = 500
n_embeds = 384
block_size = 56
n_heads = 6
n_layer = 6
head_size = 16
dropout = 0.2
torch.manual_seed(seed)

class AttentionHead(nn.Module):
    def __init__(self, head_size) :
        super().__init__()
        self.key = nn.Linear(n_embeds, head_size, bias=False)
        self.query = nn.Linear(n_embeds, head_size, bias=False)
        self.value = nn.Linear(n_embeds, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        #Compute initial weights or "affinities"
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out 
    
class MultiHead(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.head = nn.ModuleList([AttentionHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embeds, n_embeds)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        y = torch.cat([h(x) for h in self.head], dim=-1)
        y = self.proj(y)
        y = self.dropout(y)
        return y

class FeedForward(nn.Module):
    def __init__(self, n_embeds):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embeds, 4*n_embeds),
            nn.ReLU(),
            nn.Linear(4*n_embeds, n_embeds),
            nn.Dropout(dropout),
            )
        
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_heads, n_embeds):
        super().__init__()
        head_size = n_embeds // n_heads
        self.mheads = MultiHead(n_heads, head_size)
        self.ffwd = FeedForward(n_embeds)
        self.ln1 = nn.LayerNorm(n_embeds)
        self.ln2 = nn.LayerNorm(n_embeds)
    def forward(self, x):
        x_sa = self.mheads(self.ln1(x)) + x
        x_ffwd = self.ffwd(self.ln2(x_sa)) + x_sa
        return x_ffwd

class TransformerModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocabulary_size, n_embeds)
        self.positional_emb = nn.Embedding(block_size, n_embeds)
        #self.mheads = MultiHead(4, n_embeds // 4)
        #self.ffwd = FeedForward(n_embeds)
        self.block = nn.Sequential(*[Block(n_heads, n_embeds) for _ in range(n_layer)])
        self.lnorm = nn.LayerNorm(n_embeds)
        self.linear = nn.Linear(n_embeds, vocabulary_size)
       

    def forward(self, x, targets=None):
        B, t = x.shape 
        x_t = torch.arange(t, device=device)
        embedded_token = self.token_embedding_table(x) # shape B,T,C: n_embeds
        embedded_pos = self.positional_emb(x_t)
        x_pos = embedded_token + embedded_pos
        x_pos = self.block(x_pos)
        x_pos = self.lnorm(x_pos)
        logits = self.linear(x_pos)  # shape B, T, vocab_size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss 
    
    def generate(self, x, max_new_tokens):
        x_cond = x[:, -block_size:]
        for _ in range(max_new_tokens):
            logits, _ = self(x_cond)
            logits = logits[:, -1, :] #Last elem from time dimension 
            probs = F.softmax(logits, dim=1)
            x_next = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, x_next), dim=1)
        return x 
        





with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()
    
chars = sorted(list(set(text)))
vocabulary_size = len(chars)
    
    #encoder decoder functions 
enc_dic ={ch : i for i,ch in enumerate(chars)}
deco_dic = {i : ch for i,ch in enumerate(chars)}
encode = lambda s: [enc_dic[c] for c in s]
decode = lambda int_l: "".join([deco_dic[i] for i in int_l])

    #get data 
data = torch.tensor(encode(text), dtype=torch.long)


    #train/test split 

train_size = int(train_perc*len(data))
train_data = data[:train_size]
val_data = data[train_size:]


#batch the data


def get_batch(split):
    data = train_data if split == "train" else val_data
    idx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    x, y = x.to(device), y.to(device)
    return x,y

xb, yb = get_batch("train")
print(f"inputs, {xb.shape}, {xb} \n targets, {yb.shape}, {yb}")

#model
model = TransformerModel()
m = model.to(device)

#Pytorch optimizer 
optim = torch.optim.AdamW(m.parameters(), lr=learning_rate)

#evaluate loss function 
@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iter)
        for k in range(eval_iter):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item() 
        out[split] = losses.mean()
    m.train()
    return out 

#Training loop 
"""for step in range(n_steps):
    #evaluate loss on val too
    if step % eval_inter == 0:
        losses = estimate_loss()
        print(f" for step {step} training loss is {losses["train"] : .4f} validation loss is {losses["val"] : .4f}")

    #sample data 
    xb, yb = get_batch("train")

    #evaluate loss
    logits, loss = model(xb, yb)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()
    

torch.save(m.state_dict(), PATH)

#generate from model 
"""
m.load_state_dict(torch.load(PATH))
x_gen_text = "Name"
#x_gen = torch.tensor(encode(x_gen_text), dtype=torch.long, device=device)
x_gen = torch.zeros((1, 1), dtype=torch.long, device=device)
y_gen = m.generate(x_gen, 10000)
result = decode(y_gen[0].tolist())
w = open("output.txt", "w", encoding="utf-8")
w.write(result)
w.close()
