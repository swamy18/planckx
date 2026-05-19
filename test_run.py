import torch
from run_demo import PlanckXCharModel
m = PlanckXCharModel(vocab_size=256, d_model=64)
input_ids = torch.randint(0, m.vocab_size, (1,5))
logits = m(input_ids)
print('ok', logits.shape)
