from collections import OrderedDict

import torch
from transformers import GPT2LMHeadModel


model = GPT2LMHeadModel.from_pretrained("gpt2")

checkpoint = torch.load('checkpoint_step3.pth')
custom_state = checkpoint['model_state_dict']

fixed_state = OrderedDict()
for k, v in custom_state.items():
    new_key = k.replace("_orig_mod.", "")
    fixed_state[new_key] = v

for k, v in fixed_state.items():
    print(k, v.shape)


model.load_state_dict(fixed_state, strict=True)

model.to("cuda")
model.eval()