import torch

model =torch.load("k20.pth")
input_tensor = torch.ones(28,6).long()

script_model = torch.jit.trace(model,input_tensor)
script_model.save("k20.pt")