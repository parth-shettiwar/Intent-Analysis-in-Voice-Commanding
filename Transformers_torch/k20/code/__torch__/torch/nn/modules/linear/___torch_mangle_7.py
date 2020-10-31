class Linear(Module):
  __parameters__ = ["weight", "bias", ]
  weight : Tensor
  bias : Tensor
  training : bool
  def forward(self: __torch__.torch.nn.modules.linear.___torch_mangle_7.Linear,
    input: Tensor) -> Tensor:
    input0 = torch.addmm(self.bias, input, torch.t(self.weight), beta=1, alpha=1)
    return input0
