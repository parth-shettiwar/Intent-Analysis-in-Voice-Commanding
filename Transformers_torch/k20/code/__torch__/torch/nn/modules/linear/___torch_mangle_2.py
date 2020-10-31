class Linear(Module):
  __parameters__ = ["weight", "bias", ]
  weight : Tensor
  bias : Tensor
  training : bool
  def forward(self: __torch__.torch.nn.modules.linear.___torch_mangle_2.Linear,
    input: Tensor) -> Tensor:
    _0 = self.bias
    output = torch.matmul(input, torch.t(self.weight))
    return torch.add_(output, _0, alpha=1)
