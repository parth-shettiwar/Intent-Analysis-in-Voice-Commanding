class Linear(Module):
  __parameters__ = ["weight", ]
  weight : Tensor
  training : bool
  def forward(self: __torch__.torch.nn.modules.linear.Linear,
    argument_1: Tensor) -> Tensor:
    _0 = torch.matmul(argument_1, torch.t(self.weight))
    return _0
