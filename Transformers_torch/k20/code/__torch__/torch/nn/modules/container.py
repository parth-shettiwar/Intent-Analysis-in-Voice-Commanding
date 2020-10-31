class Sequential(Module):
  __parameters__ = []
  training : bool
  __annotations__["0"] = __torch__.torch.nn.modules.linear.___torch_mangle_4.Linear
  __annotations__["1"] = __torch__.torch.nn.modules.activation.ReLU
  __annotations__["2"] = __torch__.torch.nn.modules.linear.___torch_mangle_5.Linear
  def forward(self: __torch__.torch.nn.modules.container.Sequential,
    argument_1: Tensor) -> Tensor:
    _0 = getattr(self, "1")
    _1 = (getattr(self, "0")).forward(argument_1, )
    _2 = (getattr(self, "2")).forward((_0).forward(_1, ), )
    return _2
