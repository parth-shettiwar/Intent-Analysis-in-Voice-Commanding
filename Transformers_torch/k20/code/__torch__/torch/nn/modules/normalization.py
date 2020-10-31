class LayerNorm(Module):
  __parameters__ = ["weight", "bias", ]
  weight : Tensor
  bias : Tensor
  training : bool
  def forward(self: __torch__.torch.nn.modules.normalization.LayerNorm,
    input: Tensor) -> Tensor:
    _0 = self.bias
    _1 = self.weight
    input0 = torch.layer_norm(input, [50], _1, _0, 1.0000000000000001e-05, True)
    return input0
