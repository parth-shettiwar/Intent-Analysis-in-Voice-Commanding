class Embedding(Module):
  __parameters__ = ["weight", ]
  weight : Tensor
  training : bool
  def forward(self: __torch__.torch.nn.modules.sparse.Embedding,
    input: Tensor) -> Tensor:
    tokens = torch.embedding(self.weight, input, -1, False, False)
    return tokens
