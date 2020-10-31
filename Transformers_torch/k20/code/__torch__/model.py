class Transformer(Module):
  __parameters__ = []
  training : bool
  token_emb : __torch__.torch.nn.modules.sparse.Embedding
  tblocks : __torch__.torch.nn.modules.container.___torch_mangle_6.Sequential
  toprobs : __torch__.torch.nn.modules.linear.___torch_mangle_7.Linear
  def forward(self: __torch__.model.Transformer,
    input: Tensor) -> Tensor:
    _0 = self.toprobs
    _1 = (self.tblocks).forward((self.token_emb).forward(input, ), )
    input0 = torch.mean(_1, [1], False, dtype=None)
    return torch.relu((_0).forward(input0, ))
class TransformerBlock(Module):
  __parameters__ = []
  training : bool
  attention : __torch__.model.SelfAttention
  norm1 : __torch__.torch.nn.modules.normalization.LayerNorm
  norm2 : __torch__.torch.nn.modules.normalization.___torch_mangle_3.LayerNorm
  ff : __torch__.torch.nn.modules.container.Sequential
  def forward(self: __torch__.model.TransformerBlock,
    argument_1: Tensor) -> Tensor:
    _2 = self.norm2
    _3 = self.ff
    _4 = self.norm1
    _5 = (self.attention).forward(argument_1, )
    input = torch.add(_5, argument_1, alpha=1)
    _6 = (_4).forward(input, )
    input1 = torch.add((_3).forward(_6, ), _6, alpha=1)
    return (_2).forward(input1, )
class SelfAttention(Module):
  __parameters__ = []
  training : bool
  tokeys : __torch__.torch.nn.modules.linear.Linear
  toqueries : __torch__.torch.nn.modules.linear.___torch_mangle_0.Linear
  tovalues : __torch__.torch.nn.modules.linear.___torch_mangle_1.Linear
  unifyheads : __torch__.torch.nn.modules.linear.___torch_mangle_2.Linear
  def forward(self: __torch__.model.SelfAttention,
    argument_1: Tensor) -> Tensor:
    _7 = self.unifyheads
    _8 = self.tovalues
    _9 = self.tokeys
    _10 = self.toqueries
    b = ops.prim.NumToTensor(torch.size(argument_1, 0))
    _11 = int(b)
    _12 = int(b)
    _13 = int(b)
    _14 = int(b)
    _15 = int(b)
    t = ops.prim.NumToTensor(torch.size(argument_1, 1))
    _16 = int(t)
    _17 = int(t)
    _18 = int(t)
    _19 = int(t)
    _20 = int(t)
    _21 = int(t)
    _22 = int(t)
    _23 = int(t)
    k = ops.prim.NumToTensor(torch.size(argument_1, 2))
    _24 = int(k)
    _25 = int(k)
    _26 = int(k)
    _27 = int(k)
    _28 = int(k)
    _29 = int(k)
    _30 = int(k)
    queries = torch.view((_10).forward(argument_1, ), [_15, _23, 8, _30])
    keys = torch.view((_9).forward(argument_1, ), [_14, _22, 8, _29])
    values = torch.view((_8).forward(argument_1, ), [_13, _21, 8, _28])
    _31 = torch.contiguous(torch.transpose(keys, 1, 2), memory_format=0)
    _32 = [int(torch.mul(b, CONSTANTS.c0)), _20, _27]
    keys0 = torch.view(_31, _32)
    _33 = torch.contiguous(torch.transpose(queries, 1, 2), memory_format=0)
    _34 = [int(torch.mul(b, CONSTANTS.c0)), _19, _26]
    queries0 = torch.view(_33, _34)
    _35 = torch.contiguous(torch.transpose(values, 1, 2), memory_format=0)
    _36 = [int(torch.mul(b, CONSTANTS.c0)), _18, _25]
    values0 = torch.view(_35, _36)
    queries1 = torch.div(queries0, torch.pow(k, 0.25))
    keys1 = torch.div(keys0, torch.pow(k, 0.25))
    input = torch.matmul(queries1, torch.transpose(keys1, 1, 2))
    dot = torch.softmax(input, 2, None)
    out = torch.view(torch.matmul(dot, values0), [_12, 8, _17, _24])
    _37 = torch.contiguous(torch.transpose(out, 1, 2), memory_format=0)
    _38 = [_11, _16, int(torch.mul(k, CONSTANTS.c0))]
    input2 = torch.view(_37, _38)
    return (_7).forward(input2, )
