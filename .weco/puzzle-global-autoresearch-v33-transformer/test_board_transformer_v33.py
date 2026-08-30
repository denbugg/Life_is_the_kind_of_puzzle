import torch
import board_transformer_v33 as m


def test_shapes_and_sizes():
    for name,lower,upper in (("ts",2_000_000,3_500_000),("tm",5_000_000,8_000_000)):
        model,_,_=m.make_variant(name,10)
        score,local=model(torch.randn(2,32,24,24))
        assert score.shape==(2,) and local.shape==(2,3,24,24)
        assert lower<=m.parameter_count(model)<=upper
