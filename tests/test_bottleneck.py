import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.bottleneck import BottleneckFusion, BottleneckLayer


class TestBottleneck(unittest.TestCase):
    def setUp(self):
        # same fake data as shared tensors
        torch.manual_seed(42)
        self.hands = torch.randn(2, 10, 128)
        self.face = torch.randn(2, 10, 128)
        self.body = torch.randn(2, 10, 128)

    def test_one_layer_shapes_and_grads(self):
        layer = BottleneckLayer(dropout=0)
        streams, tokens = layer(self.hands, self.face, self.body)
        self.assertEqual(tokens.shape, (2, 4, 128))
        for s in streams:
            self.assertEqual(s.shape, (2, 10, 128))

        loss = tokens.pow(2).mean() + sum(s.pow(2).mean() for s in streams)
        loss.backward()
        self.assertGreater(layer.tokens.grad.abs().sum().item(), 0)

    def test_one_layer_only_mixes_through_tokens(self):
        # changing the hands should change the tokens but not the face output
        layer = BottleneckLayer(dropout=0).eval()
        streams1, tokens1 = layer(self.hands, self.face, self.body)
        streams2, tokens2 = layer(self.hands + 1, self.face, self.body)
        self.assertTrue(torch.allclose(streams1[1], streams2[1]))
        self.assertFalse(torch.allclose(tokens1, tokens2))

    def test_two_layers_mix_channels(self):
        # with 2 layers the face output should now depend on the hands
        fusion = BottleneckFusion(num_layers=2, dropout=0).eval()
        _, _, streams1, _ = fusion(self.hands, self.face, self.body)
        _, _, streams2, _ = fusion(self.hands + 1, self.face, self.body)
        self.assertFalse(torch.allclose(streams1[1], streams2[1]))

    def test_memory_shape_and_mask(self):
        fusion = BottleneckFusion(dropout=0)
        masks = [torch.zeros(2, 10, dtype=torch.bool) for _ in range(3)]
        masks[1][:, -2:] = True
        memory, memory_mask, streams, _ = fusion(self.hands, self.face, self.body, masks)
        self.assertEqual(memory.shape, (2, 34, 128))  # 10 + 10 + 10 + 4
        self.assertTrue(torch.equal(memory_mask[:, 10:20], masks[1]))
        self.assertFalse(memory_mask[:, -4:].any())
        self.assertEqual(streams[1][:, -2:].abs().sum().item(), 0)

    def test_padding_doesnt_change_output(self):
        fusion = BottleneckFusion(dropout=0).eval()
        masks = [torch.zeros(2, 10, dtype=torch.bool) for _ in range(3)]
        for m in masks:
            m[:, -3:] = True
        hands_nan = self.hands.clone()
        hands_nan[:, -3:] = float("nan")
        memory1, _, _, _ = fusion(self.hands, self.face, self.body, masks)
        memory2, _, _, _ = fusion(hands_nan, self.face, self.body, masks)
        self.assertTrue(torch.allclose(memory1, memory2))

    def test_missing_channel_no_nan(self):
        # every hand frame is missing
        masks = [torch.zeros(2, 10, dtype=torch.bool) for _ in range(3)]
        masks[0][:] = True
        memory, _, _, _ = BottleneckFusion(dropout=0)(self.hands, self.face, self.body, masks)
        self.assertTrue(torch.isfinite(memory).all())

    def test_different_lengths(self):
        memory, _, _, _ = BottleneckFusion(dropout=0)(self.hands[:, :3], self.face[:, :7], self.body)
        self.assertEqual(memory.shape, (2, 24, 128))  # 3 + 7 + 10 + 4

    def test_decoder_grads_reach_every_channel(self):
        hands = self.hands.clone().requires_grad_()
        face = self.face.clone().requires_grad_()
        body = self.body.clone().requires_grad_()
        fusion = BottleneckFusion(dropout=0)
        memory, memory_mask, _, _ = fusion(hands, face, body)

        decoder = torch.nn.TransformerDecoderLayer(128, 4, 256, dropout=0, batch_first=True)
        out = decoder(torch.randn(2, 5, 128), memory, memory_key_padding_mask=memory_mask)
        out.pow(2).mean().backward()

        for x in [hands, face, body]:
            self.assertGreater(x.grad.abs().sum().item(), 0)
        for name, p in fusion.named_parameters():
            self.assertIsNotNone(p.grad, name)


if __name__ == "__main__":
    unittest.main()
