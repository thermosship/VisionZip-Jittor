import importlib.util
import unittest
from pathlib import Path

from visionzip_jittor.config import VisionZipConfig, load_config


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from reference.pytorch_visionzip import visionzip_compress_torch
else:
    torch = None
    visionzip_compress_torch = None


ROOT = Path(__file__).resolve().parents[1]


def official_inline(hidden_states, attentions, metric, dominant_num, contextual_num):
    """Literal small-tensor transcription of official clip_encoder.py."""

    cls_attention = attentions[:, :, 0, 1:]
    cls_attention_sum = cls_attention.sum(dim=1)
    topk_indices = cls_attention_sum.topk(dominant_num, dim=1).indices + 1
    all_indices = torch.cat(
        [
            torch.zeros(
                (hidden_states.shape[0], 1),
                dtype=topk_indices.dtype,
                device=topk_indices.device,
            ),
            topk_indices,
        ],
        dim=1,
    )
    mask = torch.ones_like(hidden_states[:, :, 0], dtype=torch.bool).scatter_(
        1, all_indices, False
    )
    dominant_tokens = hidden_states.masked_select(~mask.unsqueeze(-1)).view(
        hidden_states.shape[0], dominant_num + 1, hidden_states.shape[2]
    )
    metric_filtered = metric[mask].view(
        hidden_states.shape[0],
        hidden_states.shape[1] - (dominant_num + 1),
        metric.shape[2],
    )
    hidden_filtered = hidden_states.masked_select(mask.unsqueeze(-1)).view(
        hidden_states.shape[0],
        hidden_states.shape[1] - (dominant_num + 1),
        hidden_states.shape[2],
    )
    metric_normalized = metric_filtered / metric_filtered.norm(
        dim=-1, keepdim=True
    )
    step = max(1, metric_normalized.shape[1] // contextual_num)
    target_indices = torch.arange(0, metric_normalized.shape[1], step)[
        :contextual_num
    ]
    remaining_positions = torch.arange(metric_normalized.shape[1])
    merge_mask = ~torch.isin(remaining_positions, target_indices)
    target_tokens = metric_normalized[:, target_indices, :]
    tokens_to_merge = metric_normalized[:, merge_mask, :]
    similarity = torch.bmm(tokens_to_merge, target_tokens.transpose(1, 2))
    assign_one_hot = torch.zeros(
        tokens_to_merge.shape[0],
        tokens_to_merge.shape[1],
        contextual_num,
        dtype=hidden_filtered.dtype,
    )
    assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)
    counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
    hidden_to_merge = hidden_filtered[:, merge_mask, :]
    aggregated_hidden = (
        torch.bmm(assign_one_hot.transpose(1, 2), hidden_to_merge) / counts
    )
    target_hidden = hidden_filtered[:, target_indices, :]
    contextual_tokens = target_hidden + aggregated_hidden
    return torch.cat([dominant_tokens, contextual_tokens], dim=1), all_indices


class ConfigTests(unittest.TestCase):
    def test_official_nominal_budget_excludes_cls(self):
        config = load_config(ROOT / "configs/visionzip_64.json")
        self.assertEqual(config.nominal_visual_tokens, 64)
        self.assertEqual(config.actual_output_tokens, 65)
        self.assertEqual(config.dominant_tokens, 54)
        self.assertEqual(config.contextual_tokens, 10)

    def test_invalid_budget_is_rejected(self):
        config = VisionZipConfig(dominant_tokens=5, contextual_tokens=4)
        with self.assertRaises(ValueError):
            config.validate(sequence_length=8)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed in this environment")
class TorchReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        self.hidden = torch.randn(2, 17, 8)
        logits = torch.randn(2, 3, 17, 17)
        self.attention = logits.softmax(dim=-1)
        self.metric = torch.randn(2, 17, 4)
        self.config = VisionZipConfig(
            dominant_tokens=4,
            contextual_tokens=3,
            merge_mode="code_exact",
        )

    def test_matches_literal_official_code(self):
        expected_tokens, expected_indices = official_inline(
            self.hidden,
            self.attention,
            self.metric,
            dominant_num=4,
            contextual_num=3,
        )
        actual = visionzip_compress_torch(
            self.hidden, self.attention, self.metric, self.config
        )
        torch.testing.assert_close(actual.compressed_tokens, expected_tokens)
        self.assertTrue(torch.equal(actual.selected_indices, expected_indices))

    def test_dominant_output_preserves_original_order(self):
        hidden = torch.arange(7, dtype=torch.float32).view(1, 7, 1)
        attention = torch.zeros(1, 1, 7, 7)
        attention[0, 0, 0, 5] = 10
        attention[0, 0, 0, 2] = 9
        metric = torch.randn(1, 7, 3)
        config = VisionZipConfig(dominant_tokens=2, contextual_tokens=1)
        output = visionzip_compress_torch(hidden, attention, metric, config)
        self.assertEqual(output.selected_indices.tolist(), [[0, 5, 2]])
        self.assertEqual(output.dominant_ordered_indices.tolist(), [[0, 2, 5]])
        self.assertEqual(output.compressed_tokens[0, :3, 0].tolist(), [0, 2, 5])

    def test_paper_avg_is_a_distinct_ablation(self):
        exact = visionzip_compress_torch(
            self.hidden, self.attention, self.metric, self.config
        )
        avg_config = VisionZipConfig(
            dominant_tokens=4,
            contextual_tokens=3,
            merge_mode="paper_avg",
        )
        averaged = visionzip_compress_torch(
            self.hidden, self.attention, self.metric, avg_config
        )
        self.assertFalse(
            torch.allclose(exact.contextual_tokens, averaged.contextual_tokens)
        )
        self.assertTrue(torch.equal(exact.assignments, averaged.assignments))

    def test_output_shape_includes_cls(self):
        output = visionzip_compress_torch(
            self.hidden, self.attention, self.metric, self.config
        )
        self.assertEqual(tuple(output.compressed_tokens.shape), (2, 8, 8))


if __name__ == "__main__":
    unittest.main()
