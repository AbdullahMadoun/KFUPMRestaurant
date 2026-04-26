from __future__ import annotations

import torch
from PIL import Image

from stage1_kcfd.dataset import Stage1Collator
from stage1_kcfd.qwen_io import build_assistant_conversation, build_user_conversation, processor_batch


class FakeProcessor:
    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        roles = ",".join(row["role"] for row in conversation)
        return f"{roles}:{add_generation_prompt}"

    def __call__(self, *, text, images, return_tensors, padding, **kwargs):
        assert return_tensors == "pt"
        assert padding is True
        assert len(text) == len(images)
        return {
            "input_ids": torch.ones((len(text), 4), dtype=torch.long),
            "attention_mask": torch.ones((len(text), 4), dtype=torch.long),
        }


def test_qwen_io_builds_user_and_assistant_conversations_with_fallback_processor_path():
    image = Image.new("RGB", (8, 8), "white")
    user = build_user_conversation(image, "prompt")
    full = build_assistant_conversation(image, "prompt", '{"items":[]}')

    batch = processor_batch(FakeProcessor(), [user, full], add_generation_prompt=True, fallback_images=[image, image])

    assert user[0]["content"][0]["type"] == "image"
    assert full[-1]["role"] == "assistant"
    assert tuple(batch["input_ids"].shape) == (2, 4)


class LeftPaddingProcessor(FakeProcessor):
    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False):
        roles = ",".join(row["role"] for row in conversation)
        return roles

    def __call__(self, *, text, images, return_tensors, padding, **kwargs):
        if all("assistant" in row for row in text):
            return {
                "input_ids": torch.arange(8, dtype=torch.long).view(1, 8),
                "attention_mask": torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1]], dtype=torch.long),
            }
        return {
            "input_ids": torch.arange(4, dtype=torch.long).view(1, 4),
            "attention_mask": torch.tensor([[0, 1, 1, 1]], dtype=torch.long),
        }


def test_collator_masks_prompt_tokens_correctly_when_processor_left_pads():
    image = Image.new("RGB", (8, 8), "white")
    collator = Stage1Collator(LeftPaddingProcessor(), prompt="prompt")

    batch = collator([{"image": image, "image_id": "img", "target_json": '{"items":[]}'}])

    assert batch["labels"][0, 0].item() == -100
    assert batch["labels"][0, 1].item() == -100
    assert batch["labels"][0, 2].item() == -100
    assert batch["labels"][0, 3].item() == -100
    assert batch["labels"][0, 4].item() == -100
    assert batch["labels"][0, 5].item() != -100


class PrefixTargetProcessor(FakeProcessor):
    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False):
        roles = ",".join(row["role"] for row in conversation)
        return roles

    def __call__(self, *, text, images, return_tensors, padding, **kwargs):
        if all("assistant" in row for row in text):
            return {
                "input_ids": torch.tensor(
                    [
                        [10, 11, 12, 20, 21, 0],
                        [10, 11, 12, 30, 31, 32],
                    ],
                    dtype=torch.long,
                ),
                "attention_mask": torch.tensor(
                    [
                        [1, 1, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1, 1],
                    ],
                    dtype=torch.long,
                ),
            }
        return {
            "input_ids": torch.tensor([[10, 11, 12], [10, 11, 12]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.long),
        }


def test_collator_masks_user_image_prefix_and_keeps_assistant_target_labels():
    image = Image.new("RGB", (8, 8), "white")
    collator = Stage1Collator(PrefixTargetProcessor(), prompt="prompt")

    batch = collator([
        {"image": image, "image_id": "img-1", "target_json": '{"items":[]}'},
        {"image": image, "image_id": "img-2", "target_json": '{"items":[1]}'},
    ])

    assert batch["labels"].tolist() == [
        [-100, -100, -100, 20, 21, -100],
        [-100, -100, -100, 30, 31, 32],
    ]
