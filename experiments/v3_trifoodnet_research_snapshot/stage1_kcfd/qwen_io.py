from __future__ import annotations

from typing import Any, Dict, List, Sequence


def build_user_conversation(image: Any, prompt: str) -> List[Dict[str, Any]]:
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]


def build_assistant_conversation(image: Any, prompt: str, target_text: str) -> List[Dict[str, Any]]:
    return build_user_conversation(image, prompt) + [{
        "role": "assistant",
        "content": [{"type": "text", "text": target_text}],
    }]


def processor_batch(
    processor,
    conversations: Sequence[List[Dict[str, Any]]],
    *,
    add_generation_prompt: bool,
    fallback_images: Sequence[Any],
):
    texts = [
        processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        for conversation in conversations
    ]
    try:
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(list(conversations))
        return processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )
    except Exception:
        return processor(text=texts, images=list(fallback_images), return_tensors="pt", padding=True)
