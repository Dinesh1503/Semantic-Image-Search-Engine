CAPTION_PROMPT = """Analyze this image for an image retrieval database.
Do not use poetic, emotional, or subjective language.
Instead, provide a structured, factual breakdown of the visual elements.

Please describe the image using these four categories:
1. MAIN SUBJECTS: List the specific physical objects visible.
2. SPATIAL LAYOUT: Where are these objects located?
3. ATTRIBUTES: Describe colors, textures, and states.
4. ENVIRONMENT: Describe the lighting, weather, and time of day.

Start the description directly with the main subject. Output as a single concise paragraph."""


def qwen_vl_chat_prompt(prompt_text: str = CAPTION_PROMPT) -> str:
    """Wrap a text prompt in the Qwen-VL chat template with a single image placeholder."""
    return (
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        f"{prompt_text}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
