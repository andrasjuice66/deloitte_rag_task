from typing import Callable, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from .config import Config


def build_llm(
    model_name: str,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> Callable[[str], str]:
    """
    Build a simple local Hugging Face text-generation LLM that takes a prompt string
    and returns a generated string.
    """
    max_new_tokens = max_new_tokens or max(128, Config.LLM_MAX_LENGTH // 2)
    temperature = temperature if temperature is not None else Config.LLM_TEMPERATURE

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
    )
    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        return_full_text=False,
    )

    def llm(prompt: str) -> str:
        out = gen(prompt)[0]["generated_text"]
        return out.strip()

    print(f"Loaded LLM: {model_name}")
    return llm


