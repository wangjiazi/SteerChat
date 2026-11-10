"""Fit and apply preference directions to a single Llama checkpoint."""

from collections.abc import Iterable
import math
from threading import Lock

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    GenerationConfig,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerBase,
)


class SteeringModel:
    """Collect and intervene on the output of a zero-indexed decoder layer."""

    def __init__(self, model: LlamaForCausalLM, tokenizer: PreTrainedTokenizerBase):
        if not isinstance(model, LlamaForCausalLM):
            raise ValueError("SteerChat requires a LlamaForCausalLM checkpoint.")
        if not tokenizer.chat_template:
            raise ValueError("The checkpoint's tokenizer must have a chat template.")
        self._model = model.eval().requires_grad_(False)
        self._tokenizer = tokenizer
        self._lock = Lock()

    @classmethod
    def from_pretrained(cls, model_id: str, *, device: str = "cpu") -> "SteeringModel":
        """Load a Hugging Face checkpoint or a local model directory."""
        config = AutoConfig.from_pretrained(model_id)
        if not isinstance(config, LlamaConfig):
            raise ValueError("SteerChat requires a LlamaForCausalLM checkpoint.")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = LlamaForCausalLM.from_pretrained(model_id, config=config)
        return cls(model.to(device), tokenizer)

    @property
    def num_layers(self) -> int:
        return len(self._model.model.layers)

    def _layer(self, index: int) -> torch.nn.Module:
        if type(index) is not int or not 0 <= index < self.num_layers:
            raise ValueError(f"Layer must be an integer from 0 to {self.num_layers - 1}.")
        return self._model.model.layers[index]

    def _encode(self, prompt: str, *, reserve_tokens: int = 0) -> dict[str, torch.Tensor]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Each prompt must be a nonempty string.")
        text = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        tokens = self._tokenizer(
            text, add_special_tokens=False, return_tensors="pt", return_attention_mask=True
        )
        length = tokens.input_ids.shape[1]
        context_length = self._model.config.max_position_embeddings
        if not length or length + reserve_tokens > context_length:
            raise ValueError(f"Prompt and requested continuation must fit within {context_length} tokens.")
        return {
            "input_ids": tokens.input_ids.to(self._model.device),
            "attention_mask": tokens.attention_mask.to(self._model.device),
        }

    @torch.no_grad()
    def fit(self, pairs: Iterable[tuple[str, str]], *, layer: int) -> torch.Tensor:
        """Return the mean positive-minus-negative activation as a CPU float32 vector."""
        decoder_layer = self._layer(layer)
        total = torch.zeros(self._model.config.hidden_size, dtype=torch.float32, device="cpu")
        last_activation = torch.empty_like(total)
        count = 0

        def capture(_module, _args, output: torch.Tensor) -> None:
            # Each forward pass is a single unpadded prompt.
            last_activation.copy_(output[0, -1])

        with self._lock, decoder_layer.register_forward_hook(capture):
            for pair in pairs:
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    raise ValueError("Each preference pair must contain a positive and a negative prompt.")
                positive, negative = pair
                self._model.model(**self._encode(positive), use_cache=False)
                positive_activation = last_activation.clone()
                self._model.model(**self._encode(negative), use_cache=False)
                total += positive_activation - last_activation
                count += 1

        if not count:
            raise ValueError("Provide at least one preference pair.")
        vector = total / count
        if not torch.isfinite(vector).all():
            raise ValueError("The fitted steering vector contains nonfinite values.")
        return vector

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        layer: int | None = None,
        vector: torch.Tensor | None = None,
        strength: float = 0.0,
    ) -> str:
        """Generate a greedy continuation, applying steering when strength is nonzero."""
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer.")
        if not math.isfinite(strength):
            raise ValueError("Steering strength must be finite.")

        with self._lock:
            inputs = self._encode(prompt, reserve_tokens=max_new_tokens)
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
            hook = None
            if strength:
                if layer is None or vector is None:
                    raise ValueError("A layer and vector are required for nonzero steering strength.")
                decoder_layer = self._layer(layer)
                if (
                    not isinstance(vector, torch.Tensor)
                    or vector.shape != (self._model.config.hidden_size,)
                    or not vector.is_floating_point()
                ):
                    raise ValueError("The steering vector must be floating point with one value per hidden dimension.")
                weight = next(decoder_layer.parameters())
                shift = (vector * strength).to(device=weight.device, dtype=weight.dtype)
                if not torch.isfinite(shift).all():
                    raise ValueError("The scaled steering vector contains nonfinite values.")

                def steer(_module, _args, output: torch.Tensor) -> torch.Tensor:
                    # The last position is the prefill boundary or the new cached token.
                    updated = output.clone()
                    updated[:, -1, :] += shift
                    return updated

                hook = decoder_layer.register_forward_hook(steer)

            try:
                generated = self._model.generate(
                    **inputs, generation_config=generation_config, use_model_defaults=False
                )
            finally:
                if hook is not None:
                    hook.remove()

            continuation = generated[0, inputs["input_ids"].shape[1] :].tolist()
            return self._tokenizer.decode(continuation, skip_special_tokens=True).strip()
