# SteerChat

SteerChat lets you compare how a local language model answers the same prompt at different steering strengths. It's an independent baseline inspired by Jessica Y. Bo et al.'s 2025 preprint, [*Steerable Chatbots: Personalizing LLMs with Preference-Based Activation Steering*](https://arxiv.org/abs/2505.04260v2).

## Run a comparison

Install uv, then run the following from a terminal. The first run installs the Python dependencies and downloads SmolLM2-135M-Instruct. Inference runs on the CPU by default. [uv installation](https://docs.astral.sh/uv/getting-started/installation/) · [SmolLM2 model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)

```sh
git clone https://github.com/wangjiazi/SteerChat.git
cd SteerChat
uv run steerchat --prompt "What is composting?" --strength -0.5 0 0.5
```

For the bundled preference pairs, positive strengths steer toward concise answers and negative strengths steer toward detailed answers. Strength `0` runs the unsteered baseline. All runs use the same prompt and greedy decoding. The degree of style change depends on the checkpoint, prompt, decoder layer, and strength. Larger magnitudes can degrade answer quality.

Use `--layer` to select a zero-based decoder layer, `--model` for another Transformers Llama checkpoint with a chat template, or `--device cuda` on a CUDA-enabled PyTorch installation. Run `uv run steerchat --help` for the full set of options.

## Use your own preference pairs

Pass `--pairs path/to/pairs.json` to fit a direction from your own examples. Each object contains two user messages for the same task. The `positive` message expresses the preference you want a positive strength to favor.

```json
[
  {
    "positive": "Give a concise answer. Why does the Moon have phases?",
    "negative": "Give a detailed answer. Why does the Moon have phases?"
  }
]
```

## How it works

1. Format each message with the checkpoint's chat template and capture the selected decoder block's output at the last input token.
2. Compute the steering direction as the mean of the paired differences, `v = mean(h_positive − h_negative)`.
3. Add `strength × v` to the same block's output at the final prompt token and at each subsequent decoding step.

The paper's reported experiments fit logistic regression probes and steer multiple layers selected by probe accuracy. SteerChat starts with a single-layer mean-difference baseline. The bundled pairs support a qualitative check on answer style. [Paper Appendix A.1](https://arxiv.org/html/2505.04260v3#A1.SS1)

The reproduction roadmap includes layer selection with logistic probes, held-out comparisons with prompt-based personalization, and an interactive chatbot.

## References

- [Bo et al., 2025 preprint](https://arxiv.org/abs/2505.04260v2) and [2026 revision](https://arxiv.org/abs/2505.04260v3)
- [EasySteer's replication notebook](https://github.com/ZJU-REAL/EasySteer/blob/main/replications/steerable_chatbot/steerable_chatbot.ipynb)
- [steering-vectors](https://github.com/steering-vectors/steering-vectors)
- [Microsoft's llm-steer-instruct](https://github.com/microsoft/llm-steer-instruct)

## License

[MIT](LICENSE)
