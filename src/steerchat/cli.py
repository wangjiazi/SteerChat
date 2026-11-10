"""Compare generations across steering strengths."""

import argparse
import json
import math
from importlib.resources import files
from pathlib import Path


DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def load_pairs(path: Path | None) -> list[tuple[str, str]]:
    source = path if path is not None else files("steerchat").joinpath("concise_pairs.json")
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Expected a non-empty JSON array of preference pairs.")

    pairs = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Pair {index} must be a JSON object.")
        positive, negative = item.get("positive"), item.get("negative")
        if not all(isinstance(prompt, str) and prompt.strip() for prompt in (positive, negative)):
            raise ValueError(f"Pair {index} needs non-empty positive and negative prompts.")
        pairs.append((positive, negative))
    return pairs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True, help="User message for all runs.")
    parser.add_argument(
        "--strength",
        type=float,
        nargs="+",
        default=[-0.5, 0.0, 0.5],
        help="Signed steering strengths. Use 0 for the unsteered baseline (default: -0.5 0 0.5).",
    )
    parser.add_argument("--layer", type=int, help="Zero-based decoder layer (default: middle).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Llama checkpoint with a chat template.")
    parser.add_argument("--device", default="cpu", help="PyTorch device, such as cpu or cuda.")
    parser.add_argument(
        "--pairs", type=Path, help="JSON preference pairs (default: concise versus detailed)."
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=128, help="Token budget per response (default: 128)."
    )
    args = parser.parse_args(argv)

    if not args.prompt.strip():
        parser.error("--prompt must contain text.")
    if not all(math.isfinite(strength) for strength in args.strength):
        parser.error("--strength values must be finite.")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive.")

    try:
        pairs = load_pairs(args.pairs) if any(args.strength) else []
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    from .steering import SteeringModel

    try:
        model = SteeringModel.from_pretrained(args.model, device=args.device)
        layer = model.num_layers // 2 if args.layer is None else args.layer
        if not 0 <= layer < model.num_layers:
            parser.error(f"--layer must be between 0 and {model.num_layers - 1}.")
        vector = model.fit(pairs, layer=layer) if pairs else None
        print(f"Model {args.model} | Layer {layer}", flush=True)
        for strength in args.strength:
            response = model.generate(
                args.prompt,
                max_new_tokens=args.max_new_tokens,
                layer=layer,
                vector=vector,
                strength=strength,
            )
            print(f"\nStrength {strength:g}\n{response}", flush=True)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"steerchat: {exc}\n")


if __name__ == "__main__":
    main()
