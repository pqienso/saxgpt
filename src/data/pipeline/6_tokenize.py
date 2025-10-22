from transformers import AutoProcessor
from transformers import EncodecModel
import torch
import argparse
import yaml
from tqdm import tqdm
from pathlib import Path

from ..util.tokenization import tokenize


def tokenize_clips_in_example(
    example_path: Path,
    codes_dir: Path,
    **tokenize_args,
):
    video_id = example_path.stem
    codes_dir = codes_dir / f"{video_id}.pt"
    if codes_dir.exists():
        return

    example = torch.load(example_path, weights_only=False)
    for clip in tqdm(example, leave=False):
        clip["lead"] = tokenize(clip["lead"], **tokenize_args).cpu()
        clip["backing"] = tokenize(clip["backing"], **tokenize_args).cpu()
    torch.save(example, codes_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(
        description="Cut audio, augment pitch/tempo and tokenize audio."
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use CUDA for tokenization",
    )
    parser.add_argument("--rebuild", action="store_true", help="Retokenize all clips")
    args, _ = parser.parse_known_args()

    print("\n\n" + "=" * 20)
    print("Pipeline Step 6: Tokenize Audio")
    print("=" * 20 + "\n\n")

    with open(Path(args.config), "r") as file:
        config = yaml.safe_load(file)
    try:
        aug_dir_str = config["data_paths"]["aug_dir"]
        codes_dir_str = config["data_paths"]["codes_dir"]

        encodec_chunk_len = config["encodec"]["chunk_len_s"]

        keep_aug_clips = config["intermediates"]["keep_aug_clips"]
    except KeyError as e:
        print(f"Error: Missing key in configuration file: {e}")
        raise

    codes_dir = Path(codes_dir_str)
    codes_dir.mkdir(exist_ok=True)
    aug_dir = Path(aug_dir_str)

    if args.rebuild:
        for example in codes_dir.glob("*.pt"):
            example.unlink()

    print("\nBeginning tokenization")
    print("Getting model and processor")
    device = torch.device("cuda" if args.cuda else "cpu")
    print(f"Using device {device}\n")

    model = EncodecModel.from_pretrained("facebook/encodec_32khz").to(device)
    processor = AutoProcessor.from_pretrained("facebook/encodec_32khz")

    num_files = len(list(aug_dir.glob("*.pt")))
    for example_path in tqdm(aug_dir.glob("*.pt"), total=num_files):
        tokenize_clips_in_example(
            example_path,
            codes_dir,
            processor=processor,
            model=model,
            chunk_len_s=encodec_chunk_len,
        )
    print(f"\n\nTokenization complete. Codes saved to {codes_dir}")

    if not keep_aug_clips:
        for example in aug_dir.glob("*.pt"):
            example.unlink()
