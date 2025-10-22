import torch
from tqdm import tqdm
import yaml
import argparse
from pathlib import Path

from ..util.augmenter import AudioAugmenter


def augment_example(aug_dir: Path, example: Path, augmenter: AudioAugmenter):
    video_id = example.stem

    if (aug_dir / f"{video_id}.pt").exists():
        return

    aug_clips = []
    example = torch.load(example_path, weights_only=False)
    for clip in example:
        augmented_leads = augmenter(clip["lead"])
        augmented_backings = augmenter(clip["backing"])

        for augmented_lead, augmented_backing, augmentation_info in zip(
            augmented_leads, augmented_backings, augmenter.aug_info
        ):
            aug_clip = {
                "lead": augmented_lead,
                "backing": augmented_backing,
                "start": clip["start"],
                "end": clip["end"],
            }
            aug_clip.update(augmentation_info)
            aug_clips.append(aug_clip)

    torch.save(aug_clips, aug_dir / f"{video_id}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="Augment audio clips.")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the YAML configuration file",
    )
    parser.add_argument("--rebuild", type=str, help="Restart augmentation")
    args, _ = parser.parse_known_args()

    print("\n\n" + "=" * 20)
    print("Pipeline Step 5: Augment Audio")
    print("=" * 20 + "\n\n")

    with open(Path(args.config), "r") as file:
        config = yaml.safe_load(file)
    try:
        clips_dir_str = config["data_paths"]["clips_path"]
        aug_dir_str = config["data_paths"]["aug_path"]

        aug_cfg = config["augmentation"]

        keep_clips = config["intermediates"]["keep_clips"]
    except KeyError as e:
        print(f"Error: Missing key in configuration file: {e}")
        raise

    clips_dir = Path(clips_dir_str)
    aug_dir = Path(aug_dir_str)
    aug_dir.mkdir(exist_ok=True)

    if args.rebuild:
        for example in aug_dir.glob("*.pt"):
            example.unlink()

    if aug_cfg is None:
        aug_cfg = {
            "semitone_steps": [],
            "tempo_ratios": [],
        }
    augmenter = AudioAugmenter(**aug_cfg)

    print("\nAugmenting audio clips\n")
    num_files = len(list(clips_dir.glob("*.pt")))
    for example_path in tqdm(clips_dir.glob("*.pt"), total=num_files):
        augment_example(aug_dir, example_path, augmenter)
    print("\nAugmentation complete.\n")

    if not keep_clips:
        for example in clips_dir.glob("*.pt"):
            example.unlink()
