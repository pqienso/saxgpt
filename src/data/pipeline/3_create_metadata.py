import argparse
import yaml
from pathlib import Path

from ..util.metadata import update_metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="Update audio metadata.")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Create metadata from scratch"
    )
    args, _ = parser.parse_known_args()

    print("\n\n" + "=" * 20)
    print("Pipeline Step 3: Create Metadata")
    print("=" * 20 + "\n\n")

    with open(Path(args.config), "r") as file:
        config = yaml.safe_load(file)
    try:
        dl_dir_str = config["data_paths"]["dl_dir"]
        stems_dir_str = config["data_paths"]["stems_dir"]
        metadata_path_str = config["data_paths"]["metadata_path"]

        rms_window_args = config["rms_window"]
    except KeyError as e:
        print(f"Error: Missing key in configuration file: {e}")
        raise

    dl_dir = Path(dl_dir_str)
    stems_dir = Path(stems_dir_str)
    metadata_path = Path(metadata_path_str)

    if args.rebuild:
        metadata_path.unlink()

    update_metadata(metadata_path, dl_dir, stems_dir, rms_window_args)
    print("\nMetadata updated.\n")
