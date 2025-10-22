from pathlib import Path
import argparse
import yaml

from ..util.data_ingestion import ingest_audio_url

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest audio from YouTube playlist."
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Start with clean download directory"
    )
    args, _ = parser.parse_known_args()

    print("\n\n" + "=" * 20)
    print("Pipeline Step 1: Data Ingestion")
    print("=" * 20 + "\n\n")

    with open(Path(args.config), "r") as file:
        config = yaml.safe_load(file)
    try:
        dl_dir_str = config["data_paths"]["dl_dir"]

        url = config["url"]
    except KeyError as e:
        print(f"Error: Missing key in configuration file: {e}")
        raise

    dl_dir = Path(dl_dir_str)
    dl_dir.mkdir(exist_ok=True)

    if args.rebuild:
        for audio_file in dl_dir.glob("*.wav"):
            audio_file.unlink()

    print("\nDownloading audio...\n")
    ingest_audio_url(url, dl_dir)
    print("\nAudio ingestion complete.\n")
