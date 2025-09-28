from transformers import AutoProcessor
from transformers import EncodecModel
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
import yaml

from data_processing.data_pipeline import AudioExample

EXAMPLE_SIZE = 30 * 50 # 50 tokens = 1s


def convert_wav_to_tensor(
    processor: AutoProcessor, model: EncodecModel, audio_values: torch.Tensor
):
    inputs = processor(
        raw_audio=audio_values.squeeze(),
        sampling_rate=processor.sampling_rate,
        return_tensors="pt",
    )
    output = model.encode(inputs["input_values"], inputs["padding_mask"])
    return output.audio_codes.squeeze()


def convert_tensor_to_wav(model: EncodecModel, audio_codes: torch.Tensor):
    audio_codes = audio_codes.unsqueeze(0).unsqueeze(0)
    waveform = model.decode(audio_codes, [None])
    return waveform.audio_values.detach()[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(
        description="Cut audio, augment pitch / tempo, and tokenize."
    )
    parser.add_argument(
        "--config-path",
        type=str,
        help="Path to the YAML configuration file",
        default="config/data/main.yaml",
    )
    args = parser.parse_args()

    with open(Path(args.config_path), "r") as file:
        config = yaml.safe_load(file)
    try:
        stem_path_str = config["data_paths"]["stem_dest"]
        metadata_path_str = config["data_paths"]["metadata_path"]
    except KeyError as e:
        print(f"Error: Missing key in configuration file: {e}")
        raise

    model = EncodecModel.from_pretrained("facebook/encodec_32khz")
    processor = AutoProcessor.from_pretrained("facebook/encodec_32khz")

    print("Loading examples...")
    examples: list[AudioExample] = torch.load(
        args.dataset, map_location="cpu", weights_only=False
    )
    examples = examples[args.start : args.end]

    print("Beginning tokenization")
    codes = []
    for example in tqdm(examples):
        lead_codes = convert_wav_to_tensor(processor, model, example.lead)
        backing_codes = convert_wav_to_tensor(processor, model, example.backing)
        if lead_codes.shape != backing_codes.shape:
            print("\n\nWARNING: lead and backing codes have different shape")
        codes.append((backing_codes, lead_codes))
    torch.save(
        codes,
        args.dest,
    )
