from transformers import AutoProcessor
from transformers import EncodecModel
import torch
import os
import argparse
from tqdm import tqdm

from data_processing.data_pipeline import AudioExample

EXAMPLE_SIZE = 30 * 50
STRIDE = 10 * 50


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
    print(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help=".pt dataset file location")
    parser.add_argument(
        "dest",
        help="Destination to save codes (.pt file path)",
    )
    parser.add_argument(
        "--start",
        type=int,
        help="Specify start index of audio examples",
        default=None,
    )
    parser.add_argument(
        "--end",
        type=int,
        help="Specify end index of audio examples",
        default=None,
    )
    args = parser.parse_args()

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
        assert lead_codes.shape == backing_codes.shape, (
            "length of tokenized audio must be equal"
        )
        codes.append((backing_codes, lead_codes))
    torch.save(
        codes,
        args.dest,
    )
