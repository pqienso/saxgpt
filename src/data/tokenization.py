from torch import Tensor
from transformers import AutoProcessor
from transformers import EncodecModel

model = EncodecModel.from_pretrained("facebook/encodec_32khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_32khz")


def tokenize(
    audio_values: Tensor,
    processor: AutoProcessor = processor,
    model: EncodecModel = model,
):
    inputs = processor(
        raw_audio=audio_values.squeeze(),
        sampling_rate=processor.sampling_rate,
        return_tensors="pt",
    )
    device = model.device
    output = model.encode(
        inputs["input_values"].to(device), inputs["padding_mask"].to(device)
    )
    return output.audio_codes.squeeze()


def detokenize(audio_codes: Tensor, model: EncodecModel = model):
    audio_codes = audio_codes.unsqueeze(0).unsqueeze(0)
    waveform = model.decode(audio_codes, [None])
    return waveform.audio_values.detach()[0]
