from torch import Tensor
from transformers import AutoProcessor
from transformers import EncodecModel

def tokenize(
    processor: AutoProcessor, model: EncodecModel, audio_values: Tensor
):
    inputs = processor(
        raw_audio=audio_values.squeeze(),
        sampling_rate=processor.sampling_rate,
        return_tensors="pt",
    )
    output = model.encode(inputs["input_values"], inputs["padding_mask"])
    return output.audio_codes.squeeze()


def detokenize(model: EncodecModel, audio_codes: Tensor):
    audio_codes = audio_codes.unsqueeze(0).unsqueeze(0)
    waveform = model.decode(audio_codes, [None])
    return waveform.audio_values.detach()[0]