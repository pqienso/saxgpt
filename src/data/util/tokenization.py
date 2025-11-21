from torch import Tensor
from transformers import AutoProcessor
from transformers import EncodecModel
import torch
from typing import Optional

model = EncodecModel.from_pretrained("facebook/encodec_32khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_32khz")


def tokenize(
    audio_values: Tensor,
    processor: AutoProcessor = processor,
    model: EncodecModel = model,
    chunk_len_s: Optional[float] = None,
    device: Optional = None,
):
    if audio_values.dim() > 2:
        raise NotImplementedError("Only single audio file supported for now")

    if device is not None:
        model = model.to(device)
    if chunk_len_s is None:
        chunks = [audio_values]
    else:
        chunks = torch.split(
            audio_values,
            int(chunk_len_s * processor.sampling_rate),
            dim=-1,
        )

    code_chunks = []
    for chunk in chunks:
        inputs = processor(
            raw_audio=chunk.squeeze().cpu(),
            sampling_rate=processor.sampling_rate,
            return_tensors="pt",
        )
        device = model.device
        output = model.encode(
            inputs["input_values"].to(device), inputs["padding_mask"].to(device)
        )
        # (nb_frames, batch_size, nb_quantizers, frame_len) -> (nb_quantizers, frame_len)
        audio_codes = output.audio_codes.squeeze(dim=(0, 1))
        code_chunks.append(audio_codes.cpu())

    codes = torch.cat(code_chunks, dim=-1)
    return codes


def detokenize(audio_codes: Tensor, model: EncodecModel = model, device: Optional = None):
    if device is not None:
        model = model.to(device)
    audio_codes = audio_codes.unsqueeze(0).unsqueeze(0)
    waveform = model.decode(audio_codes, [None])
    return waveform.audio_values.detach()[0]
