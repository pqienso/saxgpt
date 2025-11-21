import torch
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import torchaudio
from datetime import timedelta
from tqdm import tqdm
from argparse import ArgumentParser
import yaml

from ..training.util.checkpointing import load_checkpoint_for_inference
from ..training.util.config_model import create_model
from ..data.util.tokenization import tokenize, detokenize
from ..data.util.codes_interleaving import (
    add_delay_interleaving,
    remove_delay_interleaving,
)
from ..data.util.audio_util import resample, convert_mono, normalize_lufs, trim_audio


class SaxGPT:
    """
    Wrapper around the model to produce Saxophone solos.
    """

    MODEL_SR = 32000

    def __init__(
        self,
        model_config: Dict,
        checkpoint_path: Path,
        device=torch.device("cpu"),
        encodec_chunk_len: float = 25.0,
    ):
        self.model = create_model(model_config).to(device)
        _ = load_checkpoint_for_inference(self.model, checkpoint_path, device)
        self.device = device
        self.encodec_chunk_len = encodec_chunk_len
        self.start_tokens = torch.tensor(
            [[2048 for _ in range(model_config["model"]["num_codebooks"])]],
            dtype=int,
            device=device,
        )

    def _process_backing_audio(
        self,
        audio_path: Path,
        trim: bool = True,
        clip_length_s: Optional[float] = 30.0,
    ) -> Tuple[List[Path], List[torch.Tensor]]:
        if trim:
            assert clip_length_s is not None

        if audio_path.is_dir():
            audio_paths = [path for path in audio_path.glob("*.wav")]
        else:
            audio_paths = [audio_path]

        processed_audio = []
        for path in audio_paths:
            waveform, sample_rate = torchaudio.load(path)
            if trim:
                waveform = trim_audio(
                    waveform=waveform,
                    end=timedelta(seconds=clip_length_s),
                    sample_rate=sample_rate,
                )
            waveform = convert_mono(waveform)
            if sample_rate != self.MODEL_SR:
                waveform = resample(waveform, sample_rate, self.MODEL_SR)
            waveform = normalize_lufs(waveform, self.MODEL_SR)
            processed_audio.append(waveform)

        return audio_paths, processed_audio

    def generate_solos(
        self,
        audio_path: Path,
        temperature: float = 0.9,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        trim_audio: bool = True,
        max_token_len: int = 1505,
        clip_length_s: float = 30.0,
    ):
        print("Processing audio...")
        audio_paths, processed_audio = self._process_backing_audio(
            audio_path, trim=trim_audio, clip_length_s=clip_length_s
        )
        print("Beginning generation...")
        for audio_path, backing_audio in tqdm(zip(audio_paths, processed_audio)):
            codes = add_delay_interleaving(
                tokenize(
                    backing_audio.to(self.device),
                    chunk_len_s=self.encodec_chunk_len,
                    device=self.device,
                )
            )
            gen_codes = self.model.generate(
                codes.unsqueeze(0).to(self.device),
                max_len=max_token_len,
                start_tokens=self.start_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            ).squeeze(0)
            gen_audio = detokenize(
                remove_delay_interleaving(gen_codes), device=self.device
            )

            gen_audio_path = audio_path.parent / f"{audio_path.stem}_gen.wav"
            torchaudio.save(gen_audio_path, gen_audio, self.MODEL_SR)

            audio_len = min(gen_audio.shape[-1], backing_audio.shape[-1])
            combined_audio = torch.clamp(
                gen_audio[:, :audio_len] + backing_audio[:, :audio_len], -1.0, 1.0
            )
            combined_audio_path = audio_path.parent / f"{audio_path.stem}_combined.wav"
            torchaudio.save(combined_audio_path, combined_audio, self.MODEL_SR)


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate Jazz backing tracks with SaxGPT")

    parser.add_argument("--config", type=str, help="Path to model config .yaml")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint for model")
    parser.add_argument(
        "--audio", type=str, help="Path to audio file / directory of audio files"
    )
    parser.add_argument("--cuda", action="store_true", help="Use CUDA for inference")
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="Use full audio instead of using first 30 seconds",
    )
    parser.add_argument(
        "--temp",
        type=float,
        help="Sampling temperature",
        default=0.9,
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    model = SaxGPT(
        config,
        Path(args.checkpoint),
        device=torch.device("cuda") if args.cuda else torch.device("cpu"),
    )

    model.generate_solos(Path(args.audio), args.temp, trim_audio=not args.no_trim)
