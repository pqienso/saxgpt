from typing import List
import yt_dlp
import os
import re
import pandas as pd
from glob import glob


def extract_video_id(youtube_url: str):
    video_pattern = r"https?://(?:www\.)?youtube\.com/watch\?v=([A-Za-z0-9_-]{11})"
    match = re.search(video_pattern, youtube_url)
    return match.group(1) if match else None


def is_playlist_url(youtube_url: str) -> bool:
    playlist_pattern = r"https?://(?:www\.)?youtube\.com/playlist\?list=[A-Za-z0-9_-]+"
    return bool(re.match(playlist_pattern, youtube_url))


def is_video_url(youtube_url: str) -> bool:
    video_pattern = r"https?://(?:www\.)?youtube\.com/watch\?v=[A-Za-z0-9_-]{11}"
    return bool(re.match(video_pattern, youtube_url))


def download_youtube_wav(video_url: str, download_folder: str) -> None:
    video_id = extract_video_id(video_url)
    assert video_id is not None, "URL must be a valid YouTube video"

    wav_file_name = os.path.join(download_folder, f"{video_id}.wav")
    if os.path.exists(wav_file_name):
        print(f"File '{wav_file_name}' already exists. Skipping download.")
        return

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(download_folder, "%(id)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(video_url, download=True)
    except yt_dlp.DownloadError:
        print(f"Download of {video_url} failed, skipping.")


def extract_playlist_urls(playlist_url: str) -> List[str]:
    assert is_playlist_url(playlist_url), "URL must be a valid YouTube playlist"

    with yt_dlp.YoutubeDL({"quiet": True, "extract_flat": True}) as ydl:
        info_dict = ydl.extract_info(playlist_url, download=False)
        video_urls = [entry["url"] for entry in info_dict["entries"]]
        return video_urls


def extract_url_title(youtube_url: str) -> str:
    assert is_video_url(youtube_url) or is_playlist_url(youtube_url), (
        "URL must be of a YouTube video/playlist."
    )
    with yt_dlp.YoutubeDL({"quiet": True, "extract_flat": True}) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        return info_dict["title"]


def ingest_audio_url(youtube_url: str, download_folder: str) -> None:
    is_video = is_video_url(youtube_url)
    is_playlist = is_playlist_url(youtube_url)
    assert is_video or is_playlist, "URL must be either a YouTube playlist or video"

    if is_playlist:
        video_urls = extract_playlist_urls(youtube_url)
    else:
        video_urls = [youtube_url]

    for url in video_urls:
        download_youtube_wav(url, download_folder)


def update_audio_titles(audio_metadata_csv_path: str, audio_download_path: str) -> None:
    df = pd.read_csv(audio_metadata_csv_path).set_index("video_id", drop=True)
    audio_files = glob(os.path.join(audio_download_path, "*.wav"))
    new_audio_metadata = []
    for audio_file in audio_files:
        video_id = os.path.splitext(os.path.basename(audio_file))[0]
        if video_id not in df.index:
            video_title = extract_url_title(
                f"https://www.youtube.com/watch?v={video_id}"
            )
            new_audio_metadata.append(
                {
                    "video_id": video_id,
                    "video_title": video_title,
                    "valid_windows": pd.NA,
                }
            )
    new_df = pd.DataFrame(new_audio_metadata)
    df = pd.concat([new_df, df.reset_index()], axis="index").reset_index(drop=True)
    df.to_csv(audio_metadata_csv_path, index=False)
