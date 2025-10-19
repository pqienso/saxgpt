from typing import List
import yt_dlp
import re
from pathlib import Path


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


def download_youtube_wav(video_url: str, download_folder: Path) -> None:
    video_id = extract_video_id(video_url)
    assert video_id is not None, "URL must be a valid YouTube video"

    wav_file_path = download_folder / f"{video_id}.wav"
    
    if wav_file_path.exists():
        print(f"File '{wav_file_path}' already exists. Skipping download.")
        return

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(download_folder / "%(id)s.%(ext)s"),
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

    with yt_dlp.YoutubeDL(
        {
            "quiet": True,
            "extract_flat": True,
            "playlistend": None,
        }
    ) as ydl:
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


def ingest_audio_url(youtube_url: str, download_folder: Path) -> None:
    is_video = is_video_url(youtube_url)
    is_playlist = is_playlist_url(youtube_url)
    assert is_video or is_playlist, "URL must be either a YouTube playlist or video"

    if is_playlist:
        video_urls = extract_playlist_urls(youtube_url)
    else:
        video_urls = [youtube_url]

    for i, url in enumerate(video_urls):
        print(f"\n\n Downloading {i + 1} of {len(video_urls)}")
        download_youtube_wav(url, download_folder)

