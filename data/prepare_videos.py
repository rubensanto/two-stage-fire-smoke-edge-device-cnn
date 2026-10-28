"""
Rebuild the test videos from their sources, as listed in data/video_sources_positive.csv and
data/video_sources_negative.csv. The videos are not redistributed in this repository.

    python data/prepare_videos.py [--out data/videos] [--only P13,N1,...]

Requirements: ffmpeg in the PATH and, for the YouTube videos, yt-dlp (pip install yt-dlp).

- YouTube videos with a recorded cut (start_s/end_s, crop and output size) are downloaded, cut, cropped and
  resized (letterbox or stretch) to 640x640 at the frame rate of the original test video.
- FIRESENSE (Zenodo 836749) and KMU videos are extracted from the original archives (only the needed files are
  read from the FIRESENSE archives) and resized to 640x640.
- The remaining sources (Pexels, LinkedIn, Reddit, news websites, YouTube videos without a recorded cut) are listed
  with their link for manual download; the notes column of the manifests describes the content.

Re-encoded videos differ slightly from the originals (codec, compression), so the frame-level results of the paper
are reproduced only approximately; the onset frames in video_sources_positive.csv refer to the original test videos.
"""
import argparse
import csv
import io
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ZENODO = "https://zenodo.org/api/records/836749/files/{}/content"
KMU_ZIP = "https://cvpr.kmu.ac.kr/Dataset/Smoke_flame_like_moving_object.zip"


class HttpFile(io.RawIOBase):
    """Seekable remote file (HTTP range requests), to read single members of a large zip."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def ffmpeg(args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def scale_filter(mode, size=640):
    if mode == "stretch":
        return f"scale={size}:{size}"
    return f"scale={size}:{size}:force_original_aspect_ratio=decrease,pad={size}:{size}:(ow-iw)/2:(oh-ih)/2"


def youtube(row, dst, tmp):
    src = tmp / f"{row['url'].split('v=')[1]}.mp4"
    if not src.exists():
        subprocess.run(["yt-dlp", "-f", "bestvideo[ext=mp4]+bestaudio/best", "--merge-output-format", "mp4",
                        "-o", str(src), row["url"]], check=True)
    # crop box recorded in the pixels of the original download; rescale if the downloaded resolution differs
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", str(src)], capture_output=True, text=True, check=True).stdout
    w, h = map(int, probe.strip().split(",")[:2])
    sx, sy = w / int(row["original_width"]), h / int(row["original_height"])
    crop = (f"crop={int(int(row['crop_w']) * sx)}:{int(int(row['crop_h']) * sy)}:"
            f"{int(int(row['crop_x']) * sx)}:{int(int(row['crop_y']) * sy)}")
    mode = row["output"].split()[-1]
    ffmpeg(["-ss", row["start_s"], "-to", row["end_s"], "-i", str(src), "-an",
            "-vf", f"{crop},{scale_filter(mode)},fps={row['fps']}", "-c:v", "libx264", "-crf", "18", str(dst)])


def from_zip(zf, member, dst, tmp):
    raw = tmp / Path(member).name
    with zf.open(member) as s, open(raw, "wb") as d:
        shutil.copyfileobj(s, d)
    ffmpeg(["-i", str(raw), "-an", "-vf", "scale=640:640", "-c:v", "libx264", "-crf", "18", str(dst)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "videos"))
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    only = set(filter(None, a.only.split(",")))
    out = Path(a.out)
    manual = []
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        zips = {}
        for kind, manifest in [("positive", "video_sources_positive.csv"), ("negative", "video_sources_negative.csv")]:
            (out / kind).mkdir(parents=True, exist_ok=True)
            for row in csv.DictReader(open(HERE / manifest, encoding="utf-8")):
                if only and row["id"] not in only:
                    continue
                dst = out / kind / row["file"]
                if dst.exists():
                    continue
                try:
                    if kind == "positive" and row["source"] == "YouTube" and row["start_s"]:
                        youtube(row, dst, tmp)
                    elif kind == "negative" and "zenodo" in row["url"]:
                        archive, member = [s.strip() for s in row["source_member"].split(":")]
                        if archive not in zips:
                            zips[archive] = zipfile.ZipFile(io.BufferedReader(HttpFile(ZENODO.format(archive)), 1 << 20))
                        from_zip(zips[archive], member, dst, tmp)
                    elif kind == "negative" and "kmu" in row["url"]:
                        if "kmu" not in zips:
                            data = urllib.request.urlopen(KMU_ZIP, timeout=300).read()
                            zips["kmu"] = zipfile.ZipFile(io.BytesIO(data))
                        from_zip(zips["kmu"], row["source_member"].split(":")[1].strip(), dst, tmp)
                    else:
                        manual.append(row)
                        continue
                    print("ok", row["id"], dst.name, flush=True)
                except Exception as e:     # keep going; report at the end
                    print("FAILED", row["id"], row["file"], e, file=sys.stderr, flush=True)
                    manual.append(row)
    if manual:
        print(f"\n{len(manual)} videos to be obtained manually (save them under {out}/positive or {out}/negative "
              "with the file name given):")
        for row in manual:
            note = row.get("notes") or row.get("processing", "")
            print(f"  {row['id']:4s} {row['file']:45s} {row['url']}  [{note}]")


if __name__ == "__main__":
    main()
