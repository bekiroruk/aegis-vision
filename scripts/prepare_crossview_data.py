"""Bounded official WILDTRACK subset; data preparation only, no inference.

Only explicitly selected ZIP members are fetched with verified HTTP byte ranges.
No 6.8 GB full download, fallback mirror, automatic extraction, or model fitting.
"""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import re
import struct
from urllib.request import Request, urlopen
import zipfile

URL = "https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/Wildtrack/Wildtrack_dataset_full.zip"
SIZE = 6807496358
ETAG = '"5b86d551-195c226a6"'
FRAMES = tuple(range(0, 2000, 125))  # frozen before model inference, 16 timestamps
CAMERAS = (1, 2)
PROFILES = {
    "diagnostic": (FRAMES, CAMERAS, None, "wildtrack-c1-c2-same-frame-gt-crops-v1"),
    "development": (tuple(range(60, 2000, 50)), (3, 4), 0, "wildtrack-flip-development-v1"),
    "holdout": (tuple(range(60, 2000, 200)), (5, 6), 1, "wildtrack-flip-holdout-v1"),
}


class RemoteZip(io.RawIOBase):
    def __init__(self, url=URL):
        self.url, self.position, self.transferred = url, 0, 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        position = offset + (self.position if whence == 1 else SIZE if whence == 2 else 0)
        if whence not in (0, 1, 2) or not 0 <= position <= SIZE:
            raise ValueError("ZIP seek outside official archive")
        self.position = position
        return position

    def tell(self):
        return self.position

    def read(self, size=-1):
        size = SIZE - self.position if size < 0 else min(size, SIZE - self.position)
        if not size:
            return b""
        if size > 16 * 1024 * 1024 or self.transferred + size > 200 * 1024 * 1024:
            raise ValueError("Range/download budget exceeded")
        start, end = self.position, self.position + size - 1
        request = Request(self.url, headers={"Range": f"bytes={start}-{end}", "If-Match": ETAG,
                                           "Accept-Encoding": "identity"})
        with urlopen(request, timeout=30) as response:
            if (response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{SIZE}"
                    or response.headers.get("ETag") != ETAG):
                raise ValueError("Server did not honor pinned byte range; refusing full download")
            content = response.read(size + 1)
        if len(content) != size:
            raise ValueError("Truncated or oversized range")
        self.position += size
        self.transferred += size
        return content


def digest(content):
    return hashlib.sha256(content).hexdigest()


def header_matches(stream, info, offset):
    if not 0 <= offset < SIZE - 30:
        return False
    stream.seek(offset)
    header = stream.read(30)
    if len(header) != 30 or header[:4] != b"PK\x03\x04":
        return False
    name_size, extra_size = struct.unpack_from("<HH", header, 26)
    if not 0 < name_size <= 1024 or offset + 30 + name_size + extra_size + info.compress_size > SIZE:
        return False
    return stream.read(name_size) == info.filename.encode("utf-8")


def resolve_header(stream, info):
    # This legacy >4 GiB archive wraps classic ZIP offsets without ZIP64.
    # Resolve only bounded candidates, verifying the complete member name;
    # ZipFile subsequently verifies decompression and the member CRC.
    candidates = [info.header_offset, info.header_offset - 2**32]
    valid = [offset for offset in candidates if header_matches(stream, info, offset)]
    if len(valid) != 1:
        raise ValueError("Cannot uniquely resolve legacy ZIP member header")
    resolved = copy.copy(info)
    resolved.header_offset = valid[0]
    return resolved


def resolve_member(stream, info, entries, directory_start):
    resolved = resolve_header(stream, info)
    if not resolved.header_offset < directory_start <= SIZE:
        raise ValueError("Invalid central directory boundary")
    # ZIP32 rollover makes ZipFile's sorted raw-offset _end_offset unreliable:
    # early members shifted by 4 GiB appear inside genuine C6 payloads. Find the
    # next *verified* local header, not merely the next raw central-directory
    # offset. Retain ZipFile's overlap rejection against that physical boundary.
    candidates = []
    for other in entries:
        for offset in (other.header_offset, other.header_offset - 2**32):
            if resolved.header_offset < offset < directory_start:
                candidates.append((offset, other))
    candidates.sort(key=lambda pair: pair[0])
    boundary = directory_start
    for probe, (offset, other) in enumerate(candidates):
        if probe >= 32:
            raise ValueError("Cannot establish bounded next ZIP member header")
        if header_matches(stream, other, offset):
            boundary = offset
            break
    resolved._end_offset = boundary
    return resolved


def boxes(annotation, view, width=1920, height=1080):
    if not isinstance(annotation, list) or len(annotation) > 512:
        raise ValueError("Invalid WILDTRACK person list")
    result, seen = [], set()
    for person in annotation:
        identity = person["personID"]
        if type(identity) is not int or identity < 0 or identity in seen:
            raise ValueError("Invalid or duplicate ground-truth identity")
        seen.add(identity)
        views = [v for v in person["views"] if v["viewNum"] == view]
        if len(views) != 1:
            raise ValueError("Missing/duplicate view annotation")
        v = views[0]
        raw = [v[name] for name in ("xmin", "ymin", "xmax", "ymax")]
        if any(type(n) is not int for n in raw):
            raise ValueError("Noninteger annotation box")
        if raw == [-1] * 4:
            continue
        x1, y1, x2, y2 = raw
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        # Fixed visibility/size filter, never selected using appearance scores.
        if x2 - x1 < 20 or y2 - y1 < 64:
            continue
        result.append({"identity": identity, "bbox": [x1, y1, x2, y2]})
    return sorted(result, key=lambda row: row["identity"])


def prepare(output, inventory=False, profile="diagnostic"):
    selected_frames, cameras, parity, protocol = PROFILES[profile]
    remote = RemoteZip()
    with zipfile.ZipFile(remote) as archive:
        names = archive.namelist()
        if len(names) > 100000 or len(names) != len(set(names)):
            raise ValueError("Unexpected ZIP directory")
        if inventory:
            print(json.dumps(names[:20], indent=2))
            return
        output.mkdir(parents=True, exist_ok=False)
        assets, frames = [], []

        def fetch(suffix):
            matches = [n for n in names if n.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError(f"Missing/ambiguous archive member {suffix}")
            info = archive.getinfo(matches[0])
            if info.file_size > 16 * 1024 * 1024 or info.compress_size > 16 * 1024 * 1024:
                raise ValueError("Oversized ZIP member")
            info = resolve_member(remote, info, archive.infolist(), archive.start_dir)
            content = archive.read(info)  # validates ZIP CRC; paths are never extracted
            relative = suffix.lstrip("/")
            if not re.fullmatch(r"(?:Image_subsets/C[1-7]/\d{8}\.png|annotations_positions/\d{8}\.json)", relative):
                raise ValueError("Unexpected output member")
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            assets.append(dict(path=relative, sha256=digest(content), bytes=len(content), archive_member=info.filename))
            return relative, content

        for frame in selected_frames:
            _, annotation = fetch(f"/annotations_positions/{frame:08d}.json")
            labels = json.loads(annotation)
            views = []
            for camera in cameras:
                name, content = fetch(f"/Image_subsets/C{camera}/{frame:08d}.png")
                people = boxes(labels, camera - 1)
                if parity is not None:
                    people = [p for p in people if p["identity"] % 2 == parity]
                views.append(dict(camera=f"C{camera}", image=name, sha256=digest(content),
                                  persons=people))
            frames.append(dict(frame=frame, views=views))
            print(f"Prepared synchronized frame {frame}, camera boxes {[len(v['persons']) for v in views]}", flush=True)
        manifest = dict(version=1, protocol=protocol,
            source_url=URL, archive_size=SIZE, archive_etag=ETAG,
            subset_frames=list(selected_frames), cameras=[f"C{c}" for c in cameras], frames=frames,
            min_similarity=.85, ambiguity_margin=.05,
            warning="Diagnostic fixed subset with oracle boxes. Not detector, tracker, handoff or full-dataset accuracy.")
        (output / "provenance.json").write_text(json.dumps(dict(assets=assets, transferred_bytes=remote.transferred), indent=2), encoding="utf-8")
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Prepared {len(frames)} synchronized pairs; downloaded {remote.transferred} bytes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--profile", choices=PROFILES, default="diagnostic")
    args = parser.parse_args()
    if not args.download:
        parser.error("Explicit --download is required")
    prepare(args.output, args.inventory, args.profile)
