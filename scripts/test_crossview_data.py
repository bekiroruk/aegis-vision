"""Preparation tests: no network, model, downloaded data or third-party libs."""
import io
import struct
import unittest
from unittest.mock import patch
from zipfile import ZipInfo
import prepare_crossview_data as data


def person(identity=1, coordinates=(10, 20, 50, 100)):
    return dict(personID=identity, views=[dict(viewNum=0,
        **dict(zip(("xmin", "ymin", "xmax", "ymax"), coordinates)))])


class Response(io.BytesIO):
    def __init__(self, content=b"abc", status=206, content_range=f"bytes 0-2/{data.SIZE}", etag=data.ETAG):
        super().__init__(content)
        self.status = status
        self.headers = {"Content-Range": content_range, "ETag": etag}


class PreparationTests(unittest.TestCase):
    def test_crop_clipping_and_fixed_filter(self):
        result = data.boxes([person(2), person(1, (-10, -20, 40, 80)),
                             person(3, (-1, -1, -1, -1)), person(4, (0, 0, 19, 63))], 0)
        self.assertEqual(result, [dict(identity=1, bbox=[0, 0, 40, 80]), dict(identity=2, bbox=[10, 20, 50, 100])])

    def test_duplicate_ids(self):
        with self.assertRaises(ValueError):
            data.boxes([person(), person()], 0)

    def test_duplicate_view(self):
        p = person(); p["views"] *= 2
        with self.assertRaises(ValueError):
            data.boxes([p], 0)

    def test_wrong_types(self):
        for p in (person(True), person(1, (1.5, 0, 50, 80))):
            with self.assertRaises(ValueError):
                data.boxes([p], 0)

    def test_range_success(self):
        with patch.object(data, "urlopen", return_value=Response()) as request:
            stream = data.RemoteZip()
            self.assertEqual(stream.read(3), b"abc")
            self.assertEqual(stream.tell(), 3)
            self.assertEqual(stream.transferred, 3)
            self.assertEqual(request.call_args.args[0].get_header("Range"), "bytes=0-2")

    def test_unsafe_response_rejected(self):
        for response in (Response(status=200), Response(content_range="bytes 1-3/100"),
                         Response(etag='"changed"'), Response(content=b"ab"), Response(content=b"abcd")):
            with patch.object(data, "urlopen", return_value=response):
                with self.assertRaises(ValueError):
                    data.RemoteZip().read(3)

    def test_download_budget(self):
        with self.assertRaises(ValueError):
            data.RemoteZip().read(17 * 1024 * 1024)
        stream = data.RemoteZip(); stream.transferred = 200 * 1024 * 1024
        with self.assertRaises(ValueError):
            stream.read(1)

    def test_seek_bounds(self):
        stream = data.RemoteZip()
        self.assertEqual(stream.seek(-22, 2), data.SIZE - 22)
        with self.assertRaises(ValueError):
            stream.seek(-1)
        with self.assertRaises(ValueError):
            stream.seek(data.SIZE + 1)

    def test_legacy_wrapped_header(self):
        name = "Wildtrack_dataset/test.json"
        header = bytearray(30)
        header[:4] = b"PK\x03\x04"
        struct.pack_into("<HH", header, 26, len(name), 0)
        stream = io.BytesIO(bytes(header) + name.encode())
        entry = ZipInfo(name); entry.header_offset = 2**32; entry.compress_size = 0
        self.assertEqual(data.resolve_header(stream, entry).header_offset, 0)
        self.assertEqual(entry.header_offset, 2**32)  # source directory unchanged

    def test_bad_header_rejected(self):
        entry = ZipInfo("test"); entry.header_offset = 0; entry.compress_size = 0
        with self.assertRaises(ValueError):
            data.resolve_header(io.BytesIO(b"x" * 100), entry)

    def test_frozen_subset(self):
        self.assertEqual(data.FRAMES, tuple(range(0, 2000, 125)))
        self.assertEqual(data.CAMERAS, (1, 2))

    def test_experiment_camera_and_identity_split(self):
        dev, holdout = data.PROFILES["development"], data.PROFILES["holdout"]
        self.assertFalse(set(dev[1]) & set(holdout[1]))
        self.assertEqual((dev[2], holdout[2]), (0, 1))
        self.assertEqual(dev[0], tuple(range(60, 2000, 50)))
        self.assertFalse(set(dev[0]) & set(data.FRAMES))


if __name__ == "__main__":
    unittest.main()
