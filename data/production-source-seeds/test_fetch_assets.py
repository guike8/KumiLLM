"""Offline checks for download bounds, integrity and selective archive reads."""
import hashlib
import io
from pathlib import Path
import random
import tarfile
import tempfile
import unittest
from fetch_assets import archive_members, Budget, LimitedReader, safe_path, verify

class FetchTests(unittest.TestCase):
    def test_original_hash_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            verify(b"changed",{"bytes":7,"sha256":"0"*64})

    def test_read_cannot_pass_stream_limit(self):
        response = io.BytesIO(b"a"*100)
        reader = LimitedReader(response,8,Budget(20))
        self.assertEqual(reader.read(100),b"a"*8)
        with self.assertRaises(ValueError):
            reader.read(1)
        self.assertEqual(response.tell(),8)

    def test_paths_cannot_escape_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ["../escape.wav","/tmp/escape.wav"]:
                with self.assertRaises(ValueError):
                    safe_path(root,name)

    def test_archive_stops_after_requested_member(self):
        data = b"actual-audio-placeholder"
        member = "LibriSpeech/train-clean-100/1/2/1-2-3.flac"
        payload = io.BytesIO()
        with tarfile.open(fileobj=payload,mode="w:gz") as archive:
            info = tarfile.TarInfo(member)
            info.size = len(data)
            archive.addfile(info,io.BytesIO(data))
            later = random.Random(4).randbytes(200_000)
            info = tarfile.TarInfo("unrequested.flac")
            info.size = len(later)
            archive.addfile(info,io.BytesIO(later))
        source = payload.getvalue()
        class Opener:
            def open(self,*args,**kwargs):
                return io.BytesIO(source)
        spec = {"archive_member":member,"url":"https://openslr.trmal.net/resources/12/train-clean-100.tar.gz","archive_stream_limit_bytes":32_768,"path":"assets/voice.flac","bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}
        budget = Budget()
        with tempfile.TemporaryDirectory() as directory:
            archive_members([{"original":spec}],Path(directory),Opener(),budget)
            self.assertEqual((Path(directory)/"assets/voice.flac").read_bytes(),data)
        self.assertLess(budget.used,len(source))
        self.assertLessEqual(budget.used,32_768)

if __name__ == "__main__":
    unittest.main()
