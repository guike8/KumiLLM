import os
from pathlib import Path
import stat
import tempfile
import unittest

import download_official as downloader


class OutputProtection(unittest.TestCase):
    def test_private_link_is_refused_without_changing_target_permissions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            victim = root / 'unrelated'
            victim.mkdir(mode=0o755)
            before = stat.S_IMODE(victim.stat().st_mode)
            (root / 'private').symlink_to(victim, target_is_directory=True)
            with self.assertRaises(OSError):
                with downloader.output_directories(root):
                    self.fail('Linked directory accepted')
            self.assertEqual(stat.S_IMODE(victim.stat().st_mode), before)
            self.assertEqual(list(victim.iterdir()), [])

    def test_receipt_link_does_not_overwrite_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            victim = root / 'unrelated.txt'
            victim.write_text('preserve me')
            (root / 'download-receipt.json').symlink_to(victim)
            with self.assertRaises(RuntimeError):
                with downloader.output_directories(root):
                    self.fail('Existing receipt accepted')
            self.assertEqual(victim.read_text(), 'preserve me')
            self.assertFalse((root / 'private').exists())

    def test_directory_handles_stay_pinned_after_path_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            victim = root / 'unrelated'
            victim.mkdir()
            with downloader.output_directories(root) as (_, private_fd):
                (root / 'private').rename(root / 'original-private')
                (root / 'private').symlink_to(victim, target_is_directory=True)
                fd = os.open('part', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=private_fd)
                os.write(fd, b'example')
                os.close(fd)
            self.assertEqual(list(victim.iterdir()), [])
            self.assertEqual((root / 'original-private/part').read_bytes(), b'example')


if __name__ == '__main__':
    unittest.main()
