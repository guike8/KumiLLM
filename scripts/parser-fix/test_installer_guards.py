"""The installer must refuse unrelated environments, versions and targets."""
from pathlib import Path
from contextlib import ExitStack, redirect_stdout
import io
import json
import tempfile
import unittest
from unittest import mock

import install_parser_fix as installer
from install_parser_fix import guard_target, replace_bytes, sha256, ROOT, VERSION


class InstallerGuards(unittest.TestCase):
    def test_workspace_environment_only(self):
        target = ROOT / '.venv/lib/python3.12/site-packages/mlx_lm/tool_parsers/qwen3_coder.py'
        self.assertEqual(guard_target(ROOT / '.venv', VERSION, target), target)
        with self.assertRaises(ValueError):
            guard_target('/usr/local', VERSION, target)

    def test_selected_root_works_without_changing_default_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / '.venv/lib/python3.12/site-packages' / installer.MODULE
            target.parent.mkdir(parents=True)
            target.write_bytes(Path(__file__).with_name('qwen3_coder.original.py').read_bytes())
            distribution = mock.Mock(version=VERSION)
            distribution.locate_file.return_value = target
            output = io.StringIO()
            with mock.patch.object(installer.sys, 'prefix', str(root / '.venv')), \
                    mock.patch.object(installer.sys, 'argv', ['installer', 'check', '--root', str(root)]), \
                    mock.patch.object(installer.importlib.metadata, 'distribution', return_value=distribution), \
                    redirect_stdout(output):
                installer.main()
            self.assertEqual(json.loads(output.getvalue())['state'], 'original')
            self.assertFalse((root / 'patches').exists())

    def test_workspace_symlink_cannot_redirect_installation_outside_root(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, outside = base / 'workspace', base / 'unrelated-environment'
            root.mkdir()
            outside.mkdir()
            (root / '.venv').symlink_to(outside, target_is_directory=True)
            target = outside / 'lib/python3.12/site-packages' / installer.MODULE
            with self.assertRaisesRegex(ValueError, 'outside the selected root'):
                guard_target(root / '.venv', VERSION, target, root=root)

    def test_version_and_target(self):
        target = ROOT / '.venv/lib/python3.12/site-packages/mlx_lm/tool_parsers/qwen3_coder.py'
        with self.assertRaises(ValueError):
            guard_target(ROOT / '.venv', '0.33.0', target)
        with self.assertRaises(ValueError):
            guard_target(ROOT / '.venv', VERSION, '/tmp/mlx_lm/tool_parsers/qwen3_coder.py')
        with self.assertRaises(ValueError):
            guard_target(ROOT / '.venv', VERSION, ROOT / '.venv/other.py')

    def test_hash_guard_atomic_write_and_preserved_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'module.py'
            path.write_bytes(b'original')
            path.chmod(0o640)
            with self.assertRaises(ValueError):
                replace_bytes(path, b'wrong', sha256(b'not the current source'))
            self.assertEqual(path.read_bytes(), b'original')
            replace_bytes(path, b'patched', sha256(b'original'))
            self.assertEqual(path.read_bytes(), b'patched')
            self.assertEqual(path.stat().st_mode & 0o777, 0o640)
            self.assertEqual([p.name for p in path.parent.iterdir()], ['module.py'])

    def installer_context(self, root, target, action):
        distribution = mock.Mock(version=VERSION)
        distribution.locate_file.return_value = target
        stack = ExitStack()
        stack.enter_context(mock.patch.object(installer, 'ROOT', root))
        stack.enter_context(mock.patch.object(installer.sys, 'prefix', str(root / '.venv')))
        stack.enter_context(mock.patch.object(installer.sys, 'argv', ['installer', action]))
        stack.enter_context(mock.patch.object(installer.importlib.metadata, 'distribution', return_value=distribution))
        stack.enter_context(redirect_stdout(io.StringIO()))
        return stack

    def test_install_backup_idempotence_and_restore_in_disposable_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / '.venv/lib/python3.12/site-packages' / installer.MODULE
            target.parent.mkdir(parents=True)
            original = Path(__file__).with_name('qwen3_coder.original.py').read_bytes()
            target.write_bytes(original)
            with self.installer_context(root, target, 'install'):
                installer.main()
            self.assertEqual(sha256(target.read_bytes()), installer.PATCHED_SHA256)
            backup_dir = root / 'patches' / f'mlx-lm-{VERSION}-qwen-union'
            self.assertEqual((backup_dir / 'qwen3_coder.py.original').read_bytes(), original)
            self.assertEqual(json.loads((backup_dir / 'manifest.json').read_text())['state'], 'patched')
            with self.installer_context(root, target, 'install'):
                installer.main()
            with self.installer_context(root, target, 'restore'):
                installer.main()
            self.assertEqual(target.read_bytes(), original)
            self.assertEqual(json.loads((backup_dir / 'manifest.json').read_text())['state'], 'restored')

    def test_unknown_source_and_conflicting_backup_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / '.venv/lib/python3.12/site-packages' / installer.MODULE
            target.parent.mkdir(parents=True)
            target.write_bytes(b'unknown source')
            with self.installer_context(root, target, 'install'), self.assertRaises(ValueError):
                installer.main()
            self.assertEqual(target.read_bytes(), b'unknown source')
            original = Path(__file__).with_name('qwen3_coder.original.py').read_bytes()
            target.write_bytes(original)
            backup = root / 'patches' / f'mlx-lm-{VERSION}-qwen-union/qwen3_coder.py.original'
            backup.parent.mkdir(parents=True)
            backup.write_bytes(b'conflicting backup')
            with self.installer_context(root, target, 'install'), self.assertRaises(ValueError):
                installer.main()
            self.assertEqual(target.read_bytes(), original)
            self.assertEqual(backup.read_bytes(), b'conflicting backup')

    def test_backup_directory_symlinks_cannot_write_outside_workspace(self):
        for component in ('patches', 'version-directory'):
            for destination_exists in (False, True):
                with self.subTest(component=component, destination_exists=destination_exists), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory) / 'workspace'
                    outside = Path(directory) / 'outside'
                    target = root / '.venv/lib/python3.12/site-packages' / installer.MODULE
                    target.parent.mkdir(parents=True)
                    original = Path(__file__).with_name('qwen3_coder.original.py').read_bytes()
                    target.write_bytes(original)
                    if destination_exists:
                        outside.mkdir()
                    if component == 'patches':
                        link = root / 'patches'
                    else:
                        (root / 'patches').mkdir()
                        link = root / 'patches' / f'mlx-lm-{VERSION}-qwen-union'
                    link.symlink_to(outside, target_is_directory=True)
                    with self.installer_context(root, target, 'install'), self.assertRaisesRegex(ValueError, 'symlink'):
                        installer.main()
                    self.assertEqual(target.read_bytes(), original)
                    self.assertEqual(list(outside.iterdir()) if outside.exists() else [], [])
                    self.assertEqual(outside.exists(), destination_exists)

    def test_backup_and_manifest_file_symlinks_are_refused(self):
        for name in ('qwen3_coder.py.original', 'manifest.json'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / 'workspace'
                outside = Path(directory) / 'outside.txt'
                outside.write_bytes(b'preserve unrelated file')
                target = root / '.venv/lib/python3.12/site-packages' / installer.MODULE
                target.parent.mkdir(parents=True)
                original = Path(__file__).with_name('qwen3_coder.original.py').read_bytes()
                target.write_bytes(original)
                backup_dir = root / 'patches' / f'mlx-lm-{VERSION}-qwen-union'
                backup_dir.mkdir(parents=True)
                (backup_dir / name).symlink_to(outside)
                with self.installer_context(root, target, 'install'), self.assertRaisesRegex(ValueError, 'redirected'):
                    installer.main()
                self.assertEqual(outside.read_bytes(), b'preserve unrelated file')
                self.assertEqual(target.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
