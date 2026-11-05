"""Apply/restore a pinned MLX-LM parser fix in one workspace's .venv only."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import stat
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
VERSION = '0.32.2'
ORIGINAL_SHA256 = 'd1029aae22ed475e9d7c09d29c37e74dca8df52b0fb046f000f2e3f9b83a6583'
PATCHED_SHA256 = '5921b9378739903e18133c5237ef8866bd922fb0fb3e0725c0372e7ad6752488'
MODULE = 'mlx_lm/tool_parsers/qwen3_coder.py'


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def guard_target(prefix, version, target, *, root=None):
    root = Path(root if root is not None else ROOT).resolve()
    expected = root / '.venv'
    if not expected.resolve().is_relative_to(root):
        raise ValueError('The workspace .venv resolves outside the selected root')
    if Path(prefix).resolve() != expected.resolve():
        raise ValueError(f'Run with {expected}/bin/python; refusing other Python environments')
    if version != VERSION:
        raise ValueError(f'Expected mlx-lm {VERSION}, found {version}; rebase the patch first')
    target = Path(target).resolve()
    if not target.is_relative_to(expected.resolve()) or target.parts[-3:] != tuple(MODULE.split('/')):
        raise ValueError(f'Refusing parser target outside the workspace venv: {target}')
    return target


def replace_bytes(path, content, expected_hash):
    if sha256(path.read_bytes()) != expected_hash:
        raise ValueError('Parser changed during preparation; refusing overwrite')
    mode = stat.S_IMODE(path.stat().st_mode)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def backup_paths(root):
    """Reject redirected backup directories and files before any access."""
    root = Path(root).resolve()
    directory = root
    for component in ('patches', f'mlx-lm-{VERSION}-qwen-union'):
        directory = directory / component
        if directory.is_symlink():
            raise ValueError('Refusing a symlink in the workspace backup path')
    directory = directory.resolve()
    if not directory.is_relative_to(root):
        raise ValueError('Backup directory resolves outside the selected root')
    children = [directory / name for name in ('qwen3_coder.py.original', 'manifest.json')]
    if any(path.is_symlink() or path.resolve().parent != directory for path in children):
        raise ValueError('Refusing a redirected workspace backup or manifest file')
    return directory, *children


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'install', 'restore'))
    parser.add_argument('--root', type=Path, default=ROOT,
                        help='Workspace containing .venv (default: repository root)')
    args = parser.parse_args()
    root = args.root.resolve()
    distribution = importlib.metadata.distribution('mlx-lm')
    target = guard_target(sys.prefix, distribution.version, distribution.locate_file(MODULE), root=root)
    before = target.read_bytes()
    before_hash = sha256(before)
    if before_hash not in (ORIGINAL_SHA256, PATCHED_SHA256):
        raise ValueError(f'Unrecognized parser SHA-256 {before_hash}; refusing to alter it')
    if args.action == 'check':
        print(json.dumps({'target': str(target), 'version': VERSION, 'sha256': before_hash,
                          'state': 'patched' if before_hash == PATCHED_SHA256 else 'original'}))
        return
    backup_dir, backup, manifest_path = backup_paths(root)
    if args.action == 'install':
        patched = Path(__file__).with_name('qwen3_coder.py').read_bytes()
        if sha256(patched) != PATCHED_SHA256:
            raise ValueError('Bundled patched source does not match its reviewed SHA-256')
        compile(patched, str(target), 'exec')
        if before_hash == PATCHED_SHA256:
            if not backup.is_file() or sha256(backup.read_bytes()) != ORIGINAL_SHA256:
                raise ValueError('Patched parser lacks the expected original backup')
            print(json.dumps({'state': 'already patched', 'target': str(target), 'sha256': before_hash}))
            return
        backup_dir.mkdir(parents=True, exist_ok=True)
        # Recheck after creation, before deriving paths used by reads/writes.
        backup_dir, backup, manifest_path = backup_paths(root)
        if backup.exists():
            if sha256(backup.read_bytes()) != ORIGINAL_SHA256:
                raise ValueError('Existing backup does not match the original; refusing overwrite')
        else:
            with backup.open('xb') as handle:
                handle.write(before)
        replacement, final_hash, state = patched, PATCHED_SHA256, 'patched'
    else:
        if before_hash == ORIGINAL_SHA256:
            print(json.dumps({'state': 'already original', 'target': str(target), 'sha256': before_hash}))
            return
        replacement = backup.read_bytes()
        if sha256(replacement) != ORIGINAL_SHA256:
            raise ValueError('Original backup hash mismatch; refusing restore')
        final_hash, state = ORIGINAL_SHA256, 'restored'
    replace_bytes(target, replacement, before_hash)
    if sha256(target.read_bytes()) != final_hash:
        raise ValueError('Post-write parser verification failed')
    manifest = {'state': state, 'version': VERSION, 'target': str(target),
                'backup': str(backup), 'before_sha256': before_hash, 'after_sha256': final_hash,
                'updated_at': datetime.now(timezone.utc).isoformat()}
    # Recheck before writing metadata as well; never follow a pre-existing link.
    _, _, manifest_path = backup_paths(root)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, importlib.metadata.PackageNotFoundError) as error:
        raise SystemExit(str(error))
