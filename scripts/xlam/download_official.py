#!/usr/bin/env python3
"""Fetch the pinned, authorized xLAM release without persisting credentials."""
import hashlib
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit, urljoin
from urllib.request import Request, build_opener, HTTPRedirectHandler

REPO = 'Salesforce/xlam-function-calling-60k'
REVISION = '26d14ebfe18b1f7b524bd39b404b50af5dc97866'
FILENAME = 'xlam_function_calling_60k.json'
EXPECTED_BYTES = 96116570
ROOT = Path(__file__).resolve().parent


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None


def allowed(url):
    value = urlsplit(url)
    host = value.hostname or ''
    if value.scheme != 'https' or value.username or value.password or value.port not in (None, 443):
        raise RuntimeError('Unsafe source URL')
    if not (host == 'huggingface.co' or host.endswith('.huggingface.co') or host.endswith('.hf.co')):
        raise RuntimeError('Redirect outside Hugging Face artifact hosts')
    return host


def open_source(opener, initial, token):
    url = initial
    for _ in range(6):
        host = allowed(url)
        headers = {'User-Agent': 'KumiLLM-xLAM-subset/1', 'Accept-Encoding': 'identity'}
        # The Hugging Face token must never reach the signed artifact host.
        if host == 'huggingface.co':
            headers['Authorization'] = 'Bearer ' + token
        try:
            return opener.open(Request(url, headers=headers), timeout=60)
        except HTTPError as error:
            if error.code not in (301, 302, 303, 307, 308):
                raise RuntimeError('Official download returned HTTP %s (%s)' %
                                   (error.code, error.headers.get('X-Error-Code', 'no error code'))) from None
            url = urljoin(url, error.headers['Location'])
    raise RuntimeError('Too many artifact redirects')


def absent(directory_fd, name):
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise RuntimeError('Output already exists: ' + name)


@contextmanager
def output_directories(root):
    """Pin directory handles; never follow output links or overwrite receipts."""
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    private_fd = None
    try:
        absent(root_fd, 'download-receipt.json')
        try:
            os.mkdir('private', mode=0o700, dir_fd=root_fd)
        except FileExistsError:
            pass
        private_fd = os.open('private', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                             dir_fd=root_fd)
        os.fchmod(private_fd, 0o700)
        absent(private_fd, FILENAME)
        yield root_fd, private_fd
    finally:
        if private_fd is not None:
            os.close(private_fd)
        os.close(root_fd)


def main():
    with output_directories(ROOT) as handles:
        download(*handles)


def download(root_fd, private_fd):
    token = os.environ.get('HF_TOKEN')
    if not token:
        token = (Path.home() / '.cache/huggingface/token').read_text().strip()
    if not token:
        raise RuntimeError('Existing authorized Hugging Face credential required')
    opener = build_opener(NoRedirect())
    api = 'https://huggingface.co/api/datasets/%s/tree/%s?recursive=false&expand=false' % (REPO, REVISION)
    with open_source(opener, api, token) as response:
        inventory = json.loads(response.read(1024 * 1024))
    source = next(row for row in inventory if row.get('path') == FILENAME)
    assert source['size'] == EXPECTED_BYTES
    upstream = source.get('lfs', {}).get('oid')
    expected_sha = upstream if isinstance(upstream, str) and re.fullmatch('[0-9a-f]{64}', upstream) else None
    url = 'https://huggingface.co/datasets/%s/resolve/%s/%s' % (REPO, REVISION, FILENAME)
    temporary = FILENAME + '.part'
    digest = hashlib.sha256()
    count = 0
    started = time.time()
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=private_fd)
    with os.fdopen(descriptor, 'wb') as output, open_source(opener, url, token) as response:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            count += len(chunk)
            if count > EXPECTED_BYTES:
                raise RuntimeError('Official release exceeds expected size')
            digest.update(chunk)
            output.write(chunk)
        output.flush()
        os.fsync(output.fileno())
    actual = digest.hexdigest()
    if count != EXPECTED_BYTES or expected_sha and actual != expected_sha:
        raise RuntimeError('Official file size or upstream hash mismatch')
    os.link(temporary, FILENAME, src_dir_fd=private_fd, dst_dir_fd=private_fd,
            follow_symlinks=False)
    os.unlink(temporary, dir_fd=private_fd)
    receipt = {'schema': 'kumillm-xlam-official-download/1', 'dataset_id': REPO,
               'revision': REVISION, 'file': FILENAME, 'source_url': url,
               'bytes': count, 'sha256': actual, 'upstream_lfs_sha256': expected_sha,
               'upstream_lfs_hash_available': expected_sha is not None,
               'upstream_lfs_hash_verified': actual == expected_sha if expected_sha else None,
               'download_seconds': time.time() - started, 'downloaded_at_unix': time.time(),
               'credential_or_signed_url_stored': False, 'license_metadata': 'CC-BY-4.0',
               'access_request_authorized_by_user': True}
    descriptor = os.open('download-receipt.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=root_fd)
    with os.fdopen(descriptor, 'w') as output:
        output.write(json.dumps(receipt, indent=2) + '\n')
        output.flush()
        os.fsync(output.fileno())
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
