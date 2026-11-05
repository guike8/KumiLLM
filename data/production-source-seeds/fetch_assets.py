"""Fetch only named, hash-verified seed recordings. Python standard library only.

FFmpeg is optional and required only with --derive. Large speech archives are
never saved; a bounded prefix stream is stopped after the requested members.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import tempfile
from urllib.parse import urlparse
from urllib.request import build_opener, HTTPRedirectHandler, ProxyHandler, Request
import wave

BASE = Path(__file__).resolve().parent
MAX_DOWNLOAD = 100_000_000
HOSTS = {"archive.org", "raw.githubusercontent.com", "openslr.trmal.net", "openslr.elda.org", "openslr.magicdatatech.com", "www.openslr.org", "openslr.org"}

def check_url(url):
    p = urlparse(url)
    host = p.hostname or ""
    if p.scheme != "https" or p.username or p.password or p.port not in (None,443) or not (host in HOSTS or host.endswith(".archive.org")):
        raise ValueError("Source URL outside the fixed HTTPS host allowlist")

class CheckedRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)

class Budget:
    def __init__(self, limit=MAX_DOWNLOAD):
        self.limit, self.used = limit, 0
    def add(self, count):
        if self.used + count > self.limit:
            raise ValueError("Total source download limit exceeded")
        self.used += count

class LimitedReader:
    def __init__(self, response, limit, budget):
        self.response, self.limit, self.budget, self.used = response, limit, budget, 0
    def read(self, size):
        if size < 0:
            raise ValueError("Unbounded reads are forbidden")
        if self.used >= self.limit:
            raise ValueError("Per-source stream limit exceeded")
        data = self.response.read(min(size, self.limit-self.used))
        self.budget.add(len(data))
        self.used += len(data)
        return data

def safe_path(root, relative):
    p = PurePosixPath(relative)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise ValueError("Unsafe asset path")
    target = root.joinpath(*p.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Asset path leaves output directory")
    return target

def verify(payload, spec):
    if len(payload) != spec["bytes"] or hashlib.sha256(payload).hexdigest() != spec["sha256"]:
        raise ValueError("Original audio size/hash mismatch")

def save(root, spec, payload):
    verify(payload, spec)
    target = safe_path(root,spec["path"])
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != spec["sha256"]:
            raise ValueError("Existing output has different content; choose another output directory")
        return target
    with tempfile.NamedTemporaryFile(dir=target.parent,delete=False) as tmp:
        temporary = Path(tmp.name)
        tmp.write(payload)
    temporary.replace(target)
    return target

def open_source(opener,url,headers=None):
    check_url(url)
    return opener.open(Request(url,headers={"User-Agent":"KumiLLM-source-seeds/1.0","Accept-Encoding":"identity",**(headers or {})}),timeout=60)

def direct(asset,root,opener,budget):
    spec = asset["original"]
    with open_source(opener,spec["url"]) as response:
        reader = LimitedReader(response,spec["bytes"]+1,budget)
        chunks = []
        while True:
            chunk = reader.read(65_536)
            if not chunk:
                break
            chunks.append(chunk)
    return save(root,spec,b"".join(chunks))

def archive_members(assets,root,opener,budget):
    first = assets[0]["original"]
    wanted = {a["original"]["archive_member"]:a for a in assets}
    if any(a["original"]["url"] != first["url"] for a in assets):
        raise ValueError("Archive batch mixes source URLs")
    limit = min(a["original"]["archive_stream_limit_bytes"] for a in assets)
    with open_source(opener,first["url"],{"Range":f"bytes=0-{limit-1}"}) as response:
        reader = LimitedReader(response,limit,budget)
        with tarfile.open(fileobj=reader,mode="r|gz") as archive:
            for member in archive:
                if member.size > 20_000_000 or archive.offset > 50_000_000:
                    raise ValueError("Expanded archive scan bound exceeded")
                if member.name not in wanted:
                    continue
                asset = wanted[member.name]
                spec = asset["original"]
                if not member.isfile() or member.size != spec["bytes"]:
                    raise ValueError("Unexpected archive member type/size")
                save(root,spec,archive.extractfile(member).read())
                del wanted[member.name]
                if not wanted:
                    break
    if wanted:
        raise ValueError("Requested speech members unavailable within the bounded prefix")

def derive(asset,root):
    original = safe_path(root,asset["original"]["path"])
    working = safe_path(root,asset["working"]["path"])
    working.parent.mkdir(parents=True,exist_ok=True)
    if working.exists():
        raise ValueError("Working output already exists; choose another output directory")
    args = [str(original) if a=="{original}" else str(working) if a=="{working}" else a for a in asset["working"]["derive_argv"]]
    subprocess.run(args,check=True)
    with wave.open(str(working),"rb") as audio:
        if (audio.getnchannels(),audio.getsampwidth(),audio.getframerate()) != (2,2,44100):
            raise ValueError("Unexpected working PCM format")
        pcm = audio.readframes(audio.getnframes())
    if hashlib.sha256(pcm).hexdigest() != asset["working"]["pcm_sha256"]:
        raise ValueError("Derived PCM differs from the reference; decoder/version requires review")
    return hashlib.sha256(working.read_bytes()).hexdigest() == asset["working"]["sha256"]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset",action="append",default=[],help="Manifest asset ID; repeat to select several")
    parser.add_argument("--all",action="store_true",help="Select all nine seed recordings")
    parser.add_argument("--output",type=Path,default=BASE/"audio-cache")
    parser.add_argument("--derive",action="store_true",help="Also make working WAV files using installed ffmpeg")
    args = parser.parse_args()
    manifest = json.loads((BASE/"asset-download-manifest.json").read_text())
    assets = manifest["assets"]
    known = {a["id"] for a in assets}
    if set(args.asset)-known:
        parser.error("Unknown asset ID")
    selected = assets if args.all else [a for a in assets if a["id"] in args.asset]
    if not selected:
        print(json.dumps({"available_assets":sorted(known),"network_requests":0}))
        return
    opener = build_opener(ProxyHandler({}),CheckedRedirects())
    budget = Budget(min(MAX_DOWNLOAD,manifest["download_budget_bytes"]))
    archived = []
    for asset in selected:
        if asset["original"]["retrieval"] == "bounded_tar_gz_member":
            archived.append(asset)
        else:
            direct(asset,args.output,opener,budget)
    if archived:
        archive_members(archived,args.output,opener,budget)
    results = []
    for asset in selected:
        match = derive(asset,args.output) if args.derive else None
        results.append({"id":asset["id"],"original_hash_verified":True,"pcm_hash_verified":True if args.derive else None,"working_container_hash_matches_reference":match})
    print(json.dumps({"source_bytes_read":budget.used,"assets":results}))

if __name__ == "__main__":
    main()
