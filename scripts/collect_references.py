#!/usr/bin/env python3
"""Bounded official-source collection with the deployed Evidence AI scraper."""
import argparse
import asyncio
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
from urllib.parse import urlsplit, urlunsplit

from ai_scraper.config import Settings
from ai_scraper.pipeline import Pipeline
from ai_scraper.policy import URLPolicy
from ai_scraper.retrieval.direct import DirectHTTPRetriever
from ai_scraper.schemas import ExtractionRequest
from ai_scraper.storage import atomic_json

REQUIRED = {
    'song': 'lom_song', 'track': 'lom_track', 'clip': 'lom_clip',
    'clipslot': 'lom_clipslot', 'device': 'lom_device',
    'deviceparameter': 'lom_deviceparameter', 'mixerdevice': 'lom_mixerdevice',
}
OPTIONAL = {'songview': 'lom_songview', 'trackview': 'lom_trackview',
            'chain': 'lom_chain', 'drumpad': 'lom_drumpad'}


async def collect(destination):
    destination.mkdir(parents=False, exist_ok=False)
    (destination / 'results').mkdir()
    settings = Settings(state_dir=destination / 'state', retrieval_backend='direct',
                        allowed_domains=['docs.cycling74.com'], wait_seconds=90.0,
                        max_response_bytes=5_000_000, max_snapshot_bytes=20_000_000,
                        brightdata_api_key=None, hf_token=None, hf_provider=None, hf_model=None,
                        local_api_key=None, local_model=None)
    policy = URLPolicy(settings.allowed_domains)
    retriever = DirectHTTPRetriever(policy, max_response_bytes=settings.max_response_bytes,
                                    timeout=60.0, max_redirects=5, max_attempts=2)
    pipeline = Pipeline(settings, retriever=retriever, inference=None, renderer=None, policy=policy)
    assert pipeline.inference is None
    attempts, sources, links, requested = [], [], {}, set()

    async def fetch(source_id, url):
        if len(attempts) >= 12:
            raise ValueError('Bounded twelve-URL limit reached')
        requested.add(url)
        result = await pipeline.extract_url(ExtractionRequest(url=url, schema_id='page_v1', freshness_seconds=3600))
        document = result.model_dump(mode='json')
        assert result.usage.inference_attempts == 0
        provenance = document.get('provenance') or {}
        assert provenance.get('provider') is None and provenance.get('model') is None
        path = destination / 'results' / (source_id + '.json')
        atomic_json(path, document)
        data = document.get('data') or {}
        for link in data.get('links') or []:
            parsed = urlsplit(link.get('url', ''))
            if parsed.scheme != 'https' or parsed.hostname != 'docs.cycling74.com' or not parsed.path.startswith('/apiref/lom/'):
                continue
            canonical = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, '', ''))
            slug = re.sub('[^a-z0-9]', '', parsed.path.rstrip('/').split('/')[-1].lower())
            links.setdefault(slug, canonical)
        evidence = document.get('evidence') or {}
        text = data.get('text') or ''
        segments = text if isinstance(text, list) else [text]
        row = {'id': source_id, 'url': url, 'final_url': provenance.get('final_url'),
               'status': document['status'], 'source_hash': provenance.get('source_hash'),
               'retrieved_at': provenance.get('retrieved_at'), 'title': data.get('title'),
               'extracted_characters': len('\n'.join(segments)), 'text_segments': len(segments), 'evidence_fields': len(evidence),
               'result_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'usage': document.get('usage'), 'retrieval_method': provenance.get('retrieval_method'),
               'retention': 'Private reference snapshot; source article text is excluded from public training data.'}
        attempts.append(row)
        if row['status'] == 'success' and re.fullmatch('[0-9a-f]{64}', row['source_hash'] or ''):
            sources.append(row)
        atomic_json(destination / 'attempts.json', attempts)
        print(json.dumps({'completed': len(attempts), 'id': source_id, 'status': row['status'],
                          'characters': row['extracted_characters'], 'evidence_fields': row['evidence_fields']}), flush=True)
        await asyncio.sleep(1)

    try:
        await fetch('lom_index', 'https://docs.cycling74.com/apiref/lom/')
        if 'deviceparameter' not in links:
            await fetch('lom_deviceparameter', 'https://docs.cycling74.com/apiref/lom/deviceparameter/')
        for slug, source_id in list(REQUIRED.items()) + list(OPTIONAL.items()):
            if any(row['id'] == source_id for row in attempts):
                continue
            url = links.get(slug)
            if not url or url in requested:
                continue
            await fetch(source_id, url)
        successful = {row['id'] for row in sources}
        missing = sorted(set(REQUIRED.values()) - successful)
        manifest = {'format_version': 1, 'batch': destination.name,
                    'tool': 'Evidence AI Scraper', 'tool_version': version('evidence-ai-scraper'),
                    'execution_host': 'mac_studio', 'scope': 'official_reference_evidence_only',
                    'maximum_urls': 12, 'retrieval_concurrency': 1, 'inference_enabled': False,
                    'inference_attempts': sum(row['usage']['inference_attempts'] for row in attempts),
                    'sources': sources, 'attempts': attempts, 'missing_required_source_ids': missing,
                    'source_article_text_included_in_training': False,
                    'source_license_granted_by_scraper': False,
                    'readiness': 'ready_for_original_task_generation' if not missing else 'missing_required_references'}
        atomic_json(destination / 'source_manifest.json', manifest)
        print(json.dumps({'source_manifest': str(destination / 'source_manifest.json'),
                          'successful_sources': len(sources), 'attempted_urls': len(attempts),
                          'missing_required_source_ids': missing, 'inference_attempts': 0}), flush=True)
        return 0 if not missing else 2
    finally:
        await pipeline.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    raise SystemExit(asyncio.run(collect(parser.parse_args().output.resolve())))
