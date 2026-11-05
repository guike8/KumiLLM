#!/usr/bin/env python3
"""Independent FFmpeg R128 and PCM checks; no audio modifications."""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import wave


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise SystemExit('Existing ffmpeg installation required')
    results = []
    for path in sorted(args.assets.glob('open-goldberg-*-excerpt-30s.wav')):
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        command = [ffmpeg, '-hide_banner', '-nostats', '-i', str(path),
                   '-af', 'ebur128=peak=true', '-f', 'null', '-']
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        summary = result.stderr[result.stderr.rfind('Summary:'):].split('[out#')[0].strip()
        with wave.open(str(path)) as wav:
            if wav.getsampwidth() != 2 or wav.getcomptype() != 'NONE':
                raise ValueError('Expected PCM16 source')
            samples = array.array('h', wav.readframes(wav.getnframes()))
            if sys.byteorder != 'little':
                samples.byteswap()
        peak = max(abs(x) for x in samples) / 32768
        rms = math.sqrt(sum(x*x for x in samples) / len(samples)) / 32768
        after = hashlib.sha256(path.read_bytes()).hexdigest()
        if before != after:
            raise ValueError('Source changed during check')
        results.append({'file': path.name, 'sha256': before, 'source_unchanged': True,
                        'ffmpeg_summary': summary, 'independent_sample_peak_dbfs': 20*math.log10(peak),
                        'independent_rms_dbfs': 20*math.log10(rms),
                        'independent_crest_db': 20*math.log10(peak/rms),
                        'full_scale_sample_count': sum(abs(x) >= 32767 for x in samples),
                        'sample_count': len(samples), 'non_silent': any(samples)})
    if len(results) != 3:
        raise ValueError('Exactly three reviewed piano excerpts expected')
    args.output.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
