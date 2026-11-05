# Kumi production source seeds

This bundle documents nine real recording seeds from three independent recording groups and 30 original production scenarios. The recordings were acquired and checked locally. Audio binaries, article bodies, raw snapshots, private machine paths and model weights are excluded from the repository.

These scenarios are **pending Kumi mapping and execution**. This source collection contains zero verified Kumi training trajectories. A recording, a documented workflow or a valid scraper result does not establish that a Kumi edit succeeded.

| Recording group | Seeds | Recording license | Material |
| --- | ---: | --- | --- |
| Open Goldberg Variations, Kimiko Ishizaka | 3 | CC0 1.0 | Three piano performances; 30-second working excerpts |
| VSCO 2 Community Edition, Versilian Studios | 4 | CC0 1.0 | Bass drum, tambourine roll, sustained flute and pizzicato cello samples |
| LibriSpeech, reader kumarei | 2 | CC BY 4.0 | Unaccompanied read-speech clips from the original training split |

The original recordings and working WAV files total **34,051,666 bytes**. Actual audio source downloads, including the bounded speech-archive prefix, read 11,028,604 bytes during collection. These are small source seeds: the piano clips share one album/performer, the VSCO samples share one library, and the two speech clips share a reader and audiobook chapter. Keep each `source_group` together when defining training and evaluation splits.

## Files

- `source-catalog.json`: 15 source entries with URLs, hashes, extraction status, coverage limits and license distinctions.
- `reusable-assets.json`: nine locally acquired recording seeds and five research-corpus leads. Restricted, unavailable and license-unclear corpus leads remain metadata-only.
- `task-seeds.json`: 30 natural producer goals, clarification/recovery turns, fixture requirements and acceptance checks. No supplied numerical answer or verified tool trajectory is included.
- `acquisition-report.json`: source collection evidence and explicit limitations. Seventeen source/recording-license pages were acquired in total; 12 scraper results succeeded and five remained partial. Two supporting piano-license pages are reported separately from the 15 catalog entries.
- `asset-download-manifest.json`: all nine original/working file pairs, source URLs, expected hashes, upstream revision where available, recording licenses, attribution and derivation commands.
- `fetch_assets.py`: optional bounded downloader using the Python standard library. Installed FFmpeg is needed only to derive working WAVs.
- `downloader-live-check.json`: a successful live check of one direct percussion download and both bounded-archive speech downloads, including original and derived PCM hashes. The other six assets were verified during collection; this downloader check does not claim a fresh download of all nine.
- `bundle-integrity.json`: SHA256 hashes of the public bundle files.

All audio paths in manifests refer to a local cache. They do not mean that an audio file is committed to this repository. Source hashes identify privately retained page evidence; those article bodies are not distributed here.

## Reproduce a small subset

Python 3.10+ is sufficient. Running without selection lists the available asset IDs and performs no network request:

```sh
python fetch_assets.py
python fetch_assets.py --asset vsco-bass-drum --output audio-cache
python fetch_assets.py --asset librispeech-374-180298-0021 --asset librispeech-374-180298-0038 --output audio-cache
```

Use `--all` to select all nine originals. The downloader verifies each original's exact size and SHA256 before storing it. It restricts network requests to named HTTPS source hosts and enforces a 100 MB source-read budget. Existing files with different content are rejected.

The LibriSpeech URL identifies a large upstream archive. The downloader requests only its initial byte range, stops as soon as the named training clips are extracted, and aborts after 8,000,000 source bytes if necessary. It never saves or downloads the complete archive. If a server changes the archive layout so the clips are no longer in that prefix, this method fails explicitly; do not replace it with a full-corpus download.

To create the working stereo PCM16 / 44.1 kHz WAVs, install FFmpeg separately and add `--derive` using a fresh output directory:

```sh
python fetch_assets.py --asset vsco-flute --output audio-cache --derive
```

Reference WAVs were made with FFmpeg 9.0.2. The manifest includes both the collected whole-WAV SHA256 and the PCM-data SHA256. FFmpeg versions can change container metadata, so the tool reports whether the container matches while requiring the decoded PCM to match. A PCM mismatch requires review. The speech source is mono; its working WAV duplicates that signal into two channels. Resampling or decoding does not remove source codec artifacts or recording-room sound.

Run the offline checks with:

```sh
python -m unittest discover -s . -p 'test_fetch_assets.py'
```

## Recording provenance and reuse

Open Goldberg's official project describes the **recording and score** as public domain. The Internet Archive item explicitly identifies the Kimiko Ishizaka recording as CC0. Credit: J.S. Bach, Goldberg Variations BWV 988; Kimiko Ishizaka, piano; produced by Anne-Marie Sylvestre, Teldex Studio, Berlin, 2012. See [the project](https://opengoldbergvariations.org/) and [recording item](https://archive.org/details/The_Open_Goldberg_Variations-11823).

VSCO's publisher explicitly licenses the sample library CC0, independently of any paper. Credit: Versilian Studios; recordings by Sam Gossner and Simon Dalzell; sample cutting by Elan Hickler / Soundemote. Source WAVs are pinned to a repository revision and verified against both SHA256 and Git blob hashes. See [the publisher](https://versilian-studios.com/vsco-community/) and [repository license](https://github.com/sgossner/VSCO-2-CE/blob/master/LICENSE).

The [OpenSLR LibriSpeech release](https://www.openslr.org/12/) identifies its corpus license as CC BY 4.0. Credit: Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur, LibriSpeech, 2015; source LibriVox reader **kumarei**, *Camille*, Chapter 16. The manifest identifies the two original training-split archive members, the unchanged FLAC hashes and the WAV conversion. Retain the attribution and [CC BY 4.0 license link](https://creativecommons.org/licenses/by/4.0/) when sharing the selected recordings or derivatives.

The source catalog keeps paper, code and data licenses separate. RIME's symbolic plans have not been imported or translated into Kumi; the inspected MixAgent repository was empty; a released LP-Fx corpus/license was not established. No private chain-of-thought annotations were collected. Only permitted public instructions, short plans, actions and observable outcomes may be considered in later work.

To promote any scenario into training data, record actual supported Kumi operations, tool responses, before/after state, reproducible renders and independent outcome checks. Retain source-group provenance and reject unsupported, silent, failed or no-op trajectories.
