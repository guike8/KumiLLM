# Third-party notices

Original KumiLLM contributions are under the MIT license in [LICENSE](LICENSE), copyright (c) 2026 KumiLLM contributors. Upstream material retains the component notices and terms below. The root license does not replace those terms.

## Kumi instructions, schemas and source changes

- Source: <https://github.com/user1303836/kumi>.
- Revision used: `2ff07a5be515e0a267ea86f7c3d56fa579b607ea` (1.24.68).
- Copyright (c) 2026 Ableton MCP Beyond contributors.
- License: MIT; the complete upstream text and trademark notice are preserved in [licenses/Kumi-MIT.txt](licenses/Kumi-MIT.txt).

The dataset reproduces Kumi's instructions and tool definitions. The collection and simulator patch also contains Kumi source context. These are upstream MIT material, alongside the new synthetic goals, scripted-teacher actions, verification records and collection code. Preserve the Kumi copyright and permission notice when redistributing copies or substantial portions of that material, including the embedded instructions and schemas.

KumiLLM changes add a read-only mixer-display simulation path, focused regression checks, a bounded scripted teacher, complete request/response capture and strict episode admission. The source patch identifies the changed code; this repository does not claim to be an upstream Kumi release.

## MLX-LM Qwen3-Coder parser

- Source project: <https://github.com/ml-explore/mlx-lm>.
- Version targeted: `v0.32.2`.
- MLX-LM MIT license copyright: Copyright © 2023 Apple Inc.
- Copied parser file header: Copyright © 2025 Apple Inc.
- License text: [licenses/MLX-LM-MIT.txt](licenses/MLX-LM-MIT.txt), copied from the [v0.32.2 license](https://github.com/ml-explore/mlx-lm/blob/v0.32.2/LICENSE).

The parser's original Apple header and its `Modified from` source link are retained. KumiLLM changes the scalar schema matcher and XML argument conversion for `anyOf`, `oneOf` and type-array unions: unambiguously numeric values retain their number type, ambiguous string branches and `-inf` stay text, and invalid or uncertain conversions remain text for downstream validation. Regression checks and a reversible, version/hash-guarded installer accompany this change.

The original installed parser SHA-256 was `d1029aae22ed475e9d7c09d29c37e74dca8df52b0fb046f000f2e3f9b83a6583`. Public source annotations or packaging changes may have different file hashes from the installed Studio artifact; installation checks and provenance identify the exact artifact they target.

### Qwen/vLLM parser ancestry

The MLX-LM parser identifies this earlier source:

<https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/blob/main/qwen3coder_tool_parser.py>

That source declares:

```text
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright contributors to the vLLM project
```

The full Apache License, Version 2.0 is preserved in [licenses/QWEN-VLLM-APACHE-2.0.txt](licenses/QWEN-VLLM-APACHE-2.0.txt). The ancestry and scalar-union modification notice are also preserved in [licenses/QWEN-VLLM-NOTICE.txt](licenses/QWEN-VLLM-NOTICE.txt). The Apple adaptation's source comment does not record a Qwen source revision; this notice does not infer one.

## Model and tokenizer reference

- Reference: <https://huggingface.co/mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit>.
- Pinned revision: `6e302ea604ad9ab206367e2c501d1571023e7b6d`.

Model weights and tokenizer assets are not bundled. Users obtain them separately under their applicable terms. The released demonstrations use a scripted teacher; they are not outputs sampled from this model.

## Distribution scope

This release contains synthetic Kumi execution data, collection and verification tools, patches, documentation and provenance. It contains no recorded user music, audio, model weights, tokenizer assets, scraped papers or documentation extracts, compiled Kumi binaries, or Willington runtime files. Names of external products in Kumi's tool descriptions identify capabilities; their mention does not include those products or grant rights to them.

Ableton Live is a trademark of Ableton AG. KumiLLM is not affiliated with, endorsed by, or sponsored by Ableton AG. Upstream projects retain their names and trademarks.

Exact license-source URLs and file hashes are recorded in [licenses/SOURCES.json](licenses/SOURCES.json).
