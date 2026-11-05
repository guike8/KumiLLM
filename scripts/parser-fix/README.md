# Qwen3-Coder scalar-union parser fix

This compatibility patch targets **MLX-LM 0.32.2**. It decodes a JSON scalar
only when an `anyOf`, `oneOf`, or type-array schema unambiguously admits that
value and rules out the original string. Kumi's numeric `volumeDb` arguments
therefore retain their number type; `-inf`, ambiguous strings, invalid values,
and unknown schema assertions retain conservative behavior.

The full Apple copyright header and Qwen source link remain in both parser
copies. The modified file carries a change notice. See the repository's
`THIRD_PARTY_NOTICES.md` and `licenses/` for component licenses.

## Apply in an existing workspace

The selected workspace must contain a `.venv` with MLX-LM 0.32.2 installed.
Run the installer using that environment's Python. For this repository as
the workspace, from its root:

```sh
.venv/bin/python scripts/parser-fix/install_parser_fix.py check --root .
.venv/bin/python scripts/parser-fix/install_parser_fix.py install --root .
KUMI_TEST_INSTALLED_PARSER=1 .venv/bin/python -m unittest discover \
  -s scripts/parser-fix -p 'test_parser_fix.py' -v
```

For another workspace, pass its path to `--root` and use its `.venv/bin/python`.
The installer refuses another Python prefix, a `.venv` resolving outside the
selected root, an unexpected module path, a different package version, or
unknown source bytes. It preserves an original backup and supports idempotent
installation and guarded restoration:

```sh
.venv/bin/python scripts/parser-fix/install_parser_fix.py restore --root .
```

Restart the model server after installing or restoring so it imports the
selected parser. The installer does not contact or restart a server.

## Hashes and checks

- Original MLX-LM parser: `d1029aae22ed475e9d7c09d29c37e74dca8df52b0fb046f000f2e3f9b83a6583`
- Public modified parser: `5921b9378739903e18133c5237ef8866bd922fb0fb3e0725c0372e7ad6752488`

The original collection environment used the same implementation without the
added public change-notice comment. Its reviewed parser hash was
`996f6ccd6530349df74d6e195dee343ebcdfd5d2cce7fca24c4d60d797e4be45`.
The public file differs from that version only in the copyright/change-notice
header. The original collection environment has not been overwritten by this
public packaging change.

Run the portable regression suite without installing the parser:

```sh
python3 -m unittest discover -s scripts/parser-fix -p 'test_*.py' -v
```

The suite tests conversion and full XML parsing, and uses disposable temporary
environments to test installation, backup, restore, path guards, and source
hash checks. It does not change the running Python environment.

This fix does not repair model output that omits its opening tool-call marker;
that remains a separate model serialization failure.
