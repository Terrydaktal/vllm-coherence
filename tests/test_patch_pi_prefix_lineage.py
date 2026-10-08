from __future__ import annotations

import gzip
import json
import os
import runpy
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/pi_0842_prefix_lineage_provider.js.gz"
RELATIVE = "node_modules/@earendil-works/pi-ai/dist/api/openai-completions.js"


def patched_runtime(tmp_path):
    api = runpy.run_path(str(ROOT / "scripts/patch-pi-tool-call-integrity"))
    target = tmp_path / RELATIVE
    target.parent.mkdir(parents=True)
    target.write_bytes(gzip.decompress(FIXTURE.read_bytes()))
    target.chmod(0o600)
    prior = len(api["PATCHES"]) - len(api["LINEAGE_PATCHES"])
    assert api["classify_chain"](target.read_bytes(), api["PATCHES"]) == prior
    api["run"](tmp_path, apply=True)
    return api, target


def test_pinned_provider_installs_all_lineage_hooks_and_helper_idempotently(tmp_path):
    api, target = patched_runtime(tmp_path)
    api["run"](tmp_path, apply=False)
    api["run"](tmp_path, apply=True)
    helper = target.with_name("qwen-prefix-lineage.mjs")
    assert helper.read_bytes() == (ROOT / "integrations/pi/qwen-prefix-lineage.mjs").read_bytes()
    text = target.read_text()
    assert text.index("createPrefixLineage({ context") < text.index("let params = buildParams")
    assert text.index("prefixLineage?.converted(params)") < text.index("await options?.onPayload")
    assert text.index("await options?.onPayload") < text.index("prefixLineage?.wire(params)")
    assert "prefixLineage?.fetch ?? transport?.fetch" in text
    assert "prefixLineage?.responseId(chunk.id)" in text
    assert "prefixLineage?.assembled(output, true)" in text
    assert "prefixLineage?.assembled(output, false)" in text


def test_usage_authentication_reverses_lineage_hooks_before_its_own_checks(tmp_path):
    _, target = patched_runtime(tmp_path)
    usage = runpy.run_path(str(ROOT / "scripts/patch-pi-streaming-usage"))
    assert usage["classify"](target.read_bytes(), usage["PATCHES"][0]) == "patched"


def test_missing_changed_or_symlink_helper_is_not_a_qualified_runtime(tmp_path):
    api, target = patched_runtime(tmp_path)
    helper = target.with_name("qwen-prefix-lineage.mjs")
    helper.write_text("unexpected implementation\n")
    with pytest.raises(api["PatchError"], match="absent or outdated"):
        api["run"](tmp_path, apply=False)
    api["run"](tmp_path, apply=True)
    helper.unlink()
    helper.symlink_to(ROOT / "integrations/pi/qwen-prefix-lineage.mjs")
    with pytest.raises(api["PatchError"], match="regular file"):
        api["run"](tmp_path, apply=True)


def test_real_patched_provider_records_hooks_after_payload_callbacks(tmp_path):
    """Execute the actual pinned provider source with SDK/network CPU stubs."""
    _, target = patched_runtime(tmp_path)
    driver = ROOT / "tests/pi_prefix_lineage_provider_driver.mjs"
    result = subprocess.run(
        ["node", "--experimental-vm-modules", str(driver), str(target)],
        capture_output=True, text=True, timeout=20, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "provider boundaries qualified"


@pytest.mark.parametrize("existing_mode", [None, 0o755])
def test_default_recorder_preserves_only_numeric_comparisons_in_private_files(tmp_path, existing_mode):
    directory = tmp_path / ".local/state/qwen-r9700/diagnostics"
    if existing_mode is not None:
        directory.mkdir(parents=True, mode=existing_mode)
        directory.chmod(existing_mode)
    module = ROOT / "integrations/pi/qwen-prefix-lineage.mjs"
    source = f"""
import {{ createPrefixLineage, prefixLineageHealth }} from {json.dumps(module.as_uri())};
const payload = {{ messages: [{{ role: 'user', content: 'private-user-text' }}],
  kv_transfer_params: {{ qwen_chat: {{ id: 'a'.repeat(64), generation: 'b'.repeat(64) }} }} }};
const observer = createPrefixLineage({{ context: {{ messages: payload.messages }}, fetch: () => null }});
observer.converted(payload); observer.wire(payload);
observer.fetch('http://127.0.0.1/v1/chat/completions', {{ body: JSON.stringify(payload) }});
observer.responseId('chatcmpl-synthetic');
observer.assembled({{ role: 'assistant', content: [{{ type: 'text', text: 'private-answer-text' }}] }});
for (let i = 0; prefixLineageHealth().pending && i < 100; i++) await new Promise(r => setTimeout(r, 5));
console.log(JSON.stringify(prefixLineageHealth()));
"""
    result = subprocess.run(["node", "--input-type=module", "-e", source], env={**os.environ, "HOME": str(tmp_path)},
                            capture_output=True, text=True, timeout=10, check=True)
    health = json.loads(result.stdout)
    assert health["written"] == 6 and health["errors"] == 0
    assert directory.stat().st_mode & 0o777 == 0o700
    (path,) = directory.glob("*.jsonl")
    assert path.stat().st_mode & 0o777 == 0o600
    text = path.read_text()
    assert "private-user-text" not in text and "private-answer-text" not in text
    assert "digest" not in text and "token_ids" not in text
    assert json.loads(text.splitlines()[-1])["coverage_complete"] is True


def test_default_recorder_rejects_symlink_directory_without_affecting_request(tmp_path):
    parent = tmp_path / ".local/state/qwen-r9700"
    parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o755)
    outside.chmod(0o755)
    (parent / "diagnostics").symlink_to(outside, target_is_directory=True)
    module = ROOT / "integrations/pi/qwen-prefix-lineage.mjs"
    source = f"""
import {{ createPrefixLineage, prefixLineageHealth }} from {json.dumps(module.as_uri())};
createPrefixLineage({{ context: {{ messages: [] }} }});
for (let i = 0; prefixLineageHealth().pending && i < 100; i++) await new Promise(r => setTimeout(r, 5));
console.log(JSON.stringify(prefixLineageHealth()));
"""
    result = subprocess.run(["node", "--input-type=module", "-e", source], env={**os.environ, "HOME": str(tmp_path)},
                            capture_output=True, text=True, timeout=10, check=True)
    assert json.loads(result.stdout)["errors"] > 0
    assert list(outside.iterdir()) == []
    assert outside.stat().st_mode & 0o777 == 0o755


@pytest.mark.parametrize("unsafe_kind", ["public_file", "hardlink", "symlink", "fifo"])
def test_default_recorder_rejects_unsafe_existing_log_without_writing(tmp_path, unsafe_kind):
    module = ROOT / "integrations/pi/qwen-prefix-lineage.mjs"
    source = f"""
import {{ createPrefixLineage, prefixLineageHealth }} from {json.dumps(module.as_uri())};
import {{ chmod, link, readFile, readdir, symlink, unlink, writeFile }} from 'node:fs/promises';
import {{ execFileSync }} from 'node:child_process';
import {{ join }} from 'node:path';
const directory = join(process.env.HOME, '.local/state/qwen-r9700/diagnostics');
createPrefixLineage({{ context: {{ messages: [] }} }});
for (let i = 0; prefixLineageHealth().written < 1 && i < 100; i++) await new Promise(r => setTimeout(r, 5));
const path = join(directory, (await readdir(directory)).find(name => name.endsWith('.jsonl')));
const outside = join(process.env.HOME, 'untouched');
await writeFile(outside, 'unchanged', {{ mode: 0o600 }});
const kind = {json.dumps(unsafe_kind)};
if (kind === 'public_file') await chmod(path, 0o644);
else {{
  await unlink(path);
  if (kind === 'hardlink') await link(outside, path);
  else if (kind === 'symlink') await symlink(outside, path);
  else execFileSync('mkfifo', [path]);
}}
const before = kind === 'public_file' ? await readFile(path, 'utf8') : null;
createPrefixLineage({{ context: {{ messages: [] }} }});
for (let i = 0; !prefixLineageHealth().errors && i < 100; i++) await new Promise(r => setTimeout(r, 5));
if ((await readFile(outside, 'utf8')) !== 'unchanged') throw new Error('unsafe target was written');
if (before !== null && (await readFile(path, 'utf8')) !== before) throw new Error('unsafe file was appended');
console.log(JSON.stringify(prefixLineageHealth()));
"""
    result = subprocess.run(["node", "--input-type=module", "-e", source], env={**os.environ, "HOME": str(tmp_path)},
                            capture_output=True, text=True, timeout=10, check=True)
    health = json.loads(result.stdout)
    assert health["written"] == 1 and health["errors"] == 1 and health["dropped"] == 1
