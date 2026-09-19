import test from 'node:test';
import assert from 'node:assert/strict';
import { radianceBridgeUrl, radianceBridgeRequest } from '../integrations/pi/qwen-radiance-bridge.mjs';
import { cacheCommand } from '../integrations/pi/qwen-radiance-cache.mjs';

test('VM cache operations use the relay, with no SSH fallback', async (t) => {
  const previous = process.env.QWEN_RADIANCE_BRIDGE_URL;
  t.after(() => previous === undefined ? delete process.env.QWEN_RADIANCE_BRIDGE_URL : process.env.QWEN_RADIANCE_BRIDGE_URL = previous);
  process.env.QWEN_RADIANCE_BRIDGE_URL = 'http://127.0.0.1:18080/qwen-radiance/control';
  const requests = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    requests.push({ url, ...options, body: JSON.parse(options.body) });
    return new Response(JSON.stringify({ success: true }), { headers: { 'content-type': 'application/json' } });
  });
  const chat = { id: 'a'.repeat(64), generation: 'b'.repeat(64) };
  await cacheCommand(['list', '--json'], chat);
  await cacheCommand(['flush', '--identity-json', JSON.stringify(chat)]);
  await cacheCommand(['compact', '--identity-json', JSON.stringify(chat)]);
  assert.deepEqual(requests.map(r => r.body), [
    { operation: 'list', chat_ids: [chat.id] }, { operation: 'flush', chat }, { operation: 'compact', chat },
  ]);
  assert.ok(requests.every(r => r.redirect === 'error'));
  await assert.rejects(cacheCommand(['delete', '--all']), /unsupported/);
  await assert.rejects(cacheCommand(['list']), /identity unavailable/);
  t.mock.method(globalThis, 'fetch', async () => { throw new Error('relay offline'); });
  await assert.rejects(cacheCommand(['flush', '--identity-json', JSON.stringify(chat)]), /relay offline/);
});

test('relay endpoint validation and response limits', async (t) => {
  const previous = process.env.QWEN_RADIANCE_BRIDGE_URL;
  t.after(() => previous === undefined ? delete process.env.QWEN_RADIANCE_BRIDGE_URL : process.env.QWEN_RADIANCE_BRIDGE_URL = previous);
  for (const url of ['http://remote/qwen-radiance/control', 'http://user@127.0.0.1/qwen-radiance/control',
    'http://127.0.0.1/admin', 'http://127.0.0.1/qwen-radiance/control?url=http://other']) {
    process.env.QWEN_RADIANCE_BRIDGE_URL = url;
    assert.throws(radianceBridgeUrl, /invalid/);
  }
  process.env.QWEN_RADIANCE_BRIDGE_URL = 'http://127.0.0.1:18080/qwen-radiance/control';
  await assert.rejects(radianceBridgeRequest({ operation: 'config' }, {
    fetcher: async () => new Response('x'.repeat(8 * 1024 * 1024 + 1)),
  }), /too large/);
  await assert.rejects(radianceBridgeRequest({ operation: 'config' }, {
    fetcher: async () => new Response('{"error":"unavailable"}', { status: 503 }),
  }), /unavailable/);
});
