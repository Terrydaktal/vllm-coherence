import assert from "node:assert/strict";
import test from "node:test";
import { installToolListing } from "../integrations/pi/qwen-tools.mjs";

function fixture({ tools = [], active = [] } = {}) {
  const commands = new Map();
  const notices = [];
  const reads = { all: 0, active: 0 };
  const state = { tools, active };
  const pi = new Proxy({
    registerCommand(name, command) {
      assert.equal(commands.has(name), false, "command registered once");
      commands.set(name, command);
    },
    getAllTools() { reads.all++; return state.tools; },
    getActiveTools() { reads.active++; return state.active; },
  }, {
    get(target, name) {
      assert.ok(name in target, `unexpected Pi API access: ${String(name)}`);
      return target[name];
    },
  });
  installToolListing(pi);
  assert.deepEqual([...commands.keys()], ["tools"]);
  assert.deepEqual(reads, { all: 0, active: 0 }, "registry is read when invoked, after extensions load");
  const invoke = async () => {
    notices.length = 0;
    await commands.get("tools").handler("", { ui: { notify: (message, level) => notices.push({ message, level }) } });
    return notices.map(({ message }) => message).join("\n");
  };
  return { invoke, notices, reads, state };
}

function assertToolStatus(output, name, status) {
  const namePattern = new RegExp(`\\b${name}\\b`);
  const statusPattern = new RegExp(`\\b${status}\\b`, "i");
  assert.ok(output.split("\n").some((line) => namePattern.test(line) && statusPattern.test(line)),
    `${name} should be displayed as ${status}: ${output}`);
}

test("/tools lists registered names, descriptions and current enabled status locally", async () => {
  const tools = Object.freeze([
    Object.freeze({ name: "read", description: "Read a synthetic local file" }),
    Object.freeze({ name: "extension_lookup", description: "Search synthetic extension records" }),
    Object.freeze({ name: "write", description: "Replace a synthetic local file" }),
  ]);
  const active = Object.freeze(["read", "extension_lookup"]);
  const listing = fixture({ tools, active });
  const output = await listing.invoke();
  for (const tool of tools) {
    assert.ok(output.includes(tool.description), `${tool.name} description is displayed`);
    assertToolStatus(output, tool.name, active.includes(tool.name) ? "enabled" : "disabled");
  }
  assert.deepEqual(listing.reads, { all: 1, active: 1 });
  assert.ok(listing.notices.every(({ level }) => level !== "error"));
  assert.equal(listing.state.tools, tools, "tool registry is not replaced");
  assert.equal(listing.state.active, active, "active tools are not replaced");
});

test("/tools reflects added and removed tools and changed activation on each invocation", async () => {
  const listing = fixture({ tools: [{ name: "old_extension", description: "Previous extension" }], active: ["old_extension"] });
  assertToolStatus(await listing.invoke(), "old_extension", "enabled");
  listing.state.tools = [
    { name: "read", description: "New builtin description" },
    { name: "new_extension", description: "New extension description" },
  ];
  listing.state.active = ["new_extension"];
  let output = await listing.invoke();
  assert.doesNotMatch(output, /old_extension|Previous extension/);
  assertToolStatus(output, "read", "disabled");
  assertToolStatus(output, "new_extension", "enabled");
  assert.ok(output.includes("New builtin description"));
  listing.state.tools[0].description = "Updated builtin description";
  listing.state.active = ["read"];
  output = await listing.invoke();
  assertToolStatus(output, "read", "enabled");
  assertToolStatus(output, "new_extension", "disabled");
  assert.ok(output.includes("Updated builtin description"));
  assert.doesNotMatch(output, /New builtin description/);
  assert.deepEqual(listing.reads, { all: 3, active: 3 });
});

test("/tools clearly reports an empty registry", async () => {
  const listing = fixture();
  assert.match(await listing.invoke(), /no (?:registered )?tools|tools[^\n]*\b0\b/i);
  assert.ok(listing.notices.every(({ level }) => level !== "error"));
});

for (const source of ["all", "active"]) {
  test(`/tools surfaces ${source} registry failures locally`, async () => {
    const listing = fixture({ tools: [{ name: "read", description: "Synthetic read" }], active: ["read"] });
    Object.defineProperty(listing.state, source === "all" ? "tools" : "active", {
      get() { throw new Error(`${source} registry unavailable`); },
    });
    assert.match(await listing.invoke(), new RegExp(`${source} registry unavailable`));
    assert.ok(listing.notices.some(({ level }) => level === "error"));
  });
}
