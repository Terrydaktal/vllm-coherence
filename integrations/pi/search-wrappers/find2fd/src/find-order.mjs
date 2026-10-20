import { opendirSync, statSync } from "node:fs";
import { typeMatches } from "./find-options.mjs";

export const MAX_CANDIDATES = 100000;
export const MAX_ORDER_PARENTS = 10000;
export const MAX_DIRECTORY_ENTRIES = 10000;
export const MAX_ORDER_ENTRIES = 200000;
const key = (value) => value.toString("hex");

function components(path) {
  const parts = [];
  let start = 0;
  for (let index = 0; index <= path.length; index++) {
    if (index !== path.length && path[index] !== 47) continue;
    const part = path.subarray(start, index);
    if (!part.length || part.equals(Buffer.from(".")) || part.equals(Buffer.from(".."))) throw new Error("unnormalized fd record");
    parts.push(part); start = index + 1;
  }
  return parts;
}

export function decodeCandidates(data, root) {
  if (!data.length) return [];
  if (data.at(-1) !== 0) throw new Error("unterminated fd pathname record");
  const prefix = Buffer.from(root === "/" ? "/" : `${root}/`);
  const candidates = [], seen = new Set();
  let start = 0;
  for (let index = 0; index < data.length; index++) {
    if (data[index] !== 0) continue;
    let record = data.subarray(start, index); start = index + 1;
    if (!record.length || !record.subarray(0, prefix.length).equals(prefix)) throw new Error("fd pathname escaped its starting root");
    const directory = record.at(-1) === 47;
    if (directory) record = record.subarray(0, -1);
    const relative = record.subarray(prefix.length), id = key(relative);
    if (seen.has(id)) throw new Error("duplicate fd pathname record");
    seen.add(id);
    candidates.push({ relative, parts: components(relative), directory });
    if (candidates.length > MAX_CANDIDATES) throw new Error("fd candidate count bound");
  }
  return candidates;
}

/** Restore GNU pre-order from raw directory entries, only on needed ancestors. */
export function orderCandidates(root, candidates, options, budget = { entries: 0, parents: 0 }) {
  const parents = new Map([["", Buffer.alloc(0)]]);
  if (!candidates.length) return [];
  for (const candidate of candidates) {
    let relative = Buffer.alloc(0);
    for (let index = 0; index < candidate.parts.length - 1; index++) {
      relative = relative.length ? Buffer.concat([relative, Buffer.from("/"), candidate.parts[index]]) : candidate.parts[index];
      parents.set(key(relative), relative);
    }
  }
  if (budget.parents + parents.size > MAX_ORDER_PARENTS) throw new Error("ordering parent count bound");
  budget.parents += parents.size;
  const orderedParents = [...parents.values()].sort((left, right) => componentsDepth(left) - componentsDepth(right));
  const listings = new Map();
  for (const relative of orderedParents) {
    if (relative.length) {
      const parentEnd = relative.lastIndexOf(47);
      const parent = parentEnd < 0 ? Buffer.alloc(0) : relative.subarray(0, parentEnd);
      const name = relative.subarray(parentEnd + 1);
      const entry = listings.get(key(parent))?.get(key(name))?.entry;
      const path = Buffer.concat([Buffer.from(root === "/" ? "/" : `${root}/`), relative]);
      if (!entry || !(entry.isDirectory() || (options.dereference === "L" && entry.isSymbolicLink() && statSync(path).isDirectory())))
        throw new Error("fd ancestor changed or is not a directory");
    }
    const path = relative.length ? Buffer.concat([Buffer.from(root === "/" ? "/" : `${root}/`), relative]) : Buffer.from(root);
    const directory = opendirSync(path, { encoding: "buffer" });
    const listing = new Map();
    try {
      let entry, rank = 0;
      while ((entry = directory.readSync()) !== null) {
        if (++rank > MAX_DIRECTORY_ENTRIES) throw new Error("GNU FTS inode ordering for large directories");
        if (++budget.entries > MAX_ORDER_ENTRIES) throw new Error("ordering entry count bound");
        listing.set(key(entry.name), { rank, entry });
      }
    } finally { directory.closeSync(); }
    listings.set(key(relative), listing);
  }
  for (const candidate of candidates) {
    const ranks = [];
    let parent = Buffer.alloc(0), leaf;
    for (const part of candidate.parts) {
      leaf = listings.get(key(parent))?.get(key(part));
      if (!leaf) throw new Error("fd pathname changed or missing from directory");
      ranks.push(leaf.rank);
      parent = parent.length ? Buffer.concat([parent, Buffer.from("/"), part]) : part;
    }
    let entry = leaf.entry;
    if (options.dereference === "L" && entry.isSymbolicLink()) {
      try { entry = statSync(Buffer.concat([Buffer.from(root === "/" ? "/" : `${root}/`), candidate.relative])); }
      catch (error) { if (!["ENOENT", "ENOTDIR"].includes(error.code)) throw error; }
    }
    if (entry.isDirectory() !== candidate.directory ||
        (options.candidateFilter.types && !options.candidateFilter.types.some((type) => typeMatches(entry, type))))
      throw new Error("fd type changed or invalid response");
    candidate.ranks = ranks;
    candidate.entry = entry;
  }
  return candidates.filter((candidate) => candidate.parts.length >= options.minDepth &&
    (options.maxDepth === null || candidate.parts.length <= options.maxDepth)).sort((left, right) => {
      for (let index = 0; index < Math.min(left.ranks.length, right.ranks.length); index++) {
        if (left.ranks[index] !== right.ranks[index]) return left.ranks[index] - right.ranks[index];
      }
      return options.depth ? right.ranks.length - left.ranks.length : left.ranks.length - right.ranks.length;
    });
}

function componentsDepth(value) {
  return value.length ? value.reduce((count, byte) => count + (byte === 47 ? 1 : 0), 1) : 0;
}

export function displayPath(root, relative) {
  return Buffer.concat([Buffer.from(root.endsWith("/") ? root : `${root}/`), relative]);
}
