// Exact standalone wrapper snapshots are bundled with Pi, including their
// native fallback and fast dispatchers. User ! commands and child scripts
// receive no hook, and the original shell command is never textually rewritten.
import { withGrepWrapper } from "./search-wrappers/grep2rg/src/grep-wrapper.mjs";
import { withFindWrapper } from "./search-wrappers/find2fd/src/find-wrapper.mjs";

export function withSearchWrappers(context, shellPath) {
  return withFindWrapper(withGrepWrapper(context, shellPath), shellPath);
}
