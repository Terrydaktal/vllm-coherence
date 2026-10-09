// Pi resolves this adapter alongside its pinned runtime; the implementation is
// dependency-injected and directly testable without loading a model/provider.
import { installTaskPlan } from "./qwen-task-plan.mjs";

export default function taskPlan(pi) {
  installTaskPlan(pi);
}
