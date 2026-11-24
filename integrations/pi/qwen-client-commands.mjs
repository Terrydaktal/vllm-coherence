// Client-only aliases use Pi's normal cancellation and graceful shutdown hooks.
export default function clientCommands(pi) {
  pi.registerCommand("exit", {
    description: "Quit Pi (alias for /quit)",
    handler(_args, ctx) {
      ctx.abort();
      ctx.shutdown();
    },
  });
}
