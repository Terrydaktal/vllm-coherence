# Security reports

Do not put private sessions, reversible token streams, KV state, credentials or
raw tensors in a public issue. Use a synthetic reproduction and sanitized version
information. Contact [Terrydaktal](https://github.com/Terrydaktal) to arrange private
handling for sensitive reports.

The API/control paths target a trusted single-user GPU host and bind to loopback.
Use SSH forwarding; do not expose the unauthenticated service to untrusted networks.
