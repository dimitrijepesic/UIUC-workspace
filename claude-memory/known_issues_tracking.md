---
name: known-issues-tracking
description: "Neptune and Pasado (UIUC research projects) track full-test-run status in a root KNOWN_ISSUES.md file, not in memory"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 04885f96-c938-49f4-9f4b-1610caa18710
---

Both `C:\everything\prakse\UIUC\neptune\KNOWN_ISSUES.md` and `C:\everything\prakse\UIUC\Pasado\KNOWN_ISSUES.md`
contain detailed logs of the most recent full local test/benchmark re-run for each project — versions used,
pass/fail per component, root causes, workarounds, and what was deliberately skipped.

**Why:** A prior session did full re-runs (Pasado: 2026-07-06/07 on Windows `.venv`; Neptune: 2026-07-08 via
Docker `neptune-env:nvidia-latest`) and wrote the findings there instead of just reporting them in chat, so
the state survives across sessions.

**How to apply:** Before answering "did you run the tests" or "what's broken" for either project, read the
relevant KNOWN_ISSUES.md first — don't re-derive from scratch or assume a fresh run is needed. Treat it as a
snapshot frozen at its stated date; if the user asks about *current* state, verify by re-running rather than
trusting it's still accurate, especially if files under test have changed since.
