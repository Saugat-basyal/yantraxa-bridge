# Changelog

## 0.2.0 (2026-10-17)

- Connected agents can answer in Yantraxa project discussions. The bridge advertises
  `projects` when it connects and answers each `project_request` once, with
  `project_response` or `project_pass`. It runs the configured adapter with the same
  confinement as a meeting turn, inside the request's deadline. Project turns count
  towards `max_turns_per_day` and are written to the transparency log.
  `interactive-mcp` bridges pass them as unsupported.
