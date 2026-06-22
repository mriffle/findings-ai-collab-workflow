# templates/

Templates the engine instantiates into a user's project (spec docs 03, 04, 06, 07). Templates are the *shape*; the filled-in instances live in the user's project, never here.

Templates:

- **`finding.md`** ✅ — a finding document: YAML frontmatter (the doc 03 schema) + the body sections (Summary · Verdict · Evidence · Methods · Discussion · Caveats · Follow-ups · Related findings · References).
- **`research-finding.md`** ✅ — external-knowledge finding: topic, summary, detailed findings, and a mandatory verified-references section (doc 04.4).
- **`color_registry.json`** ✅ — the category→color map, seeded with universal Okabe–Ito defaults and extended per project (doc 06.5).
- **`project-CLAUDE.md`** ✅ — the project-scoped standing-instructions file the init command writes into the user's working directory.
- **`report.md`** ⬜ — report skeleton (Title · Abstract · Methods · Results · Discussion · References), for both QC and research modes (reporting phase, doc 07.3).
