# miniserve Security Policy

> **Status:** v1.0.0 policy. Contact below is private; the PGP fingerprint is
> `_TBD_` — it will be published under this heading the moment a PGP posting
> key is established, and commits will go to the repo RSS feed.

## Reporting a vulnerability

**DO NOT open a public GitHub issue for a security report.**

Email the maintainer privately at:

    amman.khanansari@gmail.com

Please include:

- A minimal reproduction (prompt, config, version/commit — `git rev-parse HEAD`).
- The affected component (e.g. `miniserve/scheduler.py`, `bench/regression_gate.py`).
- Your assessment of the impact, and any patch you already wrote (we apply
  credited patches, or you can request pre-publication anonymity in the
  advisory).

## Coordinated disclosure policy — 90 days

- **Day 0:** you report. We acknowledge inside **48 hours**.
- **Day 0–30:** triage → reproduce → severity (CVSS 3.1) → fix + test.
- **Day 30–90 (max):** downstream users can upgrade; the vulnerable version
  stops being commonly deployed.
- **Day 90 (deadline):** a GitHub Security Advisory is published with the
  fix, CVE requested if severity ≥ medium, and the reporter is credited in
  the advisory and in `CHANGELOG.md` (or anonymously as requested).
- If a fix needs longer than 90 days for a genuinely large migration, the
  *progress* is published at day 90 with an agreed joint delay — the clock
  never goes silent.

## In scope

- Anything shipped in a miniserve release: `miniserve/**`, `kernels/**`,
  `bench/**`, `tests/**`, and anything attached to `v*` GitHub releases
  (wheels, SBOMs, signatures, SLSA provenance).
- Supply-chain issues: malformed wheels, missing/cosign-broken signatures,
  provenance/generator mismatches, tampered SBOM entries.
- The HTTP API surface: auth bypass, queue-state injection, SSE /
  deduplication races that let one request observe another client's tokens
  (the dedup key is content-derived — that is a known, documented trade,
  not a vulnerability unless it crosses *tenant* boundaries).

## Out of scope

- **DoS by huge prompts.** The queue cap / 503 overload path (`MAX_QUEUE`,
  `MAX_INFLIGHT`) and prompt-size rejection behave as designed. Please do
  report *auth-bypassing* DoS (e.g. one client resetting another's session).
- Requires a GPG exploit or runner compromise *of the CI infrastructure
  itself* — report that to GitHub Security instead of to us.
- Scanning noise: `pip-audit`/`grype` findings in already-vendored dev deps
  that do not appear in the shipped wheel's `dependencies`.
- Any benchmark number stated without a committed `bench/data/*.csv` — that
  is a process violation, not a vulnerability; open a regular bug with links.

## Signed releases

Every release artifact under `v*` tags ships with:

1. **SLSA Build L3 provenance** (non-forgeable, generated inside an isolated
   GitHub Actions reusable workflow) — verify with `slsa-verifier`.
2. **Sigstore keyless signatures** — verify with `cosign verify-blob`.
3. **SBOMs** (SPDX + CycloneDX, built with `syft`).
4. **`SHA256SUMS.txt`**.

Verifying is a one-liner and we publish it on release day.

## PGP fingerprint

Required to securely post/acknowledge coordinated disclosures:

    Requires PGP: _TBD_

Until a fingerprint is published here, encrypt reports with the maintainer's
key only if you can verify it out-of-band; otherwise send plaintext and mark
the report clearly.
