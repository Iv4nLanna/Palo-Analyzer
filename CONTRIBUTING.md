# Contributing

Thanks for your interest in contributing.

## Development Setup
1. Create and activate a virtual environment.
2. Install dependencies:
```powershell
pip install -r requirements.txt
```
3. Run tests:
```powershell
pytest -q
```

## Branching and Commits
- Use short branches per change (example: `feat/ci-tests`, `fix/margin-swap`).
- Keep commits focused and descriptive.
- Recommended commit style:
  - `feat: ...`
  - `fix: ...`
  - `docs: ...`
  - `test: ...`

## Pull Requests
- Explain the problem and the proposed solution.
- Include validation evidence (test output, sample JSON, screenshots if UI change).
- For psychometric rules, cite the rule source used.

## Quality Bar
- Do not break existing CLI and desktop flows.
- Add or update tests for critical rule changes.
- Keep outputs auditable (traceable parameters and rule IDs in JSON).
