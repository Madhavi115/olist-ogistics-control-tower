# Team workflow (Git + GitHub)

A lightweight feature-branch workflow, similar to a small engineering team:

1. Create a GitHub repository, protect `main` (require CI + 1 review if working with others).
2. Create an issue with acceptance criteria, e.g. "Order mart has one row per order".
3. Work on `feature/phase1-order-mart`; commit small, descriptive changes.
4. Push branch and create Pull Request referencing the issue.
5. Check CI output and review the diff. Avoid merging failing tests.
6. Squash-merge into `main`; remove the branch.

Do not commit raw CSV files, generated outputs, secrets, `.env`, or notebooks
containing customer-level records. Add changes to the data contract when schema
or label definitions change.

Example conventional commits:
- `feat: build order-level logistics mart`
- `test: prevent duplicate order joins`
- `docs: define late-delivery eligibility`
