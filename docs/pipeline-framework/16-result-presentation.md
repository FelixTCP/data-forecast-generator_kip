# #16 Context Engineering: Result Presentation

## Objective

Produce user-facing outputs for technical and business audiences.

## Outputs

- `evaluation.json` (machine-readable)
- `report.md` (human-readable)
- optional lightweight plot assets

## Copilot Prompt Snippet

```markdown
Implement `build_result_package(context, output_dir)`.
Generate summary with chosen model, key metrics, strongest features, and actionable next steps.
```

## Suggested `report.md` Sections

1. Problem + selected target
2. Data quality summary
3. Candidate models + scores
4. Selected model rationale
5. Risks and caveats
6. Next iteration recommendations
