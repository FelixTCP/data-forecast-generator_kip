# #16 Context Engineering: Result Presentation

## Objective

Produce user-facing outputs for technical and business audiences.

If upstream quality flag indicates `leakage_suspected`, the report must clearly state that metrics are invalid for production forecasting and must include remediation actions.

## Outputs

- `evaluation.json` (machine-readable)
- `report.md` (human-readable)
- optional lightweight plot assets

## Copilot Prompt Snippet

```markdown
Implement `build_result_package(context, output_dir)`.
Generate summary with chosen model, key metrics, strongest features, and actionable next steps.
If leakage is suspected, do not present a "selected production model"; present diagnostics instead.
```

## Suggested `report.md` Sections

1. Problem + selected target
2. Data quality summary
3. Candidate models + scores
4. Selected model rationale
5. Risks and caveats
6. Next iteration recommendations

## Mandatory Leakage Disclosure

- Include a dedicated warning paragraph when `quality_flag` is `leakage_suspected`, `subpar`, or `no_viable_candidate`.
- Explicitly state whether the run is production-usable.
