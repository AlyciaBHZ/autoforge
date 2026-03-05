# Paper Reproduction Capability (ICLR)

AutoForge now supports a dedicated paper reproduction workflow:

- `autoforge paper infer "<goal>"`: infer likely ICLR papers from only goal text.
- `autoforge paper benchmark`: run hit@k benchmark on real ICLR papers.
- `autoforge paper reproduce "<goal>"`: generate reproduction artifacts and prompt.

## Why This Matters

In community operations, users often provide only:
- "I want the paper that does X on Y."
- "Reproduce the latest method for Z."

The platform needs to:
1. infer the candidate paper,
2. construct a reproducible engineering plan,
3. generate code and evaluate reproduction gaps.

## Workflow

1. Fetch papers from OpenReview (`ICLR.cc/<year>/Conference`).
2. Rank candidates by goal-to-paper overlap (title/keywords/abstract weighted).
3. Produce:
   - `candidate.json`
   - `reproduction_brief.md`
   - `generation_prompt.txt`
4. Optional: run `--run-generate` to start full AutoForge code generation.

Defaults are tuned for lightweight theory-first runs:
- Reuse local OpenReview cache by default (`--cache-hours`, `--refresh-corpus`).
- PDF parsing is opt-in (`--with-pdf`) to avoid extra network/CPU cost.
- Environment spec is generated in `theory-first` profile (no heavy ML deps installed by default).

Strict contract mode:
- Add `--strict-contract` to enforce required artifact outputs and report schema validation.
- On contract violation, command exits with code `2`.
- Contract spec: `docs/contracts/paper_repro_contract_v1.md`.

## Example

```bash
autoforge paper infer "improve long-context reasoning with sparse attention" --year 2025 --top-k 5
autoforge paper benchmark --year 2025 --sample-size 6 --top-k 5
autoforge paper reproduce "robust graph learning under missing features" --year 2025 --pick 1
autoforge paper reproduce "formal theorem proving with retrieval" --year 2025 --with-pdf
autoforge paper reproduce "long-context sparse reasoning" --year 2025 --run-generate --strict-contract
```

## Current Limitations

- Inference is lexical/keyword weighted (not full semantic retrieval).
- Dataset/metric extraction is abstract-level (not full PDF table parsing).
- No automatic figure-level curve alignment yet.
- Full code generation requires configured LLM credentials.

These limitations are tracked by benchmark output and should guide next iterations.
