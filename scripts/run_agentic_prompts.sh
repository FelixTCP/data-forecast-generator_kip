#!/usr/bin/env bash
set -euo pipefail

CSV_PATH="${1:-}"
TARGET_COLUMN="${2:-}"
OUTPUT_DIR="${3:-artifacts/manual_$(date -u +%Y%m%dT%H%M%SZ)}"
CODE_DIR="${4:-${OUTPUT_DIR}/code}"
SPLIT_MODE="${SPLIT_MODE:-auto}"
COPILOT_MODEL="${COPILOT_MODEL:-gpt-5-mini}"
REASONING_EFFORT="${REASONING_EFFORT:-low}"

if [[ -z "$CSV_PATH" ]]; then
  echo "Usage: $0 <csv_path> <target_column> [output_dir] [code_dir]"
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$CODE_DIR" "$OUTPUT_DIR/debug"

echo "Running direct prompt workflow"
echo "CSV_PATH=$CSV_PATH"
echo "TARGET_COLUMN=$TARGET_COLUMN"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "CODE_DIR=$CODE_DIR"

render_prompt() {
  local in_file="$1"
  local out_file="$2"
  sed \
    -e "s|{{CSV_PATH}}|$CSV_PATH|g" \
    -e "s|{{TARGET_COLUMN}}|$TARGET_COLUMN|g" \
    -e "s|{{OUTPUT_DIR}}|$OUTPUT_DIR|g" \
    -e "s|{{CODE_DIR}}|$CODE_DIR|g" \
    -e "s|{{SPLIT_MODE}}|$SPLIT_MODE|g" \
    "$in_file" > "$out_file"
}

run_prompt() {
  local label="$1"
  local template="$2"
  local rendered="$OUTPUT_DIR/debug/${label}_prompt.md"
  local response="$OUTPUT_DIR/debug/${label}_response.md"

  render_prompt "$template" "$rendered"
  copilot --allow-all-tools --allow-all-paths --allow-all-urls --no-ask-user \
    --model "$COPILOT_MODEL" --reasoning-effort "$REASONING_EFFORT" \
    -s -p "$(cat "$rendered")" > "$response"
}

run_prompt setup prompts/agentic/setup_prompt.md
run_prompt step10 prompts/agentic/step_10.md
run_prompt step11 prompts/agentic/step_11.md
run_prompt step12 prompts/agentic/step_12.md
run_prompt step13 prompts/agentic/step_13.md
run_prompt step14 prompts/agentic/step_14.md
run_prompt step15 prompts/agentic/step_15.md
run_prompt step16 prompts/agentic/step_16.md

echo "Done. Inspect: $OUTPUT_DIR"
