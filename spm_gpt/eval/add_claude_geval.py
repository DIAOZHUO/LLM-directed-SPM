"""
Adds "GEval Score(claude)" to each result file of eval_results/ (knowledge-base SLM), using Claude Sonnet 4.6
through the Claude Code CLI (`claude -p`) as an independent LLM judge.
The questions are taken from the messages of spmknowledge_test_dataset, as in generate_eval_dataset_answer.py.
Files that already have "GEval Score(claude)" are skipped unless --force is given.

The knowledge-base dataset and the generated answers (predictions / ground truth) are not included in this
repository because they are derived from copyrighted textbooks and papers; only the resulting scores are provided.
"""
import json
import os
import re
import argparse
import subprocess

from datasets import load_from_disk
from tqdm import tqdm

stop = False
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "eval_results")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "finetune", "spmknowledge_test_dataset")




def add_key_to_json(path, key, value):
    if not os.path.exists(path):
        data = {}
    else:
        with open(path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}

    data[key] = value
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)





CRITERIA = (
    "Coherence — the collective quality of all sentences. "
    "We align this dimension with the DUC quality question of structure and coherence "
    "whereby the summary should be well-structured and well-organized. "
    "The summary should not just be a heap of related information, but should build "
    "from sentence to a coherent body of information about a topic."
)

EVAL_PROMPT_TEMPLATE = """You are an evaluator. Assess the Actual Answer using the criteria below and return ONLY a JSON object with two keys: "score" (float 0.0–1.0) and "reason" (one concise sentence).

Criteria: {criteria}

Input: {input}
Expected Answer: {expected_output}
Actual Answer: {actual_output}

Return ONLY valid JSON, no other text:
{{"score": <float>, "reason": "<one sentence>"}}"""


def g_eval_with_claude_code(input_text: str, actual_output: str, expected_output: str, criteria: str) -> float:
    prompt = EVAL_PROMPT_TEMPLATE.format(
        criteria=criteria,
        input=input_text,
        expected_output=expected_output,
        actual_output=actual_output,
    )
    result = subprocess.run(
        ["claude", "--model", "claude-sonnet-4-6", "-p", prompt],
        capture_output=True,
        text=True,
        encoding="utf-8",
        stdin=subprocess.DEVNULL,
    )
    text = result.stdout or ""
    # print(text)
    try:
        data = json.loads(text.strip())
        return float(data["score"])
    except Exception:
        match = re.search(r'"score"\s*:\s*([0-9.]+)', text)
        if match:
            return min(float(match.group(1)), 1.0)
        raise RuntimeError(f"error parsing claude score: {text}")


def load_questions() -> list[str]:
    dataset = load_from_disk(DATASET_PATH)
    questions = []
    for example in dataset:
        user_content = next(
            m["content"] for m in example["messages"] if m["role"] == "user"
        )
        questions.append(user_content)
    return questions


def compute_claude_geval(questions: list[str], predictions: list[str], references: list[str]) -> float:
    scores = []
    for q, pred, ref in tqdm(zip(questions, predictions, references), total=len(predictions)):
        score = g_eval_with_claude_code(
            input_text=q,
            actual_output=pred,
            expected_output=ref,
            criteria=CRITERIA,
        )
        scores.append(score)
    return float(sum(scores) / len(scores))


def process_file(path: str, questions: list[str], force: bool = False) -> None:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if "GEval Score(claude)" in data and not force:
        print(f"  [skip] already has GEval Score(claude)")
        return

    predictions = data.get("predictions", [])
    references = data.get("ground truth", [])

    if not predictions or not references:
        print(f"  [skip] missing predictions or ground truth")
        return

    print(f"  evaluating {len(predictions)} samples...")
    score = compute_claude_geval(questions, predictions, references)
    print(f"  GEval Score(claude) = {score:.4f}")

    add_key_to_json(path, "GEval Score(claude)", score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-evaluate even if score already exists")
    parser.add_argument("--files", nargs="*", help="specific filenames to process (default: all)")
    args = parser.parse_args()

    print("Loading questions from dataset...")
    questions = load_questions()
    print(f"Loaded {len(questions)} questions.")

    if args.files:
        targets = [os.path.join(RESULTS_DIR, f) for f in args.files]
    else:
        targets = [
            os.path.join(RESULTS_DIR, f)
            for f in sorted(os.listdir(RESULTS_DIR))
            if f.endswith(".json") and "_backup" not in f
        ]

    for path in targets:
        fname = os.path.basename(path)
        print(f"\n[{fname}]")
        process_file(path, questions, force=args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()
