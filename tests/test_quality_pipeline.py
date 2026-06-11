import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_simple_yaml(path):
    result = {}
    for raw_line in (ROOT / path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        elif value.isdigit():
            value = int(value)
        else:
            try:
                value = float(value)
            except ValueError:
                value = value.strip("'\"")
        result[key.strip()] = value
    return result


class QualityPipelineTests(unittest.TestCase):
    def test_sft_config_targets_4090_quality_profile(self):
        config = load_simple_yaml("sft_config.yaml")

        self.assertIn("Qwen3-4B", config["model_name_or_path"])
        self.assertNotIn("quantization_bit", config)
        self.assertEqual(config["lora_rank"], 32)
        self.assertEqual(config["lora_alpha"], 64)
        self.assertEqual(config["lora_target"], "all")
        self.assertGreaterEqual(config["per_device_train_batch_size"], 2)
        self.assertIs(config["gradient_checkpointing"], False)
        self.assertIs(config["bf16"], True)
        self.assertGreater(config["val_size"], 0)

    def test_eval_set_is_fixed_and_broad_enough(self):
        path = ROOT / "data" / "under_acoustic_data" / "eval_underwater_qa.json"
        data = json.loads(path.read_text(encoding="utf-8"))

        self.assertGreaterEqual(len(data), 100)
        for item in data:
            self.assertEqual(set(item), {"id", "question", "reference", "keywords", "risk_flags"})
            self.assertTrue(item["id"].startswith("uwqa-"))
            self.assertTrue(item["question"].strip())
            self.assertGreaterEqual(len(item["reference"]), 80)
            self.assertGreaterEqual(len(item["keywords"]), 3)
            self.assertIsInstance(item["risk_flags"], list)

    def test_dpt_file_is_valid_json_if_present(self):
        path = ROOT / "data" / "under_acoustic_data" / "dpt.json"
        if path.exists():
            json.loads(path.read_text(encoding="utf-8"))

    def test_app_uses_low_hallucination_generation_defaults(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        assignments = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        try:
                            assignments[target.id] = ast.literal_eval(node.value)
                        except Exception:
                            pass

        self.assertIn("Qwen3-4B", assignments["BASE_MODEL_PATH"])
        self.assertIs(assignments["GENERATION_CONFIG"]["do_sample"], False)
        self.assertLessEqual(assignments["GENERATION_CONFIG"]["temperature"], 0.2)
        self.assertLessEqual(assignments["GENERATION_CONFIG"]["top_p"], 0.9)
        self.assertIn("must not fabricate", assignments["SYSTEM_PROMPT"])


if __name__ == "__main__":
    unittest.main()

