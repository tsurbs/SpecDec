"""
Utils for benchmarking Speculative Decoding across multiple model configurations.
Tests on Natural Language (The Pile) and Code (The Stack) datasets.
"""

import torch
import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple
import warnings


class SpeculativeDecodingTester:
    """
    A class to test speculative decoding across multiple model configurations
    and dataset types.
    """

    def __init__(
        self, verifier_checkpoint: str, draft_checkpoint: str, device: str = None
    ):
        """
        Initialize the tester with verifier and draft models.

        Args:
            verifier_checkpoint: HuggingFace model checkpoint for verifier
            draft_checkpoint: HuggingFace model checkpoint for draft model
            device: Device to use (cuda/cpu). Auto-detected if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verifier_checkpoint = verifier_checkpoint
        self.draft_checkpoint = draft_checkpoint

        print(f"Using device: {self.device}")
        print(f"Verifier: {verifier_checkpoint}")
        print(f"Draft: {draft_checkpoint}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(verifier_checkpoint)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load models
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_checkpoint,
            device_map="auto",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.verifier_model = AutoModelForCausalLM.from_pretrained(
            verifier_checkpoint,
            device_map="auto",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

        self.verifier_model.eval()
        self.draft_model.eval()

        print("Models loaded successfully!\n")

    def standard_autoregressive_generation(
        self, input_ids: torch.Tensor, max_new_tokens: int
    ) -> Tuple[torch.Tensor, float]:
        """
        Generates text using standard autoregressive decoding.

        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            Tuple of (output_ids, latency)
        """
        start_time = time.time()

        with torch.no_grad():
            output = self.verifier_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding
                pad_token_id=self.tokenizer.eos_token_id,
            )

        latency = time.time() - start_time
        return output, latency

    @torch.no_grad()
    def speculative_decoding(
        self, input_ids: torch.Tensor, max_new_tokens: int, gamma: int = 5
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Implementation of Speculative Decoding.

        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum number of tokens to generate
            gamma: Number of tokens the draft model guesses at once

        Returns:
            Tuple of (output_ids, latency, acceptance_rate)
        """
        start_time = time.time()

        total_draft_tokens = 0
        accepted_draft_tokens = 0

        curr_input_ids = input_ids.clone()
        target_length = input_ids.shape[1] + max_new_tokens

        while curr_input_ids.shape[1] < target_length:
            # Step 1: Draft Model generates gamma tokens
            draft_outputs = self.draft_model.generate(
                curr_input_ids,
                max_new_tokens=gamma,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            draft_tokens = draft_outputs[0, curr_input_ids.shape[1] :]

            # Step 2: Verifier checks the draft
            verifier_input = torch.cat(
                [curr_input_ids, draft_tokens.unsqueeze(0)], dim=1
            )
            verifier_outputs = self.verifier_model(verifier_input)
            logits = verifier_outputs.logits

            start_pos = curr_input_ids.shape[1] - 1
            end_pos = verifier_input.shape[1] - 1
            predicted_tokens = torch.argmax(logits[0, start_pos:end_pos], dim=-1)

            # Step 3: Acceptance Loop
            n_matches = 0
            for i in range(len(draft_tokens)):
                if draft_tokens[i] == predicted_tokens[i]:
                    n_matches += 1
                else:
                    break

            total_draft_tokens += len(draft_tokens)
            accepted_draft_tokens += n_matches

            # Step 4: Append Accepted Tokens
            accepted_sequence = draft_tokens[:n_matches]
            curr_input_ids = torch.cat(
                [curr_input_ids, accepted_sequence.unsqueeze(0)], dim=1
            )

            # Step 5: Correction
            if n_matches < len(draft_tokens):
                correction_token = predicted_tokens[n_matches]
                curr_input_ids = torch.cat(
                    [curr_input_ids, correction_token.unsqueeze(0).unsqueeze(0)], dim=1
                )

            if curr_input_ids.shape[1] >= target_length:
                break

        latency = time.time() - start_time
        acceptance_rate = (
            accepted_draft_tokens / total_draft_tokens if total_draft_tokens > 0 else 0
        )

        return curr_input_ids, latency, acceptance_rate

    def run_single_test(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        gamma: int = 5,
        verbose: bool = True,
    ) -> Dict:
        """
        Run a single test comparing baseline and speculative decoding.

        Args:
            prompt: Input prompt
            max_new_tokens: Number of tokens to generate
            gamma: Draft token count
            verbose: Print detailed output

        Returns:
            Dictionary with test results
        """
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        )
        input_ids = inputs.input_ids.to(self.device)

        # Baseline
        baseline_output, baseline_time = self.standard_autoregressive_generation(
            input_ids, max_new_tokens
        )

        # Speculative
        spec_output, spec_time, acc_rate = self.speculative_decoding(
            input_ids, max_new_tokens, gamma=gamma
        )

        # Calculate metrics
        speedup = baseline_time / spec_time if spec_time > 0 else 0

        # Verify consistency
        min_len = min(baseline_output.shape[1], spec_output.shape[1])
        baseline_trunc = baseline_output[:, :min_len]
        spec_trunc = spec_output[:, :min_len]
        matches = (baseline_trunc == spec_trunc).sum().item()
        consistency = matches / min_len if min_len > 0 else 0

        results = {
            "prompt": prompt[:100],  # Truncated for efficiency
            "baseline_time": baseline_time,
            "speculative_time": spec_time,
            "acceptance_rate": acc_rate,
            "speedup": speedup,
            "consistency": consistency,
            "tokens_generated": min_len - input_ids.shape[1],
        }

        if verbose:
            print(f"Baseline Time: {baseline_time:.4f}s")
            print(f"Speculative Time: {spec_time:.4f}s")
            print(f"Acceptance Rate: {acc_rate:.2%}")
            print(f"Speedup: {speedup:.2f}x")
            print(f"Consistency: {consistency:.2%}")

        return results

    def run_benchmark_suite(
        self, prompts: List[Dict], max_new_tokens: int = 100, gamma: int = 5
    ) -> Dict:
        """
        Run benchmark on a suite of prompts.

        Args:
            prompts: List of dictionaries with 'text' and 'type' keys
            max_new_tokens: Number of tokens to generate
            gamma: Draft token count

        Returns:
            Aggregated results
        """
        results = []

        for i, prompt_data in enumerate(prompts):
            prompt = prompt_data["text"]
            prompt_type = prompt_data.get("type", "unknown")

            print(f"Test {i+1}/{len(prompts)}")
            print(f"Type: {prompt_type}")
            print(f"Prompt: {prompt[:100]}")

            try:
                result = self.run_single_test(
                    prompt, max_new_tokens, gamma, verbose=True
                )
                result["type"] = prompt_type
                results.append(result)
            except Exception as e:
                print(f"Error in test {i+1}: {str(e)}")
                continue

        # Aggregate results by type
        aggregated = {}
        for result in results:
            prompt_type = result["type"]
            if prompt_type not in aggregated:
                aggregated[prompt_type] = {
                    "acceptance_rates": [],
                    "speedups": [],
                    "baseline_times": [],
                    "speculative_times": [],
                }

            aggregated[prompt_type]["acceptance_rates"].append(result["acceptance_rate"])
            aggregated[prompt_type]["speedups"].append(result["speedup"])
            aggregated[prompt_type]["baseline_times"].append(result["baseline_time"])
            aggregated[prompt_type]["speculative_times"].append(result["speculative_time"])

        # Calculate averages
        summary = {}
        for prompt_type, metrics in aggregated.items():
            summary[prompt_type] = {
                "avg_acceptance_rate": np.mean(metrics["acceptance_rates"]),
                "std_acceptance_rate": np.std(metrics["acceptance_rates"]),
                "avg_speedup": np.mean(metrics["speedups"]),
                "std_speedup": np.std(metrics["speedups"]),
                "avg_baseline_time": np.mean(metrics["baseline_times"]),
                "avg_speculative_time": np.mean(metrics["speculative_times"]),
                "num_samples": len(metrics["acceptance_rates"]),
            }

        return {
            "individual_results": results,
            "summary": summary,
            "verifier_model": self.verifier_checkpoint,
            "draft_model": self.draft_checkpoint,
        }
