"""
Formal testing script for Speculative Decoding across multiple model configurations.
Tests on Natural Language (The Pile) and Code (The Stack) datasets.
"""

import torch
import time
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class SpeculativeDecodingTester:
    """
    A class to test speculative decoding across multiple model configurations
    and dataset types.
    """
    
    def __init__(self, verifier_checkpoint: str, draft_checkpoint: str, device: str = None):
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
        print(f"Loading models...")
        print(f"  Verifier: {verifier_checkpoint}")
        print(f"  Draft: {draft_checkpoint}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(verifier_checkpoint)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load models
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_checkpoint,
            device_map="auto",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.verifier_model = AutoModelForCausalLM.from_pretrained(
            verifier_checkpoint,
            device_map="auto",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        
        
        self.verifier_model.eval()
        self.draft_model.eval()
        
        print("Models loaded successfully!\n")
    
    def standard_autoregressive_generation(self, input_ids: torch.Tensor, 
                                          max_new_tokens: int) -> Tuple[torch.Tensor, float]:
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
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        latency = time.time() - start_time
        return output, latency
    
    @torch.no_grad()
    def speculative_decoding(self, input_ids: torch.Tensor, 
                           max_new_tokens: int, 
                           gamma: int = 5) -> Tuple[torch.Tensor, float, float]:
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
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            draft_tokens = draft_outputs[0, curr_input_ids.shape[1]:]
            
            # Step 2: Verifier checks the draft
            verifier_input = torch.cat([curr_input_ids, draft_tokens.unsqueeze(0)], dim=1)
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
            curr_input_ids = torch.cat([curr_input_ids, accepted_sequence.unsqueeze(0)], dim=1)
            
            # Step 5: Correction
            if n_matches < len(draft_tokens):
                correction_token = predicted_tokens[n_matches]
                curr_input_ids = torch.cat([curr_input_ids, correction_token.unsqueeze(0).unsqueeze(0)], dim=1)
            
            if curr_input_ids.shape[1] >= target_length:
                break
        
        latency = time.time() - start_time
        acceptance_rate = accepted_draft_tokens / total_draft_tokens if total_draft_tokens > 0 else 0
        
        return curr_input_ids, latency, acceptance_rate
    
    def run_single_test(self, prompt: str, max_new_tokens: int = 100, 
                       gamma: int = 5, verbose: bool = True) -> Dict:
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
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
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
            "prompt": prompt[:100],
            "baseline_time": baseline_time,
            "speculative_time": spec_time,
            "acceptance_rate": acc_rate,
            "speedup": speedup,
            "consistency": consistency,
            "tokens_generated": min_len - input_ids.shape[1]
        }
        
        if verbose:
            print(f"Baseline Time: {baseline_time:.4f}s")
            print(f"Speculative Time: {spec_time:.4f}s")
            print(f"Acceptance Rate: {acc_rate:.2%}")
            print(f"Speedup: {speedup:.2f}x")
            print(f"Consistency: {consistency:.2%}")
        
        return results
    
    def run_benchmark_suite(self, prompts: List[Dict], max_new_tokens: int = 100, 
                          gamma: int = 5) -> Dict:
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
            prompt = prompt_data['text']
            prompt_type = prompt_data.get('type', 'unknown')
            
            print(f"\n{'='*80}")
            print(f"Test {i+1}/{len(prompts)} - Type: {prompt_type}")
            print(f"Prompt: {prompt[:100]}...")
            print(f"{'='*80}")
            
            try:
                result = self.run_single_test(prompt, max_new_tokens, gamma, verbose=True)
                result['type'] = prompt_type
                results.append(result)
            except Exception as e:
                print(f"Error in test {i+1}: {str(e)}")
                continue
        
        # Aggregate results by type
        aggregated = {}
        for result in results:
            ptype = result['type']
            if ptype not in aggregated:
                aggregated[ptype] = {
                    'acceptance_rates': [],
                    'speedups': [],
                    'baseline_times': [],
                    'speculative_times': []
                }
            
            aggregated[ptype]['acceptance_rates'].append(result['acceptance_rate'])
            aggregated[ptype]['speedups'].append(result['speedup'])
            aggregated[ptype]['baseline_times'].append(result['baseline_time'])
            aggregated[ptype]['speculative_times'].append(result['speculative_time'])
        
        # Calculate averages
        summary = {}
        for ptype, metrics in aggregated.items():
            summary[ptype] = {
                'avg_acceptance_rate': np.mean(metrics['acceptance_rates']),
                'std_acceptance_rate': np.std(metrics['acceptance_rates']),
                'avg_speedup': np.mean(metrics['speedups']),
                'std_speedup': np.std(metrics['speedups']),
                'avg_baseline_time': np.mean(metrics['baseline_times']),
                'avg_speculative_time': np.mean(metrics['speculative_times']),
                'num_samples': len(metrics['acceptance_rates'])
            }
        
        return {
            'individual_results': results,
            'summary': summary,
            'verifier_model': self.verifier_checkpoint,
            'draft_model': self.draft_checkpoint
        }


def load_pile_samples(num_samples: int = 5) -> List[Dict]:
    """
    Load Natural Language samples from The Pile dataset.
    
    Args:
        num_samples: Number of samples to load
        
    Returns:
        List of prompt dictionaries
    """
    print("Loading samples from The Pile...")
    try:
        # Load a subset of The Pile (using monology/pile-uncopyrighted)
        dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
        
        prompts = []
        for i, item in enumerate(dataset):
            if i >= num_samples:
                break
            text = item['text']
            
            # Skip very short texts
            if len(text) < 400:
                continue
            
            # Extract from middle section (skip first 25%, take from there)
            start_pos = len(text) // 4
            # Take ~200 chars from middle
            prompt_text = text[start_pos:start_pos + 200].strip()
            
            if len(prompt_text) > 50:  # Ensure reasonable length
                prompts.append({
                    'text': prompt_text,
                    'type': 'NL'
                })
        
        print(f"Loaded {len(prompts)} NL samples from The Pile")
        return prompts
    
    except Exception as e:
        print(f"Error loading The Pile: {e}")
        print("Using fallback NL prompts...")
        return [
            {
                'text': "The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by classical philosophers who attempted to describe",
                'type': 'NL'
            },
            {
                'text': "Climate change is one of the most pressing issues facing humanity today. The scientific consensus is clear: human activities, particularly the burning of fossil fuels and deforestation, are the primary drivers of recent global warming. The consequences",
                'type': 'NL'
            },
            {
                'text': "In the realm of quantum mechanics, particles exhibit behaviors that seem paradoxical from a classical physics perspective. The famous double-slit experiment demonstrates that electrons can behave as both particles and waves, depending on how they are observed",
                'type': 'NL'
            }
        ]


def load_stack_samples(languages: List[str] = ['python', 'c', 'go', 'rust'], 
                       num_samples: int = 3) -> List[Dict]:
    """
    Load Code samples from The Stack dataset.
    
    Args:
        languages: List of programming languages to sample
        num_samples: Number of samples per language
        
    Returns:
        List of prompt dictionaries
    """
    print(f"Loading code samples from The Stack for languages: {languages}...")
    prompts = []
    
    for lang in languages:
        try:
            # Load The Stack dataset for specific language
            dataset = load_dataset(
                "bigcode/the-stack-dedup",
                data_dir=f"data/{lang}",
                split="train",
                streaming=True
            )
            
            lang_prompts = 0
            for item in dataset:
                if lang_prompts >= num_samples:
                    break
                
                code = item['content']
                
                # Skip very short files (likely just imports/boilerplate)
                if len(code) < 600:
                    continue
                
                lines = code.split('\n')
                
                # Skip files with too few lines
                if len(lines) < 20:
                    continue
                
                # Find a good starting point in the middle (skip first 30% of lines)
                start_line = len(lines) // 3
                
                # Extract ~15 lines from middle section
                middle_lines = lines[start_line:start_line + 15]
                prompt_text = '\n'.join(middle_lines).strip()
                
                # Make sure we have substantive code (not just comments/whitespace)
                non_empty_lines = [l for l in middle_lines if l.strip() and not l.strip().startswith('#') and not l.strip().startswith('//')]
                
                if len(prompt_text) > 100 and len(non_empty_lines) >= 5:
                    prompts.append({
                        'text': prompt_text,
                        'type': f'Code ({lang})'
                    })
                    lang_prompts += 1
            
            print(f"  Loaded {lang_prompts} samples for {lang}")
        
        except Exception as e:
            print(f"  Error loading {lang}: {e}")

    print(f"Total code samples loaded: {len(prompts)}")
    return prompts


def run_full_benchmark():
    """
    Run the complete benchmark suite for all model configurations.
    """
    # Model configurations
    configs = [
        {
            'name': 'GPT-2',
            'verifier': 'gpt2-large',
            'draft': 'distilgpt2'
        },
        {
            'name': 'Qwen',
            'verifier': 'Qwen/Qwen2.5-7B',
            'draft': 'Qwen/Qwen2.5-0.5B'
        },
        {
            'name': 'Pythia',
            'verifier': 'EleutherAI/pythia-12b',
            'draft': 'EleutherAI/pythia-70m'
        }
    ]
    
    # Test parameters
    MAX_NEW_TOKENS = 100
    GAMMA = 5
    NUM_NL_SAMPLES = 100
    NUM_CODE_SAMPLES = 100
    
    # Load datasets
    nl_prompts = load_pile_samples(NUM_NL_SAMPLES)
    code_prompts = load_stack_samples(['python', 'c', 'go', 'rust'], NUM_CODE_SAMPLES)
    
    all_results = {}
    
    for config in configs:
        print(f"\n\n{'#'*80}")
        print(f"# Testing {config['name']} Configuration")
        print(f"# Verifier: {config['verifier']}")
        print(f"# Draft: {config['draft']}")
        print(f"{'#'*80}\n")
        
        try:
            tester = SpeculativeDecodingTester(
                config['verifier'],
                config['draft']
            )
            
            # Combine all prompts
            all_prompts = nl_prompts + code_prompts
            
            # Run benchmark
            results = tester.run_benchmark_suite(
                all_prompts,
                max_new_tokens=MAX_NEW_TOKENS,
                gamma=GAMMA
            )
            
            all_results[config['name']] = results
            
            # Print summary
            print(f"\n{'='*80}")
            print(f"SUMMARY FOR {config['name']}")
            print(f"{'='*80}")
            for ptype, metrics in results['summary'].items():
                print(f"\n{ptype}:")
                print(f"  Acceptance Rate: {metrics['avg_acceptance_rate']:.2%} ± {metrics['std_acceptance_rate']:.2%}")
                print(f"  Speedup: {metrics['avg_speedup']:.2f}x ± {metrics['std_speedup']:.2f}x")
                print(f"  Samples: {metrics['num_samples']}")
            
            # Clean up models to free memory
            del tester
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"Error testing {config['name']}: {str(e)}")
            continue
    
    # Save results to file
    output_file = "benchmark_results.json"
    with open(output_file, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj
        
        serializable_results = convert_to_serializable(all_results)
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n\nResults saved to {output_file}")
    
    # Generate LaTeX table
    generate_latex_table(all_results)
    
    return all_results


def generate_latex_table(results: Dict):
    """
    Generate LaTeX table from benchmark results.
    
    Args:
        results: Dictionary of benchmark results
    """
    print(f"\n\n{'='*80}")
    print("LATEX TABLE")
    print(f"{'='*80}\n")
    
    table_lines = []
    table_lines.append("\\begin{tabular}{|l|l|c|c|}")
    table_lines.append("\\hline")
    table_lines.append("Model & Completion & Acce. Rate & Speedup \\\\")
    table_lines.append("\\hline")
    
    for model_name, model_results in results.items():
        summary = model_results['summary']
        verifier = model_results['verifier_model'].split('/')[-1]
        draft = model_results['draft_model'].split('/')[-1]
        
        # Sort types: NL first, then Code types
        sorted_types = sorted(summary.keys(), key=lambda x: (0 if x == 'NL' else 1, x))
        
        for i, ptype in enumerate(sorted_types):
            metrics = summary[ptype]
            
            # Format model name only for first row
            if i == 0:
                model_col = f"{model_name} ({verifier}+{draft})"
            else:
                model_col = ""
            
            acc_rate = f"{metrics['avg_acceptance_rate']:.1%}"
            speedup = f"{metrics['avg_speedup']:.2f}x"
            
            table_lines.append(f"{model_col} & {ptype} & {acc_rate} & {speedup} \\\\")
        
        table_lines.append("\\hline")
    
    table_lines.append("\\end{tabular}")
    
    latex_table = "\n".join(table_lines)
    print(latex_table)
    
    # Save to file
    with open("latex_table.tex", 'w') as f:
        f.write(latex_table)
    
    print(f"\nLaTeX table saved to latex_table.tex")


if __name__ == "__main__":
    print("Starting Speculative Decoding Benchmark Suite")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("\n")
    
    results = run_full_benchmark()
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)
