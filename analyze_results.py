"""
Analyze and visualize benchmark results.
"""

import json
import sys


def load_results(filename="benchmark_results.json"):
    """Load benchmark results from JSON file."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {filename} not found. Run test_speculative_decoding.py first.")
        return None
    except json.JSONDecodeError:
        print(f"Error: {filename} is not valid JSON.")
        return None


def print_summary_table(results):
    """Print a formatted summary table."""
    print("\n" + "="*100)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*100)
    
    # Header
    print(f"\n{'Model':<20} {'Type':<15} {'Acc. Rate':<15} {'Speedup':<15} {'Samples':<10}")
    print("-"*100)
    
    for model_name, model_data in results.items():
        summary = model_data.get('summary', {})
        
        # Sort types: NL first, then code types
        sorted_types = sorted(summary.keys(), key=lambda x: (0 if x == 'NL' else 1, x))
        
        for i, ptype in enumerate(sorted_types):
            metrics = summary[ptype]
            
            model_col = model_name if i == 0 else ""
            acc_rate = f"{metrics['avg_acceptance_rate']*100:.1f}% ± {metrics['std_acceptance_rate']*100:.1f}%"
            speedup = f"{metrics['avg_speedup']:.2f}x ± {metrics['std_speedup']:.2f}x"
            samples = metrics['num_samples']
            
            print(f"{model_col:<20} {ptype:<15} {acc_rate:<15} {speedup:<15} {samples:<10}")
        
        print("-"*100)


def generate_latex_table_from_results(results, output_file="latex_table.tex"):
    """Generate LaTeX table from results."""
    print(f"\nGenerating LaTeX table...")
    
    lines = []
    lines.append("\\begin{tabular}{|l|l|c|c|}")
    lines.append("\\hline")
    lines.append("Model & Completion & Acce. Rate & Speedup \\\\")
    lines.append("\\hline")
    
    for model_name, model_data in results.items():
        summary = model_data.get('summary', {})
        verifier = model_data['verifier_model'].split('/')[-1]
        draft = model_data['draft_model'].split('/')[-1]
        
        # Sort types
        sorted_types = sorted(summary.keys(), key=lambda x: (0 if x == 'NL' else 1, x))
        
        for i, ptype in enumerate(sorted_types):
            metrics = summary[ptype]
            
            if i == 0:
                model_col = f"{model_name} ({verifier}+{draft})"
            else:
                model_col = ""
            
            acc_rate = f"{metrics['avg_acceptance_rate']*100:.1f}\\%"
            speedup = f"{metrics['avg_speedup']:.2f}x"
            
            lines.append(f"{model_col} & {ptype} & {acc_rate} & {speedup} \\\\")
        
        lines.append("\\hline")
    
    lines.append("\\end{tabular}")
    
    latex_content = "\n".join(lines)
    
    # Save to file
    with open(output_file, 'w') as f:
        f.write(latex_content)
    
    print(f"LaTeX table saved to {output_file}")
    
    # Also print to console
    print("\n" + "="*80)
    print("LATEX TABLE")
    print("="*80)
    print(latex_content)
    print("="*80)


def compare_models(results):
    """Compare different model configurations."""
    print("\n" + "="*100)
    print("MODEL COMPARISON")
    print("="*100)
    
    # Group by prompt type
    prompt_types = set()
    for model_data in results.values():
        prompt_types.update(model_data.get('summary', {}).keys())
    
    for ptype in sorted(prompt_types, key=lambda x: (0 if x == 'NL' else 1, x)):
        print(f"\n{ptype}:")
        print(f"  {'Model':<15} {'Acc. Rate':<20} {'Speedup':<20}")
        print("  " + "-"*60)
        
        for model_name, model_data in results.items():
            summary = model_data.get('summary', {})
            if ptype in summary:
                metrics = summary[ptype]
                acc_rate = f"{metrics['avg_acceptance_rate']*100:.1f}% ± {metrics['std_acceptance_rate']*100:.1f}%"
                speedup = f"{metrics['avg_speedup']:.2f}x ± {metrics['std_speedup']:.2f}x"
                print(f"  {model_name:<15} {acc_rate:<20} {speedup:<20}")


def analyze_individual_results(results):
    """Analyze individual test results for patterns."""
    print("\n" + "="*100)
    print("INDIVIDUAL RESULT ANALYSIS")
    print("="*100)
    
    for model_name, model_data in results.items():
        print(f"\n{model_name}:")
        individual = model_data.get('individual_results', [])
        
        if not individual:
            print("  No individual results found")
            continue
        
        # Group by type
        by_type = {}
        for result in individual:
            ptype = result.get('type', 'unknown')
            if ptype not in by_type:
                by_type[ptype] = []
            by_type[ptype].append(result)
        
        for ptype, results_list in sorted(by_type.items()):
            print(f"\n  {ptype} ({len(results_list)} samples):")
            
            for i, result in enumerate(results_list, 1):
                prompt_preview = result['prompt'][:50] + "..." if len(result['prompt']) > 50 else result['prompt']
                print(f"    {i}. {prompt_preview}")
                print(f"       Acc Rate: {result['acceptance_rate']*100:.1f}% | "
                      f"Speedup: {result['speedup']:.2f}x | "
                      f"Consistency: {result['consistency']*100:.1f}%")


def main():
    """Main analysis function."""
    # Get filename from command line or use default
    filename = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.json"
    
    print("="*100)
    print("SPECULATIVE DECODING RESULTS ANALYZER")
    print("="*100)
    print(f"\nLoading results from: {filename}")
    
    results = load_results(filename)
    
    if results is None:
        return
    
    print(f"Found results for {len(results)} model configuration(s)")
    
    # Print summary table
    print_summary_table(results)
    
    # Compare models
    compare_models(results)
    
    # Generate LaTeX table
    generate_latex_table_from_results(results)
    
    # Analyze individual results
    analyze_individual_results(results)
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
