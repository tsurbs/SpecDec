from datasets import load_dataset
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

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


def load_stack_samples(languages: List[str] = ["python", "javascript", "java", "c++"], 
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
    
    for lang in languages if languages is not None else [None]:
        try:
            # Load The Stack dataset for specific language
            dataset = load_dataset(
                "bigcode/the-stack-dedup",
                data_dir=f"data/{lang}" if lang is not None else "data/all",
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
                
                prompts.append({
                    'text': code,
                    'type': f'Code ({lang})'
                })
                lang_prompts += 1
            
            print(f"  Loaded {lang_prompts} samples for {lang}")
        
        except Exception as e:
            print(f"  Error loading {lang}: {e}")

    print(f"Total code samples loaded: {len(prompts)}")
    return prompts

def load_stack_samples_representative(dataset_proportion: float) -> List[Dict]:
    """
    Load representative Code samples from The Stack dataset across multiple languages, using bigcode/the-stack-smol.
    
    Args:
        num_samples: Total number of samples to load across languages
    Returns:
        List of prompt dictionaries
    """
    prompts = []
    languages = [
        "assembly",
        "batchfile",
        "c++",
        "c-sharp",
        "c",
        "cmake",
        "css",
        "dockerfile",
        "fortran",
        "go",
        "haskell",
        "html",
        "java",
        "javascript",
        "julia",
        "lua",
        "makefile",
        "markdown",
        "perl",
        "php",
        "powershell",
        "python",
        "ruby",
        "rust",
        "scala",
        "shell",
        "sql",
        "tex",
        "typescript",
        "visual-basic"
    ]
    
    print(f"Loading representative code samples from The Stack across multiple languages...")
    for lang in languages:
        try:
            # Load The Stack dataset for specific language
            dataset = load_dataset(
                "bigcode/the-stack-smol",
                data_dir=f"data/{lang}",
                split="train",
                streaming=True
            )
            
            lang_prompts = 0
            for item in dataset:
                
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