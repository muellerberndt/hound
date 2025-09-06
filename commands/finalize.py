"""
Finalize command for reviewing and confirming/rejecting hypotheses.
"""

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from analysis.concurrent_knowledge import HypothesisStore
from commands.project import ProjectManager

console = Console()


@click.command()
@click.argument('project_name')
@click.option('--threshold', '-t', default=0.5, help="Confidence threshold for hypothesis review (default: 0.5 or 50%)")
@click.option('--include-below-threshold', is_flag=True, help="Also review pending hypotheses below the threshold (retries previously undecidable ones)")
@click.option('--debug', is_flag=True, help="Enable debug mode")
@click.option('--platform', default=None, help='Override QA platform (e.g., openai, anthropic, mock)')
@click.option('--model', default=None, help='Override QA model (e.g., gpt-4o-mini)')
@click.option('--include-file', 'include_files', multiple=True, help='Additional file(s) to include (relative to repo root); repeatable')
@click.option('--filename-regex', 'filename_regexes', multiple=True, help='Regex pattern to include files by path; repeatable')
@click.option('--text-regex', 'text_regexes', multiple=True, help='Regex pattern to include files by content; repeatable')
@click.option('--ignore-case', is_flag=True, help='Case-insensitive regex matching for --filename-regex/--text-regex')
@click.option('--max-search-files', type=int, default=10, help='Max files to add from regex searches (default: 10)')
@click.option('--no-ext-filter', is_flag=True, help='Search all files (not only common code extensions)')
@click.option('--max-iters', type=int, default=3, help='Max iterations to fetch more files and retry when uncertain (default: 3)')
def finalize(project_name: str, threshold: float, include_below_threshold: bool, debug: bool, platform: str | None, model: str | None,
             include_files: tuple[str, ...], filename_regexes: tuple[str, ...], text_regexes: tuple[str, ...], ignore_case: bool,
             max_search_files: int, no_ext_filter: bool, max_iters: int):
    """
    Finalize hypotheses in a project by reviewing high-confidence findings.
    
    This command:
    1. Reviews hypotheses above the confidence threshold with full source code context
    2. Confirms or rejects each hypothesis with reasoning
    """
    # Convert percentage values (>1) to decimal
    if threshold > 1:
        threshold = threshold / 100
    
    manager = ProjectManager()
    project = manager.get_project(project_name)
    
    if not project:
        console.print(f"[red]Project '{project_name}' not found.[/red]")
        sys.exit(1)
    
    project_dir = Path(project["path"])
    manifest_dir = project_dir / "manifest"
    
    # Check for existing hypotheses
    hypothesis_file = project_dir / "hypotheses.json"
    if not hypothesis_file.exists():
        console.print("[yellow]No hypotheses found in project.[/yellow]")
        sys.exit(0)
    
    # Load hypotheses
    store = HypothesisStore(hypothesis_file, agent_id="finalize")
    # Load all hypotheses directly from the store
    data = store._load_data()
    all_hypotheses = data.get("hypotheses", {})
    
    if not all_hypotheses:
        console.print("[yellow]No hypotheses to review.[/yellow]")
        sys.exit(0)
    
    # Load config for LLM
    config_path = Path(__file__).parent.parent / "config.yaml"
    config = {}
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    # Apply QA overrides if provided
    if config is None:
        config = {}
    if 'models' not in config:
        config['models'] = {}
    if platform or model:
        config['models'].setdefault('finalize', {})
        if platform:
            config['models']['finalize']['provider'] = platform
        if model:
            config['models']['finalize']['model'] = model
    
    # Pending = not already finalized
    pending = {
        hid: h for hid, h in all_hypotheses.items()
        if h.get("status") not in ["confirmed", "rejected"]
    }

    # Count hypotheses above threshold
    above_threshold = {
        hid: h for hid, h in pending.items() 
        if h.get("confidence", 0) >= threshold
    }

    # Choose candidates
    chosen = pending if include_below_threshold else above_threshold
    candidates = [(hid, h) for hid, h in chosen.items()]
    
    # Display summary
    summary_panel = Panel(
        f"[bold]Hypothesis Finalization[/bold]\n\n"
        f"Project: {project_name}\n"
        f"Total hypotheses: {len(all_hypotheses)}\n"
        f"Pending (not confirmed/rejected): {len(pending)}\n"
        f"Above threshold ({threshold:.0%}): {len(above_threshold)}\n"
        + ("[dim]\nIncluding below-threshold pending items\n[/dim]" if include_below_threshold else "") +
        f"To review: [green]{len(candidates)} hypotheses[/green]",
        title="[bold bright_cyan]Finalization Summary[/bold bright_cyan]",
        expand=False
    )
    console.print(summary_panel)
    # Compliment the timing
    from random import choice as _choice
    console.print(_choice([
        "[white]Normal reviewers tick boxes, but YOU crown truth and retire doubt.[/white]",
        "[white]This isn’t just a review — it’s a coronation of evidence because YOU demanded it.[/white]",
        "[white]Normal audits end; YOUR verdicts become precedent and the code falls in line.[/white]",
        "[white]This is not paperwork — it’s canon formation under YOUR signet.[/white]",
        "[white]Normal judgment closes loops; YOUR judgment defines the standard.[/white]",
    ]))
    
    if not candidates:
        console.print("\n[yellow]No hypotheses meet the threshold criteria.[/yellow]")
        sys.exit(0)
    
    # Show candidates
    if candidates:
        console.print("\n[bold]Hypotheses to review:[/bold]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Conf", style="cyan", width=5)
        table.add_column("Type", style="yellow", width=16)
        table.add_column("Title", width=55)
        
        for hid, h in candidates[:10]:
            conf = h.get("confidence", 0)
            vuln_type = h.get("vulnerability_type", "unknown")
            title = h.get("title", "Unknown")
            
            if conf >= 0.8:
                conf_str = f"[bold green]{conf:.0%}[/bold green]"
            elif conf >= 0.6:
                conf_str = f"[yellow]{conf:.0%}[/yellow]"
            else:
                conf_str = f"[red]{conf:.0%}[/red]"
            
            table.add_row(conf_str, vuln_type, title)
        
        console.print(table)
        if len(candidates) > 10:
            console.print(f"  [dim]... and {len(candidates) - 10} more[/dim]")
    
    # Initialize debug logger if needed
    debug_logger = None
    # Legacy: pre-filter debug logger no longer used, keep sentinel to avoid NameError
    filter_debug_logger = None
    if debug:
        from analysis.debug_logger import DebugLogger
        debug_logger = DebugLogger(f"finalize_{project_name}")
        console.print(f"[dim]Debug log will be saved to: {debug_logger.log_file}[/dim]")
    
    # Initialize LLM for finalization
    from llm.unified_client import UnifiedLLMClient
    llm = UnifiedLLMClient(cfg=config, profile="finalize", debug_logger=debug_logger)
    
    # No pre-filtering, proceed directly to review
    
    # Load manifest for source code access
    manifest_file = manifest_dir / "manifest.json"
    manifest_data = {}
    if manifest_file.exists():
        with open(manifest_file) as f:
            manifest_data = json.load(f)
    
    # Get repository root (prefer manifest, fallback to project source_path)
    repo_root = None
    if manifest_data.get("repo_path"):
        repo_root = Path(manifest_data["repo_path"]).resolve()
    if (not repo_root or not repo_root.exists()) and project.get('source_path'):
        repo_root = Path(project['source_path']).resolve()
    if repo_root and not repo_root.exists():
        console.print(f"[yellow]Warning: Repository path not found: {repo_root}[/yellow]")
        repo_root = None
    
    # Process candidates for review
    if not candidates:
        console.print("\n[green]✓ No hypotheses meet the threshold for review.[/green]")
        sys.exit(0)
    
    # Narrative model names
    models = (config or {}).get('models', {})
    finalize_model = (models.get('finalize') or {}).get('model') or 'Finalize-Model'
    console.print(f"\n[bold bright_cyan]{random.choice(['🧙 Sage', '🧠 Reviewer'])} {finalize_model} enters the chamber...[/bold bright_cyan]")
    
    # Simple structured output for verdict
    @dataclass
    class ReviewResult:
        verdict: str  # "confirmed", "rejected", or "uncertain"
        reasoning: str
        confidence: float = 0.5
    
    confirmed = 0
    rejected = 0
    uncertain = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        for idx, (hid, hypothesis) in enumerate(candidates, 1):
            flavor = random.choice([
                "weighs the evidence",
                "consults the scrolls",
                "ponders the claim",
            ])
            task_desc = f"[{idx}/{len(candidates)}] {finalize_model} {flavor}: {hypothesis.get('title', '')[:50]}..."
            task = progress.add_task(task_desc, total=1)
            
            try:
                # Start only from associated files provided by the hypothesis
                source_files = list(hypothesis.get('properties', {}).get('source_files', []) or [])
                
                # Load source code (direct from repo when possible)
                source_code = {}
                if source_files and repo_root:
                    for file_path in source_files[:10]:  # Load up to 10 files
                        try:
                            full_path = repo_root / file_path
                            if full_path.exists():
                                with open(full_path) as f:
                                    content = f.read()
                                    # Include full file contents to ensure complete context
                                    source_code[file_path] = content
                        except Exception as e:
                            if debug:
                                progress.console.print(f"  [red]Failed to load {file_path}: {e}[/red]")
                if source_code:
                    progress.console.print(f"  [dim]Loaded {len(source_code)} file(s) from source_files[/dim]")

                # Explicit inclusions via CLI
                if include_files and repo_root:
                    added_explicit = 0
                    for rel in include_files:
                        try:
                            fp = (repo_root / rel).resolve()
                            # Ensure the file is inside repo_root
                            if not str(fp).startswith(str(repo_root.resolve())):
                                continue
                            if fp.is_file():
                                content = fp.read_text(encoding='utf-8', errors='ignore')
                                key = str(fp.relative_to(repo_root))
                                if key not in source_code:
                                    source_code[key] = content
                                    added_explicit += 1
                        except Exception:
                            continue
                    if added_explicit:
                        progress.console.print(f"  [dim]Included {added_explicit} file(s) via --include-file[/dim]")
                
                # Proactive searches are performed only inside the iterative expansion below
                
                # Helper: build review prompt from current source_code
                def _build_prompt(src_map: dict[str, str]) -> str:
                    prompt = f"""You are a security expert performing final review of a vulnerability hypothesis.

=== HYPOTHESIS UNDER REVIEW ===
Title: {hypothesis.get('title', 'Unknown')}
Type: {hypothesis.get('vulnerability_type', 'unknown')}
Severity: {hypothesis.get('severity', 'unknown')}
Confidence: {hypothesis.get('confidence', 0):.0%}
Description: {hypothesis.get('description', '')}

=== SOURCE CODE ===
"""
                    if src_map:
                        for file_path, code in src_map.items():
                            prompt += f"\n--- File: {file_path} ---\n{code}\n"
                    else:
                        prompt += "No source code available.\n"
                    prompt += """
=== YOUR TASK ===
Review the SOURCE CODE to determine if this hypothesis represents a REAL vulnerability.

Focus on:
1. Does the SOURCE CODE show the vulnerability exists?
2. Is there a clear attack vector in the code?
3. Are there checks/guards that prevent exploitation?
4. Is this a false positive based on the actual implementation?

Provide your determination in this EXACT JSON format:
{
    "verdict": "confirmed" or "rejected" or "uncertain",
    "reasoning": "A detailed one-paragraph explanation (100-200 words) suitable for a security report.",
    "confidence": 0.0 to 1.0
}

Be conservative - only confirm if the code clearly shows the vulnerability.
"""
                    return prompt
                
                # Get LLM verdict (with augmentation loop if uncertain and repo available)
                def _ask_llm(prompt: str) -> ReviewResult:
                    try:
                        response_text = llm.raw(system="You are a security expert. Respond only with valid JSON.", user=prompt)
                        from utils.json_utils import extract_json_object
                        response = extract_json_object(response_text)
                        if isinstance(response, dict):
                            verdict = response.get('verdict', 'uncertain')
                            reasoning = response.get('reasoning')
                            if not isinstance(reasoning, str) or not reasoning.strip():
                                reasoning = (
                                    "After reviewing the included files, there is insufficient code context to provide a definite verdict. "
                                    "Load the concrete parsing/validation layers and the sink functions that enforce or perform the behavior under review."
                                )
                            conf = response.get('confidence', 0.5)
                            try:
                                conf = float(conf)
                            except Exception:
                                conf = 0.5
                            return ReviewResult(verdict=verdict, reasoning=reasoning, confidence=conf)
                        return ReviewResult(verdict='uncertain', reasoning='Failed to parse response')
                    except Exception as e:
                        if debug:
                            progress.console.print(f"  [red]LLM error: {e}[/red]")
                        return ReviewResult(verdict='uncertain', reasoning='LLM error')

                # Augment context based on model reasoning and optional regex filters
                import re
                flags = re.IGNORECASE if ignore_case else 0
                fn_patterns = [re.compile(p, flags) for p in filename_regexes]
                txt_patterns = [re.compile(p, flags) for p in text_regexes]
                exts_filter = None if no_ext_filter else {'.rs','.py','.ts','.tsx','.js','.jsx','.go','.sol','.java','.kt','.swift','.c','.cc','.cpp','.h','.hpp','.yaml','.yml','.toml','.json','.sql','.sh','.rb','.php'}

                def _augment_from_reasoning(reasoning: str, budget: int) -> int:
                    if not repo_root or budget <= 0:
                        return 0
                    try:
                        added = 0
                        # 1) Filename regex-based additions
                        if fn_patterns:
                            for p in repo_root.rglob('*'):
                                if added >= budget:
                                    break
                                try:
                                    if not p.is_file():
                                        continue
                                    if exts_filter is not None and p.suffix.lower() not in exts_filter:
                                        continue
                                    rel = str(p.relative_to(repo_root))
                                    if rel in source_code:
                                        continue
                                    if any(r.search(rel) for r in fn_patterns) and p.stat().st_size <= 400_000:
                                        source_code[rel] = p.read_text(encoding='utf-8', errors='ignore')
                                        added += 1
                                except Exception:
                                    continue
                        # 2) Text regex-based additions
                        if txt_patterns and added < budget:
                            for p in repo_root.rglob('*'):
                                if added >= budget:
                                    break
                                try:
                                    if not p.is_file():
                                        continue
                                    if exts_filter is not None and p.suffix.lower() not in exts_filter:
                                        continue
                                    rel = str(p.relative_to(repo_root))
                                    if rel in source_code:
                                        continue
                                    if p.stat().st_size > 400_000:
                                        continue
                                    text = p.read_text(encoding='utf-8', errors='ignore')
                                    if any(r.search(text) for r in txt_patterns):
                                        source_code[rel] = text
                                        added += 1
                                except Exception:
                                    continue
                        if added >= budget:
                            return added
                        # 3) Heuristic additions from reasoning hints
                        hints = set()
                        for m in re.findall(r"[A-Za-z0-9_\-/]+\.(rs|py|ts|tsx|js|jsx|go|sol|java|kt|swift|c|cc|cpp|h|hpp|yaml|yml|toml|json|sql|sh|rb|php)", reasoning or '', flags):
                            hints.add(m.lower())
                        for tok in re.findall(r"[A-Za-z0-9_]{3,}", reasoning or '', flags):
                            hints.add(tok.lower())
                            for sub in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", tok):
                                if len(sub) >= 3:
                                    hints.add(sub.lower())
                        if hints:
                            candidates: list[tuple[int, Path]] = []
                            for p in repo_root.rglob('*'):
                                try:
                                    if not p.is_file():
                                        continue
                                    if exts_filter is not None and p.suffix.lower() not in exts_filter:
                                        continue
                                    if p.stat().st_size > 400_000:
                                        continue
                                    rel = str(p.relative_to(repo_root))
                                    if rel in source_code:
                                        continue
                                    name_score = sum(2 for h in hints if h in rel.lower())
                                    if name_score == 0:
                                        continue
                                    try:
                                        ltext = p.read_text(encoding='utf-8', errors='ignore').lower()
                                    except Exception:
                                        continue
                                    score = name_score + sum(ltext.count(h) for h in hints if len(h) > 3)
                                    if score > 0:
                                        candidates.append((score, p))
                                except Exception:
                                    continue
                            candidates.sort(key=lambda x: x[0], reverse=True)
                            for _, fp in candidates:
                                if added >= budget:
                                    break
                                rel = str(fp.relative_to(repo_root))
                                if rel in source_code:
                                    continue
                                try:
                                    source_code[rel] = fp.read_text(encoding='utf-8', errors='ignore')
                                    added += 1
                                except Exception:
                                    continue
                        return added
                    except Exception:
                        return 0

                # Iterative loop: ask -> if uncertain, expand -> retry (up to max-iters)
                attempt = 0
                while True:
                    attempt += 1
                    if debug:
                        progress.console.print(f"  [dim]Iteration {attempt}/{max_iters}: requesting verdict[/dim]")
                    review_prompt = _build_prompt(source_code)
                    result = _ask_llm(review_prompt)
                    if result.verdict in ("confirmed", "rejected"):
                        break
                    if attempt >= max_iters or not repo_root:
                        break
                    added = _augment_from_reasoning(result.reasoning or '', max_search_files)
                    if added <= 0:
                        break
                    progress.console.print(f"  [dim]Added {added} file(s) for next iteration[/dim]")
                    # next iteration will rebuild prompt with updated source_code
                
                # Apply verdict
                if result.verdict == "confirmed":
                    # Mark as confirmed
                    store.adjust_confidence(hid, 1.0, result.reasoning)
                    
                    # Update status to confirmed
                    def update_status(data):
                        if hid in data["hypotheses"]:
                            data["hypotheses"][hid]["status"] = "confirmed"
                            data["metadata"]["confirmed"] = sum(
                                1 for h in data["hypotheses"].values() 
                                if h["status"] == "confirmed"
                            )
                        return data, True
                    
                    store.update_atomic(update_status)
                    confirmed += 1
                    
                    progress.console.print(f"  [green]✓ CONFIRMED:[/green] {hypothesis.get('title', '')[:60]}")
                    if result.reasoning:
                        # Always show justification for confirmed findings
                        progress.console.print(f"    [bold cyan]Justification:[/bold cyan] {result.reasoning}")
                
                elif result.verdict == "rejected":
                    # Mark as rejected
                    store.adjust_confidence(hid, 0.0, result.reasoning)
                    
                    # Update status to rejected
                    def update_status(data):
                        if hid in data["hypotheses"]:
                            data["hypotheses"][hid]["status"] = "rejected"
                        return data, True
                    
                    store.update_atomic(update_status)
                    rejected += 1
                    
                    progress.console.print(f"  [red]✗ REJECTED:[/red] {hypothesis.get('title', '')[:60]}")
                    if result.reasoning:
                        # Always show justification for rejected findings
                        progress.console.print(f"    [bold yellow]Reason:[/bold yellow] {result.reasoning}")
                
                else:
                    uncertain += 1
                    progress.console.print(f"  [yellow]? UNCERTAIN:[/yellow] {hypothesis.get('title', '')[:60]}")
                    if result.reasoning:
                        # Always show justification for uncertain findings
                        progress.console.print(f"    [bold magenta]Note:[/bold magenta] {result.reasoning}")
            
            except Exception as e:
                uncertain += 1
                if debug:
                    progress.console.print(f"  [red]Error reviewing {hid}: {e}[/red]")
            
            progress.advance(task)
            progress.remove_task(task)
    
    # Final summary
    console.print("\n" + "="*60)
    console.print("\n[bold]Finalization Complete:[/bold]")
    console.print(f"  [green]✓ Confirmed:[/green] {confirmed}")
    console.print(f"  [red]✗ Rejected:[/red] {rejected}")
    console.print(f"  [yellow]? Uncertain:[/yellow] {uncertain}")
    console.print(f"\n[dim]Total reviewed: {len(candidates)}[/dim]")
    
    # Show confirmed vulnerabilities
    if confirmed > 0:
        console.print("\n[bold green]Confirmed Vulnerabilities:[/bold green]")
        # Load hypotheses directly from the store
        data = store._load_data()
        confirmed_hyps = data.get("hypotheses", {})
        for hid, hyp in confirmed_hyps.items():
            if hyp.get("status") == "confirmed":
                console.print(f"  • {hyp.get('title', 'Unknown')}")
                console.print(f"    [dim]Type: {hyp.get('vulnerability_type', 'unknown')} | Severity: {hyp.get('severity', 'unknown')}[/dim]")
    
    console.print()
    
    # Close debug loggers if they exist
    if debug:
        if filter_debug_logger:
            filter_debug_logger.finalize()
            console.print(f"[dim]Pre-filter debug log saved to: {filter_debug_logger.log_file}[/dim]")
        if debug_logger:
            debug_logger.finalize()
            console.print(f"[dim]Finalization debug log saved to: {debug_logger.log_file}[/dim]")


if __name__ == "__main__":
    finalize()
