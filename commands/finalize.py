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
def finalize(project_name: str, threshold: float, include_below_threshold: bool, debug: bool, platform: str | None, model: str | None):
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
                # Get source files from hypothesis and augment heuristically
                source_files = list(hypothesis.get('properties', {}).get('source_files', []) or [])
                hypothesis.get('node_refs', [])
                # Heuristic: guess file paths from title/description/reasoning/evidence text
                try:
                    from analysis.path_utils import guess_relpaths
                    extra_texts = [
                        hypothesis.get('title', ''),
                        hypothesis.get('description', ''),
                        hypothesis.get('reasoning', ''),
                    ]
                    for ev in hypothesis.get('evidence', []) or []:
                        if isinstance(ev, dict):
                            extra_texts.append(ev.get('description', '') or '')
                        elif isinstance(ev, str):
                            extra_texts.append(ev)
                    guessed = guess_relpaths("\n".join([t for t in extra_texts if t]), repo_root)
                    for rel in guessed:
                        if rel not in source_files:
                            source_files.append(rel)
                except Exception:
                    pass
                
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
                
                # Independent search: if still empty, scan repo_root for relevant files
                if not source_code and repo_root:
                    if debug:
                        progress.console.print("  [dim]Searching repository for relevant files...[/dim]")
                    try:
                        # Build simple keyword set from hypothesis text
                        kw = set()
                        for k in (hypothesis.get('title', ''), hypothesis.get('description', ''), hypothesis.get('reasoning', '')):
                            for token in (k or '').replace('\n', ' ').split():
                                token = token.strip(".,:;(){}[]'\"<>\n\r\t")
                                if len(token) >= 3 and token.isascii():
                                    kw.add(token.lower())
                        # Include affected function names
                        for fn in hypothesis.get('properties', {}).get('affected_functions', []) or []:
                            if isinstance(fn, str) and fn:
                                kw.add(fn.lower())
                        # Scan for relevant files by name/ext and keyword hits
                        exts = {'.rs','.py','.ts','.tsx','.js','.jsx','.go','.sol','.java','.kt','.swift','.c','.cc','.cpp','.h','.hpp','.yaml','.yml','.toml','.json','.sql','.sh','.rb','.php'}
                        candidates: list[tuple[int, Path]] = []
                        max_files = 10
                        for p in repo_root.rglob('*'):
                            try:
                                if not p.is_file():
                                    continue
                                if p.suffix.lower() not in exts:
                                    continue
                                # Light size cap to avoid huge files
                                if p.stat().st_size > 400_000:
                                    continue
                                name_score = 0
                                lname = p.name.lower()
                                for t in kw:
                                    if t and t in lname:
                                        name_score += 2
                                if name_score == 0 and not source_files:
                                    # Avoid reading too many files with no hint
                                    continue
                                # Read and score by keyword occurrences
                                score = name_score
                                try:
                                    text = p.read_text(encoding='utf-8', errors='ignore')
                                    ltext = text.lower()
                                    hits = sum(ltext.count(t) for t in kw if len(t) > 3)
                                    score += hits
                                except Exception:
                                    continue
                                if score > 0:
                                    candidates.append((score, p))
                            except Exception:
                                continue
                        candidates.sort(key=lambda x: x[0], reverse=True)
                        before = len(source_code)
                        for _, fp in candidates[:max_files]:
                            try:
                                rel = str(fp.relative_to(repo_root))
                                source_code[rel] = fp.read_text(encoding='utf-8', errors='ignore')
                            except Exception:
                                continue
                    except Exception:
                        pass
                    added = len(source_code) - (before if 'before' in locals() else 0)
                    if added > 0:
                        progress.console.print(f"  [dim]Repo search added {added} file(s)[/dim]")
                
                # Build review prompt
                review_prompt = f"""You are a security expert performing final review of a vulnerability hypothesis.

=== HYPOTHESIS UNDER REVIEW ===
Title: {hypothesis.get('title', 'Unknown')}
Type: {hypothesis.get('vulnerability_type', 'unknown')}
Severity: {hypothesis.get('severity', 'unknown')}
Confidence: {hypothesis.get('confidence', 0):.0%}
Description: {hypothesis.get('description', '')}

=== SOURCE CODE ===
"""
                if source_code:
                    for file_path, code in source_code.items():
                        review_prompt += f"\n--- File: {file_path} ---\n{code}\n"
                else:
                    review_prompt += "No source code available.\n"
                
                review_prompt += """
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
    "reasoning": "A detailed one-paragraph explanation (100-200 words) suitable for a security report. Start with 'Upon reviewing...' or 'After analyzing...' and explain what you examined, what you found, and why you reached your conclusion. Be specific about code elements, line numbers, and functions you reviewed.",
    "confidence": 0.0 to 1.0
}

Rules:
- "confirmed" = Vulnerability clearly exists in the code with exploitable path
  Write like: "After analyzing the [specific function/contract] at [location], I confirmed this [vulnerability type] is exploitable. [Explain what makes it vulnerable, specific code patterns observed, and why existing protections are insufficient]."

- "rejected" = Code analysis shows this is a false positive or mitigated
  Write like: "Upon reviewing the alleged [vulnerability type] in [specific location], I determined this is a false positive. [Explain what protections exist, why the vulnerability cannot be exploited, or what was misunderstood in the initial hypothesis]."

- "uncertain" = Need more code context to determine
  Write like: "After examining the [specific area], I cannot definitively confirm or reject this [vulnerability type]. [Explain what was reviewed, what remains unclear, and what additional analysis would be needed for a definitive verdict]."

Be conservative - only confirm if the code clearly shows the vulnerability.
"""
                
                # Get LLM verdict
                try:
                    # Use raw() method and parse JSON response (robust to code fences)
                    response_text = llm.raw(system="You are a security expert. Respond only with valid JSON.", user=review_prompt)
                    from utils.json_utils import extract_json_object
                    response = extract_json_object(response_text)
                    if isinstance(response, dict):
                        result = ReviewResult(
                            verdict=response.get('verdict', 'uncertain'),
                            reasoning=response.get('reasoning', 'No reasoning provided'),
                            confidence=response.get('confidence', 0.5)
                        )
                    else:
                        result = ReviewResult(verdict='uncertain', reasoning='Failed to parse response')
                except Exception as e:
                    if debug:
                        progress.console.print(f"  [red]LLM error: {e}[/red]")
                    result = ReviewResult(verdict='uncertain', reasoning='LLM error')
                
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
