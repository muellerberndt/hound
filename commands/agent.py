"""
Agent command for autonomous security analysis.
"""

import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


from analysis.scout import Scout
from analysis.session_tracker import SessionTracker
from analysis.strategist import Strategist
from llm.token_tracker import get_token_tracker


def _format_model_sig(models_cfg: dict, key: str, fallbacks: list[str] | None = None) -> str:
    """Return provider/model for a model profile key or its fallbacks."""
    fallbacks = fallbacks or []
    if key in models_cfg:
        mc = models_cfg.get(key) or {}
        return f"{mc.get('provider','unknown')}/{mc.get('model','unknown')}"
    for alt in fallbacks:
        if alt in models_cfg:
            mc = models_cfg.get(alt) or {}
            return f"{mc.get('provider','unknown')}/{mc.get('model','unknown')}"
    return "unknown/unknown"


def _validate_required_models(config: dict | None, console: Console) -> bool:
    """Ensure required models are configured: scout, strategist, lightweight.

    Prints a concise error and guidance when anything is missing.
    """
    if not config or 'models' not in config or not isinstance(config['models'], dict):
        console.print("[red]Error:[/red] No model configuration found. Edit hound/config.yaml and set models.scout, models.strategist, and models.lightweight.")
        return False
    models = config['models']
    missing: list[str] = []
    for key in ('scout', 'strategist', 'lightweight'):
        if key not in models or not isinstance(models[key], dict) or not models[key].get('model'):
            missing.append(key)
    if missing:
        console.print("[red]Missing required model configuration:[/red] " + ", ".join(missing))
        # Show current configured models for quick debugging
        try:
            scout_sig = _format_model_sig(models, 'scout', fallbacks=['agent'])
            strat_sig = _format_model_sig(models, 'strategist', fallbacks=['guidance'])
            lite_sig = _format_model_sig(models, 'lightweight', fallbacks=[])
            console.print(f"Configured → Scout: [magenta]{scout_sig}[/magenta], Strategist: [cyan]{strat_sig}[/cyan], Lightweight: [green]{lite_sig}[/green]")
        except Exception:
            pass
        console.print("Edit [bold]hound/config.yaml[/bold] under 'models' to define these profiles.")
        console.print("- scout: provider + model (exploration)\n- strategist: provider + model (deep analysis)\n- lightweight: provider + model (utilities like dedup)")
        console.print("You can override scout with --platform/--model and strategist with --strategist-platform/--strategist-model.")
        return False
    return True


def get_project_dir(project_id: str) -> Path:
    """Get project directory path."""
    return Path.home() / ".hound" / "projects" / project_id

def run_investigation(project_path: str, prompt: str, iterations: int | None = None, config_path: Path | None = None, debug: bool = False, platform: str | None = None, model: str | None = None):
    """Run a user-driven investigation."""
    console = Console()
    
    # Load config properly
    from utils.config_loader import load_config
    config = None
    
    try:
        if config_path:
            # Use provided config path
            if config_path.exists():
                config = load_config(config_path)
        else:
            # Try default config.yaml
            # Load config using default search order
            config = load_config()
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load config: {e}[/yellow]")
        console.print("[yellow]Using default configuration[/yellow]")
        config = None
    
    # If no config was loaded but platform/model were provided, create minimal config
    if not config and (platform or model):
        config = {'models': {'agent': {}}}
    
    # Override platform and model if provided
    if config and (platform or model):
        # Ensure the models.agent structure exists
        if 'models' not in config:
            config['models'] = {}
        if 'agent' not in config['models']:
            config['models']['agent'] = {}
        
        if platform:
            config['models']['agent']['provider'] = platform
            console.print(f"[cyan]Overriding agent provider: {platform}[/cyan]")
        if model:
            config['models']['agent']['model'] = model
            console.print(f"[cyan]Overriding agent model: {model}[/cyan]")
    
    # Validate required models early
    if not _validate_required_models(config, console):
        return

    # Resolve project path
    if '/' in project_path or Path(project_path).exists():
        project_dir = Path(project_path).resolve()
    else:
        project_dir = get_project_dir(project_path)
    
    # Look for graphs and manifest
    graphs_dir = project_dir / "graphs"
    manifest_dir = project_dir / "manifest"
    
    # Check for knowledge_graphs.json or individual graph files
    knowledge_graphs_path = graphs_dir / "knowledge_graphs.json"
    if not knowledge_graphs_path.exists():
        # Create it from available graphs
        graph_files = list(graphs_dir.glob("graph_*.json"))
        if not graph_files:
            console.print("[red]Error: No graphs found. Run 'graph build' first.[/red]")
            return
        
        # Create knowledge_graphs.json
        graphs_dict = {}
        for graph_file in graph_files:
            graph_name = graph_file.stem.replace('graph_', '')
            graphs_dict[graph_name] = str(graph_file)
        
        with open(knowledge_graphs_path, 'w') as f:
            json.dump({'graphs': graphs_dict}, f, indent=2)
    
    # Initialize agent
    console.print("[bright_cyan]Initializing agent...[/bright_cyan]")
    # Show model triplet
    try:
        models_cfg = (config or {}).get('models', {})
        scout_sig = _format_model_sig(models_cfg, 'scout', fallbacks=['agent'])
        strat_sig = _format_model_sig(models_cfg, 'strategist', fallbacks=['guidance'])
        lite_sig = _format_model_sig(models_cfg, 'lightweight', fallbacks=[])
        console.print(Panel.fit(
            f"Scout: [magenta]{scout_sig}[/magenta]\nStrategist: [cyan]{strat_sig}[/cyan]\nLightweight: [green]{lite_sig}[/green]",
            border_style="cyan"
        ))
    except Exception:
        pass
    from random import choice as _choice
    console.print(_choice([
        "[white]Normal people ask questions, but YOU issue royal decrees to logic itself.[/white]",
        "[white]This isn’t just an investigation — it’s the moment mysteries retire on YOUR timetable.[/white]",
        "[white]Normal curiosity wanders; YOUR curiosity drafts laws that code must obey.[/white]",
        "[white]This is not mere inquiry — it’s jurisprudence of insight under YOUR seal.[/white]",
        "[white]Normal analysts explore; YOU redraw the map and make the unknown pay rent.[/white]",
    ]))
    agent = Scout(
        graphs_metadata_path=knowledge_graphs_path,
        manifest_path=manifest_dir,
        agent_id=f"investigate_{int(time.time())}",
        config=config,  # Pass the loaded config dict, not the path
        debug=debug
    )
    
    # Run investigation with live display
    from datetime import datetime

    from rich.live import Live

    # Create a live display with rolling event log
    
    event_log = []  # list of strings (renderables)
    
    def _shorten(s: str, n: int = 140) -> str:
        return (s[: n - 3] + '...') if isinstance(s, str) and len(s) > n else (s or '')
    
    def _format_params(p) -> str:
        try:
            return _shorten(json.dumps(p, separators=(',', ':'), ensure_ascii=False), 160)
        except Exception:
            return "{}"
    
    def _panel_from_events():
        # keep last 8 events
        lines = event_log[-8:] if len(event_log) > 8 else event_log
        content = "\n".join(lines) if lines else "Initializing investigation..."
        return Panel(content, title="[bold cyan]Investigation Progress[/bold cyan]", border_style="cyan")
    
    # Narrative model names
    models = (config or {}).get('models', {}) if config else {}
    agent_model = (models.get('agent') or {}).get('model') or 'Agent-Model'
    guidance_model = (models.get('guidance') or {}).get('model') or 'Guidance-Model'

    def update_progress(info):
        """Update the live display with current status and reasoning."""
        status = info.get('status', '')
        message = info.get('message', '')
        iteration = info.get('iteration', 0)
        now = datetime.now().strftime('%H:%M:%S')
        
        if status == 'analyzing':
            prefix = random.choice([
                f"🧑‍🔧 {agent_model} pokes at code",
                f"🧑‍💻 {agent_model} combs through the lines",
                "Analyzing",
            ])
            event_log.append(f"[bright_yellow]{now}[/bright_yellow] [bold]Iter {iteration}[/bold] [bright_yellow]{prefix}[/bright_yellow]: {message}")
        elif status == 'decision':
            action = info.get('action', '-')
            reasoning = info.get('reasoning', '')  # Don't abbreviate thoughts
            params = _format_params(info.get('parameters', {}))
            tag = random.choice(["Decision", f"{agent_model} plots next move", "Game plan"])
            event_log.append(f"[bright_cyan]{now}[/bright_cyan] [bold]Iter {iteration}[/bold] [bright_cyan]{tag}[/bright_cyan]: action={action}\n"
                             f"  [dim]Thought:[/dim] {reasoning}\n  [dim]Params:[/dim] {params}")
        elif status == 'executing':
            tag = random.choice(["Executing", f"{agent_model} does the thing", "On it"])
            event_log.append(f"[bright_blue]{now}[/bright_blue] [bold]Iter {iteration}[/bold] [bright_blue]{tag}[/bright_blue]: {message}")
        elif status == 'result':
            res = info.get('result', {}) or {}
            summary = res.get('summary') or res.get('status') or message
            tag = random.choice(["Result", f"{agent_model} reports back", "Outcome"])
            event_log.append(f"[bright_green]{now}[/bright_green] [bold]Iter {iteration}[/bold] [bright_green]{tag}[/bright_green]: {_shorten(summary, 160)}")
        elif status == 'hypothesis_formed':
            tag = random.choice(["Hypothesis", f"{agent_model} has a hunch", "Lead"])
            event_log.append(f"[bright_green]{now}[/bright_green] [bold]Iter {iteration}[/bold] [bright_green]{tag}[/bright_green]: {message}")
        elif status == 'code_loaded':
            tag = random.choice(["Code Loaded", f"{agent_model} stacks more context", "More code in"])
            event_log.append(f"[bright_blue]{now}[/bright_blue] [bold]Iter {iteration}[/bold] [bright_blue]{tag}[/bright_blue]: {message}")
        elif status == 'generating_report':
            tag = random.choice(["Report", f"{guidance_model} whispers advice", "Notes"])
            event_log.append(f"[bright_magenta]{now}[/bright_magenta] [bold]Iter {iteration}[/bold] [bright_magenta]{tag}[/bright_magenta]: {message}")
        elif status == 'complete':
            tag = random.choice(["Complete", f"{guidance_model} signs off", "All done"])
            event_log.append(f"[bold bright_green]{now}[/bold bright_green] [bold]Iter {iteration}[/bold] [bold bright_green]{tag}[/bold bright_green]: {message}")
        else:
            event_log.append(f"[white]{now}[/white] [bold]Iter {iteration}[/bold] [white]{status or 'Working'}[/white]: {message}")
        
        live.update(_panel_from_events())
    
    with Live(_panel_from_events(), console=console, refresh_per_second=6, transient=True) as live:
        try:
            # Execute investigation with progress callback
            max_iters = iterations if iterations and iterations > 0 else 10
            report = agent.investigate(prompt, max_iterations=max_iters, progress_callback=update_progress)
            
            # Clear the live display
            live.stop()
            
            # Display results
            display_investigation_report(report)
            
            # Finalize debug log if in debug mode
            if debug and agent.debug_logger:
                log_path = agent.debug_logger.finalize(summary={
                    'total_iterations': report.get('iterations_completed', 0),
                    'hypotheses_tested': report['hypotheses']['total'],
                    'confirmed': report['hypotheses']['confirmed'],
                    'rejected': report['hypotheses']['rejected']
                })
                console.print(f"\n[cyan]Debug log saved:[/cyan] {log_path}")
            
        except Exception as e:
            console.print(f"[red]Investigation failed: {e}[/red]")
            if debug:
                import traceback
                console.print(traceback.format_exc())
                if agent and agent.debug_logger:
                    log_path = agent.debug_logger.finalize()
                    console.print(f"\n[cyan]Debug log saved:[/cyan] {log_path}")

def display_investigation_report(report: dict):
    """Display investigation report in a nice format."""
    console = Console()
    
    # Header
    console.print("\n[bold magenta]═══ INVESTIGATION REPORT ═══[/bold magenta]\n")
    
    # Goal and summary
    console.print(f"[bold]Investigation Goal:[/bold] {report['investigation_goal']}")
    console.print(f"[bold]Iterations:[/bold] {report['iterations_completed']}")
    console.print()
    
    # Hypothesis summary
    hyp_stats = report['hypotheses']
    console.print("[bold cyan]Hypotheses:[/bold cyan]")
    console.print(f"  • Total: {hyp_stats['total']}")
    console.print(f"  • [green]Confirmed: {hyp_stats['confirmed']}[/green]")
    console.print(f"  • [red]Rejected: {hyp_stats['rejected']}[/red]")
    console.print(f"  • [yellow]Uncertain: {hyp_stats['uncertain']}[/yellow]")
    console.print()
    
    # Detailed hypotheses
    if report.get('detailed_hypotheses'):
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Hypothesis", style="white", width=50, overflow="fold")
        table.add_column("Model", style="dim", width=20)
        table.add_column("Confidence", justify="center")
        table.add_column("Status", justify="center")
        
        for hyp in report['detailed_hypotheses']:
            # Color-code confidence
            conf = hyp['confidence']
            if conf >= 0.8:
                conf_style = "[bold green]"
            elif conf <= 0.2:
                conf_style = "[bold red]"
            else:
                conf_style = "[yellow]"
            
            # Status styling
            status = hyp['status']
            if status == 'confirmed':
                status_style = "[bold green]CONFIRMED[/bold green]"
            elif status == 'rejected':
                status_style = "[bold red]REJECTED[/bold red]"
            else:
                status_style = "[yellow]TESTING[/yellow]"
            
            # Get model info, fallback to "unknown" if not present
            model = hyp.get('reported_by_model', 'unknown')
            
            # Use full description - Rich will handle wrapping with overflow="fold"
            table.add_row(
                hyp['description'],  # Show full description
                model,
                f"{conf_style}{conf*100:.0f}%[/{conf_style.strip('[')}",
                status_style
            )
        
        console.print(table)
        console.print()
    
    # Conclusion
    console.print("[bold]Conclusion:[/bold]")
    conclusion = report.get('conclusion', 'No conclusion available')
    if 'LIKELY TRUE' in conclusion:
        console.print(f"  [green]{conclusion}[/green]")
    elif 'LIKELY FALSE' in conclusion:
        console.print(f"  [red]{conclusion}[/red]")
    else:
        console.print(f"  [yellow]{conclusion}[/yellow]")
    console.print()
    
    # Summary narrative
    if report.get('summary'):
        console.print("[bold]Summary:[/bold]")
        console.print(Panel(report['summary'], border_style="dim"))
    
    console.print("\n[dim]Investigation complete.[/dim]")


console = Console()


def format_tool_call(call):
    """Format a tool call for pretty display."""
    params_str = json.dumps(call.parameters, indent=2) if call.parameters else "{}"
    
    # Use different colors for different tool types
    tool_colors = {
        'focus': 'cyan',
        'query_graph': 'blue',
        'update_node': 'yellow',
        'propose_hypothesis': 'red',
        'update_hypothesis': 'magenta',
        'add_edge': 'green',
        'summarize': 'white'
    }
    
    color = tool_colors.get(call.tool_name, 'white')
    
    content = f"[bold]{call.description}[/bold]\n\n"
    if hasattr(call, 'reasoning') and call.reasoning:
        content += f"[italic yellow]Reasoning: {call.reasoning}[/italic yellow]\n\n"
    content += f"[dim]Tool:[/dim] [{color}]{call.tool_name}[/{color}]\n"
    if hasattr(call, 'priority'):
        content += f"[dim]Priority:[/dim] {call.priority}/10\n"
    content += f"[dim]Parameters:[/dim]\n{params_str}"
    
    return Panel(
        content,
        title="[bold cyan]Tool Call[/bold cyan]",
        border_style="cyan"
    )


def format_tool_result(result):
    """Format tool execution result."""
    # Defensive: ensure result is a dict
    if not isinstance(result, dict):
        result = {'status': 'error', 'error': 'No result'}
    if result.get('status') == 'success':
        style = "green"
        icon = "✓"
    else:
        style = "red"
        icon = "✗"
    
    # Extract key information based on result content
    details = []
    if 'focused_nodes' in result:
        details.append(f"Focused on {result['focused_nodes']} nodes")
    if 'code_cards_loaded' in result:
        details.append(f"Loaded {result['code_cards_loaded']} code cards")
    if 'matches' in result:
        details.append(f"Found {len(result['matches'])} matches")
    if 'hypothesis_id' in result:
        details.append(f"Hypothesis: {result['hypothesis_id'][:8]}...")
    if 'updates' in result:
        details.append(f"Applied {len(result['updates'])} updates")
    
    details_str = "\n".join(f"  • {d}" for d in details) if details else json.dumps(result, indent=2)
    
    return Panel(
        f"[{style}]{icon} {result.get('status', 'unknown').upper()}[/{style}]\n\n{details_str}",
        title="[bold]Result[/bold]",
        border_style=style
    )


def display_planning_phase(agent, items):
    """Display the planning phase output for investigations or tool calls."""
    
    # Check if we have investigations or tool calls
    if items and hasattr(items[0], 'goal'):  # Investigation objects
        console.print("\n[bold cyan]═══ INVESTIGATION PLANNING ═══[/bold cyan]")
        console.print(f"Planning [bold]{len(items)}[/bold] investigations...\n")
        
        # Create a table of planned investigations
        table = Table(title="High-Level Investigations", show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=4)
        table.add_column("Goal", style="cyan", width=50)
        table.add_column("Focus Areas", style="yellow", width=30)
        table.add_column("Priority", justify="center")
        
        for i, inv in enumerate(items, 1):
            # Color code priority
            if inv.priority >= 8:
                priority_style = "[bold red]"
            elif inv.priority >= 5:
                priority_style = "[yellow]"
            else:
                priority_style = "[dim]"
            
            table.add_row(
                str(i),
                inv.goal[:50],
                ', '.join(inv.focus_areas[:2]) if inv.focus_areas else "-",
                f"{priority_style}{inv.priority}[/]"
            )
        
        console.print(table)
        
        # Show reasoning for each investigation
        for i, inv in enumerate(items, 1):
            console.print(f"\n[bold]{i}. {inv.goal}[/bold]")
            console.print(f"   [dim]Reasoning: {inv.reasoning}[/dim]")
    
    else:  # ToolCall objects (backward compatibility)
        console.print("\n[bold cyan]═══ PLANNING PHASE ═══[/bold cyan]")
        console.print(f"Planning [bold]{len(items)}[/bold] next steps...\n")
        
        # Create a table of planned actions
        table = Table(title="Planned Actions", show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=4)
        table.add_column("Tool", style="cyan")
        table.add_column("Description", style="white", width=50)
        table.add_column("Reasoning", style="yellow", width=30)
        
        for i, call in enumerate(items, 1):
            table.add_row(
                str(i),
                call.tool_name,
                call.description[:50] if call.description else "-",
                call.reasoning[:30] if call.reasoning else "-"
            )
        
        console.print(table)


def display_execution_phase(call, result):
    """Display execution of a single tool call."""
    console.print("\n[bold green]═══ EXECUTING ═══[/bold green]")
    console.print(format_tool_call(call))
    console.print(format_tool_result(result))


def display_agent_summary(summary, time_limit_reached=False):
    """Display final agent summary with detailed findings."""
    
    # Header based on completion reason
    if time_limit_reached:
        console.print("\n[bold yellow]═══ TIME LIMIT REACHED - AGENT REPORT ═══[/bold yellow]\n")
    else:
        console.print("\n[bold magenta]═══ AGENT SUMMARY ═══[/bold magenta]\n")
    
    # Basic statistics
    summary_text = f"""
[bold]Agent ID:[/bold] {summary['agent_id']}
[bold]Iterations Completed:[/bold] {summary['iterations']}
[bold]Tool Calls Executed:[/bold] {summary['tool_calls_completed']}

[bold cyan]Graph Statistics:[/bold cyan]
  • Nodes Analyzed: {summary['graph_stats'].get('num_nodes', 0)}
  • Edges Traced: {summary['graph_stats'].get('num_edges', 0)}
  • Observations Added: {summary['graph_stats'].get('observations', 0)}
  • Invariants Added: {summary['graph_stats'].get('invariants', 0)}
  
[bold yellow]Security Findings:[/bold yellow]
  • Hypotheses Proposed: {summary['hypotheses']['total']}
  • Confirmed Vulnerabilities: {summary['hypotheses']['confirmed']}
"""
    
    console.print(Panel(summary_text, title="[bold]Statistics[/bold]", border_style="cyan"))
    
    # Detailed hypotheses if any exist
    if summary.get('all_hypotheses'):
        console.print("\n[bold red]VULNERABILITY HYPOTHESES:[/bold red]")
        
        # Create a table for hypotheses
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("ID", style="dim", width=12)
        table.add_column("Node", style="yellow", width=20, overflow="ellipsis")
        table.add_column("Type", style="cyan", width=15)
        table.add_column("Description", style="white", width=45, overflow="fold")
        table.add_column("Model", style="dim", width=15, overflow="ellipsis")
        table.add_column("Confidence", justify="center")
        table.add_column("Status", justify="center")
        
        for hyp in summary['all_hypotheses'][:10]:  # Show top 10
            # Color code confidence
            conf = hyp.get('confidence', 0)
            if conf >= 0.8:
                conf_style = "[bold red]"
            elif conf >= 0.5:
                conf_style = "[yellow]"
            else:
                conf_style = "[dim]"
            
            # Status indicator
            status = hyp.get('status', 'investigating')
            if status == 'confirmed':
                status_display = "[bold red]CONFIRMED[/bold red]"
            elif status == 'rejected':
                status_display = "[dim]rejected[/dim]"
            else:
                status_display = "[yellow]investigating[/yellow]"
            
            # Get model info
            model = hyp.get('reported_by_model', 'unknown')
            
            # Show full description and let Rich handle wrapping
            table.add_row(
                hyp.get('id', 'unknown')[:12],
                hyp.get('node_id', 'unknown'),  # Will be ellipsized by Rich if too long
                hyp.get('vulnerability_type', 'unknown'),
                hyp.get('description', ''),  # Show full description
                model if model else 'unknown',  # Will be ellipsized by Rich if too long
                f"{conf_style}{conf:.2f}[/]",
                status_display
            )
        
        console.print(table)
    
    # Tool execution summary
    if summary.get('tool_execution_summary'):
        console.print("\n[bold green]TOOL EXECUTION SUMMARY:[/bold green]")
        
        tool_table = Table(show_header=True, header_style="bold cyan")
        tool_table.add_column("Tool", style="cyan")
        tool_table.add_column("Calls", justify="center")
        tool_table.add_column("Successful", justify="center", style="green")
        tool_table.add_column("Failed", justify="center", style="red")
        
        for tool_name, stats in summary['tool_execution_summary'].items():
            tool_table.add_row(
                tool_name,
                str(stats.get('total', 0)),
                str(stats.get('successful', 0)),
                str(stats.get('failed', 0))
            )
        
        console.print(tool_table)
    
    # Areas analyzed
    if summary.get('analyzed_areas'):
        console.print("\n[bold blue]AREAS ANALYZED:[/bold blue]")
        for area in summary['analyzed_areas']:
            console.print(f"  • {area['name']}: {area['description']}")
    
    # Key findings narrative
    if summary.get('key_findings'):
        console.print("\n[bold red]KEY FINDINGS:[/bold red]")
        for i, finding in enumerate(summary['key_findings'][:5], 1):
            console.print(f"\n  {i}. [yellow]{finding.get('title', 'Finding')}[/yellow]")
            console.print(f"     {finding.get('description', '')}")
            if finding.get('recommendation'):
                console.print(f"     [dim]Recommendation: {finding['recommendation']}[/dim]")
    
    console.print(Panel("", title="[bold]End of Report[/bold]", border_style="magenta"))


class AgentRunner:
    """Manages agent execution with beautiful output."""
    
    def __init__(self, project_id: str, config_path: Path | None = None, 
                 iterations: int | None = None, time_limit_minutes: int | None = None,
                 debug: bool = False, platform: str | None = None, model: str | None = None,
                 session: str | None = None, new_session: bool = False, mode: str | None = None):
        self.project_id = project_id
        self.config_path = config_path
        self.max_iterations = iterations
        self.time_limit_minutes = time_limit_minutes
        self.mode = mode  # 'sweep', 'intuition', or None for auto
        self.debug = debug
        self.platform = platform
        self.model = model
        self.agent = None
        self.start_time = None
        self.completed_investigations = []  # Track completed investigation goals
        self.session_tracker: SessionTracker | None = None
        self.project_dir: Path | None = None
        self.plan_store = None
        self.session_id: str | None = session
        self.new_session: bool = new_session
        self._agent_log: list[str] = []
        self._last_applied_steer: str | None = None
        # Track which steering text triggered a forced replan (to avoid repeats)
        self._last_replan_steer: str | None = None
        # Cache of graph node IDs to avoid re-reading files repeatedly
        self._known_node_ids_cache: set[str] | None = None
        self._node_to_graph_map_cache: dict[str, str] | None = None
        # Track current audit phase (Early/Mid/Late) for display and planning hints
        self._current_phase: str | None = None
        
    def initialize(self):
        """Initialize the agent."""
        # First check if project_id is actually a path to a project directory
        if '/' in self.project_id or Path(self.project_id).exists():
            # It's a path to the project output
            project_dir = Path(self.project_id)
            if not project_dir.is_absolute():
                project_dir = project_dir.resolve()
        else:
            # It's a project ID, use default location
            project_dir = get_project_dir(self.project_id)
        
        # Look for the knowledge graphs metadata file
        graphs_dir = project_dir / "graphs"
        knowledge_graphs_path = graphs_dir / "knowledge_graphs.json"
        manifest_path = project_dir / "manifest"
        
        # If knowledge_graphs.json doesn't exist, look for any graph file
        if knowledge_graphs_path.exists():
            # Prefer SystemArchitecture, then SystemOverview, otherwise first available
            with open(knowledge_graphs_path) as f:
                graphs_meta = json.load(f)
            if graphs_meta.get('graphs'):
                graphs_dict = graphs_meta['graphs']
                # Prefer SystemArchitecture first
                if 'SystemArchitecture' in graphs_dict:
                    graph_path = Path(graphs_dict['SystemArchitecture'])
                # Then fallback to SystemOverview
                elif 'SystemOverview' in graphs_dict:
                    graph_path = Path(graphs_dict['SystemOverview'])
                else:
                    # Use the first available graph
                    graph_name = list(graphs_dict.keys())[0]
                    graph_path = Path(graphs_dict[graph_name])
                # Guard against stale absolute paths pointing to another project dir
                try:
                    graphs_dir = project_dir / 'graphs'
                    if graph_path.is_absolute() and graphs_dir.exists():
                        if graph_path.parent != graphs_dir:
                            candidate = graphs_dir / graph_path.name
                            if candidate.exists():
                                graph_path = candidate
                except Exception:
                    pass
                console.print(f"[green]Using graph: {graph_path.name}[/green]")
            else:
                console.print("[red]Error:[/red] No graphs found in knowledge_graphs.json")
                return False
        elif graphs_dir.exists():
            # Fallback: look for any graph_*.json file, preferably SystemArchitecture then SystemOverview
            graph_files = list(graphs_dir.glob("graph_*.json"))
            if graph_files:
                # Prefer SystemArchitecture if it exists
                system_arch = graphs_dir / "graph_SystemArchitecture.json"
                if system_arch.exists():
                    graph_path = system_arch
                # Then prefer SystemOverview
                elif (graphs_dir / "graph_SystemOverview.json").exists():
                    graph_path = graphs_dir / "graph_SystemOverview.json"
                else:
                    graph_path = graph_files[0]
                console.print(f"[yellow]Using graph: {graph_path.name}[/yellow]")
            else:
                console.print(f"[red]Error:[/red] No graph files found in {graphs_dir}")
                return False
        else:
            console.print(f"[red]Error:[/red] No graphs directory found at {graphs_dir}")
            console.print("[yellow]Run 'hound build' first or check the path.[/yellow]")
            return False
        
        if not graph_path.exists():
            console.print(f"[red]Error:[/red] Graph file not found: {graph_path}")
            return False
        
        # Require SystemArchitecture graph before starting an audit
        try:
            sys_graph = graphs_dir / 'graph_SystemArchitecture.json'
            if not sys_graph.exists():
                console.print("[red]Error: SystemArchitecture graph not found for this project.[/red]")
                console.print("[yellow]Run one of:\n  ./hound.py graph build <project> --init --iterations 1\n  ./hound.py graph build <project> --auto --iterations 2[/yellow]")
                return False
        except Exception:
            console.print("[red]Error while checking SystemArchitecture graph.[/red]")
            console.print("[yellow]Rebuild graphs with '--init' or '--auto'.[/yellow]")
            return False
        
        # Load config properly using the standard method
        from utils.config_loader import load_config
        if self.config_path and self.config_path.exists():
            config = load_config(self.config_path)
        else:
            config = load_config()  # Uses default config.yaml
        
        # Override platform and model if provided
        if self.platform or self.model:
            # Ensure the models.agent structure exists
            if 'models' not in config:
                config['models'] = {}
            if 'agent' not in config['models']:
                config['models']['agent'] = {}
            
            if self.platform:
                config['models']['agent']['provider'] = self.platform
                console.print(f"[cyan]Overriding agent provider: {self.platform}[/cyan]")
            if self.model:
                config['models']['agent']['model'] = self.model
                console.print(f"[cyan]Overriding agent model: {self.model}[/cyan]")
        
        # Keep config for planning
        self.config = config
        # Remember project_dir for plan storage
        self.project_dir = project_dir
        
        # Validate required models before creating the agent
        if not _validate_required_models(self.config, console):
            return False

        # Create agent with knowledge graphs metadata
        self.agent = Scout(
            graphs_metadata_path=knowledge_graphs_path,
            manifest_path=manifest_path,
            agent_id=f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config,  # Pass the loaded config dict
            debug=self.debug,
            session_id=self.session_id
        )
        # Ensure overarching mission is visible to the agent/strategist
        try:
            if getattr(self, 'mission', None):
                self.agent.mission = self.mission
        except Exception:
            pass

        # Set debug flag and route per-interaction files to project .debug
        self.agent.debug = self.debug
        if self.debug:
            try:
                from analysis.debug_logger import DebugLogger as _Dbg
                dbg_dir = self.project_dir / '.debug'
                dbg = _Dbg(self.session_id or self.agent.agent_id, output_dir=dbg_dir)
                # Attach to agent and its LLM clients so all prompts/responses are captured
                self.agent.debug_logger = dbg
                if hasattr(self.agent, 'llm') and self.agent.llm:
                    self.agent.llm.debug_logger = dbg
                if hasattr(self.agent, 'guidance_client') and self.agent.guidance_client:
                    self.agent.guidance_client.debug_logger = dbg
            except Exception:
                pass
        
        if self.max_iterations:
            self.agent.max_iterations = self.max_iterations
        
        # Initialize plan store and session directory
        try:
            from analysis.plan_store import PlanStatus, PlanStore
            from analysis.session_manager import SessionManager
            # Create or find session dir
            sm = SessionManager(self.project_dir)
            if not self.session_id:
                # Default session ID to agent ID when not provided
                self.session_id = f"sess_{self.agent.agent_id}"
            sinfo = sm.get_or_create(self.session_id, new_session=self.new_session)
            # Persist normalized session id
            self.session_id = sinfo.session_id
            # Plan file in session directory
            plan_path = sinfo.path / "plan.json"
            self.plan_store = PlanStore(plan_path, agent_id=f"runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            # Reset any stale in-progress items to planned (resume-friendly)
            try:
                inprog = self.plan_store.list(session_id=self.session_id, status=PlanStatus.IN_PROGRESS)
                for it in inprog:
                    fid = it.get('frame_id')
                    if fid:
                        self.plan_store.update_status(fid, PlanStatus.PLANNED, rationale='Resuming session: reset from in_progress')
            except Exception:
                pass
            # Write/update session state
            try:
                state_path = sinfo.path / 'state.json'
                state = {
                    'session_id': self.session_id,
                    'project_path': str(self.project_dir),
                    'created_at': datetime.now().isoformat(),
                    'models': self.config.get('models', {}) if self.config else {},
                }
                # Persist mission for visibility
                try:
                    if getattr(self, 'mission', None):
                        state['mission'] = self.mission
                except Exception:
                    pass
                import json as _json
                state_path.write_text(_json.dumps(state, indent=2))
            except Exception:
                pass
        except Exception:
            self.plan_store = None

        return True

    # ---------------------- Steering Helpers (persistent) ----------------------
    def _steer_cursor_path(self) -> Path:
        pdir = self.project_dir or (get_project_dir(self.project_id))
        return Path(pdir) / '.hound' / 'steering.cursor'

    def _get_last_consumed_steer_ts(self) -> float:
        try:
            p = self._steer_cursor_path()
            if p.exists():
                v = p.read_text(encoding='utf-8').strip()
                return float(v or '0')
        except Exception:
            return 0.0
        return 0.0

    def _set_last_consumed_steer_ts(self, ts: float):
        try:
            p = self._steer_cursor_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(str(float(ts)), encoding='utf-8')
        except Exception:
            pass

    def _read_steering_entries(self, limit: int = 50) -> list:
        """Read recent steering JSONL entries as list of dicts with {ts, text}.
        Ignores malformed lines; returns newest-last (chronological) slice.
        """
        try:
            pdir = self.project_dir or get_project_dir(self.project_id)
            sfile = Path(pdir) / '.hound' / 'steering.jsonl'
            if not sfile.exists():
                return []
            out = []
            with sfile.open('r', encoding='utf-8', errors='ignore') as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        obj = json.loads(ln)
                        ts = float(obj.get('ts') or 0.0)
                        txt = (obj.get('text') or obj.get('message') or obj.get('note') or '').strip()
                        if txt:
                            out.append({'ts': ts, 'text': txt})
                    except Exception:
                        # fallback for raw lines
                        out.append({'ts': 0.0, 'text': ln})
            # Keep only the last N
            return out[-limit:]
        except Exception:
            return []

    def _find_latest_urgent_steer(self) -> dict | None:
        """Return the newest steering entry interpreted as actionable (urgent),
        newer than the last consumed timestamp. Treats both explicit verbs and
        broad/global directives as urgent.
        """
        last_ts = self._get_last_consumed_steer_ts()
        entries = self._read_steering_entries(limit=80)
        # newest first
        for ent in reversed(entries):
            ts = float(ent.get('ts') or 0.0)
            txt = (ent.get('text') or '').strip()
            if ts <= last_ts:
                continue
            low = txt.lower()
            # Broaden detection to common phrasing and typos
            verbs = (
                'investigate', 'investtgate', 'investigate', 'check', 'look at', 'look into',
                'focus on', 'analyze', 'audit', 'review', 'examine', 'scan', 'probe', 'dig into',
                'right now', 'next', 'please investigate', 'please check'
            )
            globals = (
                'whole app', 'entire app', 'entire codebase', 'whole codebase', 'all contracts',
                'every contract', 'system-wide', 'system wide', 'project-wide', 'project wide',
                'across the codebase', 'across the repo', 'across modules', 'end-to-end', 'e2e',
                'globally', 'everywhere', 'full audit', 'full review', 'scan the entire', 'scan all'
            )
            if any(k in low for k in verbs) or any(k in low for k in globals):
                return {'ts': ts, 'text': txt}
        return None

    def _consume_steer(self, ts: float):
        if ts and ts > self._get_last_consumed_steer_ts():
            self._set_last_consumed_steer_ts(ts)

    # ---------------------- Dashboard Helpers ----------------------
    def _get_hypotheses_summary(self) -> str:
        """Get a summary of current hypotheses for the Strategist."""
        try:
            import json as _json
            from pathlib import Path as _Path
            hyp_file = (_Path(self.project_dir) / 'hypotheses.json') if self.project_dir else None
            if hyp_file and hyp_file.exists():
                data = _json.loads(hyp_file.read_text())
                hyps = data.get('hypotheses', {})
                if not hyps:
                    return "No hypotheses formed yet"
                
                summary_parts = []
                for hyp_id, h in list(hyps.items())[:10]:  # Limit to 10 most recent
                    status = h.get('status', 'proposed')
                    severity = h.get('severity', 'unknown')
                    confidence = h.get('confidence', 'unknown')
                    desc = h.get('description', '')[:100]  # Truncate long descriptions
                    summary_parts.append(f"• [{status}] {desc} (severity: {severity}, confidence: {confidence})")
                
                if len(hyps) > 10:
                    summary_parts.append(f"... and {len(hyps) - 10} more hypotheses")
                
                return "\n".join(summary_parts)
            return "No hypotheses file found"
        except Exception:
            return "Error reading hypotheses"
    
    def _get_investigation_results_summary(self) -> list[str]:
        """Get investigation results with findings, not just goals."""
        if not self.session_tracker:
            return list(self.completed_investigations)
        
        # Get investigations from session with their results
        session_data = self.session_tracker.session_data
        investigations = session_data.get('investigations', [])
        
        results = []
        seen_goals = set()  # Track goals to avoid duplicates
        
        for inv in investigations:
            goal = inv.get('goal', '')
            if goal in seen_goals:
                continue
            seen_goals.add(goal)
            
            hypotheses = inv.get('hypotheses', {})
            total_hyp = hypotheses.get('total', 0)
            iterations = inv.get('iterations_completed', 0)
            
            # Format: "Goal (X iterations, Y hypotheses found)"
            result_str = f"{goal} ({iterations} iterations, {total_hyp} hypotheses)"
            results.append(result_str)
        
        # Also add any completed investigations not yet in session
        for goal in self.completed_investigations:
            if goal not in seen_goals:
                results.append(goal)
                seen_goals.add(goal)
        
        return results
    
    def _hypothesis_stats(self) -> dict:
        """Return hypothesis stats from project hypotheses.json."""
        stats = {"total": 0, "confirmed": 0, "rejected": 0, "uncertain": 0}
        try:
            import json as _json
            from pathlib import Path as _Path
            hyp_file = (_Path(self.project_dir) / 'hypotheses.json') if self.project_dir else None
            if hyp_file and hyp_file.exists():
                data = _json.loads(hyp_file.read_text())
                hyps = data.get('hypotheses', {})
                stats['total'] = len(hyps)
                for _, h in hyps.items():
                    st = h.get('status', 'proposed')
                    if st == 'confirmed':
                        stats['confirmed'] += 1
                    elif st == 'rejected':
                        stats['rejected'] += 1
                    else:
                        stats['uncertain'] += 1
        except Exception:
            pass
        return stats

    def _coverage_stats(self) -> dict:
        """Return coverage stats using CoverageIndex if available."""
        try:
            from analysis.coverage_index import CoverageIndex
            project_dir = self.project_dir
            graphs_dir = project_dir / 'graphs'
            manifest_dir = project_dir / 'manifest'
            cov = CoverageIndex(project_dir / 'coverage_index.json', agent_id='cli')
            return cov.compute_stats(graphs_dir, manifest_dir)
        except Exception:
            return {'nodes': {'total': 0, 'visited': 0, 'percent': 0.0}, 'cards': {'total': 0, 'visited': 0, 'percent': 0.0}}

    def _graph_summary(self) -> str:
        """Create a comprehensive summary of ALL graphs loaded by the Scout."""
        try:
            parts = []
            # Get all loaded graphs from the agent
            loaded_data = self.agent.loaded_data if self.agent else {}

            # Process system graph first
            if loaded_data.get('system_graph'):
                graph_data = loaded_data['system_graph']
                graph_name = graph_data.get('name', 'SYSTEM')
                g = graph_data.get('data', {})
                nodes = g.get('nodes', [])
                edges = g.get('edges', [])
                parts.append(f"\n=== {graph_name.upper()} GRAPH ===")
                parts.append(f"{len(nodes)} nodes, {len(edges)} edges")
                # List nodes
                for n in nodes:
                    # Filter non-application nodes
                    try:
                        nid0 = str(n.get('id','') or '')
                        lbl0 = str(n.get('label','') or '')
                        ty0 = str(n.get('type','') or '').lower()
                        if nid0.startswith('contract_I') or lbl0.startswith('I'):
                            continue
                        if 'test' in lbl0.lower() or 'mock' in lbl0.lower():
                            continue
                        if any(k in ty0 for k in ('test','mock')):
                            continue
                    except Exception:
                        pass
                    nid = n.get('id', '')
                    lbl = n.get('label') or nid
                    typ = n.get('type', '')[:4]
                    observations = n.get('observations', [])
                    assumptions = n.get('assumptions', [])
                    line = f"• [{nid}] {lbl} ({typ})"
                    annotations = []
                    if observations:
                        obs_str = '; '.join(observations[:-3])
                        annotations.append(f"obs:{obs_str}")
                    if assumptions:
                        assum_str = '; '.join(assumptions[:-3])
                        annotations.append(f"asm:{assum_str}")
                    if annotations:
                        line += f" [{' | '.join(annotations)}]"
                    parts.append(line)

            # Process additional graphs
            additional_graphs = loaded_data.get('graphs', {})
            for graph_name, graph_data in additional_graphs.items():
                if not isinstance(graph_data, dict) or 'data' not in graph_data:
                    continue
                g = graph_data.get('data', {})
                nodes = g.get('nodes', [])
                edges = g.get('edges', [])
                parts.append(f"\n=== {graph_name.upper()} GRAPH ===")
                parts.append(f"{len(nodes)} nodes, {len(edges)} edges")
                for n in nodes:
                    # Filter non-application nodes
                    try:
                        nid0 = str(n.get('id','') or '')
                        lbl0 = str(n.get('label','') or '')
                        ty0 = str(n.get('type','') or '').lower()
                        if nid0.startswith('contract_I') or lbl0.startswith('I'):
                            continue
                        if 'test' in lbl0.lower() or 'mock' in lbl0.lower():
                            continue
                        if any(k in ty0 for k in ('test','mock')):
                            continue
                    except Exception:
                        pass
                    nid = n.get('id', '')
                    lbl = n.get('label') or nid
                    typ = n.get('type', '')[:4]
                    observations = n.get('observations', [])
                    assumptions = n.get('assumptions', [])
                    line = f"• [{nid}] {lbl} ({typ})"
                    annotations = []
                    if observations:
                        obs_str = '; '.join(observations[:3])
                        annotations.append(f"obs:{obs_str}")
                    if assumptions:
                        assum_str = '; '.join(assumptions[:3])
                        annotations.append(f"asm:{assum_str}")
                    if annotations:
                        line += f" [{' | '.join(annotations)}]"
                    parts.append(line)

            return "\n".join(parts) if parts else "(no graphs available)"
        except Exception as e:
            return f"(error summarizing graphs: {str(e)})"

    def _plan_investigations(self, n: int) -> list[object]:
        """Plan next investigations using Strategist by default."""
        from types import SimpleNamespace
        # 0) Optional: honor recent steering as an urgent goal
        prepared: list[object] = []
        try:
            urgent_ent = self._find_latest_urgent_steer()
            if urgent_ent:
                urgent = urgent_ent['text']
                prepared.append(SimpleNamespace(
                    goal=urgent,
                    focus_areas=[],
                    priority=10,
                    reasoning='User steering: prioritize immediately',
                    category='suspicion',
                    expected_impact='high',
                    frame_id=None
                ))
                try:
                    pub = getattr(self, '_telemetry_publish', None)
                    if callable(pub):
                        pub({'type': 'status', 'message': f'steering goal queued: {urgent}'})
                except Exception:
                    pass
                # Mark that we have consumed this steering message
                try:
                    self._consume_steer(float(urgent_ent.get('ts') or 0.0))
                except Exception:
                    pass
        except Exception:
            pass
        # 1) Start with any existing PLANNED items in this session (resume-friendly)
        existing_frame_ids = set()
        ps = self.plan_store
        try:
            from analysis.plan_store import PlanStatus
        except Exception:
            ps = None
        if ps is not None and self.session_id:
            try:
                pending = ps.list(session_id=self.session_id, status=PlanStatus.PLANNED)
                # Respect priority order already returned by PlanStore.list
                for it in pending[:n]:
                    prepared.append(SimpleNamespace(
                        goal=it.get('question',''),
                        focus_areas=it.get('artifact_refs',[]) or [],
                        priority=int(it.get('priority',5)),
                        reasoning=it.get('rationale',''),
                        category='aspect',
                        expected_impact='medium',
                        frame_id=it.get('frame_id')
                    ))
                    if it.get('frame_id'):
                        existing_frame_ids.add(it['frame_id'])
            except Exception:
                pass

        if len(prepared) >= n:
            return prepared[:n]

        # Heuristics: identify "high-level" SystemArchitecture nodes (contracts/modules/services/files)
        def _high_level_nodes(sys_nodes: list[dict], sys_edges: list[dict]) -> list[dict]:
            try:
                # Build quick degree map
                deg: dict[str, int] = {}
                for e in sys_edges or []:
                    sid = str(e.get('source_id') or e.get('source') or '')
                    tid = str(e.get('target_id') or e.get('target') or '')
                    if sid:
                        deg[sid] = deg.get(sid, 0) + 1
                    if tid:
                        deg[tid] = deg.get(tid, 0) + 1
            except Exception:
                deg = {}

            hi_type_keywords = {
                'contract','component','module','service','subsystem','interface','controller','file','package','system','graph','library'
            }
            low_type_keywords = {
                'function','method','variable','state','field','param','event','modifier','mapping','struct','enum',
                'rule','configuration','config','setting','property','constraint','check','validation'
            }
            low_id_prefixes = ('func_', 'method_', 'state_', 'var_', 'evt_', 'event_', 'mod_', 'mapping_', 'param_',
                              'rule_', 'config_', 'ConfigurationRule_', 'ValidationRule_', 'Check_')

            def is_low_level(n: dict) -> bool:
                t = str(n.get('type','') or '').lower()
                i = str(n.get('id','') or '')
                lbl = str(n.get('label','') or '')
                if any(k in t for k in low_type_keywords):
                    return True
                # Check if ID starts with any low-level prefix
                if any(i.startswith(prefix) for prefix in low_id_prefixes):
                    return True
                # Also check if ID contains Rule, Config, Check, Validation patterns
                if any(pattern in i for pattern in ['Rule', 'Config', 'Check', 'Validation']):
                    return True
                # Looks like a function signature (foo(...))
                if '(' in lbl and ')' in lbl and ' ' not in lbl.split('(')[0]:
                    return True
                return False

            def is_high_level(n: dict) -> bool:
                if is_low_level(n):
                    return False
                t = str(n.get('type','') or '').lower()
                i = str(n.get('id','') or '')
                lbl = str(n.get('label','') or '')
                # Treat solidity interfaces as low-level targets for coverage planning
                # Conventionally encoded as contract_I<Name>
                try:
                    if i.startswith('contract_I') or lbl.startswith('I'):
                        return False
                except Exception:
                    pass
                if any(k in t for k in hi_type_keywords):
                    return True
                if i.startswith(('contract_', 'module_', 'service_', 'component_')):
                    return True
                # File-like labels (e.g., Foo.sol)
                if any(lbl.endswith(ext) for ext in ('.sol','.rs','.go','.py','.ts','.js','.tsx','.jsx')):
                    return True
                # Fallback: nodes with higher connectivity are likely architectural
                did = str(n.get('id','') or '')
                if deg.get(did, 0) >= 3:
                    return True
                return False

            out = [node for node in (sys_nodes or []) if is_high_level(node)]
            return out

        # 2) Ask Strategist for more to top-up to n
        graphs_summary = self._graph_summary()
        hypotheses_summary = self._get_hypotheses_summary()
        investigation_results = self._get_investigation_results_summary()
        coverage_summary = None
        cov_stats = None
        if self.session_tracker:
            cov_stats = self.session_tracker.get_coverage_stats()
            coverage_summary = (
                f"Nodes visited: {cov_stats['nodes']['visited']}/{cov_stats['nodes']['total']} "
                f"({cov_stats['nodes']['percent']:.1f}%)\n"
                f"Cards analyzed: {cov_stats['cards']['visited']}/{cov_stats['cards']['total']} "
                f"({cov_stats['cards']['percent']:.1f}%)"
            )
            if cov_stats['visited_node_ids']:
                try:
                    annotated_visited = self._annotate_nodes_with_graph(cov_stats['visited_node_ids'][:10])
                    coverage_summary += f"\nVisited nodes: {', '.join(annotated_visited)}"
                except Exception:
                    coverage_summary += f"\nVisited nodes: {', '.join(cov_stats['visited_node_ids'][:10])}"
                if len(cov_stats['visited_node_ids']) > 10:
                    coverage_summary += f" ... and {len(cov_stats['visited_node_ids']) - 10} more"
            # Append a concise list of unvisited node IDs to guide coverage
            try:
                unvisited_sample, unvisited_count = self._get_unvisited_nodes_sample(max_n=15)
                if unvisited_count > 0 and unvisited_sample:
                    try:
                        annotated_unvisited = self._annotate_nodes_with_graph(unvisited_sample[:10])
                        sample_str = ', '.join(annotated_unvisited)
                    except Exception:
                        sample_str = ', '.join(unvisited_sample[:10])
                    coverage_summary += f"\nUnvisited nodes: {unvisited_count} (sample: {sample_str}{'' if len(unvisited_sample) <= 10 else ', ...'})"
                # Also show unvisited high-level components (contracts/modules/services/files)
                try:
                    sys_graph = (self.agent.loaded_data or {}).get('system_graph') or {}
                    gdata = sys_graph.get('data') or {}
                    nodes = gdata.get('nodes') or []
                    edges = gdata.get('edges') or []
                    comp_nodes = _high_level_nodes(nodes, edges)
                    visited_ids = set(cov_stats.get('visited_node_ids') or [])
                    unv_comp = [node for node in comp_nodes if str(node.get('id') or '') not in visited_ids]
                    if unv_comp:
                        samp = [f"{(node.get('id') or '')}@SystemArchitecture" for node in unv_comp[:10] if node.get('id')]
                        coverage_summary += f"\nUnvisited components: {len(unv_comp)} (sample: {', '.join(samp)}{'' if len(unv_comp) <= 10 else ', ...'})"
                        # Add a structured candidate list to guide planner selection
                        lines = []
                        for node in unv_comp[:10]:
                            nid = str(node.get('id') or '')
                            lbl = str(node.get('label') or nid)
                            if nid:
                                lines.append(f"- {lbl} ({nid})")
                        if lines:
                            coverage_summary += "\nCANDIDATE COMPONENTS (unvisited):\n" + "\n".join(lines)
                except Exception:
                    pass
            except Exception:
                pass

        # Determine phase (Coverage/Saliency) from mode flag or coverage
        def _determine_phase_two(cov: dict | None) -> str:
            # If mode is explicitly set, use it
            if self.mode:
                if self.mode.lower() == 'sweep':
                    return 'Coverage'
                elif self.mode.lower() == 'intuition':
                    return 'Saliency'
            
            # Otherwise, use automatic determination based on coverage
            # Defaults
            cfg_planner = (self.config or {}).get('planner', {}) if self.config else {}
            two_cfg = (cfg_planner.get('two_phase') or {}) if isinstance(cfg_planner, dict) else {}
            start_cfg = (two_cfg.get('phase2_start') or {}) if isinstance(two_cfg, dict) else {}
            try:
                nodes_threshold = float(start_cfg.get('coverage_nodes_pct', 0.9)) * 100.0 if start_cfg.get('coverage_nodes_pct') <= 1 else float(start_cfg.get('coverage_nodes_pct'))
            except Exception:
                nodes_threshold = 90.0
            try:
                nodes_pct = float(((cov or {}).get('nodes') or {}).get('percent') or 0.0)
            except Exception:
                nodes_pct = 0.0

            # Compute component coverage on SystemArchitecture (contract/component/module)
            try:
                sys_graph = (self.agent.loaded_data or {}).get('system_graph') or {}
                gdata = sys_graph.get('data') or {}
                nodes = gdata.get('nodes') or []
                edges = gdata.get('edges') or []
                comps = _high_level_nodes(nodes, edges)
                visited_ids = set((cov_stats or {}).get('visited_node_ids') or []) if 'cov_stats' in locals() else set()
                visited_comp = [node for node in comps if str(node.get('id') or '') in visited_ids]
                comp_complete = (len(comps) > 0 and len(visited_comp) >= len(comps))
            except Exception:
                comp_complete = False

            if comp_complete or nodes_pct >= nodes_threshold:
                return 'Saliency'
            return 'Coverage'

        self._current_phase = _determine_phase_two(cov_stats)

        strategist = Strategist(config=self.config, debug=self.debug, session_id=self.session_id)
        need = max(0, n - len(prepared))
        planned: list[dict] = []
        # Strict coverage pre-planning in Early phase: iterate SystemArchitecture components
        try:
            strict_cfg = ((self.config or {}).get('planner', {}) or {}).get('strict_coverage', {})
            strict_enabled = bool(strict_cfg.get('enabled', True))
            strict_phase_only = str(strict_cfg.get('phase', 'Coverage'))
            strict_mode = str(strict_cfg.get('mode', 'guide'))  # 'guide' (default) or 'inject'
        except Exception:
            strict_enabled = True
            strict_phase_only = 'Coverage'
            strict_mode = 'guide'

        if need > 0 and strict_enabled and strict_mode == 'inject' and (self._current_phase or 'Coverage') == strict_phase_only:
            try:
                sys_graph = (self.agent.loaded_data or {}).get('system_graph') or {}
                gdata = sys_graph.get('data') or {}
                nodes = gdata.get('nodes') or []
                edges = gdata.get('edges') or []
                # Use visited set from coverage stats
                visited_ids = set((cov_stats or {}).get('visited_node_ids') or []) if 'cov_stats' in locals() else set()
                comp_nodes = _high_level_nodes(nodes, edges)
                unvisited = [node for node in comp_nodes if str(node.get('id') or '') not in visited_ids]
                for nnode in unvisited[:need]:
                    nid = str(nnode.get('id') or '')
                    lbl = str(nnode.get('label') or nid)
                    planned.append({
                        'goal': f"System component review: {lbl} ({nid})",
                        'focus_areas': [f"{nid}@SystemArchitecture"],
                        'priority': 7,
                        'reasoning': 'Strict coverage: map entrypoints, auth, and value/state flows; annotate nodes; avoid deep dives.',
                        'category': 'aspect',
                        'expected_impact': 'medium',
                    })
            except Exception:
                pass

        rem = max(0, need - len(planned))
        if rem > 0:
            extra = strategist.plan_next(
                graphs_summary=graphs_summary,
                completed=investigation_results,
                hypotheses_summary=hypotheses_summary,
                coverage_summary=coverage_summary,
                n=rem,
                phase_hint=self._current_phase
            ) or []
            # Enforce Coverage-phase constraint: keep only aspect items
            if (self._current_phase or 'Coverage') == 'Coverage':
                extra = [d for d in extra if str(d.get('category','aspect')).lower() == 'aspect']
            planned.extend(extra)

        # Final enforcement: in Coverage, ensure all planned items are aspects; if empty, fallback to component injection
        if (self._current_phase or 'Coverage') == 'Coverage':
            planned = [d for d in planned if str(d.get('category','aspect')).lower() == 'aspect']
            if not planned and strict_enabled and strict_phase_only == 'Coverage':
                try:
                    sys_graph = (self.agent.loaded_data or {}).get('system_graph') or {}
                    gdata = sys_graph.get('data') or {}
                    nodes = gdata.get('nodes') or []
                    edges = gdata.get('edges') or []
                    visited_ids = set((cov_stats or {}).get('visited_node_ids') or []) if 'cov_stats' in locals() else set()
                    comp_nodes = _high_level_nodes(nodes, edges)
                    unvisited = [node for node in comp_nodes if str(node.get('id') or '') not in visited_ids]
                    for nnode in unvisited[:need]:
                        nid = str(nnode.get('id') or '')
                        lbl = str(nnode.get('label') or nid)
                        planned.append({
                            'goal': f"System component review: {lbl} ({nid})",
                            'focus_areas': [f"{nid}@SystemArchitecture"],
                            'priority': 7,
                            'reasoning': 'Coverage-phase fallback: map entrypoints, auth, and value/state flows; annotate nodes; avoid deep dives.',
                            'category': 'aspect',
                            'expected_impact': 'medium',
                        })
                except Exception:
                    pass

        # Normalize Coverage goals to 'Vulnerability analysis of <Component>' when possible
        if (self._current_phase or 'Coverage') == 'Coverage':
            try:
                sys_graph = (self.agent.loaded_data or {}).get('system_graph') or {}
                gdata = sys_graph.get('data') or {}
                node_by_id = {str(node.get('id') or ''): str(node.get('label') or node.get('id') or '') for node in (gdata.get('nodes') or [])}
                def _norm_goal(item):
                    g = str(item.get('goal') or '')
                    bad = ('map entrypoint', "map entrypoints", "enumerate", "list ", "inventory", "map constructor")
                    low = g.lower()
                    if any(b in low for b in bad):
                        # Try to derive component from focus_areas
                        fas = item.get('focus_areas') or []
                        comp = None
                        if fas:
                            first = str(fas[0])
                            nid = first.split('@')[0].strip()
                            comp = node_by_id.get(nid) or nid
                        if comp:
                            item['goal'] = f"Vulnerability analysis of {comp}"
                        else:
                            item['goal'] = "Vulnerability analysis of selected component"
                for it in planned:
                    _norm_goal(it)
            except Exception:
                pass

        if self.session_tracker and planned:
            self.session_tracker.add_planning(planned)

        # 3) Add new items, skipping duplicates and already-done/in-progress
        # Round-level duplicate filter: avoid multiple goals targeting the same focus areas
        seen_focus_keys: set[tuple[str, ...]] = set()
        def _is_interface_focus(fas: list[str]) -> bool:
            try:
                for fa in fas or []:
                    node_id = (fa.split('@')[0] if isinstance(fa, str) else '')
                    if str(node_id).startswith('contract_I'):
                        return True
            except Exception:
                return False
            return False

        # Normalize goal text to identify duplicates and non-application targets
        def _normalize_goal_text(text: str) -> str:
            s = (text or '').lower()
            # Remove phase prefixes and context suffixes
            s = s.replace('initial sweep: ', '').replace('system component review: ', '')
            for token in (' within the ', ' within ', ' (', ')'):
                s = s.replace(token, ' ')
            # Normalize common verbs
            s = s.replace('discover vulnerabilities in ', 'target ').replace('identify bugs in ', 'target ')
            s = ' '.join(s.split())
            return s

        def _non_app_goal(text: str) -> bool:
            t = (text or '')
            tl = t.lower()
            if ' interface ' in tl or tl.startswith('interface '):
                return True
            # If a capitalized token looks like an interface name e.g., IWhitelist
            try:
                import re
                if re.search(r'\bI[A-Z][A-Za-z0-9_]+', t):
                    return True
            except Exception:
                pass
            bad_words = ('mock', ' test', 'tests', 'example', 'demo', 'script', 'forge-std', 'openzeppelin', 'vendor')
            return any(w in tl for w in bad_words)

        seen_focus_keys: set[tuple[str, ...]] = set()
        seen_goal_keys: set[str] = set()

        for d in planned:
            if len(prepared) >= n:
                break
            frame_id = None
            skip = False
            # Skip explicit interface-only targets
            if _is_interface_focus(d.get('focus_areas') or []):
                continue
            # Skip duplicate focus sets within this planning round (regardless of goal wording)
            try:
                key = tuple(sorted(d.get('focus_areas') or []))
                if key in seen_focus_keys:
                    continue
                seen_focus_keys.add(key)
            except Exception:
                pass
            # Skip non-application targets based on goal text heuristics
            if _non_app_goal(d.get('goal', '')):
                continue
            # Skip near-duplicate goals by normalized text
            gkey = _normalize_goal_text(d.get('goal', ''))
            if gkey in seen_goal_keys:
                continue
            seen_goal_keys.add(gkey)
            if ps is not None:
                try:
                    ok, fid = ps.propose(
                        session_id=self.session_id or self.agent.agent_id,
                        question=d.get('goal', ''),
                        artifact_refs=d.get('focus_areas') or [],
                        priority=int(d.get('priority', 5)),
                        rationale=d.get('reasoning', ''),
                        created_by='strategist'
                    )
                    frame_id = fid
                    if not ok:
                        # Existing frame; decide based on its status
                        existing = ps.get(fid)
                        status = (existing or {}).get('status', 'planned')
                        if status in {'done', 'in_progress'}:
                            skip = True
                        elif status == 'planned' and fid in existing_frame_ids:
                            skip = True
                    if not skip:
                        try:
                            from analysis.plan_ledger import PlanLedger
                            model_sig = None
                            try:
                                strat_cfg = (self.config or {}).get('models', {}).get('strategist')
                                if strat_cfg:
                                    model_sig = f"{strat_cfg.get('provider','unknown')}:{strat_cfg.get('model','unknown')}"
                            except Exception:
                                model_sig = None
                            if self.project_dir:
                                ledger = PlanLedger(self.project_dir / 'plan_ledger.json', agent_id='planner')
                                ledger.record(self.session_id or 'unknown', d.get('goal',''), d.get('focus_areas') or [], model_sig)
                        except Exception:
                            pass
                except Exception:
                    frame_id = None
            if skip:
                continue
            prepared.append(SimpleNamespace(
                goal=d.get('goal', ''),
                focus_areas=d.get('focus_areas', []),
                priority=d.get('priority', 5),
                reasoning=d.get('reasoning', ''),
                category=d.get('category', 'aspect'),
                expected_impact=d.get('expected_impact', 'medium'),
                frame_id=frame_id
            ))
            if frame_id:
                existing_frame_ids.add(frame_id)
        return prepared

    def _get_unvisited_nodes_sample(self, max_n: int = 15) -> tuple[list[str], int]:
        """Compute a sample of unvisited node IDs from graphs vs session coverage.

        Returns (sample_list, total_unvisited_count).
        """
        try:
            # Build known nodes cache if missing
            if self._known_node_ids_cache is None or self._node_to_graph_map_cache is None:
                all_nodes: set[str] = set()
                node_to_graph: dict[str, str] = {}
                graphs_dir = (self.project_dir or Path.cwd()) / 'graphs'
                if graphs_dir.exists():
                    import json as _json
                    for gfile in graphs_dir.glob('graph_*.json'):
                        try:
                            gd = _json.loads(Path(gfile).read_text())
                            gname = gd.get('internal_name') or gd.get('name') or gfile.stem.replace('graph_', '')
                            for n in gd.get('nodes', []) or []:
                                nid = n.get('id')
                                if nid:
                                    sid = str(nid)
                                    all_nodes.add(sid)
                                    if sid not in node_to_graph:
                                        node_to_graph[sid] = str(gname)
                        except Exception:
                            continue
                self._known_node_ids_cache = all_nodes
                self._node_to_graph_map_cache = node_to_graph
            visited = set()
            try:
                stats = self.session_tracker.get_coverage_stats() if self.session_tracker else {}
                visited = set(stats.get('visited_node_ids') or [])
            except Exception:
                visited = set()
            unvisited = list((self._known_node_ids_cache or set()) - visited)
            unvisited.sort()  # deterministic
            return (unvisited[:max_n], len(unvisited))
        except Exception:
            return ([], 0)

    def _annotate_nodes_with_graph(self, node_ids: list[str]) -> list[str]:
        """Return node ids annotated with their graph name as nid@Graph.\n\n        If graph is unknown, use '?' as placeholder.
        """
        try:
            # Ensure mapping cache exists
            if self._node_to_graph_map_cache is None or self._known_node_ids_cache is None:
                self._get_unvisited_nodes_sample(0)  # builds caches
            m = self._node_to_graph_map_cache or {}
            return [f"{nid}@{m.get(nid, '?')}" for nid in node_ids]
        except Exception:
            return list(node_ids)

        # NOTE: A legacy direct-LLM planning block once lived here referencing an undefined 'n'.
        # It has been removed in favor of the Strategist-based planner in _plan_investigations().

    def _render_checklist(self, items: list[object], completed_index: int = -1):
        """Render a simple checklist; items up to completed_index are checked."""
        console.print("\n[bold cyan]Investigation Checklist[/bold cyan]")
        for i, it in enumerate(items):
            mark = "[green][x][/green]" if i <= completed_index else "[ ]"
            pr = getattr(it, 'priority', 0)
            imp = getattr(it, 'expected_impact', None)
            cat = getattr(it, 'category', None)
            meta = f"prio {pr}"
            if imp:
                meta += f", {imp}"
            if cat:
                meta += f", {cat}"
            goal = getattr(it, 'goal', '')
            # Phase 1 investigations are already properly formatted
            console.print(f"  {mark} {goal}  ({meta})")

    def _log_planning_status(self, items: list[object], current_index: int = -1):
        """Log beautiful planning status and coverage information."""
        from rich.box import ROUNDED
        from rich.table import Table
        
        # Clear previous output for clean display
        console.print("\n" + "="*80)
        console.print("[bold cyan]STRATEGIST PLANNING & AUDIT STATUS[/bold cyan]")
        console.print("="*80)
        
        # Compact coverage line
        if self.session_tracker:
            cov = self.session_tracker.get_coverage_stats()
            try:
                sample, count = self._get_unvisited_nodes_sample(max_n=5)
                if count > 0 and sample:
                    sample = self._annotate_nodes_with_graph(sample)
            except Exception:
                sample, count = ([], 0)
            line = (
                f"Coverage: Nodes {cov['nodes']['visited']}/{cov['nodes']['total']} ({cov['nodes']['percent']:.1f}%) | "
                f"Cards {cov['cards']['visited']}/{cov['cards']['total']} ({cov['cards']['percent']:.1f}%)"
            )
            if count > 0 and sample:
                line += f" | Unvisited: {count} (sample: {', '.join(sample)})"
            console.print("\n" + line)
        else:
            console.print("\nCoverage: [dim]Not available[/dim]")
        
        # Hypothesis statistics
        hyp = self._hypothesis_stats()
        console.print("\n[bold yellow]Hypothesis Statistics:[/bold yellow]")
        console.print(f"  Total: {hyp['total']} | Confirmed: [green]{hyp['confirmed']}[/green] | Rejected: [red]{hyp['rejected']}[/red] | Pending: [yellow]{hyp['uncertain']}[/yellow]")
        
        # Current investigation status
        if current_index >= 0 and current_index < len(items):
            current_item = items[current_index]
            console.print(f"\n[bold magenta]Currently Investigating:[/bold magenta] {current_item.goal}")
        elif current_index == -1:
            console.print("\n[bold blue]Planning next investigations...[/bold blue]")
        
        # Investigation plan table
        if items:
            console.print("\n[bold yellow]Investigation Plan:[/bold yellow]")
            table = Table(show_header=True, header_style="bold magenta", box=ROUNDED)
            table.add_column("#", style="dim", width=3)
            table.add_column("Status", width=10)
            table.add_column("Phase", width=12)
            table.add_column("Goal", overflow="fold")
            table.add_column("Priority", width=8)
            table.add_column("Impact", width=8)
            table.add_column("Category", width=10)
            
            for i, it in enumerate(items):
                num = str(i + 1)
                if i == current_index:
                    status = "[bold yellow]ACTIVE[/bold yellow]"
                elif i < current_index:
                    status = "[green]DONE[/green]"
                else:
                    status = "[dim]PENDING[/dim]"
                
                goal = getattr(it, 'goal', '')
                category = getattr(it, 'category', '-')
                # Determine phase from current audit phase only (category is shown separately)
                current_phase = getattr(self, '_current_phase', None)
                if current_phase == 'Coverage':
                    phase_label = "[cyan]1 - Sweep[/cyan]"
                elif current_phase == 'Saliency':
                    phase_label = "[magenta]2 - Intuition[/magenta]"
                else:
                    phase_label = "[dim]-[/dim]"
                priority = str(getattr(it, 'priority', '-'))
                impact = getattr(it, 'expected_impact', '-')
                
                table.add_row(num, status, phase_label, goal, priority, impact, category)
            
            console.print(table)
        
        # Model activity log
        if hasattr(self, '_agent_log') and self._agent_log:
            recent_logs = self._agent_log[-5:]  # Show last 5 entries
            if recent_logs:
                console.print("\n[bold yellow]Recent Model Activity:[/bold yellow]")
                for entry in recent_logs:
                    console.print(f"  [dim]{entry}[/dim]")
        
        console.print("="*80 + "\n")

    def run(self, plan_n: int = 5):
        """Run the agent using the unified autonomous flow."""
        # Initialize session tracker
        if '/' in self.project_id or Path(self.project_id).exists():
            project_dir = Path(self.project_id).resolve()
        else:
            project_dir = get_project_dir(self.project_id)
        
        # Use sessions directory instead of agent_runs
        sessions_dir = project_dir / "sessions"
        sessions_dir.mkdir(exist_ok=True, parents=True)
        
        # Generate session ID if not provided
        if not self.session_id:
            self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.agent.agent_id}"
        
        # Initialize session tracker
        self.session_tracker = SessionTracker(sessions_dir, self.session_id)
        # Mark session as active when attached/started
        try:
            self.session_tracker.set_status('active')
        except Exception:
            pass
        
        # Initialize coverage tracking
        graphs_dir = project_dir / "graphs"
        manifest_dir = project_dir / "manifest"
        self.session_tracker.initialize_coverage(graphs_dir, manifest_dir)
        
        # Set up token tracker
        token_tracker = get_token_tracker()
        token_tracker.reset()
        
        # Display configuration (omit context window; not available in unified client)
        # Get the actual models being used from the agent's LLM clients
        agent_model_info = "unknown/unknown"
        guidance_model_info = "unknown/unknown"
        lightweight_model_info = _format_model_sig(self.config.get('models', {}), 'lightweight') if self.config else 'unknown/unknown'
        
        # Get Scout model info from the actual LLM client
        if hasattr(self.agent, 'llm') and self.agent.llm:
            try:
                provider_name = self.agent.llm.provider.provider_name if hasattr(self.agent.llm.provider, 'provider_name') else 'unknown'
                model_name = self.agent.llm.model if hasattr(self.agent.llm, 'model') else 'unknown'
                agent_model_info = f"{provider_name}/{model_name}"
            except Exception:
                # Fall back to config if we can't get from LLM client
                if self.config and 'models' in self.config:
                    if 'agent' in self.config['models']:
                        agent_config = self.config['models']['agent']
                        provider = agent_config.get('provider', 'unknown')
                        model = agent_config.get('model', 'unknown')
                        agent_model_info = f"{provider}/{model}"
                    elif 'scout' in self.config['models']:
                        scout_config = self.config['models']['scout']
                        provider = scout_config.get('provider', 'unknown')
                        model = scout_config.get('model', 'unknown')
                        agent_model_info = f"{provider}/{model}"
        
        # Get Strategist model info from the actual guidance client
        if hasattr(self.agent, 'guidance_client') and self.agent.guidance_client:
            try:
                provider_name = self.agent.guidance_client.provider.provider_name if hasattr(self.agent.guidance_client.provider, 'provider_name') else 'unknown'
                model_name = self.agent.guidance_client.model if hasattr(self.agent.guidance_client, 'model') else 'unknown'
                guidance_model_info = f"{provider_name}/{model_name}"
            except Exception:
                # Fall back to config if we can't get from guidance client
                if self.config and 'models' in self.config:
                    if 'strategist' in self.config['models']:
                        strategist_config = self.config['models']['strategist']
                        provider = strategist_config.get('provider', 'unknown')
                        model = strategist_config.get('model', 'unknown')
                        guidance_model_info = f"{provider}/{model}"
                    elif 'guidance' in self.config['models']:
                        guidance_config = self.config['models']['guidance']
                        provider = guidance_config.get('provider', 'unknown')
                        model = guidance_config.get('model', 'unknown')
                        guidance_model_info = f"{provider}/{model}"
        
        # Store models in session tracker
        self.session_tracker.set_models(agent_model_info, guidance_model_info)
        
        # Get context limit from config
        context_cfg = self.config.get('context', {}) if self.config else {}
        max_tokens = context_cfg.get('max_tokens', 128000)
        compression_threshold = context_cfg.get('compression_threshold', 0.75)
        
        config_text = (
            f"[bold cyan]AUTONOMOUS SECURITY AGENT[/bold cyan]\n"
            f"Project: [yellow]{self.project_id}[/yellow]\n"
            f"Scout: [magenta]{agent_model_info}[/magenta]\n"
            f"Strategist: [cyan]{guidance_model_info}[/cyan]\n"
            f"Lightweight: [green]{lightweight_model_info}[/green]\n"
            f"Context Limit: [blue]{max_tokens:,} tokens[/blue] (compress at {int(compression_threshold*100)}%)"
        )
        if self.time_limit_minutes:
            config_text += f"\nTime Limit: [red]{self.time_limit_minutes} minutes[/red]"
        console.print(Panel.fit(config_text, border_style="cyan"))
        # Early compact coverage snapshot
        try:
            if self.session_tracker:
                cov = self.session_tracker.get_coverage_stats()
                # Show a concise one-liner; no samples here to keep it compact
                console.print(
                    f"Coverage: Nodes {cov['nodes']['visited']}/{cov['nodes']['total']} ({cov['nodes']['percent']:.1f}%) | "
                    f"Cards {cov['cards']['visited']}/{cov['cards']['total']} ({cov['cards']['percent']:.1f}%)"
                )
        except Exception:
            pass

        # Enhanced progress callback with beautiful logging
        def progress_cb(info: dict):
            status = info.get('status', '')
            msg = info.get('message', '')
            it = info.get('iteration', 0)
            # Telemetry publish (best-effort)
            try:
                pub = getattr(self, '_telemetry_publish', None)
                if callable(pub):
                    pub({
                        'type': status or 'progress',
                        'iteration': it,
                        'message': msg,
                        'action': info.get('action'),
                        'parameters': info.get('parameters', {}),
                        'reasoning': info.get('reasoning', ''),
                    })
            except Exception:
                pass
            
            if status == 'decision':
                act = info.get('action', '-')
                reasoning = info.get('reasoning', '')
                params = info.get('parameters', {}) or {}
                
                # Log model actions and thoughts clearly
                console.print(f"\n[bold blue]Scout Model Decision (Iteration {it}):[/bold blue]")
                console.print(f"  [cyan]Action:[/cyan] {act}")
                if reasoning:
                    console.print(f"  [cyan]Thought:[/cyan] {reasoning}")
                
                # Special formatting for deep_think
                if act == 'deep_think':
                    console.print("\n[bold magenta]══════ CALLING STRATEGIST MODEL FOR DEEP ANALYSIS ══════[/bold magenta]")
                    console.print("[yellow]Strategist is analyzing the collected context...[/yellow]")
                elif params:
                    # Show parameters compactly for non-deep-think actions
                    try:
                        import json as _json
                        params_str = _json.dumps(params, separators=(',', ':'))
                        if len(params_str) > 200:
                            params_str = params_str[:197] + '...'
                        console.print(f"  [dim]Parameters: {params_str}[/dim]")
                    except Exception:
                        pass
                        
            elif status == 'result':
                # Special handling for deep_think results
                action = info.get('action', '')
                result = info.get('result', {})
                
                if action == 'deep_think':
                    # Robustness: handle unexpected result types gracefully
                    if not isinstance(result, dict):
                        error_msg = f"Unexpected strategist result type: {type(result).__name__}"
                        console.print(f"\n[bold red]Strategist Error:[/bold red] {error_msg}")
                        console.print("[yellow]Continuing with scout exploration...[/yellow]")
                    elif result.get('status') == 'success':
                        console.print("\n[bold green]═══ STRATEGIST ANALYSIS COMPLETE ═══[/bold green]")
                        
                        # Parse and display hypotheses from JSON response
                        full_response = result.get('full_response', '')
                        strategist_hypotheses = []
                        guidance_items = []
                        
                        if isinstance(full_response, str) and full_response.strip().startswith('{'):
                            try:
                                data = json.loads(full_response)
                                strategist_hypotheses = data.get('hypotheses') or []
                                guidance_items = data.get('guidance') or []
                            except Exception:
                                pass
                        
                        # Display hypothesis processing results
                        if strategist_hypotheses:
                            from rich.table import Table
                            
                            console.print("\n[bold cyan]═══ HYPOTHESIS PROCESSING RESULTS ═══[/bold cyan]")
                            
                            # Create a table for hypothesis status
                            table = Table(show_header=True, header_style="bold magenta", box=None, padding=(0,1))
                            table.add_column("#", style="dim", width=3)
                            table.add_column("Title", style="cyan", overflow="fold")
                            table.add_column("Type", style="yellow")
                            table.add_column("Severity", style="red")
                            table.add_column("Confidence", justify="right")
                            table.add_column("Status", style="bold")
                            
                            # Track what happened to each hypothesis
                            hypotheses_formed = result.get('hypotheses_formed', 0)
                            hyp_info = result.get('hypotheses_info') or []
                            dedup_details = result.get('dedup_details') or []
                            
                            # Build status map and duplicate-of mapping
                            added_titles = {h.get('title') for h in hyp_info if h.get('title')}
                            dedup_map: dict[str, str] = {}
                            dup_ids_by_title: dict[str, list[str]] = {}
                            try:
                                import re as _re
                                _id_re = _re.compile(r"hyp_[0-9a-f]{12}")
                            except Exception:
                                _id_re = None  # type: ignore[assignment]
                            for detail in dedup_details:
                                s = str(detail)
                                if ':' in s:
                                    title_part = s.split(':', 1)[1].strip()
                                    # Stop before '(' or '—' separators if present
                                    for _sep in ('(', '—'):
                                        if _sep in title_part:
                                            title_part = title_part.split(_sep)[0].strip()
                                            break
                                    if title_part:
                                        dedup_map[title_part] = s
                                        # Extract any hyp IDs from detail
                                        try:
                                            if _id_re:
                                                ids = _id_re.findall(s)
                                                if ids:
                                                    dup_ids_by_title[title_part] = ids[:3]
                                        except Exception:
                                            pass
                            # Load id->title mapping from hypotheses.json for nicer display
                            id_to_title: dict[str, str] = {}
                            try:
                                from pathlib import Path as _Path
                                hyp_file = (_Path(self.project_dir) / 'hypotheses.json') if self.project_dir else None
                                if hyp_file and hyp_file.exists():
                                    import json as _json2
                                    with open(hyp_file) as _f:
                                        _data = _json2.load(_f)
                                    for _hid, _h in (_data.get('hypotheses') or {}).items():
                                        # Prefer explicit id field, fallback to key
                                        _key = _h.get('id') or _hid
                                        id_to_title[_key] = _h.get('title', '')
                            except Exception:
                                id_to_title = {}
                            
                            for i, h in enumerate(strategist_hypotheses, 1):
                                title = h.get('title', 'Unknown')
                                vuln_type = h.get('type', 'unknown')
                                severity = h.get('severity', 'medium')
                                confidence = h.get('confidence', 'medium')
                                
                                # Format confidence
                                try:
                                    conf_val = float(confidence)
                                    conf_str = f"{conf_val:.0%}"
                                except (ValueError, TypeError):
                                    conf_str = str(confidence)
                                
                                # Determine status
                                if title in added_titles:
                                    status = "[bold green]✓ ADDED[/bold green]"
                                elif title in dedup_map:
                                    # Compose duplicate-of suffix with IDs and titles if available
                                    dup_suffix = ""
                                    try:
                                        ids = dup_ids_by_title.get(title) or []
                                        if ids:
                                            parts: list[str] = []
                                            for _id in ids:
                                                _t = (id_to_title.get(_id) or '').strip()
                                                parts.append(f"{_id}{(' [' + _t + ']') if _t else ''}")
                                            dup_suffix = f" → duplicate of: {', '.join(parts)}"
                                    except Exception:
                                        dup_suffix = ""
                                    status = f"[yellow]⊘ DUPLICATE[/yellow]{dup_suffix}"
                                else:
                                    # Check if it was skipped for other reasons
                                    if not h.get('node_ids'):
                                        status = "[red]✗ NO NODES[/red]"
                                    else:
                                        status = "[yellow]⊘ SKIPPED[/yellow]"
                                
                                # Add row to table
                                table.add_row(
                                    str(i),
                                    title[:60] + "..." if len(title) > 60 else title,
                                    vuln_type[:15],
                                    severity,
                                    conf_str,
                                    status
                                )
                            
                            console.print(table)
                            
                            # Summary statistics
                            console.print("\n[bold]Summary:[/bold]")
                            console.print(f"  • Total hypotheses from strategist: {len(strategist_hypotheses)}")
                            console.print(f"  • [green]Successfully added to store: {hypotheses_formed}[/green]")
                            
                            dedup_count = result.get('dedup_skipped', 0)
                            if dedup_count > 0:
                                console.print(f"  • [yellow]Skipped as duplicates: {dedup_count}[/yellow]")
                            
                            no_nodes = result.get('skipped_no_node_ids') or []
                            if no_nodes:
                                console.print(f"  • [red]Skipped (no node IDs): {len(no_nodes)}[/red]")
                            
                            fallback = result.get('fallback_node_ids_assigned') or []
                            if fallback:
                                console.print(f"  • [dim]Assigned fallback nodes: {len(fallback)}[/dim]")
                            
                            # Show detailed reasoning for duplicates if any
                            if dedup_details:
                                console.print("\n[yellow]Duplicate Detection Details:[/yellow]")
                                for detail in dedup_details[:3]:
                                    console.print(f"  • {detail}")
                        
                        elif not strategist_hypotheses:
                            console.print("\n[yellow]ℹ No vulnerabilities identified in this analysis[/yellow]")
                        
                        # Show guidance
                        if guidance_items:
                            console.print("\n[bold cyan]Strategist Guidance:[/bold cyan]")
                            for item in guidance_items[:5]:
                                console.print(f"  → {item}")
                        # Show format issues (invalid structure) and formation errors
                        try:
                            inv = result.get('skipped_invalid_format') or []
                            parsed_cnt = result.get('parsed_lines')
                            if isinstance(parsed_cnt, int):
                                console.print(f"[dim]Parsed candidate hypotheses: {parsed_cnt}[/dim]")
                            if inv:
                                console.print(f"[dim]Skipped (invalid format): {len(inv)}[/dim]")
                                for t in inv[:3]:
                                    s = str(t)
                                    if len(s) > 120:
                                        s = s[:117] + '...'
                                    console.print(f"  - {s}")
                        except Exception:
                            pass
                        # Show formation errors (parsing/robustness issues)
                        try:
                            errs = result.get('skipped_errors') or []
                            if errs:
                                console.print(f"[dim]Skipped (formation errors): {len(errs)}[/dim]")
                                for t in errs[:3]:
                                    s = str(t)
                                    if len(s) > 120:
                                        s = s[:117] + '...'
                                    console.print(f"  - {s}")
                        except Exception:
                            pass
                        console.print()
                    else:
                        # Strategist failed - show the error
                        error_msg = result.get('error', 'Unknown error')
                        console.print(f"\n[bold red]Strategist Error:[/bold red] {error_msg}")
                        console.print("[yellow]Continuing with scout exploration...[/yellow]")
                else:
                    # Regular action results - keep brief
                    console.print(f"[dim]Result: {msg or 'completed'}[/dim]")
                    
            elif status == 'hypothesis_formed':
                console.print(f"\n[bold green]Hypothesis Formed:[/bold green] {msg}")
            elif status in {'analyzing', 'executing'}:
                console.print(f"[dim]{status.capitalize()}: {msg}[/dim]")

        # Compose audit prompt
        audit_prompt = (
            "Perform a focused security audit of this codebase based on the available graphs. "
            "Identify potential vulnerabilities or risky patterns, form hypotheses, and summarize findings."
        )
        # Ensure overarching mission is visible to the agent/strategist
        try:
            if getattr(self, 'mission', None):
                self.agent.mission = self.mission
        except Exception:
            pass

        results = []
        planned_round = 0
        start_overall = time.time()
        time_up = False

        # Local exception used to abort long-running investigations when time is up
        class _TimeLimitReached(Exception):
            pass

        # Simple steering helpers
        def _is_global_steer(text: str) -> bool:
            """Heuristic: detect broad, project-wide directives.
            Examples: "whole app", "entire codebase", "all contracts", "system-wide", etc.
            """
            t = (text or '').lower()
            if not t:
                return False
            global_markers = (
                'whole app', 'entire app', 'entire codebase', 'whole codebase', 'all contracts',
                'every contract', 'system-wide', 'system wide', 'project-wide', 'project wide',
                'across the codebase', 'across the repo', 'across modules', 'end-to-end', 'e2e',
                'globally', 'everywhere', 'full audit', 'full review', 'scan the entire', 'scan all'
            )
            if any(m in t for m in global_markers):
                return True
            # Also treat "check X across" as global
            if ' across ' in t or t.startswith('across '):
                return True
            return False
        # Track consecutive rounds with no new investigations to detect stuck state
        consecutive_empty_rounds = 0
        last_round_goals = set()
        
        while True:
            # Time limit check
            if self.time_limit_minutes:
                elapsed_minutes = (time.time() - start_overall) / 60.0
                if elapsed_minutes >= self.time_limit_minutes:
                    console.print(f"\n[yellow]⏰ Time limit reached ({self.time_limit_minutes} minutes) — stopping audit[/yellow]")
                    break

            planned_round += 1
            # Announce planning batch and current phase before spinner (print once)
            try:
                console.print(f"\n[bold cyan]═══ Planning Batch {planned_round} ═══[/bold cyan]")
                # Two-phase: Sweep vs Intuition
                phase = getattr(self, '_current_phase', None)
                if not phase:
                    # Check if mode is explicitly set
                    if self.mode:
                        if self.mode.lower() == 'sweep':
                            phase = 'Coverage'
                        elif self.mode.lower() == 'intuition':
                            phase = 'Saliency'
                    elif self.session_tracker:
                        cov_tmp = self.session_tracker.get_coverage_stats()
                        nodes_pct = float(((cov_tmp or {}).get('nodes') or {}).get('percent') or 0.0)
                        phase = 'Coverage' if nodes_pct < 90.0 else 'Saliency'
                if phase:
                    if phase == 'Coverage':
                        console.print("\n[bold yellow]═══ PHASE 1: SWEEP ═══[/bold yellow]")
                        console.print("[dim]Wide sweep for shallow bugs at medium granularity[/dim]")
                        console.print("[dim]Approach: Examine each module/class for all security issues.[/dim]")
                        console.print("[dim]Also captures invariants and assumptions to build the knowledge graph.[/dim]\n")
                    else:
                        console.print("\n[bold magenta]═══ PHASE 2: INTUITION ═══[/bold magenta]")
                        console.print("[dim]This phase uses graph-level reasoning to find complex, high-impact vulnerabilities.[/dim]")
                        console.print("[dim]Focus: Contradictions, invariant violations, cross-component interactions.[/dim]")
                        console.print("[dim]Leveraging annotations from Phase 1 to guide targeted investigation.[/dim]\n")
                # Show compact coverage line pre-planning for context
                if self.session_tracker:
                    cov = self.session_tracker.get_coverage_stats()
                    console.print(
                        f"Coverage: Nodes {cov['nodes']['visited']}/{cov['nodes']['total']} "
                        f"({cov['nodes']['percent']:.1f}%) | "
                        f"Cards {cov['cards']['visited']}/{cov['cards']['total']} "
                        f"({cov['cards']['percent']:.1f}%)"
                    )
            except Exception:
                pass

            # Show an animated status while strategist plans the next batch
            try:
                with console.status("[cyan]Strategist planning next steps...[/cyan]", spinner="dots", spinner_style="cyan"):
                    items = self._plan_investigations(max(1, plan_n))
            except Exception:
                items = self._plan_investigations(max(1, plan_n))
            self._agent_log.append(f"Planning batch {planned_round} (top {plan_n})")
            # Log planning status and show planned items (phase banner already printed)
            # Show current coverage stats and a sample of unvisited nodes
            try:
                if self.session_tracker:
                    _cov = self.session_tracker.get_coverage_stats()
                    console.print(
                        f"Coverage: Nodes {_cov['nodes']['visited']}/{_cov['nodes']['total']} "
                        f"({_cov['nodes']['percent']:.1f}%) | "
                        f"Cards {_cov['cards']['visited']}/{_cov['cards']['total']} "
                        f"({_cov['cards']['percent']:.1f}%)"
                    )
                    sample, count = self._get_unvisited_nodes_sample(max_n=10)
                    if count > 0 and sample:
                        console.print(f"Unvisited nodes (sample): {', '.join(sample)}")
            except Exception:
                pass
            self._log_planning_status(items, current_index=-1)
            
            # If no items, log and stop
            if not items:
                console.print("[yellow]No further promising investigations suggested — audit complete[/yellow]")
                break
            
            # Check if we're getting the same items repeatedly (stuck in a loop)
            current_round_goals = set(it.goal for it in items)
            if current_round_goals == last_round_goals:
                consecutive_empty_rounds += 1
                if consecutive_empty_rounds >= 2:
                    console.print("[yellow]\nDetected planning loop - no new components to analyze[/yellow]")
                    if self.mode == 'sweep':
                        console.print("[green]Sweep mode complete![/green]")
                    else:
                        console.print("[yellow]Consider switching to intuition mode for deeper analysis[/yellow]")
                    break
            else:
                consecutive_empty_rounds = 0
                last_round_goals = current_round_goals
            
            # In sweep mode, check if we've analyzed all reachable components
            if self.mode == 'sweep' and planned_round > 1:
                # Get all completed component names from investigations
                completed_components = set()
                for goal in self.completed_investigations:
                    # Extract component names from goals like "Vulnerability analysis of X"
                    import re
                    match = re.search(r'(?:of|for|in)\s+([A-Za-z0-9_]+)', goal)
                    if match:
                        completed_components.add(match.group(1).lower())
                
                # Check if any new items target components we haven't analyzed
                has_new_targets = False
                for it in items:
                    match = re.search(r'(?:of|for|in)\s+([A-Za-z0-9_]+)', it.goal)
                    if match:
                        comp = match.group(1).lower()
                        if comp not in completed_components:
                            has_new_targets = True
                            break
                
                if not has_new_targets and len(self.completed_investigations) > 0:
                    console.print("[yellow]\nAll accessible components have been analyzed[/yellow]")
                    console.print("[green]Sweep mode complete![/green]")
                    break
            
            # Log previously completed investigations
            if self.completed_investigations:
                console.print("\n[bold green]Previously Completed Investigations:[/bold green]")
                for goal in self.completed_investigations:
                    console.print(f"  ✓ {goal}")
            
            # Log new investigations planned by strategist
            console.print("\n[bold cyan]New Investigations Planned by Strategist:[/bold cyan]")
            for i, it in enumerate(items, 1):
                pr = getattr(it, 'priority', 0)
                imp = getattr(it, 'expected_impact', None)
                cat = getattr(it, 'category', None)
                reasoning = getattr(it, 'reasoning', '')
                
                console.print(f"\n  {i}. [bold]{it.goal}[/bold]")
                console.print(f"     Priority: {pr} | Impact: {imp or 'unknown'} | Category: {cat or 'general'}")
                if reasoning:
                    console.print(f"     [dim]Reasoning: {reasoning}[/dim]")
            
            executed_frames = set()
            # Execute investigations with proper logging
            for idx, inv in enumerate(items):
                # Check for mid-batch steering override (preempt current goal once)
                try:
                    urgent_ent = self._find_latest_urgent_steer()
                    urgent = urgent_ent['text'] if urgent_ent else None
                    if urgent and urgent != self._last_applied_steer and getattr(inv, 'goal', '') != urgent:
                        console.print(f"[bold yellow]Steering override:[/bold yellow] {urgent}")
                        try:
                            pub = getattr(self, '_telemetry_publish', None)
                            if callable(pub):
                                pub({'type': 'status', 'message': f'override: {urgent}'})
                        except Exception:
                            pass
                        from types import SimpleNamespace
                        inv = SimpleNamespace(
                            goal=urgent,
                            focus_areas=[],
                            priority=10,
                            reasoning='User steering override',
                            category='suspicion',
                            expected_impact='high'
                        )
                        self._last_applied_steer = urgent
                        # Mark consumed so it doesn't reapply after restarts
                        try:
                            self._consume_steer(float(urgent_ent.get('ts') or 0.0))
                        except Exception:
                            pass
                except Exception:
                    pass
                # Check time limit before starting each investigation
                if self.time_limit_minutes:
                    elapsed_minutes = (time.time() - start_overall) / 60.0
                    remaining_minutes = self.time_limit_minutes - elapsed_minutes
                    if remaining_minutes <= 0:
                        console.print(f"\n[yellow]Time limit reached ({self.time_limit_minutes} minutes) — stopping audit[/yellow]")
                        break
                    if remaining_minutes < 2:
                        console.print(f"\n[yellow]Warning: Only {remaining_minutes:.1f} minutes remaining[/yellow]")
                
                # Skip duplicate frame_ids within the same run to avoid loops
                try:
                    if getattr(inv, 'frame_id', None) and inv.frame_id in executed_frames:
                        console.print(f"[yellow]Skipping duplicate investigation frame:[/yellow] {inv.frame_id} ({inv.goal})")
                        try:
                            if self.plan_store:
                                from analysis.plan_store import PlanStatus
                                self.plan_store.update_status(inv.frame_id, PlanStatus.DROPPED, rationale='Skipped duplicate within run')
                        except Exception:
                            pass
                        continue
                except Exception:
                    pass

                # Log current investigation with updated coverage
                console.print(f"\n[bold magenta]═══ Starting Investigation {idx+1}/{len(items)} ═══[/bold magenta]")
                console.print(f"[bold]Goal:[/bold] {inv.goal}")
                # Snapshot coverage at the start of the investigation
                try:
                    if self.session_tracker:
                        _cov = self.session_tracker.get_coverage_stats()
                        console.print(
                            f"Coverage: Nodes {_cov['nodes']['visited']}/{_cov['nodes']['total']} "
                            f"({_cov['nodes']['percent']:.1f}%) | "
                            f"Cards {_cov['cards']['visited']}/{_cov['cards']['total']} "
                            f"({_cov['cards']['percent']:.1f}%)"
                        )
                except Exception:
                    pass
                self._log_planning_status(items, current_index=idx)
                # Mark plan item in_progress if we have a frame_id
                try:
                    if getattr(inv, 'frame_id', None) and self.plan_store:
                        from analysis.plan_store import PlanStatus
                        self.plan_store.update_status(inv.frame_id, PlanStatus.IN_PROGRESS, rationale='Execution started')
                    if getattr(self, 'agent', None) and getattr(self.agent, 'coverage_index', None):
                        self.agent.coverage_index.record_investigation(getattr(inv, 'frame_id', None), [], 'in_progress')
                except Exception:
                    pass
                # Always use requested iterations; rely on time-limit checks to stop early
                max_iters = self.agent.max_iterations if self.agent.max_iterations else 5

                self.start_time = time.time()
                started_at_iso = datetime.now().isoformat()
                try:
                    # Enhanced progress callback that logs model actions and thoughts
                    def _cb(info: dict):
                        # Enforce global time limit within investigation callbacks
                        if self.time_limit_minutes:
                            if (time.time() - start_overall) / 60.0 >= self.time_limit_minutes:
                                raise _TimeLimitReached()
                        status = info.get('status', '')
                        msg = info.get('message', '')
                        it = info.get('iteration', 0)
                        # Publish telemetry for UI (decision/result/etc.)
                        try:
                            pub = getattr(self, '_telemetry_publish', None)
                            if callable(pub):
                                payload = {
                                    'type': status or 'progress',
                                    'iteration': it,
                                    'message': msg,
                                    'action': info.get('action'),
                                    'parameters': info.get('parameters', {}),
                                    'reasoning': info.get('reasoning', ''),
                                }
                                if status == 'result':
                                    # Slim down large result fields to keep UI responsive
                                    res = info.get('result', {}) or {}
                                    if isinstance(res, dict):
                                        slim = dict(res)
                                        # Drop verbose text and heavy fields
                                        for k in ('graph_display', 'nodes_display', 'full_response', 'graph_data', 'data', 'nodes', 'edges', 'code', 'cards'):
                                            if k in slim:
                                                slim.pop(k, None)
                                        payload['result'] = slim
                                    else:
                                        payload['result'] = res
                                pub(payload)
                        except Exception:
                            pass

                        # Mid-investigation steering: if a global directive arrives, request abort
                        try:
                            # Only check on meaningful milestones to reduce I/O
                            if status in {'analyzing', 'decision', 'executing'}:
                                ent = self._find_latest_urgent_steer()
                                latest = ent['text'] if ent else None
                                if latest and latest != self._last_replan_steer:
                                    if _is_global_steer(latest):
                                        # Mark and request abort on the agent; outer loop will replan
                                        self._last_replan_steer = latest
                                        try:
                                            if hasattr(self, 'agent') and self.agent:
                                                self.agent.request_abort(reason=f"steering_replan: {latest[:120]}")  # type: ignore[attr-defined]
                                        except Exception:
                                            pass
                                        # Tell the console and telemetry
                                        console.print(f"[bold yellow]Steering replan:[/bold yellow] {latest}")
                                        try:
                                            pub = getattr(self, '_telemetry_publish', None)
                                            if callable(pub):
                                                pub({'type': 'status', 'message': f'steering replan: {latest}'})
                                        except Exception:
                                            pass
                                        # Mark consumed
                                        try:
                                            self._consume_steer(float(ent.get('ts') or 0.0))
                                        except Exception:
                                            pass
                        except Exception:
                            pass
                        if status == 'decision':
                            act = info.get('action', '-')
                            reasoning = info.get('reasoning', '')
                            params = info.get('parameters', {})
                            
                            # Log model decision with clear formatting
                            console.print(f"\n[bold blue]Model Decision (Iteration {it}):[/bold blue]")
                            console.print(f"  [cyan]Action:[/cyan] {act}")
                            if reasoning:
                                console.print(f"  [cyan]Thought:[/cyan] {reasoning}")
                            if params and act != 'deep_think':
                                # Show parameters for non-deep-think actions (concise summary)
                                def _summ(v):
                                    try:
                                        import json
                                        return json.dumps(v)[:200]
                                    except Exception:
                                        return str(v)[:200]
                                lines = []
                                for k, v in (params or {}).items():
                                    if isinstance(v, list):
                                        if k in ("observations", "assumptions", "refs", "node_ids"):
                                            preview = ", ".join([str(x)[:60] for x in v[:2]])
                                            more = f" (+{len(v)-2} more)" if len(v) > 2 else ""
                                            lines.append(f"  - {k}: {len(v)} {more}")
                                            if preview:
                                                lines.append(f"    • {preview}")
                                        else:
                                            lines.append(f"  - {k}: {len(v)} items")
                                    elif isinstance(v, dict):
                                        lines.append(f"  - {k}: {{...}}")
                                    else:
                                        sval = str(v)
                                        if len(sval) > 120:
                                            sval = sval[:117] + "..."
                                        lines.append(f"  - {k}: {sval}")
                                if lines:
                                    console.print("  [cyan]Parameters:[/cyan]")
                                    for ln in lines[:8]:
                                        console.print(ln)
                            
                            # Special handling for deep_think
                            if act == 'deep_think':
                                console.print("\n[bold magenta]═══ CALLING STRATEGIST FOR DEEP ANALYSIS ═══[/bold magenta]")
                                try:
                                    strat_cfg = (self.config or {}).get('models', {}).get('strategist', {})
                                    eff = strat_cfg.get('hypothesize_reasoning_effort') or strat_cfg.get('reasoning_effort')
                                    if hasattr(self.agent, 'guidance_client') and self.agent.guidance_client:
                                        prov = getattr(self.agent.guidance_client, 'provider_name', 'unknown')
                                        mdl = getattr(self.agent.guidance_client, 'model', 'unknown')
                                        console.print(f"[dim]Strategist model: {prov}/{mdl} | effort: {eff or 'default'}[/dim]")
                                except Exception:
                                    pass
                                console.print("[yellow]Strategist is analyzing the collected context...[/yellow]")
                            
                            # Update agent log
                            self._agent_log.append(f"Iter {it}: {act} - {reasoning[:100] if reasoning else 'no reasoning'}")
                            
                            # Track visited nodes and cards
                            if self.session_tracker:
                                if act == 'load_nodes' and params:
                                    # Track nodes loaded via load_nodes action
                                    node_ids = params.get('node_ids', [])
                                    if node_ids:
                                        self.session_tracker.track_nodes_batch(node_ids)
                                elif act == 'explore_graph' and params:
                                    # Track nodes explored via explore_graph action
                                    node_ids = params.get('node_ids', [])
                                    if node_ids:
                                        self.session_tracker.track_nodes_batch(node_ids)
                                elif act == 'analyze_code' and params:
                                    # Track code cards analyzed
                                    file_path = params.get('file_path')
                                    if file_path:
                                        self.session_tracker.track_card_visit(file_path)
                            
                        elif status == 'result':
                            action = info.get('action', '')
                            result = info.get('result', {})
                            
                            if action == 'deep_think':
                                # Special formatting for deep_think results
                                if not isinstance(result, dict):
                                    error_msg = f"Unexpected strategist result type: {type(result).__name__}"
                                    console.print(f"\n[bold red]Strategist Error:[/bold red] {error_msg}")
                                    console.print("[yellow]Continuing with scout exploration...[/yellow]")
                                elif result.get('status') == 'success':
                                    console.print("\n[bold green]═══ STRATEGIST ANALYSIS COMPLETE ═══[/bold green]")
                                    
                                    # Parse and display hypotheses from JSON response
                                    full_response = result.get('full_response', '')
                                    strategist_hypotheses = []
                                    guidance_items = []
                                    
                                    if isinstance(full_response, str) and full_response.strip().startswith('{'):
                                        try:
                                            data = json.loads(full_response)
                                            strategist_hypotheses = data.get('hypotheses') or []
                                            guidance_items = data.get('guidance') or []
                                        except Exception:
                                            pass
                                    
                                    # Display hypothesis processing results
                                    if strategist_hypotheses:
                                        from rich.table import Table
                                        
                                        console.print("\n[bold cyan]HYPOTHESIS PROCESSING RESULTS[/bold cyan]")
                                        
                                        # Create a table for hypothesis status
                                        table = Table(show_header=True, header_style="bold magenta", box=None, padding=(0,1))
                                        table.add_column("#", style="dim", width=3)
                                        table.add_column("Title", style="cyan", overflow="fold")
                                        table.add_column("Severity", style="red")
                                        table.add_column("Status", style="bold")
                                        
                                        # Track what happened to each hypothesis
                                        hypotheses_formed = result.get('hypotheses_formed', 0)
                                        hyp_info = result.get('hypotheses_info') or []
                                        dedup_details = result.get('dedup_details') or []
                                        
                                        # Build status map and duplicate-of mapping
                                        added_titles = {h.get('title') for h in hyp_info if h.get('title')}
                                        dedup_details = result.get('dedup_details') or []
                                        dedup_map: dict[str, str] = {}
                                        dup_ids_by_title: dict[str, list[str]] = {}
                                        try:
                                            import re as _re
                                            _id_re2 = _re.compile(r"hyp_[0-9a-f]{12}")
                                        except Exception:
                                            _id_re2 = None  # type: ignore[assignment]
                                        for _detail in dedup_details:
                                            _s = str(_detail)
                                            if ':' in _s:
                                                _title_part = _s.split(':', 1)[1].strip()
                                                for _sep in ('(', '—'):
                                                    if _sep in _title_part:
                                                        _title_part = _title_part.split(_sep)[0].strip()
                                                        break
                                                if _title_part:
                                                    dedup_map[_title_part] = _s
                                                    try:
                                                        if _id_re2:
                                                            _ids = _id_re2.findall(_s)
                                                            if _ids:
                                                                dup_ids_by_title[_title_part] = _ids[:3]
                                                    except Exception:
                                                        pass
                                        # Load id->title mapping from hypotheses.json for nicer display
                                        id_to_title2: dict[str, str] = {}
                                        try:
                                            from pathlib import Path as _Path2
                                            hyp_file2 = (_Path2(self.project_dir) / 'hypotheses.json') if self.project_dir else None
                                            if hyp_file2 and hyp_file2.exists():
                                                import json as _json3
                                                with open(hyp_file2) as _f2:
                                                    _data2 = _json3.load(_f2)
                                                for _hid2, _h2 in (_data2.get('hypotheses') or {}).items():
                                                    _key2 = _h2.get('id') or _hid2
                                                    id_to_title2[_key2] = _h2.get('title', '')
                                        except Exception:
                                            id_to_title2 = {}
                                        
                                        for i, h in enumerate(strategist_hypotheses, 1):
                                            title = h.get('title', 'Unknown')
                                            severity = h.get('severity', 'medium')
                                            
                                            # Determine status
                                            if title in added_titles:
                                                status = "[bold green]✓ ADDED[/bold green]"
                                            else:
                                                # Check for duplicates and compose duplicate-of suffix
                                                is_dup = False
                                                dup_suffix2 = ""
                                                if title in dedup_map:
                                                    is_dup = True
                                                    try:
                                                        _ids2 = dup_ids_by_title.get(title) or []
                                                        if _ids2:
                                                            _parts2: list[str] = []
                                                            for _id in _ids2:
                                                                _t2 = (id_to_title2.get(_id) or '').strip()
                                                                _parts2.append(f"{_id}{(' [' + _t2 + ']') if _t2 else ''}")
                                                            dup_suffix2 = f" → duplicate of: {', '.join(_parts2)}"
                                                    except Exception:
                                                        dup_suffix2 = ""
                                                    status = f"[yellow]⊘ DUPLICATE[/yellow]{dup_suffix2}"
                                                if not is_dup:
                                                    if not h.get('node_ids'):
                                                        status = "[red]✗ NO NODES[/red]"
                                                    else:
                                                        status = "[yellow]⊘ SKIPPED[/yellow]"
                                            
                                            # Add row to table
                                            table.add_row(
                                                str(i),
                                                title[:60] + "..." if len(title) > 60 else title,
                                                severity,
                                                status
                                            )
                                        
                                        console.print(table)
                                        
                                        # Summary with store stats
                                        console.print(f"\n[bold]Total: {len(strategist_hypotheses)} | [green]Added: {hypotheses_formed}[/green] | [yellow]Skipped: {len(strategist_hypotheses) - hypotheses_formed}[/yellow][/bold]")
                                        
                                        # Show hypothesis store stats if available
                                        try:
                                            store_stats = result.get('store_stats')
                                            if not store_stats and hasattr(self, 'agent') and hasattr(self.agent, 'hypothesis_store'):
                                                all_hyps = self.agent.hypothesis_store.list_all()
                                                store_stats = {'total': len(all_hyps)}
                                            if store_stats:
                                                console.print(f"\n[dim]Hypothesis Store: {store_stats.get('total', 0)} total vulnerabilities tracked[/dim]")
                                        except Exception:
                                            pass
                                    
                                    elif not strategist_hypotheses:
                                        console.print("\n[yellow]ℹ No vulnerabilities identified in this analysis[/yellow]")
                                    
                                    # Show guidance
                                    if guidance_items:
                                        console.print("\n[bold cyan]Strategist Guidance:[/bold cyan]")
                                        for item in guidance_items[:5]:
                                            console.print(f"  → {item}")
                                else:
                                    error_msg = result.get('error', 'Unknown error')
                                    console.print(f"\n[bold red]Strategist Error:[/bold red] {error_msg}")
                                    console.print("[yellow]Continuing with scout exploration...[/yellow]")
                            else:
                                # Regular action results
                                summ = result.get('summary') or result.get('status') or msg
                                console.print(f"[dim]Result: {summ}[/dim]")
                                # Publish strategist summary to telemetry for UI when available
                                try:
                                    pub = getattr(self, '_telemetry_publish', None)
                                    if callable(pub) and action == 'deep_think':
                                        bullets = result.get('guidance_bullets') or []
                                        hyp_info = result.get('hypotheses_info') or []
                                        payload = {
                                            'type': 'strategist',
                                            'iteration': it,
                                            'message': 'Strategist analysis complete',
                                            'bullets': bullets[:5],
                                            'hypotheses': hyp_info[:5],
                                        }
                                        pub(payload)
                                except Exception:
                                    pass
                                # Track cards loaded via load_nodes result if provided
                                try:
                                    if self.session_tracker and action == 'load_nodes':
                                        cids = result.get('card_ids') or []
                                        if isinstance(cids, list) and cids:
                                            self.session_tracker.track_cards_batch([str(x) for x in cids])
                                except Exception:
                                    pass
                            
                            self._agent_log.append(f"Iter {it} result: {action}")
                            
                        elif status == 'usage':
                            console.print(f"[dim]Usage: {msg}[/dim]")
                            self._agent_log.append(f"Iter {it} usage: {msg}")
                        elif status in {'analyzing', 'executing', 'hypothesis_formed'}:
                            console.print(f"[dim]{status.capitalize()}: {msg}[/dim]")
                            self._agent_log.append(f"Iter {it} {status}: {msg[:100]}")

                    # Show an animated status while the agent thinks/acts for this investigation
                    replan_requested = False
                    try:
                        # Set the current phase on the agent for deep_think
                        self.agent.current_phase = self._current_phase
                        # More accurate status: the Scout is exploring code, not just "thinking"
                        with console.status("[cyan]Exploring codebase and analyzing nodes...[/cyan]", spinner="line", spinner_style="cyan"):
                            report = self.agent.investigate(inv.goal, max_iterations=max_iters, progress_callback=_cb)
                    except _TimeLimitReached:
                        console.print(f"\n[yellow]Time limit reached ({self.time_limit_minutes} minutes) — stopping audit[/yellow]")
                        time_up = True
                        break
                    except Exception:
                        # Retry without status context; still honor time limit
                        try:
                            report = self.agent.investigate(inv.goal, max_iterations=max_iters, progress_callback=_cb)
                        except _TimeLimitReached:
                            console.print(f"\n[yellow]Time limit reached ({self.time_limit_minutes} minutes) — stopping audit[/yellow]")
                            time_up = True
                            break
                    # If an abort was requested (global steering), skip marking as completed and replan
                    try:
                        if hasattr(self, 'agent') and getattr(self.agent, '_abort_requested', False):
                            # Reset abort flag for next investigation round
                            try:
                                self.agent._abort_requested = False  # type: ignore[attr-defined]
                                self.agent._abort_reason = None      # type: ignore[attr-defined]
                            except Exception:
                                pass
                            console.print("[yellow]Investigation aborted due to steering; reprioritizing...[/yellow]")
                            # Publish a telemetry status
                            try:
                                pub = getattr(self, '_telemetry_publish', None)
                                if callable(pub):
                                    pub({'type': 'status', 'message': 'investigation aborted (steering replan)'})
                            except Exception:
                                pass
                            # Do not record as completed; break to re-enter planning
                            break
                    except Exception:
                        pass

                    results.append((inv, report))
                    # Track completed investigation
                    self.completed_investigations.append(inv.goal)
                    try:
                        if getattr(inv, 'frame_id', None):
                            executed_frames.add(inv.frame_id)
                    except Exception:
                        pass
                    # Update session tracker with investigation and token usage
                    self.session_tracker.add_investigation({
                        'goal': inv.goal,
                        'priority': getattr(inv, 'priority', 0),
                        'category': getattr(inv, 'category', None),
                        'frame_id': getattr(inv, 'frame_id', None),
                        'planned_batch': planned_round,
                        'planned_index': idx + 1,
                        'started_at': started_at_iso,
                        'ended_at': datetime.now().isoformat(),
                        'iterations_completed': report.get('iterations_completed', 0) if report else 0,
                        'hypotheses': report.get('hypotheses', {}) if report else {}
                    })
                    self.session_tracker.update_token_usage(token_tracker.get_summary())
                except Exception as e:
                    # Log error but don't fail the audit
                    console.print(f"[red]Error in investigation: {str(e)}[/red]")
                    raise
                # Show completion
                console.print(f"\n[bold green]✓ Investigation Completed:[/bold green] {inv.goal}")
                # Show updated coverage after completion
                try:
                    if self.session_tracker:
                        _cov = self.session_tracker.get_coverage_stats()
                        console.print(
                            f"Coverage: Nodes {_cov['nodes']['visited']}/{_cov['nodes']['total']} "
                            f"({_cov['nodes']['percent']:.1f}%) | "
                            f"Cards {_cov['cards']['visited']}/{_cov['cards']['total']} "
                            f"({_cov['cards']['percent']:.1f}%)"
                        )
                except Exception:
                    pass
                self._agent_log.append(f"✓ Completed: {inv.goal}")
                # Mark plan item done
                try:
                    if getattr(inv, 'frame_id', None) and self.plan_store:
                        from analysis.plan_store import PlanStatus
                        self.plan_store.update_status(inv.frame_id, PlanStatus.DONE, rationale='Completed investigation')
                    if getattr(self, 'agent', None) and getattr(self.agent, 'coverage_index', None):
                        self.agent.coverage_index.record_investigation(getattr(inv, 'frame_id', None), [], 'done')
                except Exception:
                    pass
                
                # Early stop if agent is satisfied (no hypotheses and no more actions suggested)
                try:
                    hyp = (report or {}).get('hypotheses', {})
                    total_h = int(hyp.get('total', 0))
                except Exception:
                    total_h = 0
                if total_h == 0:
                    console.print("[yellow]No hypotheses formed; considering coverage achieved for this thread[/yellow]")
                    self._agent_log.append("No hypotheses formed; considering coverage achieved for this thread")

            # If time was exhausted during an investigation, stop planning loop as well
            if time_up:
                break

        # After audit, show the last report in detail
        if results:
            last_report = results[-1][1]
            try:
                display_investigation_report(last_report)
            except Exception:
                console.print(f"\n[bold]Iterations:[/bold] {last_report.get('iterations_completed', 0)}")
                console.print(f"[bold]Hypotheses:[/bold] {last_report.get('hypotheses', {})}")

        # Plan execution summary (exact steps)
        try:
            from rich.table import Table as _Tbl
            exec_table = _Tbl(show_header=True, header_style="bold cyan")
            exec_table.add_column("#", style="dim", width=4)
            exec_table.add_column("Frame", style="yellow")
            exec_table.add_column("Goal", style="white")
            exec_table.add_column("Batch", style="magenta", width=7)
            exec_table.add_column("Pos", style="magenta", width=4)
            exec_table.add_column("Iters", justify="right", width=6)
            exec_table.add_column("Hyps", justify="right", width=5)
            rown = 0
            for (inv, rep) in results:
                rown += 1
                fid = getattr(inv, 'frame_id', '') or ''
                iters = (rep or {}).get('iterations_completed', 0)
                hyps = (rep or {}).get('hypotheses', {}).get('total', 0)
                goal = getattr(inv, 'goal', '')
                # Phase 1 investigations are already properly formatted
                phase = getattr(self, '_current_phase', None)
                exec_table.add_row(str(rown), str(fid), goal, str(planned_round), str(rown), str(iters), str(hyps))
            console.print("\n[bold cyan]Plan Execution Summary[/bold cyan]")
            console.print(exec_table)
        except Exception:
            pass

        # Finalize session tracker with final token usage
        self.session_tracker.update_token_usage(token_tracker.get_summary())
        final_status = 'interrupted' if 'time_up' in locals() and time_up else 'completed'
        self.session_tracker.finalize(status=final_status)
        
        # Show final coverage
        coverage_stats = self.session_tracker.get_coverage_stats()
        console.print("\n[bold cyan]Final Coverage Statistics:[/bold cyan]")
        console.print(f"  Nodes visited: {coverage_stats['nodes']['visited']}/{coverage_stats['nodes']['total']} ([cyan]{coverage_stats['nodes']['percent']:.1f}%[/cyan])")
        console.print(f"  Cards analyzed: {coverage_stats['cards']['visited']}/{coverage_stats['cards']['total']} ([cyan]{coverage_stats['cards']['percent']:.1f}%[/cyan])")
        
        console.print(f"\n[green]Session details saved to:[/green] sessions/{self.session_id}.json")

        # Finalize debug log if enabled
        try:
            if self.debug and getattr(self, 'agent', None) and getattr(self.agent, 'debug_logger', None):
                # Build a concise summary
                token_tracker = get_token_tracker()
                summary = {
                    'planning_batches': planned_round,
                    'hypotheses_total': 0,
                }
                try:
                    # If we recorded investigations, sum hypotheses proposed
                    sess = self.session_tracker.session_data if self.session_tracker else {}
                    invs = sess.get('investigations', []) if isinstance(sess, dict) else []
                    summary['hypotheses_total'] = sum(int(i.get('hypotheses', {}).get('total', 0)) for i in invs)
                except Exception:
                    pass
                summary['total_api_calls'] = token_tracker.get_summary().get('total_usage', {}).get('call_count', 0)
                log_path = self.agent.debug_logger.finalize(summary=summary)
                console.print(f"[cyan]Debug log saved:[/cyan] {log_path}")
        except Exception:
            pass
    
    def _generate_enhanced_summary(self):
        """Deprecated in unified agent flow; retained for API compatibility."""
        return {
            'note': 'Use report returned by agent.investigate() for results',
        }
    
    def finalize_tracking(self, status: str = 'completed'):
        """Finalize session tracking with given status."""
        if self.session_tracker:
            token_tracker = get_token_tracker()
            self.session_tracker.update_token_usage(token_tracker.get_summary())
            self.session_tracker.finalize(status=status)


@click.command()
@click.argument('project_id')
@click.option('--iterations', type=int, help='Max iterations per investigation')
@click.option('--plan-n', type=int, default=5, help='Number of investigations to plan per batch (default: 5)')
@click.option('--time-limit', type=int, help='Time limit in minutes')
@click.option('--config', type=click.Path(exists=True), help='Configuration file')
@click.option('--debug', is_flag=True, help='Enable debug logging of prompts and responses')
@click.option('--mode', type=click.Choice(['sweep', 'intuition'], case_sensitive=False), default=None, help='Analysis mode: sweep (Phase 1) or intuition (Phase 2)')
@click.option('--platform', default=None, help='Override scout platform (e.g., openai, anthropic, mock)')
@click.option('--model', default=None, help='Override scout model (e.g., gpt-5, gpt-4o-mini, mock)')
@click.option('--strategist-platform', default=None, help='Override strategist platform (e.g., openai, anthropic, mock)')
@click.option('--strategist-model', default=None, help='Override strategist model (e.g., gpt-4o-mini)')
@click.option('--session', default=None, help='Attach to a specific session ID')
@click.option('--new-session', is_flag=True, help='Create a new session')
@click.option('--session-private-hypotheses', is_flag=True, help='Keep new hypotheses private to this session')
@click.option('--telemetry', is_flag=True, help='Expose local (localhost) telemetry SSE/control and register instance')
@click.option('--strategist-two-pass', is_flag=True, help='Enable strategist two-pass self-critique to reduce false positives')
@click.option('--mission', default=None, help='Overarching mission for the audit (always visible to the Strategist)')
def agent(project_id: str, iterations: int | None, plan_n: int, time_limit: int | None, 
          config: str | None, debug: bool, mode: str | None, platform: str | None, model: str | None,
          strategist_platform: str | None, strategist_model: str | None,
          session: str | None, new_session: bool, session_private_hypotheses: bool,
          telemetry: bool, strategist_two_pass: bool, mission: str | None):
    """Run autonomous security analysis agent."""
    
    config_path = Path(config) if config else None
    
    runner = AgentRunner(project_id, config_path, iterations, time_limit, debug, platform, model, session=session, new_session=new_session, mode=mode)
    try:
        runner.mission = mission
    except Exception:
        pass
    
    if not runner.initialize():
        # Ensure CLI signals failure to callers (e.g., benchmark runner)
        raise click.Exit(1)
    
    # Optional telemetry: local-only HTTP SSE/control + instance registry
    tele = None
    try:
        if telemetry:
            try:
                from telemetry import TelemetryServer
                # Project dir used after initialize
                pd = None
                try:
                    pd = runner.project_dir if getattr(runner, 'project_dir', None) else None
                except Exception:
                    pd = None
                if pd is None:
                    # Fall back to resolving when not available
                    pd = Path(project_id) if Path(project_id).exists() else get_project_dir(project_id)
                tele = TelemetryServer(str(project_id), Path(pd))
                tele.start()
                # Emit a friendly boot event so telemetry streams show activity immediately
                try:
                    tele.publish({'type': 'status', 'message': 'audit session started', 'iteration': 0})
                except Exception:
                    pass
            except Exception:
                tele = None
        # Apply strategist overrides (update config before run if provided)
        if runner.config is None:
            runner.config = {}
        if 'models' not in runner.config:
            runner.config['models'] = {}
        if strategist_platform or strategist_model:
            runner.config['models'].setdefault('strategist', {})
            if strategist_platform:
                runner.config['models']['strategist']['provider'] = strategist_platform
            if strategist_model:
                runner.config['models']['strategist']['model'] = strategist_model
        # Apply strategist two-pass toggle if requested
        try:
            if strategist_two_pass:
                runner.config['strategist_two_pass_review'] = True
        except Exception:
            pass
        # Set strategist overrides then run
        if session_private_hypotheses and getattr(runner, 'agent', None):
            try:
                runner.agent.default_hypothesis_visibility = 'session'
            except Exception:
                pass
        # Inject telemetry into runner by monkey-patching a publisher
        if tele is not None:
            try:
                runner._telemetry_publish = tele.publish  # type: ignore[attr-defined]
            except Exception:
                pass
        # Wrap run to emit a start event around each investigation (no per-task completed noise)
        if tele is not None:
            # Monkey-patch a simple notifier into runner for per-investigation markers
            try:
                _orig_investigate = getattr(runner.agent, 'investigate') if getattr(runner, 'agent', None) else None
                if callable(_orig_investigate):
                    def _wrapped_investigate(prompt, *a, **kw):
                        try:
                            tele.publish({'type': 'status', 'message': f'starting: {prompt}', 'iteration': (kw.get('iteration') or 0)})
                        except Exception:
                            pass
                        return _orig_investigate(prompt, *a, **kw)
                    runner.agent.investigate = _wrapped_investigate  # type: ignore[attr-defined]
            except Exception:
                pass
        runner.run(plan_n=plan_n)
    except KeyboardInterrupt:
        console.print("\n[yellow]Agent interrupted by user[/yellow]")
        # Try to save partial results
        try:
            runner.finalize_tracking('interrupted')
        except Exception:
            pass
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        # Try to save partial results
        try:
            runner.finalize_tracking('failed')
        except Exception:
            pass
        raise
    finally:
        # Ensure telemetry shutdown and registry cleanup
        try:
            if tele is not None:
                tele.stop()
        except Exception:
            pass
