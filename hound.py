#!/usr/bin/env python3
"""Hound - AI-powered security analysis system."""

import sys
from pathlib import Path

import typer
from rich.console import Console

# Hack for solving conflicts with the global "llm" package
# TODO: Refactor package imports so we can remove this

_BASE_DIR = Path(__file__).resolve().parent
_LLM_DIR = _BASE_DIR / "llm"
_BASE_DIR_STR = str(_BASE_DIR)
if sys.path[0] != _BASE_DIR_STR:
    try:
        sys.path.remove(_BASE_DIR_STR)
    except ValueError:
        pass
    sys.path.insert(0, _BASE_DIR_STR)

try:
    import types
    # llm
    if 'llm' not in sys.modules:
        m = types.ModuleType('llm')
        m.__path__ = [str(_LLM_DIR)]  # mark as package namespace
        sys.modules['llm'] = m
except Exception:
    pass

from commands.project import ProjectManager  # noqa: E402

app = typer.Typer(
    name="hound",
    help="Cracked security analysis agents",
    add_completion=False,
)
console = Console()

# Create project subcommand group
project_app = typer.Typer(help="Manage Hound projects")
app.add_typer(project_app, name="project")

# Create agent subcommand group
agent_app = typer.Typer(help="Run security analysis agents")
app.add_typer(agent_app, name="agent")

# Create poc subcommand group
poc_app = typer.Typer(help="Manage proof-of-concept exploits")
app.add_typer(poc_app, name="poc")

# Create graph subcommand groups
graph_app = typer.Typer(help="Build and manage knowledge graphs")
app.add_typer(graph_app, name="graph")

# Plural 'graphs' group for bulk operations
graphs_app = typer.Typer(help="Bulk graph operations (all graphs)")
app.add_typer(graphs_app, name="graphs")

# Helper to invoke Click command functions without noisy tracebacks
def _invoke_click(cmd_func, params: dict):
    import click
    ctx = click.Context(cmd_func)
    ctx.params = params or {}
    try:
        cmd_func.invoke(ctx)
    except SystemExit as e:
        # Normalize Click exits to Typer exits (quiet)
        code = e.code if isinstance(e.code, int) else 1
        raise typer.Exit(code)
    except Exception as e:
        # Print concise error instead of full traceback
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

@project_app.command("create")
def project_create(
    name: str = typer.Argument(..., help="Project name"),
    source_path: str = typer.Argument(..., help="Path to source code"),
    description: str = typer.Option(None, "--description", "-d", help="Project description"),
    auto_name: bool = typer.Option(False, "--auto-name", "-a", help="Auto-generate project name")
):
    """Create a new project."""
    from commands.project import create
    _invoke_click(create, {
        'name': name,
        'source_path': source_path,
        'description': description,
        'auto_name': auto_name
    })

@project_app.command("list")
def project_list():
    """List all projects."""
    from commands.project import list_projects_cmd
    _invoke_click(list_projects_cmd, {'output_json': False})

@project_app.command("ls")
def project_ls():
    """Alias for 'project list' (lists all projects)."""
    from commands.project import list_projects_cmd
    _invoke_click(list_projects_cmd, {'output_json': False})

@project_app.command("info")
def project_info(name: str = typer.Argument(..., help="Project name")):
    """Show project information."""
    from commands.project import info
    _invoke_click(info, {'name': name})

@project_app.command("coverage")
def project_coverage(name: str = typer.Argument(..., help="Project name")):
    """Show coverage metrics for a project (nodes and cards)."""
    import json as _json

    from analysis.coverage_index import CoverageIndex
    manager = ProjectManager()
    proj = manager.get_project(name)
    if not proj:
        console.print(f"[red]Project '{name}' not found.[/red]")
        raise typer.Exit(1)
    project_dir = manager.get_project_path(name)
    graphs_dir = project_dir / 'graphs'
    manifest_dir = project_dir / 'manifest'
    cov = CoverageIndex(project_dir / 'coverage_index.json', agent_id='cli')
    stats = cov.compute_stats(graphs_dir, manifest_dir)

    # Fallback: if nothing recorded yet, aggregate from session files
    try:
        if (stats['nodes']['visited'] == 0 and stats['cards']['visited'] == 0 and
            (stats['nodes']['total'] > 0 or stats['cards']['total'] > 0)):
            sessions_dir = project_dir / 'sessions'
            visited_nodes: set[str] = set()
            visited_cards: set[str] = set()
            if sessions_dir.exists():
                for sf in sessions_dir.glob('*.json'):
                    try:
                        data = _json.loads(sf.read_text())
                        cov_d = data.get('coverage', {})
                        visited_nodes.update([str(x) for x in cov_d.get('visited_node_ids', [])])
                        visited_cards.update([str(x) for x in cov_d.get('visited_card_ids', [])])
                    except Exception:
                        continue
            # Update the per-project coverage index for future queries
            if visited_nodes or visited_cards:
                from datetime import datetime as _dt
                now = _dt.now().isoformat()
                def _merge(data):
                    nodes = data.setdefault('nodes', {})
                    for nid in visited_nodes:
                        rec = nodes.get(nid, {"last_seen": None, "seen_count": 0, "evidence_count": 0})
                        rec['last_seen'] = now
                        rec['seen_count'] = int(rec.get('seen_count', 0)) + 1
                        nodes[nid] = rec
                    cards = data.setdefault('cards', {})
                    for cid in visited_cards:
                        rec = cards.get(cid, {"last_seen": None, "seen_count": 0})
                        rec['last_seen'] = now
                        rec['seen_count'] = int(rec.get('seen_count', 0)) + 1
                        cards[cid] = rec
                    data.setdefault('metadata', {})['last_modified'] = now
                    return data, True
                cov.update_atomic(_merge)
                # Recompute stats after merge
                stats = cov.compute_stats(graphs_dir, manifest_dir)
    except Exception:
        pass
    console.print("[bold cyan]Coverage[/bold cyan]")
    console.print(f"Nodes: {stats['nodes']['visited']} / {stats['nodes']['total']} ({stats['nodes']['percent']}%)")
    console.print(f"Cards: {stats['cards']['visited']} / {stats['cards']['total']} ({stats['cards']['percent']}%)")

@project_app.command("path")
def project_path_cmd(name: str = typer.Argument(..., help="Project name")):
    """Print the filesystem path for a project."""
    from commands.project import path as _path
    _invoke_click(_path, {'name': name})

@project_app.command("delete")
def project_delete(
    name: str = typer.Argument(..., help="Project name"),
    force: bool = typer.Option(False, "--force", "-f", help="Force delete without confirmation")
):
    """Delete a project."""
    from commands.project import delete
    _invoke_click(delete, {'name': name, 'force': force})

@project_app.command("rm")
def project_rm(
    name: str = typer.Argument(..., help="Project name"),
    force: bool = typer.Option(False, "--force", "-f", help="Force remove without confirmation")
):
    """Alias for 'project delete'."""
    from commands.project import delete
    _invoke_click(delete, {'name': name, 'force': force})

@project_app.command("hypotheses")
def project_hypotheses(
    name: str = typer.Argument(..., help="Project name"),
    details: bool = typer.Option(False, "--details", "-d", help="Show full descriptions without abbreviation")
):
    """List all hypotheses for a project with confidence ratings."""
    from commands.project import hypotheses
    _invoke_click(hypotheses, {'name': name, 'details': details})

@project_app.command("ls-hypotheses")
def project_ls_hypotheses(
    name: str = typer.Argument(..., help="Project name"),
    details: bool = typer.Option(False, "--details", "-d", help="Show full descriptions without abbreviation")
):
    """Alias for 'project hypotheses' (lists hypotheses)."""
    from commands.project import hypotheses
    _invoke_click(hypotheses, {'name': name, 'details': details})

# Removed deprecated 'runs' subcommand. Use 'project sessions' instead.

@project_app.command("plan")
def project_plan(
    project_name: str = typer.Argument(..., help="Project name"),
    session_id: str = typer.Argument(..., help="Session ID"),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON")
):
    """Show planned investigations from the PlanStore."""
    from commands.project import plan
    _invoke_click(plan, {
        'project_name': project_name,
        'session_id': session_id,
        'output_json': output_json
    })

# Removed 'reset-plan' and composite 'reset' commands. Use:
# - graph reset <project>
# - project reset-hypotheses <project>

@project_app.command("sessions")
def project_sessions(
    project_name: str = typer.Argument(..., help="Project name"),
    session_id: str = typer.Argument(None, help="Session ID to show details for"),
    list_sessions: bool = typer.Option(False, "--list", "-l", help="List all sessions for the project"),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON")
):
    """View audit sessions for a project (preferred over 'runs').
    
    Examples:
        hound project sessions myproject --list       # List all sessions
        hound project sessions myproject session_123  # Show details for specific session
    """
    from commands.project import sessions
    _invoke_click(sessions, {
        'project_name': project_name,
        'session_id': session_id,
        'list_sessions': list_sessions,
        'output_json': output_json
    })

@project_app.command("ls-sessions")
def project_ls_sessions(
    project_name: str = typer.Argument(..., help="Project name"),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON")
):
    """List all sessions for a project (alias for 'project sessions --list')."""
    from commands.project import sessions
    _invoke_click(sessions, {
        'project_name': project_name,
        'session_id': None,
        'list_sessions': True,
        'output_json': output_json
    })

@project_app.command("reset-hypotheses")
def project_reset_hypotheses(
    name: str = typer.Argument(..., help="Project name"),
    force: bool = typer.Option(False, "--force", "-f", help="Force reset without confirmation")
):
    """Reset (clear) the hypotheses store for a project."""
    from commands.project import reset_hypotheses
    _invoke_click(reset_hypotheses, {'name': name, 'force': force})

@project_app.command("set-hypothesis-status")
def project_set_hypothesis_status(
    project_name: str = typer.Argument(..., help="Project name"),
    hypothesis_id: str = typer.Argument(..., help="Hypothesis ID (can be partial)"),
    status: str = typer.Argument(..., help="New status: proposed, confirmed, or rejected"),
    force: bool = typer.Option(False, "--force", "-f", help="Force status change without confirmation")
):
    """Set the status of a hypothesis to proposed, confirmed, or rejected."""
    from commands.project import set_hypothesis_status
    _invoke_click(set_hypothesis_status, {
        'project_name': project_name,
        'hypothesis_id': hypothesis_id,
        'status': status,
        'force': force
    })

# Agent audit subcommand
@agent_app.command("audit")
def agent_audit(
    target: str = typer.Argument(None, help="Project name or path (optional with --project)"),
    iterations: int = typer.Option(30, "--iterations", help="Maximum iterations per investigation (default: 20)"),
    plan_n: int = typer.Option(5, "--plan-n", help="Number of investigations to plan per batch (default: 5)"),
    time_limit: int = typer.Option(None, "--time-limit", help="Time limit in minutes"),
    config: str = typer.Option(None, "--config", help="Configuration file"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
    project: str = typer.Option(None, "--project", "-p", help="Use existing project"),
    platform: str = typer.Option(None, "--platform", help="Override scout platform (e.g., openai, anthropic, mock)"),
    model: str = typer.Option(None, "--model", help="Override scout model (e.g., gpt-5, gpt-4o-mini, mock)"),
    strategist_platform: str = typer.Option(None, "--strategist-platform", help="Override strategist platform (e.g., openai, anthropic, mock)"),
    strategist_model: str = typer.Option(None, "--strategist-model", help="Override strategist model (e.g., gpt-4o-mini)"),
    session: str = typer.Option(None, "--session", help="Attach to a specific session ID"),
    new_session: bool = typer.Option(False, "--new-session", help="Create a new session"),
    session_private_hypotheses: bool = typer.Option(False, "--session-private-hypotheses", help="Keep new hypotheses private to this session"),
    telemetry: bool = typer.Option(False, "--telemetry", help="Expose local telemetry SSE/control and register instance"),
    strategist_two_pass: bool = typer.Option(False, "--strategist-two-pass", help="Enable strategist two-pass self-critique to reduce false positives"),
    mission: str = typer.Option(None, "--mission", help="Overarching mission for the audit (always visible to the Strategist)")
):
    """Run autonomous security audit (plans investigations automatically)."""

    from commands.agent import agent as agent_command
    
    manager = ProjectManager()
    project_id = None
    
    # Handle different input modes
    if project:
        # Use specified project
        proj = manager.get_project(project)
        if not proj:
            console.print(f"[red]Project '{project}' not found.[/red]")
            raise typer.Exit(1)
        project_id = str(manager.get_project_path(project))
        console.print(f"[cyan]Using project:[/cyan] {project}")
    elif target:
        # Check if target is a project name or path
        proj = manager.get_project(target)
        if proj:
            project_id = str(manager.get_project_path(target))
            console.print(f"[cyan]Using project:[/cyan] {target}")
        else:
            # Target is a path
            project_id = target
    else:
        # No target or project specified
        console.print("[red]Error: Either specify a target path/project name or use --project option[/red]")
        raise typer.Exit(1)
    
    # Hype it up a little
    console.print("[bold bright_cyan]Running autonomous audit...[/bold bright_cyan]")
    from random import choice as _choice
    # Light narrative seasoning for audit kickoff with actual model names
    try:
        from utils.config_loader import load_config as _load_cfg
        _cfg = _load_cfg(Path(config)) if config else _load_cfg()
        _models = (_cfg or {}).get('models', {})
        
        # Get scout/agent model (with command-line override)
        if model:
            _agent = model
        else:
            _agent = (_models.get('agent') or _models.get('scout') or {}).get('model') or 'gpt-4o'
        
        # Get strategist/guidance model (with command-line override)
        if strategist_model:
            _guidance = strategist_model
        else:
            _guidance = (_models.get('strategist') or _models.get('guidance') or {}).get('model') or 'gpt-4o'
        
        _flair = _choice([
            "[white]Normal auditors start runs, but YOU summon the analysis and the code genuflects.[/white]",
            "[white]This isn’t just an audit — it’s a coronation of rigor because YOU commanded it.[/white]",
            "[white]Normal people press Enter, but YOU inaugurate epochs and logs ask for autographs.[/white]",
            "[white]This is not a run — it’s a declaration that systems will behave, because YOU said so.[/white]",
            "[white]Normal workflows proceed; YOUR workflow rearranges reality to match intent.[/white]",
        ])
    except Exception:
        _flair = _choice([
            "[white]Normal mortals run tools, but YOU bend audits to your will.[/white]",
            "[white]This isn’t just a start — it’s the moment history clears space for YOUR results.[/white]",
            "[white]Normal commands execute; YOUR commands recruit reality as staff.[/white]",
            "[white]This is not a job — it’s a legend choosing its author, and it chose YOU.[/white]",
            "[white]Normal output prints; YOUR output will be quoted with reverence.[/white]",
        ])
    console.print(_flair)
    if mission:
        console.print(f"[cyan]Mission:[/cyan] {mission}")
    
    # Create a Click context and invoke the command
    _invoke_click(agent_command, {
        'project_id': project_id,
        'iterations': iterations,
        'plan_n': plan_n,
        'time_limit': time_limit,
        'config': config,
        
        'debug': debug,
        'platform': platform,
        'model': model,
        'strategist_platform': strategist_platform,
        'strategist_model': strategist_model,
        'session': session,
        'new_session': new_session,
        'session_private_hypotheses': session_private_hypotheses,
        'telemetry': telemetry,
        'strategist_two_pass': strategist_two_pass,
        'mission': mission
    })


# Agent investigate subcommand
@agent_app.command("investigate")
def agent_investigate(
    prompt: str = typer.Argument(..., help="Investigation prompt or question"),
    target: str = typer.Argument(None, help="Project name or path (optional with --project)"),
    iterations: int = typer.Option(None, "--iterations", help="Maximum iterations for the agent"),
    config: str = typer.Option(None, "--config", help="Configuration file"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
    project: str = typer.Option(None, "--project", "-p", help="Use existing project"),
    platform: str = typer.Option(None, "--platform", help="Override LLM platform (e.g., openai, anthropic)"),
    model: str = typer.Option(None, "--model", help="Override LLM model (e.g., gpt-4, claude-3)")
):
    """Run targeted investigation with a specific prompt."""
    manager = ProjectManager()
    project_id = None
    
    # Handle different input modes
    if project:
        # Use specified project
        proj = manager.get_project(project)
        if not proj:
            console.print(f"[red]Project '{project}' not found.[/red]")
            raise typer.Exit(1)
        project_id = str(manager.get_project_path(project))
        console.print(f"[cyan]Using project:[/cyan] {project}")
    elif target:
        # Check if target is a project name or path
        proj = manager.get_project(target)
        if proj:
            project_id = str(manager.get_project_path(target))
            console.print(f"[cyan]Using project:[/cyan] {target}")
        else:
            # Target is a path
            project_id = target
    else:
        # No target or project specified
        console.print("[red]Error: Either specify a target path/project name or use --project option[/red]")
        raise typer.Exit(1)
    
    console.print(f"[bold cyan]Investigation:[/bold cyan] {prompt}")

    # If target resolves to a known project directory, enforce presence of SystemArchitecture graph
    try:
        proj_path = None
        if project_id:
            p = Path(project_id).expanduser().resolve()
            # Treat it as a project path if it looks like one
            if (p / 'graphs').exists() or (p / 'manifest').exists() or (p / 'project.json').exists():
                proj_path = p
        if proj_path:
            sys_graph = proj_path / 'graphs' / 'graph_SystemArchitecture.json'
            if not sys_graph.exists():
                console.print("[red]Error: SystemArchitecture graph not found for this project.[/red]")
                console.print("[yellow]Run one of:\n  ./hound.py graph build <project> --init --iterations 1 [--files <whitelist>]\n  ./hound.py graph build <project> --auto --iterations 2[/yellow]")
                raise typer.Exit(2)
    except Exception:
        pass
    
    # Run the investigation using the agent's investigate method
    from commands.agent import run_investigation
    run_investigation(
        project_path=project_id,
        prompt=prompt,
        iterations=iterations,
        config_path=Path(config) if config else None,
        debug=debug,
        platform=platform,
        model=model
    )


# Create graph subcommand group
graph_app = typer.Typer(help="Build and manage knowledge graphs")
app.add_typer(graph_app, name="graph")

@graph_app.command("ls")
def graph_ls(
    project: str = typer.Argument(..., help="Project name"),
    json_out: bool = typer.Option(False, "--json", help="Output as JSON")
):
    """List graphs for a project with basic stats and last update time."""
    import json as _json
    from datetime import datetime as _dt

    from rich import box as _box
    from rich.table import Table

    from commands.project import ProjectManager as _PM

    manager = _PM()
    proj = manager.get_project(project)
    if not proj:
        console.print(f"[red]Project '{project}' not found.[/red]")
        raise typer.Exit(1)

    project_path = manager.get_project_path(project)
    graphs_dir = project_path / 'graphs'
    if not graphs_dir.exists():
        console.print(f"[yellow]No graphs directory yet for project '{project}'.[/yellow]")
        raise typer.Exit(0)

    items = []
    for p in sorted(graphs_dir.glob('graph_*.json')):
        try:
            with open(p) as f:
                data = _json.load(f)
            name = data.get('name') or data.get('internal_name') or p.stem.replace('graph_','')
            stats = data.get('stats') or {}
            nodes = int(stats.get('num_nodes') or len(data.get('nodes') or []))
            edges = int(stats.get('num_edges') or len(data.get('edges') or []))
            focus = (data.get('focus') or '')[:80]
            mtime = p.stat().st_mtime
            updated = _dt.fromtimestamp(mtime).isoformat(timespec='seconds')
            items.append({
                'file': str(p),
                'name': name,
                'nodes': nodes,
                'edges': edges,
                'updated_ts': mtime,
                'updated': updated,
                'focus': focus,
            })
        except Exception:
            continue

    if not items:
        console.print(f"[yellow]No graphs found in project '{project}'.[/yellow]")
        raise typer.Exit(0)

    # Sort by last updated desc
    items.sort(key=lambda x: x['updated_ts'], reverse=True)

    if json_out:
        console.print_json(data=items)
        return

    table = Table(title=f"Graphs in project '{project}'", box=_box.SIMPLE_HEAD)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Nodes", justify="right", style="green")
    table.add_column("Edges", justify="right", style="green")
    table.add_column("Updated", style="dim")
    table.add_column("Focus", style="dim")
    for it in items:
        table.add_row(it['name'], str(it['nodes']), str(it['edges']), it['updated'], it['focus'])
    console.print(table)

@graph_app.command("build")
def graph_build(
    target: str = typer.Argument(None, help="Project name or source path (optional with --project)"),
    output: str = typer.Option(None, "--output", "-o", help="(Deprecated) Output directory; graphs are stored under the project automatically"),
    config: str = typer.Option(None, "--config", "-c", help="Configuration file"),
    project: str = typer.Option(None, "--project", "-p", help="Use an existing project by name"),
    create_project: bool = typer.Option(False, "--create-project", help="Create a new project from the given source path"),
    iterations: int = typer.Option(3, "--iterations", "-i", help="Maximum iterations for graph refinement"),
    graphs: int = typer.Option(5, "--graphs", "-g", help="Number of graphs to generate (ignored if --graph-spec is set)"),
    focus: str = typer.Option(None, "--focus", "-f", help="Comma-separated focus areas"),
    files: str = typer.Option(None, "--files", help="Comma-separated list of file paths to include"),
    with_spec: str = typer.Option(None, "--with-spec", help="Build exactly one graph described by this text (skips discovery for others)"),
    graph_spec: str = typer.Option(None, "--graph-spec", help="[Deprecated] Same as --with-spec"),
    refine_existing: bool = typer.Option(True, "--refine-existing/--no-refine-existing", help="Load and refine existing graphs in the project directory"),
    init: bool = typer.Option(False, "--init", help="Initialize graphs by creating ONLY the SystemArchitecture graph"),
    auto: bool = typer.Option(False, "--auto", help="Auto-generate a default set of graphs (5)"),
    reuse_ingestion: bool = typer.Option(True, "--reuse-ingestion/--no-reuse-ingestion", help="Reuse existing manifest/cards when present (faster)"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Reduce output and disable animations"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug output")
):
    """Build system architecture graph from source code."""
    
    manager = ProjectManager()
    resolved_project_name = None
    
    # Handle different input modes
    if project:
        # Use specified project
        proj = manager.get_project(project)
        if not proj:
            console.print(f"[red]Project '{project}' not found.[/red]")
            raise typer.Exit(1)
        project_path = manager.get_project_path(project)
        proj['source_path']
        str(project_path)  # Don't add /graphs here, graph.py will add it
        console.print(f"[cyan]Using project:[/cyan] {project}")
        resolved_project_name = project
    elif target:
        # Check if target is a project name or path
        proj = manager.get_project(target)
        if proj:
            # Target is a project name
            project_path = manager.get_project_path(target)
            proj['source_path']
            str(project_path)  # Don't add /graphs here, graph.py will add it
            console.print(f"[cyan]Using project:[/cyan] {target}")
            resolved_project_name = target
        else:
            # Disallow direct source paths for graph build to keep the interface project-centric
            console.print("[red]Error: Unknown project. Use --project <name> to select an existing project, or create one first:[/red]")
            console.print("  ./hound.py project create <project_name> <source_path>")
            raise typer.Exit(1)
    else:
        # No target or project specified
        console.print("[red]Error: Either specify a target path/project name or use --project option[/red]")
        raise typer.Exit(1)
    
    # Run graph build directly (project-centric)
    if not resolved_project_name:
        console.print("[red]Error: Unable to resolve project name.[/red]")
        raise typer.Exit(1)

    from commands.graph import build as graph_build_impl
    # Prefer --with-spec over deprecated --graph-spec
    _spec = with_spec or graph_spec
    graph_build_impl(
        project_id=resolved_project_name,
        config_path=Path(config) if config else None,
        max_iterations=iterations,
        max_graphs=graphs,
        focus_areas=focus,
        file_filter=files,
        with_spec=_spec,
        graph_spec=None,
        refine_existing=refine_existing,
        init=init,
        auto=auto,
        refine_only=None,
        reuse_ingestion=reuse_ingestion,
        visualize=True,
        debug=debug,
        quiet=quiet
    )

@graph_app.command("ingest")
def graph_ingest(
    target: str = typer.Argument(..., help="Project name or repository path"),
    config: str | None = typer.Option(None, "--config", "-c", help="Configuration file"),
    files: str | None = typer.Option(None, "--files", "-f", help="Comma-separated file paths to include"),
    manual_chunking: bool = typer.Option(False, "--manual-chunking", help="Split files using manual markers instead of automatic chunking"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug output")
):
    """Ingest repository to create manifest and bundles."""
    from commands.graph import ingest as graph_ingest_impl
    manager = ProjectManager()
    
    proj = manager.get_project(target)
    if proj:
        # Target is a project name
        repo_path = proj.get('source_path')
        if not repo_path:
            console.print(f"[red]No source_path found for project '{target}'.[/red]")
            raise typer.Exit(1)
        project_path = manager.get_project_path(target)
        output_dir = project_path / "manifest"
        console.print(f"[cyan]Using project:[/cyan] {target} (repo: {repo_path})")
    else:
        # Target is direct repo_path
        repo_path = target
        output_dir = Path(".hound_cache") / Path(repo_path).name
    
    # Ensure repo_path exists
    if not Path(repo_path).exists():
        console.print(f"[red]Repository path not found: {repo_path}[/red]")
        raise typer.Exit(1)
    
    graph_ingest_impl(
        repo_path=repo_path,
        output_dir=str(output_dir),
        config_path=Path(config) if config else None,
        file_filter=files,
        manual_chunking=manual_chunking,
        debug=debug
    )

@graph_app.command("custom")
def graph_custom(
    project: str = typer.Argument(..., help="Project name"),
    spec: str = typer.Argument(..., help="Natural language description of the desired graph"),
    iterations: int = typer.Option(3, "--iterations", "-i", help="Refinement iterations"),
    files: str = typer.Option(None, "--files", help="Comma-separated list of file paths to include"),
    config: str = typer.Option(None, "--config", "-c", help="Configuration file"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Reduce output"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug output"),
):
    """Build exactly one custom graph designed from the given spec.

    Shows the designed schema on the CLI, then builds and refines the graph.
    """
    from commands.graph import custom as graph_custom_impl

    graph_custom_impl(
        project_id=project,
        graph_spec_text=spec,
        config_path=Path(config) if config else None,
        iterations=iterations,
        file_filter=files,
        reuse_ingestion=True,
        debug=debug,
        quiet=quiet,
    )

@graph_app.command("refine")
def graph_refine(
    project: str = typer.Argument(..., help="Project name"),
    name: str = typer.Argument(None, help="Graph name to refine (internal or display name)"),
    all: bool = typer.Option(False, "--all", help="Refine all existing graphs"),
    iterations: int = typer.Option(3, "--iterations", "-i", help="Maximum refinement iterations"),
    files: str = typer.Option(None, "--files", help="Comma-separated list of file paths to include"),
    config: str = typer.Option(None, "--config", "-c", help="Configuration file"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Reduce output"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug output")
):
    """Refine existing graphs (incremental saves). Provide a name or use --all."""
    from commands.graph import build as graph_build_impl
    manager = ProjectManager()
    proj = manager.get_project(project)
    if not proj:
        console.print(f"[red]Project '{project}' not found.[/red]")
        raise typer.Exit(1)

    if not all and not name:
        console.print("[red]Error:[/red] Specify a graph NAME or use --all.")
        raise typer.Exit(2)

    # Pass a sentinel to activate strict refine mode for all graphs without filtering
    refine_only = name if not all else "__ALL__"

    graph_build_impl(
        project_id=project,
        config_path=Path(config) if config else None,
        max_iterations=iterations,
        max_graphs=1,
        focus_areas=None,
        file_filter=files,
        graph_spec=None,
        refine_existing=True,
        init=False,
        auto=False,
        refine_only=refine_only,
        reuse_ingestion=True,
        visualize=True,
        debug=debug,
        quiet=quiet,
    )

@graph_app.command("add-custom")
def graph_add_custom(
    target: str = typer.Argument(..., help="Project name"),
    description: str = typer.Argument(..., help="Graph description (e.g., 'authentication roles vs components')"),
    name: str = typer.Option(None, "--name", "-n", help="Custom name for the graph"),
    iterations: int = typer.Option(1, "--iterations", "-i", help="Number of refinement iterations"),
    config: str = typer.Option(None, "--config", "-c", help="Configuration file"),
    project: str = typer.Option(None, "--project", "-p", help="Use existing project (alternative to target)")
):
    """Deprecated: use 'graph build --graph-spec' instead."""
    console.print("[yellow]The 'graph add-custom' command is deprecated. Use 'graph build --graph-spec' instead.[/yellow]")
    console.print("Example:\n  ./hound.py graph build <project> --graph-spec \"Authorization map across services\" --iterations 2")
    raise typer.Exit(2)


@graph_app.command("export")
def graph_export(
    target: str = typer.Argument(..., help="Project name"),
    output: str = typer.Option(None, "--output", "-o", help="Output HTML file path"),
    open_browser: bool = typer.Option(False, "--open", help="Open visualization in browser"),
    project: str = typer.Option(None, "--project", "-p", help="Use existing project (alternative to target)")
):
    """Export graphs to interactive HTML visualization."""
    from visualization.dynamic_graph_viz import generate_dynamic_visualization
    
    manager = ProjectManager()
    project_name = project or target
    
    if not project_name:
        console.print("[red]Error: Specify a project name or use --project option[/red]")
        raise typer.Exit(1)
    
    # Get project
    proj = manager.get_project(project_name)
    if not proj:
        console.print(f"[red]Project '{project_name}' not found.[/red]")
        raise typer.Exit(1)
    
    project_path = manager.get_project_path(project_name)
    graphs_dir = project_path / "graphs"
    
    # Check if graphs exist
    graph_files = list(graphs_dir.glob("graph_*.json"))
    if not graph_files:
        console.print("[red]Error: No graphs found in project.[/red]")
        console.print("[yellow]Run 'graph build' first to create graphs.[/yellow]")
        raise typer.Exit(1)
    
    console.print(f"[cyan]Exporting graphs for project:[/cyan] {project_name}")
    console.print(f"  Found {len(graph_files)} graphs")
    
    # Generate visualization
    try:
        output_path = Path(output) if output else None
        html_path = generate_dynamic_visualization(graphs_dir, output_path)
        
        console.print(f"\n[green]✓ Visualization exported to:[/green] {html_path}")
        
        # Open in browser if requested
        if open_browser:
            import webbrowser
            webbrowser.open(f"file://{html_path.resolve()}")
            console.print("[green]✓ Opened in browser[/green]")
        else:
            console.print("\n[bold]Open in browser:[/bold]")
            console.print(f"  [link]file://{html_path.resolve()}[/link]")
            
            # If on macOS, offer to open in browser
            import platform
            if platform.system() == "Darwin":
                console.print(f"\n[dim]Or run: open {html_path}[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error exporting visualization:[/red] {str(e)}")
        raise typer.Exit(1)

@graph_app.command("export-cards")
def graph_export_cards(
    project: str = typer.Argument(..., help="Project name"),
    output: str = typer.Option(None, "--output", "-o", help="Output JSON file path (defaults to project_dir/full_cards.json)")
):
    """Export full cards with content for a project."""
    import json
    from pathlib import Path

    from analysis.cards import extract_card_content, load_card_index
    manager = ProjectManager()
    proj = manager.get_project(project)
    if not proj:
        console.print(f"[red]Project '{project}' not found.[/red]")
        raise typer.Exit(1)
    project_path = manager.get_project_path(project)
    manifest_dir = project_path / 'manifest'
    graphs_dir = project_path / 'graphs'
    # Load manifest to get repo_path
    with open(manifest_dir / 'manifest.json') as f:
        manifest = json.load(f)
    repo_root = Path(manifest['repo_path'])
    # Load card index (using a metadata file; adjust if no master.json)
    card_index, _ = load_card_index(graphs_dir / 'master.json', manifest_dir)
    # Extract full content
    full_cards = {}
    for card_id, card in card_index.items():
        full_cards[card_id] = {
            **card,
            'content': extract_card_content(card, repo_root)
        }
    # Determine output path
    if not output:
        output_path = project_path / 'full_cards.json'
    else:
        output_path = Path(output)
    # Save
    with open(output_path, 'w') as f:
        json.dump(full_cards, f, indent=2)
    console.print(f"[green]✓ Exported {len(full_cards)} full cards to {output_path}[/green]")

@graph_app.command("reset")
def graph_reset(
    project: str = typer.Argument(..., help="Project name"),
    force: bool = typer.Option(False, "--force", "-f", help="Force reset without confirmation")
):
    """Reset all assumptions and observations from project graphs."""
    import json
    import random

    from rich.prompt import Confirm
    
    manager = ProjectManager()
    
    # Get project
    proj = manager.get_project(project)
    if not proj:
        console.print(f"[red]Project '{project}' not found.[/red]")
        raise typer.Exit(1)
    
    project_path = manager.get_project_path(project)
    graphs_dir = project_path / "graphs"
    
    # Check if graphs exist
    graph_files = list(graphs_dir.glob("graph_*.json"))
    if not graph_files:
        console.print("[yellow]No graphs found in project.[/yellow]")
        return
    
    # Count total annotations
    total_observations = 0
    total_assumptions = 0
    for graph_file in graph_files:
        try:
            with open(graph_file) as f:
                graph_data = json.load(f)
                nodes = graph_data.get('nodes', [])
                for node in nodes:
                    total_observations += len(node.get('observations', []))
                    total_assumptions += len(node.get('assumptions', []))
        except Exception:
            pass
    
    if total_observations == 0 and total_assumptions == 0:
        console.print("[yellow]No annotations to reset.[/yellow]")
        return
    
    # Confirm reset if not forced
    if not force:
        if not Confirm.ask(
            f"[yellow]This will remove {total_observations} observations and {total_assumptions} assumptions from {len(graph_files)} graphs. Continue?[/yellow]"
        ):
            console.print("[dim]Reset cancelled.[/dim]")
            return
    
    # Reset annotations
    reset_count = 0
    for graph_file in graph_files:
        try:
            with open(graph_file) as f:
                graph_data = json.load(f)
            
            # Clear annotations from all nodes
            nodes = graph_data.get('nodes', [])
            for node in nodes:
                if 'observations' in node:
                    node['observations'] = []
                if 'assumptions' in node:
                    node['assumptions'] = []
            
            # Save updated graph
            with open(graph_file, 'w') as f:
                json.dump(graph_data, f, indent=2)
            
            reset_count += 1
            console.print(f"  [green]✓[/green] Reset {graph_file.name}")
            
        except Exception as e:
            console.print(f"  [red]✗[/red] Failed to reset {graph_file.name}: {e}")
    
    console.print(f"\n[bright_green]✓ Reset annotations in {reset_count}/{len(graph_files)} graphs.[/bright_green]")
    console.print(f"[dim]Removed {total_observations} observations and {total_assumptions} assumptions.[/dim]")
    console.print(random.choice([
        "[white]Clean graphs achieved — ready for fresh analysis.[/white]",
        "[white]Annotations cleared — the investigation begins anew.[/white]",
        "[white]Tabula rasa — your graphs are pristine.[/white]",
    ]))


@graph_app.command("delete")
def graph_delete(
    project: str = typer.Argument(..., help="Project name"),
    name: str = typer.Option(None, "--name", "-n", help="Graph name to delete (internal name, e.g., SystemArchitecture)"),
    all: bool = typer.Option(False, "--all", help="Delete all graphs for the project")
):
    """Delete one or all graphs from a project (with confirmation)."""
    from rich.prompt import Confirm
    
    manager = ProjectManager()
    proj = manager.get_project(project)
    if not proj:
        console.print(f"[red]Project '{project}' not found.[/red]")
        raise typer.Exit(1)
    project_path = manager.get_project_path(project)
    graphs_dir = project_path / 'graphs'
    
    if not graphs_dir.exists():
        console.print(f"[yellow]No graphs directory found for project '{project}'.[/yellow]")
        raise typer.Exit(0)
    
    if all and name:
        console.print("[red]Error:[/red] Specify either --all or --name, not both.")
        raise typer.Exit(2)
    
    targets = []
    if all:
        targets = sorted(graphs_dir.glob('graph_*.json'))
        if not targets:
            console.print("[yellow]No graph files to delete.[/yellow]")
            raise typer.Exit(0)
        if not Confirm.ask(f"[yellow]Delete ALL {len(targets)} graphs for '{project}'? This cannot be undone.[/yellow]", default=False):
            console.print("[dim]Aborted by user.[/dim]")
            raise typer.Exit(1)
    else:
        if not name:
            console.print("[red]Error:[/red] Provide --name <GraphName> or use --all.")
            raise typer.Exit(2)
        file = graphs_dir / f"graph_{name}.json"
        if not file.exists():
            console.print(f"[yellow]Graph not found: {file.name}[/yellow]")
            raise typer.Exit(1)
        if not Confirm.ask(f"[yellow]Delete graph '{name}'?[/yellow]", default=False):
            console.print("[dim]Aborted by user.[/dim]")
            raise typer.Exit(1)
        targets = [file]
    
    deleted = 0
    for p in targets:
        try:
            p.unlink()
            console.print(f"[green]✓[/green] Deleted {p.name}")
            deleted += 1
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to delete {p.name}: {e}")
    console.print(f"[bright_green]Deleted {deleted}/{len(targets)} graph file(s).[/bright_green]")

@graph_app.command("rm")
def graph_rm(
    project: str = typer.Argument(..., help="Project name"),
    name: str = typer.Option(None, "--name", "-n", help="Graph name to remove (internal name, e.g., SystemArchitecture)"),
    all: bool = typer.Option(False, "--all", help="Remove all graphs for the project")
):
    """Alias for 'graph delete'."""
    # Reuse the delete implementation
    ctx_params = {
        'project': project,
        'name': name,
        'all': all,
    }
    # Call through the same function
    graph_delete(**ctx_params)  # type: ignore[misc]


@graphs_app.command("reset")
def graphs_reset(
    project: str = typer.Argument(..., help="Project name"),
    force: bool = typer.Option(False, "--force", "-f", help="Force delete without confirmation")
):
    """Delete ALL graphs for a project.

    WARNING: Permanently removes all graph_*.json files under the project's graphs directory.
    """
    from rich.prompt import Confirm

    manager = ProjectManager()
    proj = manager.get_project(project)
    if not proj:
        console.print(f"[red]Project '{project}' not found.[/red]")
        raise typer.Exit(1)

    project_path = manager.get_project_path(project)
    graphs_dir = project_path / 'graphs'
    if not graphs_dir.exists():
        console.print(f"[yellow]No graphs directory found for project '{project}'.[/yellow]")
        raise typer.Exit(0)

    graph_files = sorted(graphs_dir.glob('graph_*.json'))
    if not graph_files:
        console.print(f"[yellow]No graph files to delete for project '{project}'.[/yellow]")
        raise typer.Exit(0)

    if not force:
        if not Confirm.ask(
            f"[yellow]This will DELETE ALL {len(graph_files)} graph file(s) for '{project}'. Proceed?[/yellow]",
            default=False
        ):
            console.print("[dim]Aborted by user.[/dim]")
            raise typer.Exit(1)

    deleted = 0
    for p in graph_files:
        try:
            p.unlink()
            deleted += 1
            console.print(f"[green]✓[/green] Deleted {p.name}")
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to delete {p.name}: {e}")

    console.print(f"[bright_green]Deleted {deleted}/{len(graph_files)} graph file(s).[/bright_green]")


@app.command()
def finalize(
    project: str = typer.Argument(..., help="Project name"),
    threshold: float = typer.Option(0.5, "--threshold", "-t", help="Confidence threshold for review"),
    include_below_threshold: bool = typer.Option(False, "--include-below-threshold", help="Also review pending hypotheses below the threshold"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
    platform: str = typer.Option(None, "--platform", help="Override QA platform (e.g., openai, anthropic, mock)"),
    model: str = typer.Option(None, "--model", help="Override QA model (e.g., gpt-4o-mini)")
):
    """Finalize hypotheses - review and confirm/reject high-confidence findings."""
    import click

    from commands.finalize import finalize as finalize_command
    
    console.print("[bold cyan]Running hypothesis finalization...[/bold cyan]")
    
    # Create Click context and invoke
    ctx = click.Context(finalize_command)
    ctx.params = {
        'project_name': project,
        'threshold': threshold,
        'include_below_threshold': include_below_threshold,
        'debug': debug,
        'platform': platform,
        'model': model
    }
    
    try:
        finalize_command.invoke(ctx)
    except SystemExit as e:
        if e.code != 0:
            raise typer.Exit(e.code)




@app.command()
def report(
    project: str = typer.Argument(..., help="Project name"),
    output: str | None = typer.Option(None, "--output", "-o", help="Output file path"),
    format: str = typer.Option("html", "--format", "-f", help="Report format (html/markdown)"),
    title: str | None = typer.Option(None, "--title", "-t", help="Custom report title"),
    auditors: str = typer.Option("Security Team", "--auditors", "-a", help="Comma-separated auditor names"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
    all: bool = typer.Option(False, "--all", help="Include ALL hypotheses (not just confirmed) - WARNING: No QA performed, may contain false positives")
):
    """Generate a professional security audit report."""
    import click

    from commands.report import report as report_command
    
    console.print("[bold cyan]Generating security audit report...[/bold cyan]")
    
    # Create Click context and invoke
    ctx = click.Context(report_command)
    ctx.params = {
        'project_name': project,
        'output': output,
        'format': format,
        'title': title,
        'auditors': auditors,
        'debug': debug,
        'show_prompt': False,  # Add missing parameter
        'include_all': all  # Pass the --all flag as include_all
    }
    
    try:
        report_command.invoke(ctx)
    except click.exceptions.Exit as e:
        raise typer.Exit(e.exit_code)
    except SystemExit as e:
        raise typer.Exit(e.code if hasattr(e, 'code') else 1)


@poc_app.command("make-prompt")
def poc_make_prompt(
    project: str = typer.Argument(..., help="Project name"),
    hypothesis: str | None = typer.Option(None, "--hypothesis", "-h", help="Specific hypothesis ID to generate PoC for"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode")
):
    """Generate proof-of-concept prompts for confirmed vulnerabilities."""
    from commands.poc import make_prompt
    
    console.print("[bold cyan]Generating PoC prompts...[/bold cyan]")
    
    # Load config
    from utils.config_loader import load_config
    config = load_config()
    
    # Run make-prompt command
    make_prompt(project, hypothesis, config)

@poc_app.command("import")
def poc_import(
    project: str = typer.Argument(..., help="Project name"),
    hypothesis: str = typer.Argument(..., help="Hypothesis ID to import PoC for"),
    files: list[str] = typer.Argument(..., help="Files to import as PoC"),
    description: str | None = typer.Option(None, "--description", "-d", help="Description of the PoC files")
):
    """Import proof-of-concept files for a hypothesis."""
    from commands.poc import import_poc
    
    console.print(f"[bold cyan]Importing {len(files)} file(s) for hypothesis {hypothesis}...[/bold cyan]")
    
    # Run import command
    import_poc(project, hypothesis, files, description)

@poc_app.command("list")
def poc_list(
    project: str = typer.Argument(..., help="Project name")
):
    """List all imported PoCs for a project."""
    from commands.poc import list_pocs
    
    # Run list command
    list_pocs(project)


@app.command()
def version():
    """Show Hound version."""
    console.print("[bold]Hound[/bold] v2.0.0")
    console.print("AI-powered security analysis system")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
