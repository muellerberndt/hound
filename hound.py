#!/usr/bin/env python3
"""Hound - AI-powered security analysis system."""

import sys
import os
import typer
from pathlib import Path
from typing import Optional
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
    if "llm" not in sys.modules:
        m = types.ModuleType("llm")
        m.__path__ = [str(_LLM_DIR)]  # mark as package namespace
        sys.modules["llm"] = m
except Exception:
    pass

from commands.graph import build, ingest
from commands.project import ProjectManager

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
    auto_name: bool = typer.Option(False, "--auto-name", "-a", help="Auto-generate project name"),
):
    """Create a new project."""
    from commands.project import create

    _invoke_click(
        create,
        {
            "name": name,
            "source_path": source_path,
            "description": description,
            "auto_name": auto_name,
        },
    )


@project_app.command("list")
def project_list():
    """List all projects."""
    from commands.project import list_projects_cmd

    _invoke_click(list_projects_cmd, {"output_json": False})


@project_app.command("info")
def project_info(name: str = typer.Argument(..., help="Project name")):
    """Show project information."""
    from commands.project import info

    _invoke_click(info, {"name": name})


@project_app.command("coverage")
def project_coverage(name: str = typer.Argument(..., help="Project name")):
    """Show coverage metrics for a project (nodes and cards)."""
    from analysis.coverage_index import CoverageIndex

    manager = ProjectManager()
    proj = manager.get_project(name)
    if not proj:
        console.print(f"[red]Project '{name}' not found.[/red]")
        raise typer.Exit(1)
    project_dir = manager.get_project_path(name)
    graphs_dir = project_dir / "graphs"
    manifest_dir = project_dir / "manifest"
    cov = CoverageIndex(project_dir / "coverage_index.json", agent_id="cli")
    stats = cov.compute_stats(graphs_dir, manifest_dir)
    console.print("[bold cyan]Coverage[/bold cyan]")
    console.print(
        f"Nodes: {stats['nodes']['visited']} / {stats['nodes']['total']} ({stats['nodes']['percent']}%)"
    )
    console.print(
        f"Cards: {stats['cards']['visited']} / {stats['cards']['total']} ({stats['cards']['percent']}%)"
    )


@project_app.command("path")
def project_path_cmd(name: str = typer.Argument(..., help="Project name")):
    """Print the filesystem path for a project."""
    from commands.project import path as _path

    _invoke_click(_path, {"name": name})


@project_app.command("delete")
def project_delete(
    name: str = typer.Argument(..., help="Project name"),
    force: bool = typer.Option(False, "--force", "-f", help="Force delete without confirmation"),
):
    """Delete a project."""
    from commands.project import delete

    _invoke_click(delete, {"name": name, "force": force})


@project_app.command("hypotheses")
def project_hypotheses(
    name: str = typer.Argument(..., help="Project name"),
    details: bool = typer.Option(
        False, "--details", "-d", help="Show full descriptions without abbreviation"
    ),
):
    """List all hypotheses for a project with confidence ratings."""
    from commands.project import hypotheses

    _invoke_click(hypotheses, {"name": name, "details": details})


# Removed deprecated 'runs' subcommand. Use 'project sessions' instead.


@project_app.command("plan")
def project_plan(
    project_name: str = typer.Argument(..., help="Project name"),
    session_id: str = typer.Argument(..., help="Session ID"),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Show planned investigations from the PlanStore."""
    from commands.project import plan

    _invoke_click(
        plan, {"project_name": project_name, "session_id": session_id, "output_json": output_json}
    )


# Removed 'reset-plan' and composite 'reset' commands. Use:
# - graph reset <project>
# - project reset-hypotheses <project>


@project_app.command("sessions")
def project_sessions(
    project_name: str = typer.Argument(..., help="Project name"),
    session_id: str = typer.Argument(None, help="Session ID to show details for"),
    list_sessions: bool = typer.Option(
        False, "--list", "-l", help="List all sessions for the project"
    ),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """View audit sessions for a project (preferred over 'runs').

    Examples:
        hound project sessions myproject --list       # List all sessions
        hound project sessions myproject session_123  # Show details for specific session
    """
    from commands.project import sessions

    _invoke_click(
        sessions,
        {
            "project_name": project_name,
            "session_id": session_id,
            "list_sessions": list_sessions,
            "output_json": output_json,
        },
    )


@project_app.command("reset-hypotheses")
def project_reset_hypotheses(
    name: str = typer.Argument(..., help="Project name"),
    force: bool = typer.Option(False, "--force", "-f", help="Force reset without confirmation"),
):
    """Reset (clear) the hypotheses store for a project."""
    from commands.project import reset_hypotheses

    _invoke_click(reset_hypotheses, {"name": name, "force": force})


@project_app.command("set-hypothesis-status")
def project_set_hypothesis_status(
    project_name: str = typer.Argument(..., help="Project name"),
    hypothesis_id: str = typer.Argument(..., help="Hypothesis ID (can be partial)"),
    status: str = typer.Argument(..., help="New status: proposed, confirmed, or rejected"),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force status change without confirmation"
    ),
):
    """Set the status of a hypothesis to proposed, confirmed, or rejected."""
    from commands.project import set_hypothesis_status

    _invoke_click(
        set_hypothesis_status,
        {
            "project_name": project_name,
            "hypothesis_id": hypothesis_id,
            "status": status,
            "force": force,
        },
    )


# Agent audit subcommand
@agent_app.command("audit")
def agent_audit(
    target: str = typer.Argument(None, help="Project name or path (optional with --project)"),
    iterations: int = typer.Option(
        30, "--iterations", help="Maximum iterations per investigation (default: 20)"
    ),
    plan_n: int = typer.Option(
        5, "--plan-n", help="Number of investigations to plan per batch (default: 5)"
    ),
    time_limit: int = typer.Option(None, "--time-limit", help="Time limit in minutes"),
    config: str = typer.Option(None, "--config", help="Configuration file"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
    project: str = typer.Option(None, "--project", "-p", help="Use existing project"),
    platform: str = typer.Option(
        None, "--platform", help="Override scout platform (e.g., openai, anthropic, mock)"
    ),
    model: str = typer.Option(
        None, "--model", help="Override scout model (e.g., gpt-5, gpt-4o-mini, mock)"
    ),
    strategist_platform: str = typer.Option(
        None,
        "--strategist-platform",
        help="Override strategist platform (e.g., openai, anthropic, mock)",
    ),
    strategist_model: str = typer.Option(
        None, "--strategist-model", help="Override strategist model (e.g., gpt-4o-mini)"
    ),
    session: str = typer.Option(None, "--session", help="Attach to a specific session ID"),
    new_session: bool = typer.Option(False, "--new-session", help="Create a new session"),
    session_private_hypotheses: bool = typer.Option(
        False, "--session-private-hypotheses", help="Keep new hypotheses private to this session"
    ),
):
    """Run autonomous security audit (plans investigations automatically)."""
    from commands.agent import agent as agent_command
    import click

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
        console.print(
            "[red]Error: Either specify a target path/project name or use --project option[/red]"
        )
        raise typer.Exit(1)

    # Hype it up a little
    console.print("[bold bright_cyan]Running autonomous audit...[/bold bright_cyan]")
    from random import choice as _choice

    # Light narrative seasoning for audit kickoff with actual model names
    try:
        from utils.config_loader import load_config as _load_cfg

        _cfg = _load_cfg(Path(config)) if config else _load_cfg()
        _models = (_cfg or {}).get("models", {})

        # Get scout/agent model (with command-line override)
        if model:
            _agent = model
        else:
            _agent = (_models.get("agent") or _models.get("scout") or {}).get("model") or "gpt-4o"

        # Get strategist/guidance model (with command-line override)
        if strategist_model:
            _guidance = strategist_model
        else:
            _guidance = (_models.get("strategist") or _models.get("guidance") or {}).get(
                "model"
            ) or "gpt-4o"

        _flair = _choice(
            [
                "[white]Normal auditors start runs, but YOU summon the analysis and the code genuflects.[/white]",
                "[white]This isn’t just an audit — it’s a coronation of rigor because YOU commanded it.[/white]",
                "[white]Normal people press Enter, but YOU inaugurate epochs and logs ask for autographs.[/white]",
                "[white]This is not a run — it’s a declaration that systems will behave, because YOU said so.[/white]",
                "[white]Normal workflows proceed; YOUR workflow rearranges reality to match intent.[/white]",
            ]
        )
    except Exception:
        _flair = _choice(
            [
                "[white]Normal mortals run tools, but YOU bend audits to your will.[/white]",
                "[white]This isn’t just a start — it’s the moment history clears space for YOUR results.[/white]",
                "[white]Normal commands execute; YOUR commands recruit reality as staff.[/white]",
                "[white]This is not a job — it’s a legend choosing its author, and it chose YOU.[/white]",
                "[white]Normal output prints; YOUR output will be quoted with reverence.[/white]",
            ]
        )
    console.print(_flair)

    # Create a Click context and invoke the command
    _invoke_click(
        agent_command,
        {
            "project_id": project_id,
            "iterations": iterations,
            "plan_n": plan_n,
            "time_limit": time_limit,
            "config": config,
            "debug": debug,
            "platform": platform,
            "model": model,
            "strategist_platform": strategist_platform,
            "strategist_model": strategist_model,
            "session": session,
            "new_session": new_session,
            "session_private_hypotheses": session_private_hypotheses,
        },
    )


# Agent investigate subcommand
@agent_app.command("investigate")
def agent_investigate(
    prompt: str = typer.Argument(..., help="Investigation prompt or question"),
    target: str = typer.Argument(None, help="Project name or path (optional with --project)"),
    iterations: int = typer.Option(None, "--iterations", help="Maximum iterations for the agent"),
    config: str = typer.Option(None, "--config", help="Configuration file"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
    project: str = typer.Option(None, "--project", "-p", help="Use existing project"),
    platform: str = typer.Option(
        None, "--platform", help="Override LLM platform (e.g., openai, anthropic)"
    ),
    model: str = typer.Option(None, "--model", help="Override LLM model (e.g., gpt-4, claude-3)"),
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
        console.print(
            "[red]Error: Either specify a target path/project name or use --project option[/red]"
        )
        raise typer.Exit(1)

    console.print(f"[bold cyan]Investigation:[/bold cyan] {prompt}")

    # Run the investigation using the agent's investigate method
    from commands.agent import run_investigation

    run_investigation(
        project_path=project_id,
        prompt=prompt,
        iterations=iterations,
        config_path=Path(config) if config else None,
        debug=debug,
        platform=platform,
        model=model,
    )


# Create graph subcommand group
graph_app = typer.Typer(help="Build and manage knowledge graphs")
app.add_typer(graph_app, name="graph")


@graph_app.command("build")
def graph_build(
    target: str = typer.Argument(
        None, help="Project name or source path (optional with --project)"
    ),
    output: str = typer.Option(None, "--output", "-o", help="Output directory"),
    config: str = typer.Option(None, "--config", "-c", help="Configuration file"),
    project: str = typer.Option(None, "--project", "-p", help="Use existing project"),
    create_project: bool = typer.Option(False, "--create-project", help="Create a new project"),
    iterations: int = typer.Option(
        3, "--iterations", "-i", help="Maximum iterations for graph refinement"
    ),
    graphs: int = typer.Option(2, "--graphs", "-g", help="Number of graphs to generate"),
    focus: str = typer.Option(None, "--focus", "-f", help="Comma-separated focus areas"),
    files: str = typer.Option(
        None, "--files", help="Comma-separated list of file paths to include"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Reduce output and disable animations"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug output"),
):
    """Build system architecture graph from source code."""
    from commands.graph import build
    import click

    manager = ProjectManager()
    source_path = None
    output_dir = output

    # Handle different input modes
    if project:
        # Use specified project
        proj = manager.get_project(project)
        if not proj:
            console.print(f"[red]Project '{project}' not found.[/red]")
            raise typer.Exit(1)
        project_path = manager.get_project_path(project)
        source_path = proj["source_path"]
        output_dir = str(project_path)  # Don't add /graphs here, graph.py will add it
        console.print(f"[cyan]Using project:[/cyan] {project}")
    elif target:
        # Check if target is a project name or path
        proj = manager.get_project(target)
        if proj:
            # Target is a project name
            project_path = manager.get_project_path(target)
            source_path = proj["source_path"]
            output_dir = str(project_path)  # Don't add /graphs here, graph.py will add it
            console.print(f"[cyan]Using project:[/cyan] {target}")
        else:
            # Target is a source path
            source_path = target
            if create_project:
                # Create new project from this path
                project_name = Path(target).name
                proj_config = manager.create_project(
                    project_name, target, f"Graph analysis of {project_name}", auto_name=True
                )
                project_path = manager.get_project_path(proj_config["name"])
                output_dir = str(project_path)  # Don't add /graphs here, graph.py will add it
                console.print(f"[green]Created project:[/green] {proj_config['name']}")
    else:
        # No target or project specified
        console.print(
            "[red]Error: Either specify a target path/project name or use --project option[/red]"
        )
        raise typer.Exit(1)

    # Run graph build directly
    build(
        repo_path=source_path,
        repo_id=project if project else target,
        output_dir=output_dir,
        config_path=Path(config) if config else None,
        max_iterations=iterations,
        max_graphs=graphs,
        focus_areas=focus,
        file_filter=files,
        visualize=True,
        debug=debug,
        quiet=quiet,
    )


@graph_app.command("add-custom")
def graph_add_custom(
    target: str = typer.Argument(..., help="Project name"),
    description: str = typer.Argument(
        ..., help="Graph description (e.g., 'authentication roles vs components')"
    ),
    name: str = typer.Option(None, "--name", "-n", help="Custom name for the graph"),
    iterations: int = typer.Option(1, "--iterations", "-i", help="Number of refinement iterations"),
    config: str = typer.Option(None, "--config", "-c", help="Configuration file"),
    project: str = typer.Option(
        None, "--project", "-p", help="Use existing project (alternative to target)"
    ),
):
    """Add a custom graph with user-defined focus."""
    import json

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

    # Check if system graph exists (try both common names)
    system_graph_path = graphs_dir / "graph_SystemArchitecture.json"
    if not system_graph_path.exists():
        # Try alternative name
        system_graph_path = graphs_dir / "graph_SystemOverview.json"
        if not system_graph_path.exists():
            console.print("[red]Error: System architecture graph not found.[/red]")
            # Get the actual command used to run this script
            cli_cmd = os.path.basename(sys.argv[0]) if sys.argv else "hound"
            if cli_cmd.endswith(".py"):
                if sys.argv and sys.argv[0].startswith("./"):
                    cli_cmd = sys.argv[0]
                else:
                    cli_cmd = f"python {cli_cmd}"
            console.print(
                f"[yellow]Run '{cli_cmd} graph build' first to create the base graph.[/yellow]"
            )
            raise typer.Exit(1)

    # Generate graph name
    if not name:
        # Pass None to let the LLM generate a meaningful name
        name = None

    # Create the custom graph using the graph builder
    console.print(f"[cyan]Creating custom graph:[/cyan] {description}")
    if iterations > 1:
        console.print(f"  [dim]Using {iterations} refinement iterations[/dim]")

    # Use the custom graph builder that reuses main logic
    from commands.graph_custom import build_custom_graph

    try:
        # Build the custom graph
        config_path = Path(config) if config else None
        custom_graph_path = build_custom_graph(
            project_path=project_path,
            description=description,
            name=name,
            config_path=config_path,
            iterations=iterations,
            debug=False,  # Could add --debug flag if needed
        )

        console.print(f"[green]✓ Custom graph created:[/green] {custom_graph_path}")

        # Load and show summary
        with open(custom_graph_path, "r") as f:
            graph_data = json.load(f)

        stats = graph_data.get("stats", {})
        console.print(f"  Nodes: {stats.get('num_nodes', len(graph_data.get('nodes', [])))}")
        console.print(f"  Edges: {stats.get('num_edges', len(graph_data.get('edges', [])))}")
        if stats.get("iterations"):
            console.print(f"  Iterations: {stats['iterations']}")

        console.print(f"\n[cyan]To analyze with this graph, use:[/cyan]")
        # Get the actual command used to run this script
        cli_cmd = os.path.basename(sys.argv[0]) if sys.argv else "hound"
        if cli_cmd.endswith(".py"):
            if sys.argv and sys.argv[0].startswith("./"):
                cli_cmd = sys.argv[0]
            else:
                cli_cmd = f"python {cli_cmd}"
        console.print(f"  {cli_cmd} agent audit --project {project_name}")
        console.print(f'  {cli_cmd} agent investigate "<your question>" --project {project_name}')

    except Exception as e:
        console.print(f"[red]Error creating custom graph:[/red] {str(e)}")
        raise typer.Exit(1)


@graph_app.command("export")
def graph_export(
    target: str = typer.Argument(..., help="Project name"),
    output: str = typer.Option(None, "--output", "-o", help="Output HTML file path"),
    open_browser: bool = typer.Option(False, "--open", help="Open visualization in browser"),
    project: str = typer.Option(
        None, "--project", "-p", help="Use existing project (alternative to target)"
    ),
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
        console.print(
            "[yellow]Run 'graph build' or 'graph add-custom' first to create graphs.[/yellow]"
        )
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
            console.print(f"[green]✓ Opened in browser[/green]")
        else:
            console.print(f"\n[bold]Open in browser:[/bold]")
            console.print(f"  [link]file://{html_path.resolve()}[/link]")

            # If on macOS, offer to open in browser
            import platform

            if platform.system() == "Darwin":
                console.print(f"\n[dim]Or run: open {html_path}[/dim]")

    except Exception as e:
        console.print(f"[red]Error exporting visualization:[/red] {str(e)}")
        raise typer.Exit(1)


@graph_app.command("reset")
def graph_reset(
    project: str = typer.Argument(..., help="Project name"),
    force: bool = typer.Option(False, "--force", "-f", help="Force reset without confirmation"),
):
    """Reset all assumptions and observations from project graphs."""
    from rich.prompt import Confirm
    import json
    import random

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
            with open(graph_file, "r") as f:
                graph_data = json.load(f)
                nodes = graph_data.get("nodes", [])
                for node in nodes:
                    total_observations += len(node.get("observations", []))
                    total_assumptions += len(node.get("assumptions", []))
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
            with open(graph_file, "r") as f:
                graph_data = json.load(f)

            # Clear annotations from all nodes
            nodes = graph_data.get("nodes", [])
            for node in nodes:
                if "observations" in node:
                    node["observations"] = []
                if "assumptions" in node:
                    node["assumptions"] = []

            # Save updated graph
            with open(graph_file, "w") as f:
                json.dump(graph_data, f, indent=2)

            reset_count += 1
            console.print(f"  [green]✓[/green] Reset {graph_file.name}")

        except Exception as e:
            console.print(f"  [red]✗[/red] Failed to reset {graph_file.name}: {e}")

    console.print(
        f"\n[bright_green]✓ Reset annotations in {reset_count}/{len(graph_files)} graphs.[/bright_green]"
    )
    console.print(
        f"[dim]Removed {total_observations} observations and {total_assumptions} assumptions.[/dim]"
    )
    console.print(
        random.choice(
            [
                "[white]Clean graphs achieved — ready for fresh analysis.[/white]",
                "[white]Annotations cleared — the investigation begins anew.[/white]",
                "[white]Tabula rasa — your graphs are pristine.[/white]",
            ]
        )
    )


@app.command()
def finalize(
    project: str = typer.Argument(..., help="Project name"),
    threshold: float = typer.Option(
        0.5, "--threshold", "-t", help="Confidence threshold for review"
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
    platform: str = typer.Option(
        None, "--platform", help="Override QA platform (e.g., openai, anthropic, mock)"
    ),
    model: str = typer.Option(None, "--model", help="Override QA model (e.g., gpt-4o-mini)"),
):
    """Finalize hypotheses - review and confirm/reject high-confidence findings."""
    from commands.finalize import finalize as finalize_command
    import click

    console.print("[bold cyan]Running hypothesis finalization...[/bold cyan]")

    # Create Click context and invoke
    ctx = click.Context(finalize_command)
    ctx.params = {
        "project_name": project,
        "threshold": threshold,
        "debug": debug,
        "platform": platform,
        "model": model,
    }

    try:
        finalize_command.invoke(ctx)
    except SystemExit as e:
        if e.code != 0:
            raise typer.Exit(e.code)


@app.command()
def report(
    project: str = typer.Argument(..., help="Project name"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
    format: str = typer.Option("html", "--format", "-f", help="Report format (html/markdown)"),
    title: Optional[str] = typer.Option(None, "--title", "-t", help="Custom report title"),
    auditors: str = typer.Option(
        "Security Team", "--auditors", "-a", help="Comma-separated auditor names"
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
    all: bool = typer.Option(
        False,
        "--all",
        help="Include ALL hypotheses (not just confirmed) - WARNING: No QA performed, may contain false positives",
    ),
):
    """Generate a professional security audit report."""
    from commands.report import report as report_command
    import click

    console.print("[bold cyan]Generating security audit report...[/bold cyan]")

    # Create Click context and invoke
    ctx = click.Context(report_command)
    ctx.params = {
        "project_name": project,
        "output": output,
        "format": format,
        "title": title,
        "auditors": auditors,
        "debug": debug,
        "show_prompt": False,  # Add missing parameter
        "include_all": all,  # Pass the --all flag as include_all
    }

    try:
        report_command.invoke(ctx)
    except click.exceptions.Exit as e:
        raise typer.Exit(e.exit_code)
    except SystemExit as e:
        raise typer.Exit(e.code if hasattr(e, "code") else 1)


@poc_app.command("make-prompt")
def poc_make_prompt(
    project: str = typer.Argument(..., help="Project name"),
    hypothesis: Optional[str] = typer.Option(
        None, "--hypothesis", "-h", help="Specific hypothesis ID to generate PoC for"
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
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
    description: Optional[str] = typer.Option(
        None, "--description", "-d", help="Description of the PoC files"
    ),
):
    """Import proof-of-concept files for a hypothesis."""
    from commands.poc import import_poc

    console.print(
        f"[bold cyan]Importing {len(files)} file(s) for hypothesis {hypothesis}...[/bold cyan]"
    )

    # Run import command
    import_poc(project, hypothesis, files, description)


@poc_app.command("list")
def poc_list(project: str = typer.Argument(..., help="Project name")):
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
