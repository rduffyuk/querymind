#!/usr/bin/env python3
"""
QueryMind CLI - Command-line interface for intelligent vault search

Usage:
    python -m querymind.cli search "your query"
    python -m querymind.cli index /path/to/vault
    python -m querymind.cli health
"""

import click
from rich.console import Console
from rich.table import Table
from rich import print as rprint
import time

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """QueryMind - Intelligent RAG with Smart Query Routing"""
    pass


@cli.command()
@click.argument("query")
@click.option("-n", "--results", default=5, help="Number of results to return")
@click.option("-v", "--verbose", is_flag=True, help="Show detailed output")
def search(query: str, results: int, verbose: bool):
    """Search your vault with intelligent routing"""
    try:
        from querymind import auto_search

        console.print(f"\n🔍 Searching for: [bold cyan]{query}[/bold cyan]\n")

        start_time = time.time()
        response = auto_search(query, n_results=results, verbose=verbose)
        elapsed = time.time() - start_time

        if response.status == "success" and response.results:
            # Create results table
            table = Table(title=f"Search Results ({response.result_count} found)")
            table.add_column("#", style="cyan", no_wrap=True)
            table.add_column("File", style="green")
            table.add_column("Score", justify="right", style="yellow")
            table.add_column("Preview", style="white")

            for i, result in enumerate(response.results[:results], 1):
                score = f"{result.get('score', 0):.2f}"
                preview = result.get('content', '')[:80] + "..."
                table.add_row(
                    str(i),
                    result.get('file', 'unknown'),
                    score,
                    preview
                )

            console.print(table)
            console.print(f"\n📊 [bold]Agent:[/bold] {response.agent_type}")
            console.print(f"⏱️  [bold]Time:[/bold] {elapsed:.2f}s")
            if verbose:
                console.print(f"🎯 [bold]Status:[/bold] {response.status}")
                console.print(f"📝 [bold]Total Results:[/bold] {response.result_count}")

        else:
            console.print(f"[yellow]No results found for '{query}'[/yellow]")
            if response.error:
                console.print(f"[red]Error: {response.error}[/red]")

    except Exception as e:
        console.print(f"[red]❌ Error: {str(e)}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())


@cli.command()
def health():
    """Check system health"""
    console.print("\n🏥 [bold]QueryMind Health Check[/bold]\n")

    checks = []

    # Check ChromaDB
    try:
        from querymind.core.embeddings import ChromaDBManager
        mgr = ChromaDBManager()
        checks.append(("ChromaDB", "✅ Connected", "green"))
    except Exception as e:
        checks.append(("ChromaDB", f"❌ {str(e)[:50]}", "red"))

    # Check Redis
    try:
        from querymind.core.cache import get_cache
        cache = get_cache()
        checks.append(("Redis Cache", "✅ Connected", "green"))
    except Exception as e:
        checks.append(("Redis Cache", f"❌ {str(e)[:50]}", "red"))

    # Check Config
    try:
        from querymind.core.config import config
        checks.append(("Configuration", f"✅ Loaded (vault: {config.vault_path})", "green"))
    except Exception as e:
        checks.append(("Configuration", f"❌ {str(e)[:50]}", "red"))

    # Display results
    table = Table(title="Health Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="white")

    for component, status, color in checks:
        table.add_row(component, f"[{color}]{status}[/{color}]")

    console.print(table)
    console.print()


@cli.command()
@click.argument("path")
@click.option("--recursive", is_flag=True, help="Index subdirectories")
@click.option("--limit", default=None, type=int, help="Limit number of files")
def index(path: str, recursive: bool, limit: int):
    """Index markdown files into ChromaDB"""
    console.print(f"\n📥 Indexing files from: [bold]{path}[/bold]\n")

    try:
        import os
        from pathlib import Path
        from querymind.core.embeddings import ChromaDBManager

        vault_path = Path(path)
        if not vault_path.exists():
            console.print(f"[red]Error: Path does not exist: {path}[/red]")
            return

        # Get markdown files
        if recursive:
            files = list(vault_path.rglob("*.md"))
        else:
            files = list(vault_path.glob("*.md"))

        if limit:
            files = files[:limit]

        console.print(f"Found {len(files)} markdown files")

        # TODO: Implement actual indexing
        console.print("[yellow]⚠️  Indexing not yet implemented[/yellow]")
        console.print("Files found:")
        for i, f in enumerate(files[:10], 1):
            console.print(f"  {i}. {f.name}")
        if len(files) > 10:
            console.print(f"  ... and {len(files) - 10} more")

    except Exception as e:
        console.print(f"[red]❌ Error: {str(e)}[/red]")


if __name__ == "__main__":
    cli()
