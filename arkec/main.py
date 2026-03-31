# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke CLI — Command-line interface for the Arke toolchain."""

import click


@click.group()
@click.version_option(version="0.1.0-dev", prog_name="arke")
def cli():
    """Arke — AI-First Operator Description Language & Compiler Toolchain."""
    pass


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", help="Output file path")
def parse(input_file: str, output: str | None):
    """Parse an .ak file to Arke IR (JSON)."""
    click.echo(f"[TODO] Parsing {input_file}...")
    # TODO: Implement parser invocation


@cli.command()
@click.argument("ir_file", type=click.Path(exists=True))
def inspect(ir_file: str):
    """Inspect an Arke IR file (human-readable view)."""
    click.echo(f"[TODO] Inspecting {ir_file}...")
    # TODO: Implement IR pretty-printer


@cli.command()
@click.argument("ir_file", type=click.Path(exists=True))
@click.option("--target", required=True, help="Target hardware (e.g., ampere, ascend_a3)")
@click.option("--budget", default=100, help="Optimization search budget")
def optimize(ir_file: str, target: str, budget: int):
    """Optimize an Arke IR using AI agent."""
    click.echo(f"[TODO] Optimizing {ir_file} for {target} (budget={budget})...")
    # TODO: Implement AI optimization loop


@cli.command()
@click.argument("ir_file", type=click.Path(exists=True))
@click.option("--target", required=True, help="Code generation target (triton, cuda)")
@click.option("-o", "--output", help="Output file path")
def codegen(ir_file: str, target: str, output: str | None):
    """Generate code from Arke IR."""
    click.echo(f"[TODO] Generating {target} code from {ir_file}...")
    # TODO: Implement code generation


@cli.command()
@click.argument("ir_file", type=click.Path(exists=True))
@click.option("--ref", "reference", help="Reference Python file for correctness check")
def verify(ir_file: str, reference: str | None):
    """Verify correctness of an Arke IR."""
    click.echo(f"[TODO] Verifying {ir_file}...")
    # TODO: Implement verification


if __name__ == "__main__":
    cli()
