"""
Command-line interface for FLAIR benchmark.

Usage:
    flair build-datasets --task task1_discharged_home
    flair validate /path/to/submission
    flair table1 task1_discharged_home
"""

import click
import json
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@click.group()
@click.option("--config", "-c", default="flair_config.yaml", help="Config file path")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx, config, verbose):
    """FLAIR - Federated Learning Assessment for ICU Research

    A privacy-first benchmark for evaluating ML/AI methods on ICU prediction
    tasks using CLIF-formatted data.
    """
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config
    ctx.obj["verbose"] = verbose

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command("build-datasets")
@click.option("--task", "-t", multiple=True, help="Task(s) to build (default: all)")
@click.pass_context
def build_datasets(ctx, task):
    """Build datasets for specified tasks using clifpy."""
    from flair.config.loader import load_config
    from flair.datasets.builder import FLAIRDatasetBuilder

    config_path = ctx.obj["config_path"]

    try:
        config = load_config(config_path)
        builder = FLAIRDatasetBuilder(config.model_dump())

        if task:
            for t in task:
                click.echo(f"Building dataset for {t}...")
                output_path = builder.build_task(t)
                click.echo(f"  -> Saved to {output_path}")
        else:
            click.echo("Building all enabled tasks...")
            outputs = builder.build_all_tasks()
            for task_name, output_path in outputs.items():
                click.echo(f"  {task_name} -> {output_path}")

        click.echo("Done!")

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if ctx.obj["verbose"]:
            import traceback

            traceback.print_exc()
        sys.exit(1)


@cli.command("build-cohort")
@click.option("--output", "-o", default=None, help="Output path (default: from config)")
@click.pass_context
def build_cohort(ctx, output):
    """Build the ICU cohort from CLIF data.

    Uses clifpy to build the shared ICU cohort with:
    - Adults (age >= 18)
    - At least one ICU ADT record
    - Length of stay > 0
    - imv_at_24hr flag for Task 3 & 4
    """
    from flair.config.loader import load_config
    from flair.cohort.builder import FLAIRCohortBuilder

    config_path = ctx.obj["config_path"]

    try:
        config = load_config(config_path)
        cohort_config = config.cohort

        # Determine output path
        output_path = output or cohort_config.output_path

        click.echo("Building FLAIR cohort from CLIF data...")
        click.echo(f"  CLIF config: {config.data.clif_config_path}")
        click.echo(f"  Min age: {cohort_config.min_age}")
        click.echo(f"  Min LOS: {cohort_config.min_los_days} days")
        click.echo(f"  Output: {output_path}")
        click.echo("")

        builder = FLAIRCohortBuilder(config.data.clif_config_path)
        cohort, stats = builder.build_cohort(
            output_path=output_path,
            min_age=cohort_config.min_age,
            min_los_days=cohort_config.min_los_days,
            skip_time_filter=cohort_config.skip_time_filter,
        )

        click.echo("")
        click.echo(click.style("Cohort built successfully!", fg="green", bold=True))
        click.echo(f"  Total hospitalizations: {cohort.height:,}")
        click.echo(f"  Output saved to: {output_path}")

        # Show exclusion statistics
        click.echo("")
        click.echo("Exclusion statistics:")
        click.echo(f"  Initial: {stats.get('initial', 0):,}")
        click.echo(f"  After null date filter: {stats.get('after_null_filter', 0):,}")
        click.echo(f"  After age filter: {stats.get('after_age_filter', 0):,}")
        click.echo(f"  After LOS filter: {stats.get('after_los_filter', 0):,}")
        click.echo(f"  After ICU filter: {stats.get('after_icu_filter', 0):,}")
        click.echo(f"  Final: {stats.get('final', 0):,}")

    except ImportError as e:
        click.echo(f"Error: clifpy is required for cohort building", err=True)
        click.echo(f"Install with: pip install clifpy", err=True)
        sys.exit(1)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if ctx.obj["verbose"]:
            import traceback

            traceback.print_exc()
        sys.exit(1)


@cli.command("validate")
@click.argument("submission_dir")
@click.pass_context
def validate_submission(ctx, submission_dir):
    """Validate a method submission.

    Checks for:
    - Required files (train.py, predict.py, requirements.txt)
    - Banned package imports
    - Dangerous code patterns
    """
    from flair.submission.validator import SubmissionValidator

    validator = SubmissionValidator(Path(submission_dir))
    is_valid, errors, warnings = validator.validate()

    if errors:
        click.echo(click.style("ERRORS:", fg="red", bold=True))
        for e in errors:
            click.echo(click.style(f"  [X] {e}", fg="red"))

    if warnings:
        click.echo(click.style("WARNINGS:", fg="yellow", bold=True))
        for w in warnings:
            click.echo(click.style(f"  [!] {w}", fg="yellow"))

    if is_valid:
        click.echo(click.style("Validation PASSED", fg="green", bold=True))
        sys.exit(0)
    else:
        click.echo(click.style("Validation FAILED", fg="red", bold=True))
        sys.exit(1)


@cli.command("table1")
@click.argument("task")
@click.option("--output", "-o", default=None, help="Output file (default: stdout)")
@click.option("--format", "fmt", type=click.Choice(["json", "markdown"]), default="json")
@click.pass_context
def generate_table1(ctx, task, output, fmt):
    """Generate Table 1 for a task (aggregated statistics only)."""
    import polars as pl
    from flair.config.loader import load_config
    from flair.helpers.table1 import get_table1, format_table1_markdown

    config_path = ctx.obj["config_path"]

    try:
        config = load_config(config_path)
        datasets_dir = Path(config.output.datasets_dir) / task

        if not datasets_dir.exists():
            click.echo(f"Error: Dataset not found for {task}", err=True)
            click.echo(f"Run 'flair build-datasets --task {task}' first", err=True)
            sys.exit(1)

        # Load data
        demographics = pl.read_parquet(datasets_dir / "demographics.parquet")
        labels = pl.read_parquet(datasets_dir / "labels.parquet")

        # Generate Table 1
        table1 = get_table1(
            demographics,
            labels,
            min_cell_count=config.privacy.min_cell_count,
        )

        # Format output
        if fmt == "markdown":
            result = format_table1_markdown(table1)
        else:
            result = json.dumps(table1, indent=2)

        # Output
        if output:
            with open(output, "w") as f:
                f.write(result)
            click.echo(f"Table 1 saved to {output}")
        else:
            click.echo(result)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command("package-results")
@click.argument("task")
@click.argument("metrics_file")
@click.option("--site", required=True, help="Site name")
@click.option("--model", required=True, help="Model/method name")
@click.option("--output-dir", "-o", default="flair_output/submissions", help="Output directory")
@click.pass_context
def package_results(ctx, task, metrics_file, site, model, output_dir):
    """Package results for submission."""
    from flair.submission.packager import ResultsPackager
    from flair.config.loader import load_config
    from flair.helpers.table1 import get_table1
    import polars as pl

    config_path = ctx.obj["config_path"]

    try:
        config = load_config(config_path)

        # Load metrics
        with open(metrics_file, "r") as f:
            metrics = json.load(f)

        # Load and generate Table 1
        datasets_dir = Path(config.output.datasets_dir) / task
        demographics = pl.read_parquet(datasets_dir / "demographics.parquet")
        labels = pl.read_parquet(datasets_dir / "labels.parquet")
        table1 = get_table1(demographics, labels)

        # Package results
        packager = ResultsPackager(output_dir, site, task, model)
        output_path = packager.package(metrics, table1)

        click.echo(f"Results packaged to {output_path}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command("init")
@click.option("--site", default="my_site", help="Site name")
@click.option("--output", "-o", default="flair_config.yaml", help="Output file")
def init_config(site, output):
    """Create a default FLAIR configuration file."""
    from flair.config.loader import create_default_config

    output_path = create_default_config(output, site)
    click.echo(f"Created configuration file: {output_path}")
    click.echo("Edit this file to configure your FLAIR instance.")


@cli.command("version")
def show_version():
    """Show FLAIR version."""
    from flair import __version__

    click.echo(f"FLAIR version {__version__}")


@cli.command("privacy-warning")
def show_privacy_warning():
    """Display the FLAIR privacy policy warning."""
    warning = """
╔════════════════════════════════════════════════════════════════╗
║                     FLAIR PRIVACY POLICY                        ║
╠════════════════════════════════════════════════════════════════╣
║ 1. NO NETWORK REQUESTS                                          ║
║    - All network access is blocked at Python socket level       ║
║    - Packages like requests, urllib3, httpx are banned          ║
║    - Violation = immediate submission rejection                 ║
║                                                                  ║
║ 2. PHI PROTECTION                                                ║
║    - All outputs scanned for PHI patterns                       ║
║    - Cell counts < 10 are suppressed                            ║
║    - Individual-level data never leaves the site                ║
║                                                                  ║
║ 3. REVIEW PROCESS                                                ║
║    - PIs at each site review code before execution              ║
║    - PIs have final say - they are not required to run          ║
║    - Code inspection for data exfiltration attempts             ║
║                                                                  ║
║ 4. CONSEQUENCES                                                  ║
║    - If data is found shared to internet during review:         ║
║      → Submitter is BANNED from FLAIR                           ║
║      → Incident reported to institution                         ║
║                                                                  ║
║ PI's are doing you a favor by running your code on their data.  ║
║ Respect their trust and protect patient privacy.                ║
╚════════════════════════════════════════════════════════════════╝
"""
    click.echo(warning)


if __name__ == "__main__":
    cli()
