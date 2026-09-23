"""Command-line entry point for local tools."""

import argparse


def main() -> None:
    """Show the available command-line entry point during bootstrap."""
    parser = argparse.ArgumentParser(prog="llmbench")
    parser.parse_args()
    parser.print_help()
