# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Allow running benchmarks as ``python -m benchmarks``.

Subcommands:
    python -m benchmarks gate G0       # Gate verification
    python -m benchmarks --layer L1    # Benchmark layers (default)
"""

import sys


def _main() -> None:
    # Route "gate" subcommand to benchmarks.gate
    if len(sys.argv) > 1 and sys.argv[1] == "gate":
        # Strip the "gate" token so gate.main() sees only its own args
        sys.argv = [sys.argv[0] + " gate", *sys.argv[2:]]
        from benchmarks.gate import main as gate_main

        gate_main()
    elif len(sys.argv) > 1 and sys.argv[1] == "status":
        # Resume / progress reporting CLI.
        from benchmarks.progress_cli import main as status_main

        sys.exit(status_main(sys.argv[2:]))
    else:
        from benchmarks.cli import main

        main()


_main()
