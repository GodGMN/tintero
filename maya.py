#!/usr/bin/env python3
"""Start the native Tintero server with an existing JSON config."""

import sys

from serve.server import main


if __name__ == "__main__":
    sys.exit(main(["--engine", "maya", *sys.argv[1:]]))
