"""STRATA generator package; python main.py runs it.  Program name and version.  Importing the package sets the
process environment before torch is loaded (CUDA allocator, BLAS threads, duplicate OpenMP runtimes) and lets the
console streams replace characters they cannot print."""

import os
import sys


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8,max_split_size_mb:256")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

RFMG_PROGRAM_NAME = "RFMG-Gen"
RFMG_VERSION = "22.2.0"
RFMG_SCHEMA_VERSION = 160
_LOG = "[RFMG-Gen]"

try:  # never die on a console that cannot print a character (GBK consoles, piped stdout)
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass
