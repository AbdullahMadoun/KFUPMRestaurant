# =============================================================================
# FILE: tests/conftest.py
# CATEGORY: TEST
# PURPOSE: Pytest path bootstrap so the snapshot test suite can import top-level modules when run from the snapshot root.
# DEPENDENCIES: None
# USED BY: tests/test_allocation.py, tests/test_dataset.py, tests/test_item_processing.py, tests/test_pipeline_contracts.py, tests/test_sam3_allocation.py, tests/test_stage2_sam.py
# KEY CLASSES/FUNCTIONS: None
# LAST MODIFIED: 2026-03-30T00:00:00+00:00
# SNAPSHOT NOTES: added during snapshot finalization so `pytest ./tests` works without extra PYTHONPATH setup
# =============================================================================
from __future__ import annotations

import sys
from pathlib import Path


SNAPSHOT_ROOT = Path(__file__).resolve().parents[1]

if str(SNAPSHOT_ROOT) not in sys.path:
    sys.path.insert(0, str(SNAPSHOT_ROOT))
