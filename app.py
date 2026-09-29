"""
Reserved entry point for the Flask backend (ARCHITECTURE.md section 21,
Phase 11). Not implemented yet.

Until Phase 2 this file was a broken duplicate of training/build_dataset.py:
it imported `add_technical_indicators`, which no longer exists (ImportError),
and would have overwritten the dataset without a raw snapshot.

To build the dataset:      python -m training.build_dataset
To evaluate models:        python -m training.evaluation_harness
"""

import sys

if __name__ == "__main__":
    sys.exit(
        "app.py is reserved for the Flask backend (Phase 11) and does nothing yet.\n"
        "Build the dataset with:  python -m training.build_dataset"
    )
