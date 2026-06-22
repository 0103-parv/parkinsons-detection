"""Run the whole Parkinson's detection project end to end.

    cd ~/mentat && PYTHONPATH=. ~/swechats/.venv/bin/python -m reference.parkinsons

Runs: dataset-1 honest report -> mentat feature-panel search -> train+save the deployable
model -> (if the larger Sakar dataset is present) the independent replication.
"""
from __future__ import annotations


def _section(title: str):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def main() -> int:
    from . import detect, final_model, panel_search
    from .detect_sakar import DATA as SAKAR

    _section("1/4  DATASET 1 — UCI voice (Little 2007): honest subject-level detection")
    detect.main()
    _section("2/4  mentat FEATURE-PANEL SEARCH — minimal voice panel (gated by held-out people)")
    panel_search.main()
    _section("3/4  DEPLOYABLE MODEL — train on all data, save, predict")
    final_model.main()
    _section("4/4  INDEPENDENT REPLICATION — Sakar 2018 (252 people, 752 feats)")
    if SAKAR.exists():
        from . import detect_sakar
        detect_sakar.main()
    else:
        print("  Sakar dataset not downloaded — skipping.")
        print("  Fetch it:  PYTHONPATH=. python -m reference.parkinsons.download")

    _section("DONE — a complete, honestly-validated Parkinson's detector on REAL data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
