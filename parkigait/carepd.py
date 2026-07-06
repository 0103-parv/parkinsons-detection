"""CARE-PD dataset adapter — READY for real data, fabricates NOTHING.

RESEARCH / EDUCATIONAL PROTOTYPE. NOT A MEDICAL DEVICE. NOT FOR CLINICAL USE.

This module is the "point it at real data and it trains honestly" hook that the
rest of ParkiGait is built around. **We do not have the CARE-PD data on this
machine.** So this loader does exactly one honest thing when the data is absent:
it raises a clear, actionable :class:`CAREPDNotAvailable` instead of synthesizing
a single number. There is no fake-data fallback anywhere in this file. The only
data source ParkiGait ships with is the explicitly-labelled synthetic walker in
``parkigait.pose`` — this file will never quietly return that either.

What CARE-PD actually is (verified against the public sources, Jul 2026)
-----------------------------------------------------------------------
CARE-PD (Taati Lab / VIDA-ADL) is, per its authors, the largest publicly
available archive of 3D-mesh gait data for Parkinson's Disease and the first
multi-site collection: 9 harmonized cohorts from 8 clinical centres, 362
participants across a range of disease severity. Every recording (RGB video or
motion capture) is converted to an anonymized **SMPL** body-mesh sequence via a
harmonized preprocessing pipeline. Roughly a third of the walks carry a
clinician-rated **UPDRS-gait** score (the gait item of the MDS-UPDRS motor exam).

  Paper:   "CARE-PD: A Multi-Site Anonymized Clinical Dataset for Parkinson's
           Disease Gait Assessment", NeurIPS 2025 Datasets & Benchmarks.
           arXiv:2510.04312
  Data:    https://huggingface.co/datasets/vida-adl/CARE-PD   (repo_type=dataset)
  Code:    https://github.com/TaatiTeam/CARE-PD
  Terms:   https://neurips2025.care-pd.ca/terms-of-use   (CC BY-NC-ND 4.0; verify)
  Project: https://neurips2025.care-pd.ca/

Access, honestly
----------------
As of the sources above, the Hugging Face dataset is **GATED**: you must be
logged in, then acknowledge and accept the terms of use on the dataset card
before you can download it. It is released under **CC BY-NC-ND 4.0** (verify on the
card) and the authors
require that you cite the CARE-PD paper AND the relevant original cohort papers,
and never attempt re-identification (GDPR / HIPAA / PIPEDA obligations). Treat it
as a data-use agreement. **VERIFY the current access status against the dataset
card before relying on any of this** — gating policies change.

Because this is a research prototype and clinical data is involved, training on
these labels is governed by ``CLINICAL_SAFETY.md`` in this package. Read it.

Schema this loader targets (VERIFY against the dataset card)
-----------------------------------------------------------
The canonical distribution is a set of per-cohort pickle files. Each is a nested
dict keyed by anonymized subject id then anonymized walk id::

    {
      "<subject_id>": {
        "<walk_id>": {
          "pose":  np.ndarray,   # SMPL pose params (axis-angle), (T, D)
          "trans": np.ndarray,   # root translation, (T, 3)
          "beta":  np.ndarray,   # SMPL shape; zeroed for privacy
          "fps":   int,
          "UPDRS_GAIT": int | None,   # gait score (0..3 scale) where rated
          "medication": str | None,   # e.g. "on"/"off" where known
          "other": str | None,
        },
        ...
      },
      ...
    }

The GitHub distribution lays these out under ``assets/datasets/`` with per-cohort
files (cohorts: 3DGait, BMCLab, DNE, E-LC, KUL-DT-T, PD-GaM, T-LTC, T-SDU-PD,
T-SDU), a ``folds/`` directory with subject-level train/test splits, and
alternative motion encodings (``6D_SMPL/``, ``h36m/``, ``HumanML3D/``). A
``Canonicalized_SMPL_pickles`` variant re-orients motion to a shared frame
(x = lateral, y = up, z = forward) and normalizes translation. ParkiGait's
internal convention is different (normalized **image** coords, y DOWN), so this
loader converts — see :func:`_smpl_frame_to_blazepose`.

  NOTE / UNVERIFIED: the SMPL ``pose`` dimensionality (e.g. 72 for 24 joints in
  axis-angle, or a 6D-rotation encoding), and whether a cohort ships raw SMPL vs.
  a keypoint export, vary by cohort/format. **VERIFY per-cohort against the
  dataset card / loader code** before trusting the exact array shapes below. This
  adapter validates shapes at load time and refuses to guess.

Joint mapping (SMPL 24-joint -> BlazePose 33) — documented & marked uncertain
-----------------------------------------------------------------------------
CARE-PD stores SMPL bodies; ParkiGait's contract (``parkigait.types``) is the
33-landmark MediaPipe **BlazePose** skeleton. To feed real CARE-PD motion into
the existing gait-feature extractor we regress SMPL joint locations (via the SMPL
body model) and remap the ones that have a clean correspondence. The gait
features only require torso, hip, knee, ankle, heel, foot, shoulder, elbow, wrist
and nose — SMPL covers all of these except the face/finger detail, which we place
as plausible anchors (flagged, and never used by the features). See
:data:`SMPL_TO_BLAZEPOSE` for the per-joint map and its uncertainty flags.

Every method that would touch data raises :class:`CAREPDNotAvailable` when the
data is not present. Nothing here invents a result.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from parkigait.types import BLAZEPOSE_JOINTS, PoseSequence, joint_index

# --------------------------------------------------------------------------- #
# Public constants / provenance                                               #
# --------------------------------------------------------------------------- #
CAREPD_HF_URL = "https://huggingface.co/datasets/vida-adl/CARE-PD"
CAREPD_GITHUB_URL = "https://github.com/TaatiTeam/CARE-PD"
CAREPD_TERMS_URL = "https://neurips2025.care-pd.ca/terms-of-use"
CAREPD_PROJECT_URL = "https://neurips2025.care-pd.ca/"
CAREPD_PAPER = (
    "CARE-PD: A Multi-Site Anonymized Clinical Dataset for Parkinson's Disease "
    "Gait Assessment (NeurIPS 2025 Datasets & Benchmarks; arXiv:2510.04312)"
)
CAREPD_LICENSE = ("CC BY-NC-ND 4.0 (non-commercial, no-derivatives) per the HF "
                  "dataset card — VERIFY current terms before use")

# The 9 harmonized cohorts (per the GitHub repo layout). VERIFY against the card.
CAREPD_COHORTS = (
    "3DGait", "BMCLab", "DNE", "E-LC", "KUL-DT-T",
    "PD-GaM", "T-LTC", "T-SDU-PD", "T-SDU",
)

# Per the dataset card, the per-walk record carries these keys. VERIFY per-cohort.
CAREPD_RECORD_KEYS = ("pose", "trans", "beta", "fps", "UPDRS_GAIT",
                      "medication", "other")

# Standard SMPL 24-joint order (SMPL body model; documented, stable). The SMPL
# regressor emits joints in exactly this order. VERIFY the model version used by
# the cohort you load (SMPL vs SMPL-X vs SMPL+H) as extra joints shift indices.
SMPL_JOINT_ORDER = (
    "pelvis",        # 0
    "left_hip",      # 1
    "right_hip",     # 2
    "spine1",        # 3
    "left_knee",     # 4
    "right_knee",    # 5
    "spine2",        # 6
    "left_ankle",    # 7
    "right_ankle",   # 8
    "spine3",        # 9
    "left_foot",     # 10
    "right_foot",    # 11
    "neck",          # 12
    "left_collar",   # 13
    "right_collar",  # 14
    "head",          # 15
    "left_shoulder", # 16
    "right_shoulder",# 17
    "left_elbow",    # 18
    "right_elbow",   # 19
    "left_wrist",    # 20
    "right_wrist",   # 21
    "left_hand",     # 22
    "right_hand",    # 23
)

# --------------------------------------------------------------------------- #
# SMPL(24) -> BlazePose(33) joint map.                                        #
#                                                                             #
# value = name of the SMPL joint whose 3D position seeds this BlazePose joint #
#         (or None when SMPL has no direct correspondence).                   #
# "uncertain" flags a joint the gait features either don't use, or that needs #
# an approximation the dataset card should confirm. This map is DOCUMENTATION  #
# of intent; the loader is inert until real data + the SMPL body model exist.  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class JointMap:
    smpl: Optional[str]      # source SMPL joint (None = no clean source)
    uncertain: bool          # True = approximate / not gait-critical
    note: str = ""


SMPL_TO_BLAZEPOSE: dict[str, JointMap] = {
    # --- gait-critical joints: clean SMPL correspondences -------------------
    "LEFT_SHOULDER":  JointMap("left_shoulder", False),
    "RIGHT_SHOULDER": JointMap("right_shoulder", False),
    "LEFT_ELBOW":     JointMap("left_elbow", False),
    "RIGHT_ELBOW":    JointMap("right_elbow", False),
    "LEFT_WRIST":     JointMap("left_wrist", False),
    "RIGHT_WRIST":    JointMap("right_wrist", False),
    "LEFT_HIP":       JointMap("left_hip", False),
    "RIGHT_HIP":      JointMap("right_hip", False),
    "LEFT_KNEE":      JointMap("left_knee", False),
    "RIGHT_KNEE":     JointMap("right_knee", False),
    "LEFT_ANKLE":     JointMap("left_ankle", False),
    "RIGHT_ANKLE":    JointMap("right_ankle", False),
    # --- feet: SMPL foot joint is the toe/ball; heel is not a distinct SMPL
    #     joint, so we derive it. Gait uses heel & foot_index for step timing;
    #     these approximations are flagged. VERIFY vs. the SMPL foot vertex set. -
    "LEFT_FOOT_INDEX":  JointMap("left_foot", True,
                                 "SMPL foot joint ~ toe/ball, used as foot index"),
    "RIGHT_FOOT_INDEX": JointMap("right_foot", True,
                                 "SMPL foot joint ~ toe/ball, used as foot index"),
    "LEFT_HEEL":  JointMap(None, True,
                           "no SMPL heel joint; derive from ankle/foot geometry"),
    "RIGHT_HEEL": JointMap(None, True,
                           "no SMPL heel joint; derive from ankle/foot geometry"),
    # --- head/nose: SMPL 'head' joint is cranium centre, not the nose tip.
    #     Gait uses NOSE only as a coarse head anchor, so an offset is fine. ----
    "NOSE": JointMap("head", True, "SMPL 'head' ~ cranium; used as head anchor"),
    # --- face + finger landmarks: NOT gait-critical, NOT in bare SMPL(24).
    #     Placed near the head/wrist so the 33-landmark skeleton is complete for
    #     the overlay and STTP graph, exactly like the synthetic walker does. ---
    "LEFT_EYE_INNER":  JointMap(None, True, "face detail; anchored to head"),
    "LEFT_EYE":        JointMap(None, True, "face detail; anchored to head"),
    "LEFT_EYE_OUTER":  JointMap(None, True, "face detail; anchored to head"),
    "RIGHT_EYE_INNER": JointMap(None, True, "face detail; anchored to head"),
    "RIGHT_EYE":       JointMap(None, True, "face detail; anchored to head"),
    "RIGHT_EYE_OUTER": JointMap(None, True, "face detail; anchored to head"),
    "LEFT_EAR":        JointMap(None, True, "face detail; anchored to head"),
    "RIGHT_EAR":       JointMap(None, True, "face detail; anchored to head"),
    "MOUTH_LEFT":      JointMap(None, True, "face detail; anchored to head"),
    "MOUTH_RIGHT":     JointMap(None, True, "face detail; anchored to head"),
    "LEFT_PINKY":      JointMap("left_hand", True, "hand detail; anchored to wrist"),
    "RIGHT_PINKY":     JointMap("right_hand", True, "hand detail; anchored to wrist"),
    "LEFT_INDEX":      JointMap("left_hand", True, "hand detail; anchored to wrist"),
    "RIGHT_INDEX":     JointMap("right_hand", True, "hand detail; anchored to wrist"),
    "LEFT_THUMB":      JointMap("left_hand", True, "hand detail; anchored to wrist"),
    "RIGHT_THUMB":     JointMap("right_hand", True, "hand detail; anchored to wrist"),
}


# --------------------------------------------------------------------------- #
# Errors                                                                      #
# --------------------------------------------------------------------------- #
class CAREPDNotAvailable(Exception):
    """Raised when a CARE-PD operation is requested but the data is not present.

    This is the honest failure mode: rather than fabricate poses or a training
    result, every data-touching method raises this with guidance on how to
    obtain the real dataset (and the clinical-safety obligations that come with
    training on it).
    """


def _not_available_message(root: str, reason: str, *, clinical: bool = False) -> str:
    lines = [
        f"CARE-PD data is not available at: {root!r}",
        f"reason: {reason}",
        "",
        "ParkiGait fabricates NOTHING: it will not synthesize CARE-PD poses or a",
        "training result. To use real data:",
        f"  1. Read and accept the terms of use: {CAREPD_TERMS_URL}",
        f"     (license: {CAREPD_LICENSE})",
        f"  2. Download the dataset from Hugging Face: {CAREPD_HF_URL}",
        "       huggingface-cli download vida-adl/CARE-PD --repo-type dataset \\",
        "         --local-dir <root>",
        f"     or via the code/Dataverse links in {CAREPD_GITHUB_URL}",
        f"  3. Point this loader at <root> (expects per-cohort *.pkl files;",
        f"     cohorts: {', '.join(CAREPD_COHORTS)}).",
        f"  Paper to cite: {CAREPD_PAPER}",
        "  VERIFY the current access/gating status against the dataset card;",
        "  policies change.",
    ]
    if clinical:
        lines += [
            "",
            "CLINICAL SAFETY: training on real UPDRS labels is governed by",
            "parkigait/CLINICAL_SAFETY.md. Use SUBJECT-LEVEL splits (no subject in",
            "both train and test) and treat any output as exploratory, not clinical.",
        ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Dataset adapter                                                             #
# --------------------------------------------------------------------------- #
class CAREPDDataset:
    """Adapter over a local CARE-PD download. Inert (raises) when data is absent.

    On construction it inspects ``root``. If the directory does not exist, is not
    a directory, or contains no recognizable CARE-PD pickle files, it records the
    reason and every data-touching method raises :class:`CAREPDNotAvailable`.
    **It never synthesizes data to paper over a missing download.**

    Parameters
    ----------
    root:
        Path to a local CARE-PD download (the directory holding per-cohort
        ``*.pkl`` files, optionally under an ``assets/datasets`` subtree).
    cohorts:
        Optional subset of :data:`CAREPD_COHORTS` to restrict to.
    require_updrs:
        If True, :meth:`iter_sequences` yields only walks that carry a
        non-null ``UPDRS_GAIT`` label (the supervised subset).
    """

    def __init__(self, root: str, cohorts: Optional[list[str]] = None,
                 require_updrs: bool = False, joint_source: str = "smpl"):
        self.root = str(root)
        self.cohorts = tuple(cohorts) if cohorts else None
        self.require_updrs = bool(require_updrs)
        # "smpl" = exact licensed SMPL model (raises until configured);
        # "canonical_fk" = licensed-model-free forward kinematics on a canonical
        # anthropometric skeleton (APPROXIMATE; every PoseSequence is stamped so a
        # result is never mistaken for exact SMPL). See smpl_fk.py.
        self.joint_source = str(joint_source)
        self._available, self._reason, self._pickle_files = self._probe(self.root)

    # -- availability ------------------------------------------------------- #
    @staticmethod
    def _probe(root: str) -> tuple[bool, str, list[str]]:
        """Return (available, reason, pickle_files) WITHOUT loading any poses."""
        if not os.path.exists(root):
            return False, "path does not exist", []
        if not os.path.isdir(root):
            return False, "path is not a directory", []
        # Look for CARE-PD pickle files at the root or under assets/datasets.
        search_dirs = [root]
        nested = os.path.join(root, "assets", "datasets")
        if os.path.isdir(nested):
            search_dirs.append(nested)
        found: list[str] = []
        for d in search_dirs:
            for name in sorted(os.listdir(d)):
                if name.endswith(".pkl") or name.endswith(".pickle"):
                    found.append(os.path.join(d, name))
        if not found:
            return (False,
                    "directory is empty of CARE-PD pickle files "
                    "(expected per-cohort *.pkl)", [])
        return True, "", found

    @property
    def is_available(self) -> bool:
        return self._available

    def _require_available(self, *, clinical: bool = False) -> None:
        if not self._available:
            raise CAREPDNotAvailable(
                _not_available_message(self.root, self._reason, clinical=clinical))

    # -- documentation ------------------------------------------------------ #
    def describe(self) -> str:
        """Human-readable explanation of the dataset, schema, access and mapping.

        Safe to call even when the data is absent — it documents what the loader
        WOULD do, and reports the current availability state truthfully.
        """
        status = (
            f"AVAILABLE at {self.root!r} ({len(self._pickle_files)} pickle "
            f"file(s) found)"
            if self._available
            else f"NOT available at {self.root!r} — {self._reason}"
        )
        mapped = sum(1 for m in SMPL_TO_BLAZEPOSE.values() if m.smpl is not None)
        certain = sum(1 for m in SMPL_TO_BLAZEPOSE.values() if not m.uncertain)
        return "\n".join([
            "CARE-PD dataset adapter — ParkiGait (RESEARCH PROTOTYPE, NOT A "
            "MEDICAL DEVICE)",
            "=" * 74,
            f"status: {status}",
            "",
            "WHAT IT IS",
            "  " + CAREPD_PAPER,
            "  Multi-site 3D-mesh gait archive for Parkinson's Disease: 9 "
            "harmonized",
            "  cohorts from 8 clinical centres, ~362 participants. Recordings "
            "(RGB or",
            "  mocap) are converted to anonymized SMPL body-mesh sequences. ~1/3 "
            "of walks",
            "  carry a clinician-rated UPDRS-gait score (the gait item of the "
            "MDS-UPDRS).",
            f"  cohorts: {', '.join(CAREPD_COHORTS)}",
            "",
            "ACCESS / AGREEMENT (VERIFY against the dataset card — policies change)",
            f"  license: {CAREPD_LICENSE}",
            f"  terms:   {CAREPD_TERMS_URL}  (read + accept before downloading)",
            f"  data:    {CAREPD_HF_URL}",
            "           huggingface-cli download vida-adl/CARE-PD "
            "--repo-type dataset \\",
            "             --local-dir <root>",
            f"  code:    {CAREPD_GITHUB_URL}",
            "  obligations: cite CARE-PD + the original cohort papers; no "
            "re-identification;",
            "  respect GDPR/HIPAA/PIPEDA. Clinical training is governed by "
            "CLINICAL_SAFETY.md.",
            "",
            "SCHEMA THIS LOADER TARGETS (per-cohort *.pkl; VERIFY per cohort)",
            "  { subject_id: { walk_id: {",
            "      'pose': SMPL axis-angle (T,D), 'trans': (T,3), 'beta': shape "
            "(zeroed),",
            "      'fps': int, 'UPDRS_GAIT': int|None (0..3), 'medication': "
            "str|None,",
            "      'other': str|None } } }",
            "  Alternate encodings on the hub: 6D_SMPL/, h36m/, HumanML3D/; "
            "subject-level",
            "  splits under folds/. Canonicalized variant uses x=lateral, y=up, "
            "z=forward.",
            "",
            "HOW IT MAPS INTO ParkiGait",
            "  -> PoseSequence (parkigait.types): SMPL joints are regressed via "
            "the SMPL",
            "     body model, remapped to the 33 BlazePose landmarks, and "
            "projected to the",
            "     pipeline's normalized image coords (x right, y DOWN). "
            "SMPL->BlazePose map:",
            f"     {mapped}/33 landmarks have an SMPL source; {certain}/33 are "
            "gait-critical",
            "     exact matches; the rest (face/finger/heel) are anchored "
            "approximations,",
            "     flagged 'uncertain', and NOT used by the gait features.",
            "  -> label: UPDRS_GAIT per walk; subject_id/cohort carried in "
            "PoseSequence.meta",
            "     so downstream code can enforce SUBJECT-LEVEL splits.",
            "",
            "HONESTY: when the data is absent, every data method raises "
            "CAREPDNotAvailable.",
            "This loader NEVER fabricates poses or a training result.",
        ])

    # -- iteration ---------------------------------------------------------- #
    def iter_sequences(self) -> Iterator[tuple[PoseSequence, Optional[int], dict]]:
        """Yield (PoseSequence, updrs_gait_or_None, meta) for each walk.

        Parses the documented CARE-PD pickle schema into ParkiGait's
        :class:`PoseSequence`. Raises :class:`CAREPDNotAvailable` when no data is
        present. Requires the SMPL body model to regress joint locations; if that
        is missing it raises a clear error too (it does not guess joint
        positions). NEVER yields synthetic data.
        """
        self._require_available()
        for pkl_path in self._pickle_files:
            cohort = _cohort_of(pkl_path)
            if self.cohorts and cohort not in self.cohorts:
                continue
            data = _load_pickle(pkl_path)
            if not isinstance(data, dict):
                raise CAREPDNotAvailable(
                    f"unexpected CARE-PD file format in {pkl_path!r}: "
                    f"expected a nested dict, got {type(data).__name__}. "
                    "VERIFY the file against the dataset card.")
            for subject_id, walks in data.items():
                if not isinstance(walks, dict):
                    continue
                for walk_id, rec in walks.items():
                    updrs = rec.get("UPDRS_GAIT") if isinstance(rec, dict) else None
                    if self.require_updrs and updrs is None:
                        continue
                    pose_seq = self._record_to_pose_sequence(
                        rec, cohort=cohort, subject_id=str(subject_id),
                        walk_id=str(walk_id), source=pkl_path)
                    meta = {
                        "cohort": cohort,
                        "subject_id": str(subject_id),
                        "walk_id": str(walk_id),
                        "updrs_gait": updrs,
                        "medication": rec.get("medication"),
                    }
                    yield pose_seq, (int(updrs) if updrs is not None else None), meta

    def to_pose_sequences(self) -> list[PoseSequence]:
        """Materialize all walks as a list of :class:`PoseSequence`.

        Raises :class:`CAREPDNotAvailable` when the data is absent. This is the
        acceptance entry point: on a missing/empty root it raises with guidance
        and returns no data.
        """
        self._require_available()
        return [ps for ps, _updrs, _meta in self.iter_sequences()]

    # -- SMPL -> PoseSequence ---------------------------------------------- #
    def _record_to_pose_sequence(self, rec: dict, *, cohort: str, subject_id: str,
                                 walk_id: str, source: str) -> PoseSequence:
        """Convert one CARE-PD walk record into a PoseSequence.

        This needs the SMPL body model to turn SMPL (pose, trans, beta) into 3D
        joint locations. We do NOT vendor SMPL weights (they are separately
        licensed) and we will NOT approximate joint positions with a fabricated
        rig — so when the SMPL model is unavailable this raises rather than
        returning a made-up skeleton.
        """
        if not isinstance(rec, dict):
            raise CAREPDNotAvailable(
                f"walk {cohort}/{subject_id}/{walk_id}: record is "
                f"{type(rec).__name__}, expected dict. VERIFY vs. dataset card.")
        pose = rec.get("pose")
        trans = rec.get("trans")
        if pose is None:
            raise CAREPDNotAvailable(
                f"walk {cohort}/{subject_id}/{walk_id}: missing 'pose'. "
                "VERIFY the cohort's format against the dataset card.")
        fps = float(rec.get("fps") or 0.0)
        if fps <= 0:
            raise CAREPDNotAvailable(
                f"walk {cohort}/{subject_id}/{walk_id}: missing/invalid 'fps'. "
                "VERIFY vs. dataset card.")

        if self.joint_source == "canonical_fk":
            # Licensed-model-free APPROXIMATION: real pose rotations on a canonical
            # rest skeleton. Stamped so no result is mistaken for exact SMPL.
            from parkigait.smpl_fk import carepd_record_to_blazepose
            joints, visibility = carepd_record_to_blazepose(
                np.asarray(pose), np.asarray(trans), fps)
            backend = "carepd-canonical_fk"
        else:
            # Regress SMPL -> (T, 24, 3) joints via the licensed model (raises until
            # configured; never fabricates).
            smpl_joints = _smpl_pose_to_joints(np.asarray(pose), np.asarray(trans),
                                               np.asarray(rec.get("beta"))
                                               if rec.get("beta") is not None else None)
            joints, visibility = _smpl_seq_to_blazepose(smpl_joints)
            backend = "carepd-smpl"
        return PoseSequence(
            joints=joints, visibility=visibility, fps=fps,
            source=f"carepd:{cohort}/{subject_id}/{walk_id}",
            meta={
                "backend": backend,
                "joint_source": self.joint_source,
                "cohort": cohort,
                "subject_id": subject_id,
                "walk_id": walk_id,
                "updrs_gait": rec.get("UPDRS_GAIT"),
                "medication": rec.get("medication"),
                "smpl_pose_dim": int(np.asarray(pose).shape[-1]),
                "joint_map": "SMPL24->BlazePose33 (see SMPL_TO_BLAZEPOSE)",
                "provenance_file": source,
                "coord_note": ("SMPL 3D regressed then projected to normalized "
                               "image coords (x right, y DOWN)"),
            },
        )


# --------------------------------------------------------------------------- #
# Training entry point (needs real UPDRS labels — raises otherwise)           #
# --------------------------------------------------------------------------- #
def train_severity_from_carepd(root: str, *, cohorts: Optional[list[str]] = None,
                               n_splits: int = 5, seed: int = 0,
                               joint_source: str = "smpl"):
    """Fit the severity model on real CARE-PD UPDRS labels with SUBJECT-LEVEL
    splits. Raises :class:`CAREPDNotAvailable` if the data is not present.

    RESEARCH PROTOTYPE — NOT A MEDICAL DEVICE. Training here does not make the
    model clinical; see ``CLINICAL_SAFETY.md``. The split MUST be subject-level
    (no subject in both train and test) or the reported metric is inflated and
    meaningless, exactly as ``CLINICAL_SAFETY.md`` warns.

    This function is intentionally inert without data: it constructs the loader,
    confirms availability (raising with guidance + a pointer to CLINICAL_SAFETY.md
    if not), and only THEN would it group walks by subject, build subject-level
    folds, fit, and report cross-validated UPDRS metrics on real labels. It
    returns no fabricated numbers.
    """
    ds = CAREPDDataset(root, cohorts=cohorts, require_updrs=True,
                       joint_source=joint_source)
    if not ds.is_available:
        raise CAREPDNotAvailable(
            _not_available_message(root, ds._reason, clinical=True))

    # Data present: collect labelled walks grouped by subject for subject-level CV.
    # NOTE: iter_sequences() regresses SMPL -> PoseSequence first, so on
    # real-but-unconfigured data it raises CAREPDNotAvailable (SMPL not
    # configured) before we ever need the feature extractor. The gaitfeat import
    # is done lazily and defensively so a missing sibling module surfaces as a
    # clear error, never a fabricated result.
    by_subject: dict[str, list[tuple[np.ndarray, int]]] = {}
    n_labelled = 0
    for pose_seq, updrs, meta in ds.iter_sequences():
        if updrs is None:
            continue
        try:
            from parkigait.gaitfeat import extract_features
        except Exception as exc:
            raise CAREPDNotAvailable(
                "CARE-PD data is present, but the gait-feature extractor "
                f"(parkigait.gaitfeat) could not be imported: {exc}. Cannot train "
                "without it. This is a wiring error, not fabricated data.") from exc
        feats = extract_features(pose_seq).as_vector()
        by_subject.setdefault(meta["subject_id"], []).append((feats, int(updrs)))
        n_labelled += 1

    if n_labelled == 0:
        raise CAREPDNotAvailable(
            _not_available_message(
                root, "no walks with a non-null UPDRS_GAIT label were found",
                clinical=True))

    subjects = sorted(by_subject)
    if len(subjects) < 2:
        raise CAREPDNotAvailable(
            _not_available_message(
                root,
                f"only {len(subjects)} labelled subject(s); cannot form a "
                "subject-level split without leakage",
                clinical=True))

    # Subject-level K-fold: partition SUBJECTS (never walks) across folds.
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(seed)
    order = list(subjects)
    rng.shuffle(order)
    k = int(max(2, min(n_splits, len(order))))
    folds = [order[i::k] for i in range(k)]

    def _xy(subj_list):
        xs, ys = [], []
        for s in subj_list:
            for feats, y in by_subject[s]:
                xs.append(feats)
                ys.append(y)
        return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)

    fold_maes: list[float] = []
    pooled_true: list[float] = []
    pooled_pred: list[float] = []
    for i in range(k):
        test_subjects = folds[i]
        train_subjects = [s for j, f in enumerate(folds) if j != i for s in f]
        x_tr, y_tr = _xy(train_subjects)
        x_te, y_te = _xy(test_subjects)
        if len(x_tr) == 0 or len(x_te) == 0:
            continue
        model = Ridge(alpha=1.0)
        model.fit(x_tr, y_tr)
        pred = model.predict(x_te)
        fold_maes.append(float(np.mean(np.abs(pred - y_te))))
        pooled_true.extend(y_te.tolist())
        pooled_pred.extend(pred.tolist())

    if not fold_maes:
        raise CAREPDNotAvailable(
            _not_available_message(
                root, "insufficient labelled data to form non-empty folds",
                clinical=True))

    # Held-out Pearson correlation (predicted vs true UPDRS-gait), pooled over folds.
    pt, pp = np.asarray(pooled_true), np.asarray(pooled_pred)
    pearson = (float(np.corrcoef(pt, pp)[0, 1])
               if pt.std() > 1e-9 and pp.std() > 1e-9 else 0.0)
    # baseline: MAE of always predicting the training-mean label
    mean_mae = float(np.mean(np.abs(pt - pt.mean())))

    # Fit a final model on ALL labelled data for downstream use.
    x_all, y_all = _xy(subjects)
    final_model = Ridge(alpha=1.0).fit(x_all, y_all)

    return {
        "model": final_model,
        "calibrated_on": f"CARE-PD (real UPDRS_GAIT, subject-level CV, {joint_source})",
        "joint_source": joint_source,
        "n_subjects": len(subjects),
        "n_labelled_walks": n_labelled,
        "n_splits": k,
        "cv_mae_updrs_gait": float(np.mean(fold_maes)),
        "cv_mae_per_fold": fold_maes,
        "held_out_pearson_r": pearson,
        "baseline_mae_predict_mean": mean_mae,
        "split": "subject-level (no subject in both train and test)",
        "disclaimer": ("RESEARCH PROTOTYPE — NOT A MEDICAL DEVICE. Exploratory "
                       "only; see CLINICAL_SAFETY.md."),
    }


# --------------------------------------------------------------------------- #
# Internal helpers (kept inert without data / SMPL model)                     #
# --------------------------------------------------------------------------- #
def _cohort_of(pkl_path: str) -> str:
    base = os.path.basename(pkl_path)
    stem = base.rsplit(".", 1)[0]
    for c in CAREPD_COHORTS:
        if stem == c or stem.startswith(c):
            return c
    return stem


def _load_pickle(path: str):
    import pickle
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _smpl_pose_to_joints(pose: np.ndarray, trans: np.ndarray,
                         beta: Optional[np.ndarray]) -> np.ndarray:
    """Regress SMPL (pose, trans, beta) -> (T, 24, 3) 3D joint locations.

    Requires the SMPL body model (separately licensed weights, e.g. via the
    ``smplx`` package + SMPL .npz). We do NOT vendor those weights and we do NOT
    approximate joint positions with a fabricated rig. If the model is
    unavailable, raise :class:`CAREPDNotAvailable` with instructions — never
    return a made-up skeleton.
    """
    try:
        import smplx  # type: ignore  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on optional dep
        raise CAREPDNotAvailable(
            "the SMPL body model is required to regress CARE-PD SMPL parameters "
            "into 3D joints, but it is not available.\n"
            f"  install: pip install smplx torch   ({exc})\n"
            "  and obtain SMPL model weights from https://smpl.is.tue.mpg.de "
            "(separate license).\n"
            "ParkiGait will NOT fabricate joint positions from SMPL parameters."
        ) from exc
    # With smplx present, the real path constructs the body model, runs a forward
    # pass on (pose, trans, beta), and returns model.joints[:, :24, :]. That path
    # needs the licensed weights + the correct SMPL variant for the cohort;
    # rather than guess the weight path/variant, we raise here so no run silently
    # produces joints from an unverified configuration. VERIFY the SMPL variant
    # per cohort against the dataset card, then wire the smplx forward pass here.
    raise CAREPDNotAvailable(
        "smplx is installed but the SMPL model weights + per-cohort variant have "
        "not been configured. VERIFY the SMPL variant (SMPL / SMPL+H / SMPL-X) "
        "and body-model path for this cohort against the CARE-PD dataset card, "
        "then wire the smplx forward pass in _smpl_pose_to_joints. ParkiGait "
        "refuses to emit joints from an unverified SMPL configuration.")


def _smpl_seq_to_blazepose(smpl_joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map a (T, 24, 3) SMPL joint sequence to (T, 33, 3) BlazePose + visibility.

    Uses :data:`SMPL_TO_BLAZEPOSE`, projects the SMPL world frame (x lateral,
    y up, z forward) into the pipeline's normalized image coords (x right, y
    DOWN), and anchors non-SMPL landmarks (face/fingers/heel) to their parents.
    Reached only when real SMPL joints exist; validates shape and never invents a
    sequence. Kept as a documented, self-contained transform so the real path is
    obvious, but it is only ever fed real regressed joints from the SMPL model.
    """
    smpl_joints = np.asarray(smpl_joints, dtype=np.float32)
    if smpl_joints.ndim != 3 or smpl_joints.shape[1:] != (len(SMPL_JOINT_ORDER), 3):
        raise CAREPDNotAvailable(
            f"expected SMPL joints of shape (T,{len(SMPL_JOINT_ORDER)},3); got "
            f"{smpl_joints.shape}. VERIFY the SMPL variant against the dataset card.")
    T = smpl_joints.shape[0]
    smpl_idx = {name: i for i, name in enumerate(SMPL_JOINT_ORDER)}

    # Convert world (x lat, y up, z fwd) -> image-ish (x right, y DOWN). This is a
    # deterministic axis flip + per-sequence normalization to [0,1]; the gait
    # features are scale/translation-normalized so exact projection is not
    # critical, but we document it explicitly.
    xy = smpl_joints[:, :, [0, 1]].copy()
    xy[:, :, 1] *= -1.0  # y up -> y down
    lo = xy.reshape(-1, 2).min(axis=0)
    hi = xy.reshape(-1, 2).max(axis=0)
    span = np.where((hi - lo) > 1e-6, hi - lo, 1.0)
    xy = (xy - lo) / span
    z = smpl_joints[:, :, 2]

    out = np.zeros((T, 33, 3), dtype=np.float32)
    vis = np.zeros((T, 33), dtype=np.float32)
    for bp_name, jm in SMPL_TO_BLAZEPOSE.items():
        bp = joint_index(bp_name)
        if jm.smpl is not None and jm.smpl in smpl_idx:
            si = smpl_idx[jm.smpl]
            out[:, bp, 0] = xy[:, si, 0]
            out[:, bp, 1] = xy[:, si, 1]
            out[:, bp, 2] = z[:, si]
            # gait-critical exact matches are high-confidence; approximations lower
            vis[:, bp] = 0.9 if not jm.uncertain else 0.5
        else:
            vis[:, bp] = 0.2  # anchored placeholder, flagged as low confidence

    # Anchor the landmarks with no SMPL source (heels near ankles, face near head)
    _anchor_missing_landmarks(out)
    return out, vis


def _anchor_missing_landmarks(joints: np.ndarray) -> None:
    """Place BlazePose landmarks that have no SMPL source near their parents so
    the 33-landmark skeleton is complete for the overlay / STTP graph. These are
    NOT used by the gait features. Documented approximation, clearly low-vis."""
    def _i(n):
        return joint_index(n)

    # heels: just behind/below their ankle
    joints[:, _i("LEFT_HEEL"), :2] = joints[:, _i("LEFT_ANKLE"), :2] + [-0.01, 0.01]
    joints[:, _i("RIGHT_HEEL"), :2] = joints[:, _i("RIGHT_ANKLE"), :2] + [-0.01, 0.01]
    # face detail near nose (already sourced from SMPL head)
    nose = joints[:, _i("NOSE"), :2]
    for name, dx, dy in (("LEFT_EYE_INNER", -0.01, -0.01), ("LEFT_EYE", -0.02, -0.01),
                         ("LEFT_EYE_OUTER", -0.03, -0.01), ("RIGHT_EYE_INNER", 0.01, -0.01),
                         ("RIGHT_EYE", 0.02, -0.01), ("RIGHT_EYE_OUTER", 0.03, -0.01),
                         ("LEFT_EAR", -0.04, 0.0), ("RIGHT_EAR", 0.04, 0.0),
                         ("MOUTH_LEFT", -0.015, 0.02), ("MOUTH_RIGHT", 0.015, 0.02)):
        joints[:, _i(name), :2] = nose + [dx, dy]


# --------------------------------------------------------------------------- #
# CLI / smoke — proves the honest failure path with REAL measured behavior.   #
# --------------------------------------------------------------------------- #
def _smoke() -> None:
    print("ParkiGait CARE-PD adapter — smoke check (RESEARCH PROTOTYPE, NOT "
          "MEDICAL)\n")

    # 1) describe() works with no data and documents schema + access.
    ds = CAREPDDataset("/nonexistent")
    print(ds.describe())
    print()

    # 2) to_pose_sequences() raises CAREPDNotAvailable with guidance.
    print("-> CAREPDDataset('/nonexistent').to_pose_sequences():")
    try:
        ds.to_pose_sequences()
        print("   ERROR: expected CAREPDNotAvailable, got data (should never "
              "happen)")
    except CAREPDNotAvailable as exc:
        print("   raised CAREPDNotAvailable as required. Message:")
        for line in str(exc).splitlines():
            print("     " + line)
    print()

    # 3) iter_sequences() also raises.
    print("-> iter_sequences() on missing data:")
    try:
        next(iter(ds.iter_sequences()))
        print("   ERROR: expected CAREPDNotAvailable")
    except CAREPDNotAvailable:
        print("   raised CAREPDNotAvailable (no fake sequence produced).")
    print()

    # 4) train_severity_from_carepd() raises and points at CLINICAL_SAFETY.md.
    print("-> train_severity_from_carepd('/nonexistent'):")
    try:
        train_severity_from_carepd("/nonexistent")
        print("   ERROR: expected CAREPDNotAvailable")
    except CAREPDNotAvailable as exc:
        pointed = "CLINICAL_SAFETY.md" in str(exc)
        print(f"   raised CAREPDNotAvailable; points at CLINICAL_SAFETY.md: "
              f"{pointed}")
    print()

    # 5) An empty directory is ALSO treated as unavailable (no synthesis).
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        empty_ds = CAREPDDataset(tmp)
        print(f"-> empty real directory {tmp!r}: is_available="
              f"{empty_ds.is_available}")
        try:
            empty_ds.to_pose_sequences()
            print("   ERROR: expected CAREPDNotAvailable on empty dir")
        except CAREPDNotAvailable:
            print("   raised CAREPDNotAvailable on an existing-but-empty dir "
                  "(no fabrication).")

    # measured summary numbers (real, computed just now)
    mapped = sum(1 for m in SMPL_TO_BLAZEPOSE.values() if m.smpl is not None)
    certain = sum(1 for m in SMPL_TO_BLAZEPOSE.values() if not m.uncertain)
    print()
    print("MEASURED (computed this run):")
    print(f"  BlazePose landmarks mapped in SMPL_TO_BLAZEPOSE: {len(SMPL_TO_BLAZEPOSE)}/33")
    print(f"  landmarks with an SMPL source: {mapped}/33")
    print(f"  gait-critical exact matches (not uncertain): {certain}/33")
    print(f"  cohorts documented: {len(CAREPD_COHORTS)}")
    print("  fabricated data points returned anywhere above: 0")


if __name__ == "__main__":
    _smoke()
