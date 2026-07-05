# Clinical safety & scope — read this first

**ParkiGait is a research and educational prototype. It is NOT a medical device.
It must not be used to diagnose, screen, triage, or make any care decision for a
real person.** This file explains that boundary plainly, because the project
brief mentioned hospitals wanting to use it, and that cannot happen with this
software as it stands.

## Why not — the honest reasons

1. **It is not trained on real, labelled clinical data.** The severity model
   ships fit on *synthetic* gait signatures (and, for the voice track, on small
   public research datasets). It has never seen a real Parkinson's patient's
   walk paired with a neurologist's UPDRS score, because we do not have CARE-PD
   access on this machine. An uncalibrated model that has never been validated
   against ground truth cannot be trusted to be right about a real person.

2. **It has not been clinically validated.** There is no prospective study, no
   held-out real-patient test set, no measured sensitivity/specificity on
   patients, no comparison against a gold-standard clinical exam. "Works on a
   demo video" is not evidence of clinical accuracy. A missed sign (false
   negative) can delay real care; a false positive can cause real harm — anxiety,
   unnecessary tests, wrong treatment.

3. **Deploying it to patients would, in most jurisdictions, be illegal.** A tool
   that outputs a Parkinson's-related score for clinical use is a *regulated
   medical device*. In India that is **CDSCO** (Medical Devices Rules, 2017); in
   the US, the **FDA** (Software as a Medical Device); the EU has the **MDR**;
   many African national regulators have their own frameworks. Clinical use also
   requires ethics-board / **IRB** approval and informed consent. None of that
   exists here.

4. **Camera, lighting, clothing, and demographic bias are unquantified.** Phone
   cameras, frame rates, occlusion, loose clothing, skin tone, walking aids, and
   room layout all move the numbers, and we have not measured by how much. A
   model that is accurate for one population can be silently wrong for another.

## What real hospital deployment would actually require

Roughly, in order:

1. A **data-use agreement** for CARE-PD (or an equivalent labelled cohort), and
   ideally a partnership to collect data from the *actual* clinics that would use
   it (their cameras, their patients).
2. **Training and internal validation** on real UPDRS labels with strict
   **subject-level** splits (no patient in both train and test — see the voice
   track's `detect.py` for why this matters; naive splits inflate scores).
3. **External validation** on an independent site the model never trained on.
4. A **prospective clinical study** with a pre-registered protocol, powered to
   estimate sensitivity/specificity against a gold standard, with **fairness
   audits** across skin tone (Fitzpatrick), age, sex, and mobility aids.
5. **IRB / ethics approval and informed consent** for every study.
6. **Regulatory clearance** (CDSCO / FDA / MDR as applicable) as Software as a
   Medical Device, including a quality system, risk management (ISO 14971), and a
   clinical evaluation report.
7. **Human-in-the-loop** framing: even then, a tool like this is a *decision
   support aid* for a clinician, not an autonomous diagnostician.

## What ParkiGait *is* good for, honestly

- A **methods reference**: a real, runnable pipeline from video to gait features
  to an exploratory score, with the topology-preserving pruning (STTP) and
  mixed-precision quantization (LieQ) ideas implemented and measured.
- A **science-fair / research artifact** you can demo truthfully: "here is a
  working prototype and here are the *real* numbers I measured; here is exactly
  what would be needed to make it clinical."
- A **starting point** that is already wired for real data: point `carepd.py` at
  a real labelled cohort and the same pipeline trains and evaluates honestly.

If someone asks "can a hospital use this today?", the correct answer is **no, and
here is the checklist above for what it would take.** Saying otherwise would be
unsafe and untrue.
