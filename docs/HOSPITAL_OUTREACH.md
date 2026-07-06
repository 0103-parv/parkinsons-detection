# Hospital / clinician outreach kit

The goal is a **research collaboration**, not deployment. You want a movement-disorder
neurologist (or a physical-therapy gait lab) to give feedback and, eventually, co-run a
small IRB-approved validation study. That is the credible ask and the strong college-app line.

---

## 1. Cold email (copy, personalize the first line and the doctor's name)

> **Subject:** High school researcher, on device gait screening for Parkinson's, seeking your guidance
>
> Dear Dr. [Last name],
>
> My name is Parv Mehndiratta and I am a high school student researcher in California. Over the
> past several months I built an open source tool that analyzes a short video of someone walking
> and screens for the gait changes associated with Parkinson's, and I would be grateful for a few
> minutes of your time and your honest feedback.
>
> The tool runs entirely on a phone or laptop, so no patient video ever leaves the device, which
> keeps it private by design. I trained and tested it on the public CARE-PD research dataset, which
> includes real UPDRS gait ratings from clinicians. Using strict subject level validation, meaning
> no patient appears in both the training and testing sets, it reaches an area under the curve of
> about 0.86 for detecting impaired gait. I checked carefully that this number is real and not the
> result of data leakage.
>
> I want to be very clear about what it is and is not. It is a research and screening aid, not a
> diagnostic device, and it is not ready to make any decision about a patient. I built it to be
> honest about its limits, and I documented exactly what would be needed before it could ever be
> used in a clinic.
>
> My hope is to learn from a clinician who actually treats these patients. If you are open to it, I
> would love a short call to hear where a tool like this could genuinely help, and whether you would
> ever consider a small research collaboration, for example comparing the tool's output to your
> assessment on anonymized videos under proper ethics approval.
>
> I have attached a one page summary, and the code is open at
> github.com/0103-parv/parkinsons-detection. Thank you so much for your time and for the work you do.
>
> Warm regards,
> Parv Mehndiratta
> [your email] · [your school]

**Why this email works:** it is humble, it leads with a real validated number, it states the
limits before they have to ask (clinicians respect this), and it asks for guidance and a call, not
for them to use unproven software on patients.

---

## 2. One page brief (attach as a PDF)

**ParkiGait: on-device gait screening for Parkinson's (research prototype)**

*What it is.* A tool that takes a short walking video, extracts a skeleton on the device, computes
clinically grounded gait features (walking speed, cadence, stride length, joint range of motion,
arm swing, trunk flexion, step to step variability, freezing index), and outputs a flag for
possible parkinsonian gait plus an exploratory severity estimate. It runs fully on a phone or
laptop with no cloud, so patient video never leaves the device.

*The problem.* Parkinsonian gait change is early and measurable, but assessment needs a specialist
and a clinic visit, and cloud video raises privacy concerns. A private, low cost, on-device screen
could help route patients to a neurologist sooner and track gait over time.

*Current results (measured on the public CARE-PD dataset, 110 patients with clinician UPDRS gait
scores, strict subject level cross validation):*

| Task | Result |
|---|---|
| Detect impaired gait (screening) | area under the curve 0.86, about 79 percent sensitivity and specificity |
| Estimate severity (harder) | correlation 0.70 with clinician UPDRS gait |
| Leakage check | passes a label permutation control and an independent audit |
| Speed and memory | about 27 ms per frame, under 400 MB, on a laptop CPU |

*What it is not.* Not a diagnostic device. Not a replacement for a clinician or the standard exam.
Not ready for any patient care decision. It is a research and screening aid.

*What I am seeking.* Feedback from a clinician on where this could genuinely help, and, if you are
open to it, a small research collaboration under ethics approval to compare the tool's flag to your
own assessment on anonymized gait videos.

*Honest limits.* The 3D joints are approximated from a canonical body model rather than the
licensed research model. Terrain, walking speed, and clothing can affect the features. Validation
so far is on one public dataset, not a prospective clinical study.

*Contact.* Parv Mehndiratta, [email]. Open source code and full honest documentation:
github.com/0103-parv/parkinsons-detection

---

## 3. Who to contact (in priority order)

1. **The CARE-PD dataset authors** (the group that released the data you trained on). They already
   care about exactly this problem and are the most likely to reply to a student. Find them on the
   dataset's Hugging Face page and the paper.
2. **Movement-disorder neurologists** at nearby academic medical centers (UCSF, Stanford, UC Davis
   if you are in the Bay Area). Look for faculty pages that mention "Parkinson's" or "movement
   disorders." Email is usually listed.
3. **Physical therapy / biomechanics gait labs** at universities. They run gait analysis daily and
   love new tools.
4. **The hospitals in India and Africa you mentioned** as research partners, framed the same way:
   a validation collaboration, not a deployment.

**How to find the email:** university faculty directory, the "contact" line on their lab website,
or the corresponding-author email on one of their papers.

## 4. Etiquette and follow-up

- Personalize the first sentence (mention one of their papers or their clinic).
- Keep it short. One email, one attachment.
- If no reply in about ten days, send one brief and polite follow up, then move on.
- Send a handful at a time so you can respond quickly when someone says yes.

## 5. For the college application (keep it honest)

- Log every email, reply, and call with dates. That correspondence is your evidence.
- The strong, true claims: "built and open sourced a gait analysis tool, trained and validated it
  on real clinical data with an area under the curve of 0.86 under honest subject level validation,
  and initiated outreach / a collaboration with clinicians to validate it."
- Do **not** claim it is deployed in a hospital or diagnoses patients. It is not, and reviewers can
  tell. The honest version is more impressive and it will hold up in an interview.
- If a clinician agrees to collaborate or you start an IRB conversation, that is a genuinely
  standout line. That is the goal.
