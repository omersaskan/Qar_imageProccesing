# Data Retention and Privacy

## 1. Purpose

Meshysiz Asset Factory stores visual data, derived metrics, 3D assets, and training manifests.

The system must be designed so that future ML and fine-tuning are possible without violating privacy, consent, or retention expectations.

The key rule is:

```text
Every session may be dataset-ready,
but not every session is training-eligible.
```

---

## 2. Data Classes

| Data Class | Examples |
|---|---|
| Raw video | Uploaded or captured source video |
| Original frames | Selected camera frames |
| Masked frames | Object-centric frames |
| Masks | Binary or alpha masks |
| Thumbnails | Small preview images |
| Derived metrics | Quality, coverage, reconstruction, export metrics |
| Final GLB | Published or draft 3D asset |
| User feedback | User approval/rejection |
| Human review labels | QA labels |
| Training manifest | Privacy-safe structured ML record |
| Training registry | JSONL index of manifests |

---

## 3. Recommended Retention

| Artifact Type | Recommended Retention | Notes |
|---|---:|---|
| Raw video | Short, configurable | Should not be retained forever by default |
| Selected frames | Consent-based | Longer retention only if eligible |
| Masks | Long retention | Valuable for ML and debugging |
| Reports / metrics | Long retention | Privacy-safe if anonymized |
| Failed session frames | Limited debug retention | Example: 14 days |
| Reconstruction scratch | Short retention | Heavy data, example: 48 hours |
| Final GLB | Product lifecycle | Retain while asset/product is active |
| Training manifests | Versioned long-term | Must not contain raw identity |
| Training registry | Versioned long-term | Consent-aware |
| Audit history | Long-term | Operational traceability |

---

## 4. Consent

Training usage must be explicit and configurable.

Consent states:

| Status | Meaning | Eligible for Training |
|---|---|---|
| `unknown` | No explicit consent | No |
| `denied` | Training use denied | No |
| `granted` | Training use granted | Yes |
| `internal_only` | Internal test/session data | Policy-dependent |

Default should be:

```text
eligible_for_training = false
```

unless training consent is granted or the environment explicitly marks data as internal-only.

---

## 5. Anonymization Rules

Training manifests must:

- hash `product_id`
- avoid user identifiers
- avoid email, phone, billing, or account identifiers
- strip or ignore EXIF location data
- avoid storing raw device identifiers
- separate user identity from training data

Recommended product hash:

```text
sha256(product_id + configured_salt)
```

---

## 6. Deletion and Revocation

If a user or product owner requests deletion:

- raw video should be deleted if retained
- selected frames should be deleted if required
- training eligibility should be revoked
- future dataset exports should exclude the session
- registry should mark the manifest as revoked or ineligible
- published assets should follow product deletion policy

---

## 7. Training Dataset Export Policy

A dataset export should include only sessions where:

- `eligible_for_training = true`
- consent allows training
- deletion has not been requested
- labels are valid
- required metrics are present

Dataset export should be versioned and auditable.

---

## 8. Operator Responsibilities

Operators must not manually copy raw visual data into training folders without consent.

Operators should verify:

- manifest exists
- consent status exists
- eligibility is correct
- no raw user identity is present
- revoked sessions are excluded