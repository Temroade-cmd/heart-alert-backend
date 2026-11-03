# Heart Alert, backend

The REST API behind **Heart Alert**, a cardiovascular risk screening platform for doctors.
A doctor enters 11 clinical measurements, an XGBoost model returns a risk score,
and HeartBot, an LLM assistant, helps discuss the result in the context of that patient.

FastAPI · PostgreSQL · XGBoost · OpenRouter · Stripe · deployed on Render

> **Medical disclaimer.** Heart Alert is a decision support prototype built for a Master's thesis.
> It is not a certified medical device and has not been clinically validated.
> Its output must never replace the judgment of a qualified clinician.

---

## Contents

- [Features](#features)
- [The prediction model](#the-prediction-model)
- [Project structure](#project-structure)
- [API overview](#api-overview)
- [Getting started](#getting-started)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Limitations](#limitations)
- [Project context and team](#project-context-and-team)

---

## Features

**Risk prediction**
- `POST /predict` scores a patient from 11 clinical features and returns a probability, a risk category (low, moderate, high) and a yes/no prediction.
- Every prediction is stored and can be reviewed later in the doctor's history.

**Patient records**
- Create, update, list and delete patients, then link predictions to them.

**HeartBot, the clinical assistant**
- `POST /chat` sends the question to an LLM through OpenRouter (Gemini 2.0 Flash in the thesis version, set with `OPENROUTER_MODEL`).
- The patient's data and risk score are injected into the prompt, so the answer is about this patient and not generic.
- A system prompt keeps it in a supporting role (no diagnosis, no drug dosages), and a medical disclaimer is appended when the answer has none.

**Accounts and roles**
- Four roles, `doctor`, `assistant`, `hospital_admin` and `admin`.
- New doctors upload their medical license and an ID at signup, and stay `pending` until an admin reviews and approves them.
- A doctor can request an assistant, who then works on the doctor's patients and history.
- A hospital admin can invite doctors by email and follow their activity.

**Subscriptions**
- Freemium, Pro and Hospital plans, with a prediction quota per plan.
- Payments through Stripe Checkout, kept in sync by a Stripe webhook.

**Security**
- JWT access and refresh tokens, passwords hashed with bcrypt (older sha256 hashes are still accepted).
- Email verification and password reset through Brevo, phone number check through NumLookup.
- Rate limiting with SlowAPI on login, signup, password reset, verification, support and chat (in memory, or Redis when `REDIS_URL` is set).
- Security headers on every response, HTTPS redirect in production, strict CORS origins.
- Uploaded files checked for size and extension.

---

## The prediction model

### Data

`clev-dataset.csv` is the combined UCI Heart Disease dataset, **1,190 patients** from four sites
(Cleveland, Hungary, Switzerland, Long Beach VA). 629 patients have heart disease, 561 do not.

### The 11 input features

| Field | Meaning | Values |
|---|---|---|
| `age` | Age | years |
| `sex` | Sex | 1 male, 0 female |
| `cp` | Chest pain type | 1 typical angina, 2 atypical angina, 3 non-anginal pain, 4 asymptomatic |
| `trestbps` | Resting blood pressure | mm Hg |
| `chol` | Serum cholesterol | mg/dl |
| `fbs` | Fasting blood sugar > 120 mg/dl | 1 yes, 0 no |
| `restecg` | Resting ECG | 0 normal, 1 ST-T abnormality, 2 left ventricular hypertrophy |
| `thalach` | Maximum heart rate reached | bpm |
| `exang` | Exercise-induced angina | 1 yes, 0 no |
| `oldpeak` | ST depression induced by exercise | mm |
| `slope` | Slope of the peak exercise ST segment | 1 upsloping, 2 flat, 3 downsloping |

The Cleveland data alone has two more features, `ca` (vessels seen by fluoroscopy) and `thal` (thallium scintigraphy).
They were not collected consistently across the four cohorts, so the model uses the 11 features shared by all of them.
It also keeps the model usable without these specialised exams, the 11 features come from routine exams or an exercise stress test.

### Results

Six algorithms were compared on a stratified 80/20 split of the 1,190 patients, so 238 test patients. Scores in %.

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---:|---:|---:|---:|---:|
| **XGBoost** | **93.70** | **95.12** | **92.86** | **93.98** | 97.17 |
| Random Forest | 91.60 | 92.06 | 92.06 | 92.06 | **97.41** |
| LightGBM | 91.18 | 91.34 | 92.06 | 91.70 | 96.53 |
| CatBoost | 88.66 | 88.37 | 90.48 | 89.41 | 95.76 |
| SVM | 87.82 | 86.47 | 91.27 | 88.80 | 93.52 |
| Logistic Regression | 84.03 | 84.38 | 85.71 | 85.04 | 90.41 |

XGBoost was kept for production. It has the best accuracy, precision, recall and F1, an AUC close to the best one, and it is fast to serve.
On the 238 test patients it made 15 errors, 6 false positives and 9 false negatives.
A 5-fold cross-validation gives a mean accuracy of 92.35% (± 3.48).

The production model is an `XGBClassifier` with 100 trees, a maximum depth of 6 and a learning rate of 0.3.
Its two most important features are the ST slope (42.3%) and the chest pain type (16.3%).

### Benchmark on the Cleveland subset

To compare with the literature, the same six algorithms were also trained on the classic Cleveland subset
(303 patients, 13 features). There, Random Forest is the best model, with 88.52% accuracy and an AUC-ROC of 95.13%.
XGBoost reaches 85.25% on this smaller set, and only becomes the best model on the larger combined dataset.

### From probability to risk category

| Risk score | Category | Recommendation shown |
|---|---|---|
| below 0.30 | low | continue a healthy lifestyle and routine screening |
| 0.30 to 0.60 | moderate | monitor closely, lifestyle changes, regular check-ups |
| 0.60 and above | high | cardiology consultation recommended |

### Files

- `xgboost_model.pkl` is the model served by the API, loaded once at startup by `app/ml/model.py`.
- `train_11_features.py` trains the Random Forest baseline on the same 11 features (`heart_disease_model_11features.pkl`).
- `model_comparison_visuals.py` draws the comparison chart of the six models.

---

## Project structure

```
app/
  main.py             FastAPI app, middleware and most of the endpoints
  routers/auth.py     /auth routes (login, refresh token, hospital admin)
  ml/model.py         loads the XGBoost model and turns a probability into a risk category
  models.py           SQLAlchemy tables (doctors, patients, predictions, plans, tickets...)
  schemas.py          Pydantic request and response models
  database.py         database engine, PostgreSQL in production, SQLite fallback locally
  auth_config.py      password hashing and JWT tokens
  dependencies.py     current user, role and status checks
  rate_limiter.py     SlowAPI limiter
  middleware.py       HTTPS redirect and security headers
  validators.py       file upload and password strength checks
  email_config.py     transactional emails through Brevo
  config.py           checks the environment variables at startup
migrations/           one-off database migration scripts
clev-dataset.csv      training data
xgboost_model.pkl     production model
```

---

## API overview

Once the server runs, the full interactive documentation is generated by FastAPI at **`/docs`**.
The main groups of endpoints are below.

| Area | Endpoints |
|---|---|
| Accounts | `POST /signup`, `POST /login`, `POST /auth/refresh`, `GET /me`, `PUT /me`, `POST /change-password` |
| Verification | `POST /verify-phone`, `POST /send-verification-email`, `POST /verify-email`, `POST /forgot-password`, `POST /reset-password` |
| Prediction | `POST /predict`, `GET /history` |
| Patients | `POST /patients`, `GET /patients`, `GET /patients/{id}`, `PUT /patients/{id}`, `DELETE /patients/{id}` |
| HeartBot | `POST /chat` (30 requests per minute) |
| Doctors and assistants | `POST /doctor/request-assistant`, `GET /doctor/assistant`, `GET /assistant/doctor`, `POST /upload-document` |
| Hospitals | `POST /hospital/invite-doctor`, `GET /hospital/my-doctors`, `GET /hospital/stats`, `POST /accept-invitation` |
| Billing | `GET /pricing-plans`, `POST /create-checkout-session`, `POST /stripe-webhook`, `GET /my-subscription`, `POST /cancel-subscription` |
| Support | `POST /support/create`, `GET /my-tickets`, `POST /contact-support` |
| Admin | `GET /admin/pending-doctors`, `PUT /admin/approve-doctor/{id}`, `GET /admin/stats`, `GET /admin/tickets` |

Example prediction request (the doctor must be logged in and approved)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"patient_name": "John Doe", "age": 60, "sex": 1, "cp": 4, "trestbps": 130,
       "chol": 210, "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
       "oldpeak": 0.8, "slope": 1}'
```

```json
{ "prediction_id": 1, "risk_score": 0.52, "risk_category": "moderate", "has_disease": true }
```

---

## Getting started

You need Python 3.11 or newer. PostgreSQL is optional for local work, the API falls back to SQLite.

```bash
git clone https://github.com/Temroade-cmd/heart-alert-backend.git
cd heart-alert-backend

python -m venv .venv
source .venv/bin/activate          # Windows  .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file at the root (see [Configuration](#configuration)). For a first local run,
an empty file is enough. A temporary JWT key is generated and the database is `heart_disease.db` (SQLite).

Create the tables, then start the server.

```bash
python -c "from app.database import Base, engine; from app import models; Base.metadata.create_all(bind=engine)"
uvicorn app.main:app --reload
```

The API runs on `http://localhost:8000` and the documentation on `http://localhost:8000/docs`.

---

## Configuration

All settings are read from environment variables (or a `.env` file).

| Variable | Required | Purpose |
|---|---|---|
| `DATABASE_URL` | in production | PostgreSQL URL, `postgresql://user:password@host:5432/db` |
| `JWT_SECRET_KEY` | in production | secret used to sign tokens, at least 32 characters |
| `ENVIRONMENT` | no | `development` or `production` (production enables the HTTPS redirect) |
| `ALLOWED_ORIGINS` | no | comma-separated list of frontend origins allowed by CORS |
| `FRONTEND_URL` | no | frontend URL used in emails and redirects |
| `BACKEND_URL` | no | public URL of this API, used in email links |
| `OPENROUTER_API_KEY` | for HeartBot | OpenRouter key, the chatbot is disabled without it |
| `OPENROUTER_MODEL` | no | model used by HeartBot |
| `BREVO_API_KEY` | for emails | Brevo key for verification and reset emails |
| `MAIL_FROM`, `MAIL_FROM_NAME` | no | sender address and name |
| `NUMLOOKUP_API_KEY` | for phone check | NumLookup key |
| `PHONE_VERIFICATION_ENABLED` | no | `true` or `false` |
| `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET` | for payments | Stripe keys |
| `REDIS_URL` | no | shared storage for rate limiting across several instances |
| `ACCESS_TOKEN_EXPIRE_MINUTES`, `REFRESH_TOKEN_EXPIRE_DAYS` | no | token lifetimes (default 1440 minutes and 7 days) |
| `DB_POOL_SIZE`, `DB_MAX_OVERFLOW` | no | PostgreSQL connection pool |

Never commit the `.env` file, it is already in `.gitignore`.

---

## Deployment

The API is deployed on [Render](https://render.com) with a managed PostgreSQL database.
The environment variables above are set in the Render dashboard. A typical start command is

```bash
gunicorn app.main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
```

---

## Limitations

- The model is trained on public research data collected in the 1980s, 1,190 patients from four centres. Missing values are stored as 0 in the source data (172 patients have a cholesterol of 0). It has not been validated on other populations.
- The data comes mostly from Western populations. No Algerian data was available, so the model may not fit local patients.
- The risk thresholds (0.30 and 0.60) are fixed by hand, not calibrated on clinical outcomes.
- HeartBot depends on an external LLM. Its answers can be wrong and must be checked by the clinician.

---

## Project context and team

Heart Alert was built as a Master's thesis project (M2, Information Systems Engineering) at
Université Djillali Liabès de Sidi Bel Abbès, Algeria, in 2025/2026, within the national *Startup 1275* programme.
A Flutter mobile app uses this API.

| Team | Supervision |
|---|---|
| Sawsen Benturquia | Mohamed Amine Nemmich, supervisor |
| Tamer Benturquia | Mahmoud Fahsi, co-supervisor |
| Abdelkader Boukhalfa | Asmaa Boudali, co-supervisor |
| Yazid Bessadet | |
