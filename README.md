# FinoraXpense — Personal Finance and Budget Tracker

FinoraXpense is a full-stack personal finance management web application built with Python Flask and PostgreSQL. It lets users track income and expenses, manage monthly budgets, and build savings goals. Transaction categorization and personalized financial-health insights are both powered by **two custom-trained machine learning models that run locally inside the app** — no third-party AI API is called at request time. Users can sign in with email/password or with Google via OAuth 2.0.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Database Schema](#database-schema)
- [The AI / Machine Learning Engine](#the-ai--machine-learning-engine)
- [Local Setup](#local-setup)
- [Environment Variables](#environment-variables)
- [Running the Test Suite](#running-the-test-suite)
- [Deploying to Render](#deploying-to-render)
- [Configuring Google OAuth on Render](#configuring-google-oauth-on-render)
- [Uploading to GitHub Securely](#uploading-to-github-securely)
- [API Routes Reference](#api-routes-reference)
- [Security Notes](#security-notes)

---

## Features

### User Features

- Registration and login with role selection (User or Admin), plus Google OAuth 2.0 sign-in
- Personal dashboard with real-time income, expense, and balance summary
- Add, edit, and delete income and expense transactions, with a recent-activity widget and a full transaction history table
- Automatic transaction categorization powered by a **custom-trained local NLP model** (Transport, Food & Dining, Bills & Utilities, Shopping, Healthcare, Education, Entertainment, Savings, Income Source, General) — understands Bangla, Banglish, and English, and runs instantly with no external API call
- Monthly budget tracking with a live usage bar that turns critical (red) once spending crosses 90% of the set limit
- Savings goals with fund transfers straight from the main balance
- Analytics tab with an income vs. expense bar chart (auto-refreshes every 5 seconds) and a category-wise doughnut chart
- AI-generated personalized "Smart Action Plan" insights from a **custom-trained Random Forest model**, each paired with a quick-action shortcut (Cut Expense / Start Saving)
- Profile management with name, contact info, date of birth, gender, division/district, NID number, and profile picture upload
- Dark and light mode toggle
- Mobile-responsive layout with a bottom tab bar and touch-optimized interactions
- Custom dual-element animated cursor on the public pages (landing, login, register) — automatically disabled on touch devices

### Admin Features

- Admin dashboard with platform-wide user count, transaction count, total income, and total expenses
- Full user table with NID number, phone, verification status, and registration date
- Verify or permanently delete any user account (an admin cannot delete their own account)
- User-growth trend, category-wise expense breakdown, and platform activity, rendered as line/doughnut/bar charts
- All admin actions logged with the acting admin's IP address in the `admin_logs` table

### Security Features

- Password hashing using PBKDF2:SHA256 via Werkzeug
- Session-based authentication with a secret key loaded from environment variables and `SameSite=Lax` cookies
- Admin registration gated behind a server-side `ADMIN_SECRET`
- NID duplicate check prevents multiple registrations with the same national ID
- Profile picture uploads validated by file type (PNG, JPG, JPEG, GIF, WEBP) and size (maximum 2MB), on both frontend and backend
- Google OAuth CSRF protection using the `state` parameter, with automatic HTTPS enforcement on the redirect URI outside localhost
- PostgreSQL connections require SSL (`sslmode=require`) and are pooled via `psycopg2.pool.SimpleConnectionPool`
- All credentials and secrets are loaded exclusively from the `.env` file, and every query is parameterized to prevent SQL injection

---

## Tech Stack

| Layer            | Technology                                                  |
|-------------------|--------------------------------------------------------------|
| Backend           | Python 3.10+, Flask                                          |
| Database          | PostgreSQL hosted on Neon, psycopg2 with connection pooling  |
| Machine Learning  | scikit-learn — `MultinomialNB` + `HashingVectorizer` for categorization, `RandomForestClassifier` for financial-health scoring |
| Frontend          | HTML5, Tailwind CSS (CDN), Font Awesome 6.4                  |
| Charts            | Chart.js                                                      |
| Templating        | Jinja2                                                        |
| Auth              | Flask sessions, Google OAuth 2.0 (manual authorization-code flow) |
| Deployment        | Render Web Service (Gunicorn)                                |
| Font              | Plus Jakarta Sans via Google Fonts                            |

---

## Project Structure

```
FinoraXpense/
|
|-- app.py                            # Main Flask application, all routes and logic
|-- train_model.py                    # Trains the Random Forest financial-health model
|-- train_nlp.py                      # Builds the synthetic Bangla/Banglish/English dataset and trains the categorizer
|-- finance_dataset.csv               # ~3,000-row dataset consumed by train_model.py
|-- financial_health_model.pkl        # Output of train_model.py (generated locally, see AI section below)
|-- nlp_vectorizer.pkl                # Output of train_nlp.py
|-- nlp_category_model.pkl            # Output of train_nlp.py
|-- .env                              # Secret environment variables — never commit this
|-- .env.example                      # Safe template for environment variables
|-- .gitignore                        # Git ignore rules
|-- requirements.txt                  # Python package dependencies
|
|-- tests/
|   |-- test_app.py                   # Pytest suite covering routes, auth, and ML integration
|
|-- static/
|   |-- FinoraXpense_Logo.png         # Application logo
|   |-- uploads/
|       |-- avatars/                  # User profile pictures (auto-created at runtime)
|
|-- templates/
    |-- home.html                     # Public landing page
    |-- login.html                    # Login page with User, Admin, and Google OAuth options
    |-- register.html                 # Registration page with role selection, division/district, and NID validation
    |-- user.html                     # User dashboard (transactions, analytics, budgets, savings, AI insights)
    |-- admin.html                    # Admin control panel
    |-- error.html                    # 404 and 500 error pages
```

---

## Database Schema

All tables are created automatically on first run via `init_db()`, except `transactions`, which is created the first time a logged-in user opens `/dashboard`.

| Table            | Purpose                                                                 |
|-------------------|--------------------------------------------------------------------------|
| `users`           | Registration data, role, division/district, NID, avatar, login timestamps |
| `transactions`    | The primary income/expense ledger — feeds the dashboard, charts, and AI insights |
| `budgets`         | One upserted monthly budget limit per user                              |
| `financial_goals` | Savings goals with target and current amounts                           |
| `admin_logs`      | Audit log of admin actions, with IP address                             |
| `analytics`       | Event log (login, registration, social login, logout)                   |
| `expenses`        | Carried over from an earlier schema version; still created on startup but no longer written to by any current route |
| `user_sessions`   | Reserved for future per-session tracking; created on startup but not yet populated |

---

## The AI / Machine Learning Engine

FinoraXpense does not call any external AI API for its intelligence — both models are trained offline by the two scripts in the project root and loaded into memory with `joblib` when `app.py` starts. If either `.pkl` file is missing, the app doesn't crash: categorization silently falls back to "General" and the dashboard shows a "model missing" notice instead of AI insights.

### 1. Transaction categorization — `train_nlp.py`

- Pipeline: `HashingVectorizer` (4,194,304 features, 1–3 word n-grams) feeding a `MultinomialNB` classifier (`alpha=0.01`)
- The script procedurally builds its own training set by combining prefixes, category-specific items, and action words — verified to generate **19,348,651** synthetic text patterns across Bangla, Banglish, and English, including all 64 Bangladeshi districts, major global cities, local banks, and mobile financial services (bKash, Nagad, Rocket, etc.)
- Outputs 10 categories: Transport, Food & Dining, Income Source, Bills & Utilities, Shopping, Healthcare, Education, Entertainment, Savings, and General
- Trained with `partial_fit`, so the model is architecturally capable of incremental online learning. `app.py` includes a `train_ai_model_realtime()` helper that would let a user's manual category correction retrain the model on the spot — it currently isn't called from any route, so this is available for wiring up rather than active in the UI today

### 2. Financial-health scoring — `train_model.py`

- A `RandomForestClassifier` (200 trees, `max_depth=15`, `class_weight='balanced'`) trained on `finance_dataset.csv`, with a synthetic-data generator as a fallback if that file is missing
- Input features: total income, total expense, savings rate, budget utilization, and the food/entertainment/shopping spend ratios
- Predicts 6 classes, each mapped directly to a "Smart Action Plan" card on the dashboard: Critical Stress, Moderate, Excellent, High Food Spend, High Entertainment Spend, and High Shopping Spend
- Running the script against the bundled dataset produces **100.00% training accuracy and 99.67% testing accuracy**

---

## Local Setup

### Prerequisites

- Python 3.10 or higher
- pip
- Git
- A Neon PostgreSQL account — free tier available at neon.tech
- A Google Cloud Console project with OAuth 2.0 credentials (only needed if you want Google sign-in)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/finoraxpense.git
cd finoraxpense
```

### Step 2 — Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac or Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
pip install scikit-learn pandas numpy joblib
```

`requirements.txt` currently only lists the core web dependencies. The four packages on the second line power `train_model.py`, `train_nlp.py`, and the model-loading code in `app.py` — install them explicitly, or add them to `requirements.txt` yourself.

### Step 4 — Train the AI Models

```bash
python train_model.py
python train_nlp.py
```

- `train_model.py` finishes in seconds and prints its own accuracy report.
- `train_nlp.py` generates upward of 19 million synthetic text patterns before it starts training — expect it to take noticeably longer and use significant RAM. Let it run to completion.
- Both scripts write `.pkl` files into the project root, which `app.py` loads at startup. You can skip this step and run the app anyway, just without categorization or AI insights.

### Step 5 — Set Up Environment Variables

```bash
cp .env.example .env
```

Open `.env` and add your credentials. See [Environment Variables](#environment-variables) below.

### Step 6 — Run the Application

```bash
python app.py
```

Open your browser and go to `http://localhost:5000`

---

## Environment Variables

Create a `.env` file in the project root with the following variables. Never commit this file to GitHub.

```
SECRET_KEY=a_very_long_random_string_at_least_32_characters
FLASK_ENV=development
FLASK_HOST=0.0.0.0
FLASK_PORT=5000

DATABASE_URL=postgresql://username:password@host/dbname?sslmode=require

ANTHROPIC_API_KEY=sk-ant-your_key_here

ADMIN_SECRET=your_chosen_admin_registration_secret

GOOGLE_CLIENT_ID=your_client_id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=GOCSPX-your_client_secret
```

> **Note on `ANTHROPIC_API_KEY`:** this is left over from an earlier iteration of the project that categorized transactions via the Claude API. `app.py` no longer reads this variable anywhere — categorization now runs entirely on the local model trained by `train_nlp.py`. It's safe to leave blank or delete the line; nothing in the current codebase depends on it.

### Where to get each value

- `SECRET_KEY` — Generate with `python -c "import secrets; print(secrets.token_hex(32))"`
- `DATABASE_URL` — From your Neon project dashboard, Connection Details section
- `ADMIN_SECRET` — Choose any strong string. Anyone who has this can register as admin.
- `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` — From Google Cloud Console, APIs and Services, Credentials

---

## Running the Test Suite

The project includes a pytest suite at `tests/test_app.py` covering registration and login validation, role-based access control, Google OAuth edge cases (including state-mismatch and cancellation paths), transaction CRUD, budget and goal logic, all six AI financial-health outcomes, and the admin API. The most recent run bundled with this repo shows all **57 tests passing**.

```bash
pip install pytest pytest-html
pytest tests/ -v
```

To regenerate an HTML report like the one included in this repo:

```bash
pytest tests/ --html=report.html --self-contained-html
```

---

## Deploying to Render

### Step 1 — Create a Render Account

Go to render.com and sign up using your GitHub account.

### Step 2 — Create a New Web Service

1. From the Render dashboard, click New and then Web Service.
2. Connect your GitHub repository.
3. Fill in the following settings:

| Setting        | Value                            |
|----------------|----------------------------------|
| Name           | finoraxpense (or your choice)    |
| Runtime        | Python 3                         |
| Build Command  | pip install -r requirements.txt  |
| Start Command  | gunicorn app:app                 |
| Instance Type  | Free                             |

### Step 3 — Add Environment Variables

Go to the Environment tab in your Render service settings. Add each variable below. Do not upload the `.env` file itself.

```
SECRET_KEY
DATABASE_URL
ADMIN_SECRET
GOOGLE_CLIENT_ID
GOOGLE_CLIENT_SECRET
FLASK_ENV           -> production
```

### Step 4 — Handle the Trained Model Files

`financial_health_model.pkl`, `nlp_vectorizer.pkl`, and `nlp_category_model.pkl` are not excluded by `.gitignore`, so if you ran both training scripts locally before your first commit, they'll already be part of the repo Render deploys. If you'd rather not commit the trained files, add them to `.gitignore` instead and change the Build Command to:

```
pip install -r requirements.txt && python train_model.py && python train_nlp.py
```

Keep in mind `train_nlp.py` generates on the order of 19 million rows before it trains, so this will noticeably slow down — or on Render's free tier, may strain — the build step.

### Step 5 — Deploy

Click Create Web Service. Render will build and deploy your application. Once complete, your app will be live at:

```
https://your-app-name.onrender.com
```

Note: The free tier on Render will spin down after 15 minutes of inactivity. The first request after a period of inactivity may take 20 to 30 seconds to respond.

---

## Configuring Google OAuth on Render

After deployment, you must update your Google Cloud Console credentials to allow login from the Render domain. Without this step, Google sign-in will fail on the live site.

### Step 1 — Open Google Cloud Console

Go to console.cloud.google.com and select your project.

### Step 2 — Navigate to OAuth Credentials

From the left menu, go to APIs and Services, then Credentials. Click on your OAuth 2.0 Client ID.

### Step 3 — Add the Authorized Redirect URI

Under Authorized redirect URIs, click Add URI and enter:

```
https://your-app-name.onrender.com/login/google/callback
```

Replace `your-app-name` with the actual name of your Render service.

### Step 4 — Add the Authorized JavaScript Origin

Under Authorized JavaScript origins, click Add URI and enter:

```
https://your-app-name.onrender.com
```

### Step 5 — Keep Localhost for Development

If you want Google login to also work on your local machine, keep these entries in the same credential:

```
Redirect URI:          http://localhost:5000/login/google/callback
JavaScript Origin:     http://localhost:5000
```

Both production and local entries can exist at the same time.

### Step 6 — Save

Click Save. Changes may take a few minutes to propagate.

---

## Uploading to GitHub Securely

### Step 1 — Verify Your .gitignore is Correct

Make sure the `.gitignore` file in your project root includes at minimum:

```
.env
*.env
__pycache__/
venv/
static/uploads/avatars/
*.pyc
instance/
.pytest_cache/
```

### Step 2 — Verify the .env File is Not Being Tracked

Run this command before pushing anything:

```bash
git status
```

If `.env` appears in the list, do not proceed. Check that your `.gitignore` file is in the correct location and has `.env` listed.

You can also run:

```bash
git ls-files | grep "\.env"
```

If this command returns no output, the `.env` file is not being tracked and it is safe to push.

### Step 3 — Create a GitHub Repository

1. Go to github.com and click New Repository.
2. Give it a name such as `finoraxpense`.
3. Set visibility to Private if you want to keep your code private.
4. Do not initialize with a README since you already have one.
5. Click Create Repository.

### Step 4 — Initialize Git and Push

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/finoraxpense.git
git branch -M main
git push -u origin main
```

---

## API Routes Reference

| Method     | Route                                    | Description                                       | Auth Required |
|------------|--------------------------------------------|-----------------------------------------------------|----------------|
| GET        | /                                         | Public landing page                                | No             |
| GET, POST  | /login                                    | Login page and credential authentication           | No             |
| GET, POST  | /register                                 | Registration page and account creation             | No             |
| GET        | /logout                                   | Logout and clear session                           | No             |
| GET        | /login/google                             | Redirect to Google OAuth consent screen            | No             |
| GET        | /login/google/callback                    | Handle the Google OAuth callback                   | No             |
| GET        | /dashboard                                | User financial dashboard                           | Yes            |
| POST       | /api/profile/update                       | Update profile info and avatar                     | Yes            |
| POST       | /api/transaction                          | Add an income or expense transaction (auto-categorized) | Yes       |
| POST       | /api/transaction/int:tx_id/update         | Edit an existing transaction                        | Yes            |
| POST       | /api/transaction/int:tx_id/delete         | Delete a transaction                                | Yes            |
| GET        | /api/chart-data                           | Chart data as JSON (polled every 5 seconds)         | Yes            |
| POST       | /api/budget/update                        | Set or update the current month's budget limit      | Yes            |
| POST       | /api/goal/add                             | Create a new savings goal                           | Yes            |
| POST       | /api/goal/int:goal_id/add_funds           | Transfer funds to a savings goal                    | Yes            |
| GET        | /admin                                    | Admin control panel                                 | Admin only     |
| GET        | /api/admin/users                          | Get all users (JSON)                                | Admin only     |
| POST       | /api/admin/users/int:user_id/verify       | Verify a user account                               | Admin only     |
| POST       | /api/admin/users/int:user_id/delete       | Delete a user account (self-deletion blocked)       | Admin only     |
| GET        | /api/admin/analytics                      | Platform analytics data (JSON)                      | Admin only     |

---

## Security Notes

- Never commit the `.env` file to any repository, public or private.
- Use a `SECRET_KEY` that is at least 32 characters long and fully random.
- `ADMIN_SECRET` controls who can register as an admin. Keep it confidential.
- Set `FLASK_ENV=production` on Render so that debug mode is disabled.
- The application enforces HTTPS redirect URIs for Google OAuth in all non-localhost environments.
- File uploads are validated on both the frontend and backend to prevent oversized or malformed files.
- All database queries use parameterized statements through psycopg2 to prevent SQL injection.
- Database connections require SSL (`sslmode=require`).

---

## License

MIT License. Free for personal and educational use.
