# FinoraXpense : Personal Finance and Budget Tracker

FinoraXpense is a full-stack personal finance management web application built with Python Flask and PostgreSQL. It allows users to track their income and expenses, set monthly budgets, manage savings goals, and visualize their financial data through interactive charts. Transactions are automatically categorized using the Anthropic Claude AI API, and users can sign in with Google via OAuth 2.0.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Database Schema](#database-schema)
- [Local Setup](#local-setup)
- [Environment Variables](#environment-variables)
- [Deploying to Render](#deploying-to-render)
- [Configuring Google OAuth on Render](#configuring-google-oauth-on-render)
- [Uploading to GitHub Securely](#uploading-to-github-securely)
- [API Routes Reference](#api-routes-reference)
- [Security Notes](#security-notes)

---

## Features

### User Features

- Registration and login with role selection (User or Admin)
- Google OAuth 2.0 sign-in
- Personal dashboard with real-time income, expense, and balance summary
- Add, edit, and delete income and expense transactions
- AI-powered automatic transaction categorization using Claude API (Transport, Food and Dining, Bills and Utilities, Shopping, Healthcare, Education, Entertainment, Savings, Income Source, General)
- Monthly budget tracking with real-time percentage alerts at 70%, 90%, and 100%
- Savings goals with fund transfer from main balance
- Analytics tab with income vs expense line chart and category-wise pie chart
- AI-generated personalized spending insights based on transaction history
- Profile management with name, contact info, NID number, and profile picture upload
- Dark and light mode toggle
- Mobile-responsive layout with smooth touch interactions
- Custom animated cursor on desktop

### Admin Features

- Admin dashboard with platform-wide user count, transaction count, total income, and total expenses
- Full user table with NID number, phone, verification status, and registration date
- Verify or delete any user account
- Category-wise analytics pulled from the transactions table
- All admin actions are logged with IP address in the admin_logs table

### Security Features

- Password hashing using PBKDF2:SHA256 via Werkzeug
- Secure session management with a secret key loaded from environment variables
- Admin registration protected by a server-side secret key (ADMIN_SECRET)
- NID duplicate check prevents multiple registrations with the same national ID
- Profile picture upload validated by file type (PNG, JPG, JPEG, GIF, WEBP) and size (maximum 2MB)
- Google OAuth CSRF protection using the state parameter
- PostgreSQL connection pooling via psycopg2 SimpleConnectionPool
- All credentials and secrets are loaded exclusively from the .env file

---

## Tech Stack

| Layer       | Technology                                      |
|-------------|--------------------------------------------------|
| Backend     | Python 3.10+, Flask 2.x                         |
| Database    | PostgreSQL hosted on Neon, psycopg2             |
| Frontend    | HTML5, Tailwind CSS (CDN), Font Awesome 6       |
| Templating  | Jinja2                                           |
| AI          | Anthropic Claude API (claude-haiku-4-5-20251001)|
| Auth        | Flask Sessions, Google OAuth 2.0                |
| Deployment  | Render Web Service                              |
| Font        | Plus Jakarta Sans via Google Fonts              |

---

## Project Structure

```
FinoraXpense/
|
|-- app.py                            # Main Flask application, all routes and logic
|-- .env                              # Secret environment variables — never commit this
|-- .env.example                      # Safe template for environment variables
|-- .gitignore                        # Git ignore rules
|-- requirements.txt                  # Python package dependencies
|
|-- static/
|   |-- style.css                     # Global stylesheet
|   |-- uploads/
|       |-- FinoraXpense_Logo.png     # Application logo
|       |-- 330px-Mastercard_2019_logo.svg.png
|       |-- 330px-PayPal.svg.png
|       |-- 330px-UPI-Logo-vector.svg.png
|       |-- 330px-Visa_Inc._logo_(2021-present).svg.png
|       |-- avatars/                  # User profile pictures (auto-created at runtime)
|
|-- templates/
    |-- home.html                     # Public landing page
    |-- login.html                    # Login page with User, Admin, and Google OAuth options
    |-- register.html                 # Registration page with role selection
    |-- user.html                     # User dashboard (finance tracker, analytics, goals)
    |-- admin.html                    # Admin control panel
    |-- error.html                    # 404 and 500 error pages
```

---

## Database Schema

The application automatically creates all tables on first run using `init_db()`.

| Table            | Purpose                                                       |
|------------------|---------------------------------------------------------------|
| users            | Stores registration data, role, NID, avatar, login timestamps |
| transactions     | Primary income and expense records per user                   |
| budgets          | Monthly budget limits set by each user                        |
| financial_goals  | Savings goals with target and current amounts                 |
| expenses         | Legacy expense table retained for backward compatibility      |
| admin_logs       | Audit log of all admin actions with IP address                |
| user_sessions    | Session tracking per user                                     |
| analytics        | Event log (login, register, social login)                     |

---

## Local Setup

### Prerequisites

- Python 3.10 or higher
- pip
- Git
- A Neon PostgreSQL account — free tier available at neon.tech
- An Anthropic API key from console.anthropic.com
- A Google Cloud Console project with OAuth 2.0 credentials

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
```

### Step 4 — Set Up Environment Variables

Copy the example file and fill in your real values:

```bash
cp .env.example .env
```

Open `.env` and add your credentials. See the Environment Variables section below for details.

### Step 5 — Run the Application

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

### Where to get each value

- `SECRET_KEY` — Generate with `python -c "import secrets; print(secrets.token_hex(32))"`
- `DATABASE_URL` — From your Neon project dashboard, Connection Details section
- `ANTHROPIC_API_KEY` — From console.anthropic.com, API Keys section
- `ADMIN_SECRET` — Choose any strong string. Anyone who has this can register as admin.
- `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` — From Google Cloud Console, APIs and Services, Credentials

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

Go to the Environment tab in your Render service settings. Add each variable from your `.env` file one by one. Do not upload the `.env` file itself.

```
SECRET_KEY
DATABASE_URL
ANTHROPIC_API_KEY
ADMIN_SECRET
GOOGLE_CLIENT_ID
GOOGLE_CLIENT_SECRET
FLASK_ENV           -> production
```

### Step 4 — Deploy

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
__pycache__/
venv/
static/uploads/avatars/
*.pyc
instance/
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

| Method | Route                                    | Description                              | Auth Required |
|--------|------------------------------------------|------------------------------------------|---------------|
| GET    | /                                        | Public landing page                      | No            |
| GET    | /login                                   | Login page                               | No            |
| POST   | /login                                   | Process login form                       | No            |
| GET    | /register                                | Registration page                        | No            |
| POST   | /register                                | Process registration form                | No            |
| GET    | /logout                                  | Logout and clear session                 | No            |
| GET    | /login/google                            | Redirect to Google OAuth                 | No            |
| GET    | /login/google/callback                   | Handle Google OAuth callback             | No            |
| GET    | /dashboard                               | User financial dashboard                 | Yes           |
| POST   | /api/profile/update                      | Update user profile and avatar           | Yes           |
| POST   | /api/transaction                         | Add income or expense transaction        | Yes           |
| POST   | /api/transaction/int:id/update           | Edit an existing transaction             | Yes           |
| POST   | /api/transaction/int:id/delete           | Delete a transaction                     | Yes           |
| GET    | /api/chart-data                          | Real-time chart data (JSON)              | Yes           |
| POST   | /api/budget/update                       | Set or update monthly budget limit       | Yes           |
| POST   | /api/goal/add                            | Create a new savings goal                | Yes           |
| POST   | /api/goal/int:id/add_funds               | Transfer funds to a savings goal         | Yes           |
| GET    | /admin                                   | Admin control panel                      | Admin only    |
| GET    | /api/admin/users                         | Get all users (JSON)                     | Admin only    |
| POST   | /api/admin/users/int:id/verify           | Verify a user account                    | Admin only    |
| POST   | /api/admin/users/int:id/delete           | Delete a user account                    | Admin only    |
| GET    | /api/admin/analytics                     | Platform analytics data (JSON)           | Admin only    |

---

## Security Notes

- Never commit the `.env` file to any repository, public or private.
- Use a SECRET_KEY that is at least 32 characters long and fully random.
- The ADMIN_SECRET controls who can register as an admin. Keep it confidential.
- Set `FLASK_ENV=production` on Render so that debug mode is disabled.
- The application enforces HTTPS redirect URIs for Google OAuth in all non-localhost environments.
- File uploads are validated on both the frontend and backend to prevent oversized or malformed files.
- All database queries use parameterized statements through psycopg2 to prevent SQL injection.

---

## License

MIT License. Free for personal and educational use.
