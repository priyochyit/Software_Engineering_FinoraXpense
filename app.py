import os
import secrets
import hashlib
import re
import json
from datetime import datetime, timedelta
from functools import wraps
from dotenv import load_dotenv

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Securely loaded from .env without hardcoded fallback
app.secret_key = os.getenv('SECRET_KEY')

# File Upload Configuration for Profile Pictures
UPLOAD_FOLDER = os.path.join('static', 'uploads', 'avatars')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Set max upload size to 2MB (2 * 1024 * 1024 bytes)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024

# Fix cookie issues across different domain redirects
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Neon PostgreSQL Database Configuration (Securely loaded from .env)
DATABASE_URL = os.getenv('DATABASE_URL')

# Connection Pool for better performance
try:
    connection_pool = SimpleConnectionPool(1, 5, DATABASE_URL, sslmode='require')
except Exception as e:
    print(f"Database connection error: {e}")
    connection_pool = None

# ==================== DATABASE UTILITIES ====================

def get_db():
    """Get database connection from pool"""
    if connection_pool:
        return connection_pool.getconn()
    return None

def release_db(conn):
    """Release database connection back to pool"""
    if connection_pool and conn:
        connection_pool.putconn(conn)

def init_db():
    """Initialize database tables"""
    try:
        conn = get_db()
        if not conn:
            print("Database connection failed")
            return False
        
        cursor = conn.cursor()
        
        # Users Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                email VARCHAR(255) UNIQUE NOT NULL,
                name VARCHAR(255) NOT NULL,
                last_name VARCHAR(255),
                password_hash VARCHAR(255) NOT NULL,
                phone VARCHAR(20),
                date_of_birth DATE,
                gender VARCHAR(20),
                division VARCHAR(100),
                district VARCHAR(100),
                nid_number VARCHAR(17),
                is_admin BOOLEAN DEFAULT FALSE,
                is_verified BOOLEAN DEFAULT FALSE,
                mfa_enabled BOOLEAN DEFAULT TRUE,
                mfa_secret VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                avatar_url TEXT
            )
        ''')

        # Add new columns if they don't exist (for existing databases)
        new_columns = [
            ("phone", "VARCHAR(20)"),
            ("date_of_birth", "DATE"),
            ("gender", "VARCHAR(20)"),
            ("division", "VARCHAR(100)"),
            ("district", "VARCHAR(100)"),
            ("nid_number", "VARCHAR(17)"),
            ("is_admin", "BOOLEAN DEFAULT FALSE") 
        ]
        for col_name, col_type in new_columns:
            cursor.execute(f'''
                DO $$ BEGIN
                    ALTER TABLE users ADD COLUMN IF NOT EXISTS {col_name} {col_type};
                EXCEPTION WHEN others THEN NULL;
                END $$;
            ''')
        
        # Expenses Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS expenses (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                category VARCHAR(100) NOT NULL,
                amount DECIMAL(12, 2) NOT NULL,
                description TEXT,
                date DATE NOT NULL,
                payment_method VARCHAR(50),
                tags TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Budgets Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS budgets (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                category VARCHAR(100) NOT NULL,
                limit_amount DECIMAL(12, 2) NOT NULL,
                current_month DATE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, category, current_month)
            )
        ''')
        
        # Financial Goals Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS financial_goals (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                goal_name VARCHAR(255) NOT NULL,
                target_amount DECIMAL(12, 2) NOT NULL,
                current_amount DECIMAL(12, 2) DEFAULT 0,
                deadline DATE,
                category VARCHAR(100),
                priority VARCHAR(20),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Admin Logs Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS admin_logs (
                id SERIAL PRIMARY KEY,
                admin_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                action VARCHAR(255) NOT NULL,
                target_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                details TEXT,
                ip_address VARCHAR(45),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Sessions Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                session_token VARCHAR(255) UNIQUE NOT NULL,
                ip_address VARCHAR(45),
                user_agent TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            )
        ''')
        
        # Analytics Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analytics (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                event_type VARCHAR(100) NOT NULL,
                event_data JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_expenses_user_id ON expenses(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_expenses_date ON expenses(date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_budgets_user_id ON budgets(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_goals_user_id ON financial_goals(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_admin_logs_admin_id ON admin_logs(admin_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_nid_number ON users(nid_number)')
        
        conn.commit()
        cursor.close()
        release_db(conn)
        print("Database initialized successfully")
        return True
    except Exception as e:
        print(f"Database initialization error: {e}")
        return False

# ==================== AUTHENTICATION ====================

def hash_password(password):
    """Hash password with salt"""
    return generate_password_hash(password, method='pbkdf2:sha256')

def verify_password(stored_hash, password):
    """Verify password"""
    return check_password_hash(stored_hash, password)

def login_required(f):
    """Login required decorator"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login first', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    """Admin required decorator"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login first', 'error')
            return redirect(url_for('login'))
        
        conn = get_db()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute('SELECT is_admin FROM users WHERE id = %s', (session['user_id'],))
        user = cursor.fetchone()
        cursor.close()
        release_db(conn)
        
        if not user or not user['is_admin']:
            flash('Admin access required', 'error')
            return redirect(url_for('dashboard'))
        
        return f(*args, **kwargs)
    return decorated_function

def get_current_user():
    """Get current logged in user"""
    if 'user_id' not in session:
        return None
    
    try:
        conn = get_db()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute('''
            SELECT id, email, name, last_name, phone, date_of_birth, gender, division, district, nid_number,
                   is_admin, is_verified, avatar_url, created_at FROM users WHERE id = %s
        ''', (session['user_id'],))
        user = cursor.fetchone()
        cursor.close()
        release_db(conn)
        return user
    except Exception as e:
        print(f"Error getting current user: {e}")
        return None


# ==================== AI CATEGORIZATION ====================

def ai_categorize(description):
    """Use Claude AI API to intelligently categorize transaction description"""
    import requests as req
    try:
        api_key = os.getenv('ANTHROPIC_API_KEY', '')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not configured")
        content = (
            'You are a financial transaction categorizer. '
            'Categorize the following transaction description into EXACTLY ONE of these categories: '
            'Transport, Food & Dining, Income Source, Bills & Utilities, Shopping, '
            'Healthcare, Education, Entertainment, Savings, General. '
            'Reply with ONLY the category name, no explanation, no punctuation. '
            'Transaction: "' + description + '"'
        )
        response = req.post(
            'https://api.anthropic.com/v1/messages',
            headers={
                'Content-Type': 'application/json',
                'x-api-key': api_key,
                'anthropic-version': '2023-06-01'
            },
            json={
                'model': 'claude-haiku-4-5-20251001',
                'max_tokens': 20,
                'messages': [{'role': 'user', 'content': content}]
            },
            timeout=5
        )
        if response.status_code == 200:
            ai_cat = response.json()['content'][0]['text'].strip()
            for cat in ['Transport', 'Food & Dining', 'Income Source', 'Bills & Utilities',
                        'Shopping', 'Healthcare', 'Education', 'Entertainment', 'Savings', 'General']:
                if cat.lower() in ai_cat.lower():
                    return cat
        return "General"
    except Exception as e:
        print("AI categorization fallback: " + str(e))
        d = description.lower()
        if any(w in d for w in ['uber', 'pathao', 'ride', 'bus', 'train', 'rickshaw', 'cng']):
            return "Transport"
        if any(w in d for w in ['food', 'restaurant', 'burger', 'lunch', 'dinner', 'breakfast', 'coffee']):
            return "Food & Dining"
        if any(w in d for w in ['salary', 'freelance', 'bonus', 'income']):
            return "Income Source"
        if any(w in d for w in ['bill', 'electricity', 'internet', 'wifi', 'gas', 'water']):
            return "Bills & Utilities"
        if any(w in d for w in ['shop', 'buy', 'purchase', 'clothes', 'amazon', 'daraz']):
            return "Shopping"
        return "General"

# ==================== ROUTES - PUBLIC ====================

@app.route('/')
def home():
    """Home page"""
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        
        # NEW ADDITION: Get the role from the login form
        role = request.form.get('role', 'user')
        
        if not email or not password:
            flash('Email and password are required', 'error')
            return redirect(url_for('login'))
        
        try:
            conn = get_db()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
            user = cursor.fetchone()
            cursor.close()
            release_db(conn)
            
            if user and verify_password(user['password_hash'], password):
                
                # NEW ADDITION: Verify Role Check
                if role == 'admin' and not user['is_admin']:
                    flash('Access Denied: You do not have admin privileges.', 'error')
                    return redirect(url_for('login'))
                elif role == 'user' and user['is_admin']:
                    flash('Please select Administrator role to login.', 'error')
                    return redirect(url_for('login'))
                
                session['user_id'] = user['id']
                session['email'] = user['email']
                session['is_admin'] = user['is_admin']
                
                # Update last login
                conn = get_db()
                cursor = conn.cursor()
                cursor.execute(
                    'UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = %s',
                    (user['id'],)
                )
                conn.commit()
                cursor.close()
                release_db(conn)
                
                # Log analytics
                log_analytics(user['id'], 'user_login')
                
                flash(f"Welcome back, {user['name']}!", 'success')
                return redirect(url_for('admin' if user['is_admin'] else 'dashboard'))
            else:
                flash('Invalid email or password', 'error')
        except Exception as e:
            print(f"Login error: {e}")
            flash('An error occurred during login', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        last_name = request.form.get('last_name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        phone = request.form.get('phone', '').strip()
        date_of_birth = request.form.get('date_of_birth', '').strip() or None
        gender = request.form.get('gender', '').strip()
        division = request.form.get('division', '').strip()
        district = request.form.get('district', '').strip()
        nid_number = request.form.get('nid_number', '').strip()
        terms = request.form.get('terms')
        
        # NEW ADDITIONS: Get role and admin secret from the form
        role = request.form.get('role', 'user')
        admin_secret = request.form.get('admin_secret', '').strip()
        
        is_admin = False
        if role == 'admin':
            expected_secret = os.getenv('ADMIN_SECRET')
            if admin_secret != expected_secret:
                flash('Invalid Admin Secret Key! Registration failed.', 'error')
                return redirect(url_for('register'))
            is_admin = True
        
        # Validation
        if not all([name, email, password]):
            flash('All required fields must be filled', 'error')
            return redirect(url_for('register'))
        
        if len(password) < 8:
            flash('Password must be at least 8 characters', 'error')
            return redirect(url_for('register'))
        
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            flash('Invalid email format', 'error')
            return redirect(url_for('register'))
        
        if not terms:
            flash('You must agree to the terms', 'error')
            return redirect(url_for('register'))

        # NID validation: Bangladesh NID must be exactly 10 or 17 digits
        if nid_number:
            if not re.match(r'^\d{10}$|^\d{17}$', nid_number):
                flash('NID Number must be exactly 10 or 17 digits', 'error')
                return redirect(url_for('register'))
        
        try:
            conn = get_db()
            cursor = conn.cursor()
            
            # Check if email already exists
            cursor.execute('SELECT id FROM users WHERE email = %s', (email,))
            if cursor.fetchone():
                flash('Email already registered', 'error')
                cursor.close()
                release_db(conn)
                return redirect(url_for('register'))

            # Check if NID already exists and block with name message
            if nid_number:
                cursor.execute('SELECT name, last_name FROM users WHERE nid_number = %s', (nid_number,))
                existing_user = cursor.fetchone()
                if existing_user:
                    registered_name = f"{existing_user[0]} {existing_user[1] if existing_user[1] else ''}".strip()
                    flash(f'Sorry! This NID Number is already registered under the name: "{registered_name}". You cannot register again with the same NID.', 'error')
                    cursor.close()
                    release_db(conn)
                    return redirect(url_for('register'))
            
            # Create new user
            password_hash = hash_password(password)
            mfa_secret = secrets.token_urlsafe(32)
            
            # UPDATED QUERY: Included is_admin field for saving the role
            cursor.execute('''
                INSERT INTO users 
                (email, name, last_name, password_hash, phone, date_of_birth,
                 gender, division, district, nid_number, mfa_secret, is_verified, is_admin)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, TRUE, %s)
                RETURNING id
            ''', (email, name, last_name, password_hash,
                   phone or None, date_of_birth, gender or None,
                   division or None, district or None, nid_number or None, mfa_secret, is_admin))
            
            user_id = cursor.fetchone()[0]
            conn.commit()
            cursor.close()
            release_db(conn)

            log_analytics(user_id, 'user_registered')
            
            flash('Account created successfully! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            print(f"Registration error: {e}")
            flash('An error occurred during registration', 'error')
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    """User logout"""
    if 'user_id' in session:
        log_analytics(session['user_id'], 'user_logout')
    
    session.clear()
    flash('You have been logged out', 'success')
    return redirect(url_for('home'))


# ==================== PROFILE MANAGEMENT API ====================

def allowed_file(filename):
    """Check if the uploaded file type is permitted"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/profile/update', methods=['POST'])
@login_required
def update_profile():
    """API endpoint to Update and Modify user's personal profile securely"""
    user_id = session['user_id']
    conn = get_db()
    try:
        name = request.form.get('name', '').strip()
        last_name = request.form.get('last_name', '').strip()
        phone = request.form.get('phone', '').strip() or None
        date_of_birth = request.form.get('date_of_birth', '').strip() or None
        gender = request.form.get('gender', '').strip() or None
        division = request.form.get('division', '').strip() or None
        district = request.form.get('district', '').strip() or None
        nid_number = request.form.get('nid_number', '').strip() or None

        avatar_url = None

        # Handle Secure Profile Picture Upload Processing
        if 'profile_picture' in request.files:
            file = request.files['profile_picture']
            if file and file.filename != '':
                if allowed_file(file.filename):
                    # Strict Backend validation for file size (Double security guard)
                    file.seek(0, os.SEEK_END)
                    size = file.tell()
                    file.seek(0)
                    
                    if size <= 2 * 1024 * 1024:  # Max limit 2MB
                        filename = secure_filename(f"user_{user_id}_{file.filename}")
                        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        file.save(filepath)
                        avatar_url = f"/static/uploads/avatars/{filename}"
                    else:
                        flash('Upload Failed! Image size must be under 2MB limit.', 'error')
                        return redirect(url_for('dashboard'))
                else:
                    flash('Invalid file format. Please upload JPG, PNG or WEBP.', 'error')
                    return redirect(url_for('dashboard'))

        cursor = conn.cursor()
        
        if avatar_url:
            cursor.execute('''
                UPDATE users SET name=%s, last_name=%s, phone=%s, date_of_birth=%s, gender=%s, division=%s, district=%s, nid_number=%s, avatar_url=%s, updated_at=CURRENT_TIMESTAMP
                WHERE id=%s
            ''', (name, last_name, phone, date_of_birth, gender, division, district, nid_number, avatar_url, user_id))
        else:
            cursor.execute('''
                UPDATE users SET name=%s, last_name=%s, phone=%s, date_of_birth=%s, gender=%s, division=%s, district=%s, nid_number=%s, updated_at=CURRENT_TIMESTAMP
                WHERE id=%s
            ''', (name, last_name, phone, date_of_birth, gender, division, district, nid_number, user_id))
        
        conn.commit()
        cursor.close()
        release_db(conn)
        
        flash('Profile settings successfully updated!', 'success')
        
    except Exception as e:
        print(f"Profile Sync Error: {e}")
        if conn: release_db(conn)
        flash('A server error occurred while updating profile.', 'error')
        
    return redirect(url_for('dashboard'))


# ==================== ROUTES - ADMIN ====================

@app.route('/admin')
@admin_required
def admin():
    """Admin dashboard"""
    user = get_current_user()
    
    try:
        conn = get_db()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get statistics (ORIGINAL)
        cursor.execute('SELECT COUNT(*) as total FROM users WHERE is_admin = FALSE')
        user_count = cursor.fetchone()['total']
        
        cursor.execute('SELECT COUNT(*) as total FROM expenses')
        expense_count = cursor.fetchone()['total']
        
        cursor.execute('SELECT SUM(amount) as total FROM expenses')
        total_expenses = cursor.fetchone()['total'] or 0
        
        # Get all users (ORIGINAL)
        cursor.execute('''
            SELECT id, email, name, last_name, is_verified, created_at 
            FROM users WHERE is_admin = FALSE
            ORDER BY created_at DESC
        ''')
        all_users = cursor.fetchall()
        
        # ====== NEW ADDITIONS FOR ADMIN DASHBOARD & TRANSACTIONS ==
        # 1. Total records in transactions table
        cursor.execute("SELECT COUNT(*) as total FROM transactions")
        sys_tx_count = cursor.fetchone()['total']
        
        # 2. Total platform expenses from transactions table
        cursor.execute("SELECT COALESCE(SUM(amount), 0) as total FROM transactions WHERE type='expense'")
        sys_total_expense = cursor.fetchone()['total']
        
        # 3. Total platform income from transactions table
        cursor.execute("SELECT COALESCE(SUM(amount), 0) as total FROM transactions WHERE type='income'")
        sys_total_income = cursor.fetchone()['total']
        
        # 4. Detailed user list including NID and Phone for Admin Table rendering
        cursor.execute('''
            SELECT id, email, name, last_name, is_verified, created_at, phone, nid_number, is_admin 
            FROM users 
            ORDER BY created_at DESC
        ''')
        detailed_users = cursor.fetchall()

        cursor.close()
        release_db(conn)
        
        # ADDED new variables to render_template without modifying old ones
        return render_template('admin.html',
                             user=user,
                             user_count=user_count,
                             expense_count=expense_count,
                             total_expenses=total_expenses,
                             all_users=all_users,
                             sys_tx_count=sys_tx_count,
                             sys_total_expense=sys_total_expense,
                             sys_total_income=sys_total_income,
                             detailed_users=detailed_users)
    except Exception as e:
        print(f"Admin page error: {e}")
        flash('Error loading admin dashboard', 'error')
        return redirect(url_for('home'))

@app.route('/api/admin/users', methods=['GET'])
@admin_required
def admin_get_users():
    """Get all users for admin"""
    try:
        conn = get_db()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute('''
            SELECT id, email, name, last_name, is_verified, 
                   is_admin, created_at, last_login
            FROM users
            ORDER BY created_at DESC
        ''')
        
        users = cursor.fetchall()
        cursor.close()
        release_db(conn)
        
        return jsonify([dict(u) for u in users]), 200
    except Exception as e:
        print(f"Error fetching users: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/users/<int:user_id>/verify', methods=['POST'])
@admin_required
def admin_verify_user(user_id):
    """Verify user account"""
    user = get_current_user()
    
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        cursor.execute('UPDATE users SET is_verified = TRUE WHERE id = %s',
                      (user_id,))
        
        conn.commit()
        cursor.close()
        release_db(conn)
        
        log_admin_action(user['id'], 'user_verified', user_id)
        
        return jsonify({'success': True}), 200
    except Exception as e:
        print(f"Error verifying user: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/users/<int:user_id>/delete', methods=['POST'])
@admin_required
def admin_delete_user(user_id):
    """Delete user account"""
    user = get_current_user()
    
    if user['id'] == user_id:
        return jsonify({'error': 'Cannot delete your own account'}), 400
    
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM users WHERE id = %s', (user_id,))
        
        conn.commit()
        cursor.close()
        release_db(conn)
        
        log_admin_action(user['id'], 'user_deleted', user_id)
        
        return jsonify({'success': True}), 200
    except Exception as e:
        print(f"Error deleting user: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/analytics', methods=['GET'])
@admin_required
def admin_analytics():
    """Get analytics data"""
    try:
        conn = get_db()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # User growth (ORIGINAL)
        cursor.execute('''
            SELECT DATE_TRUNC('day', created_at) as date, COUNT(*) as count
            FROM users
            WHERE created_at >= NOW() - INTERVAL '30 days'
            GROUP BY DATE_TRUNC('day', created_at)
            ORDER BY date ASC
        ''')
        user_growth = cursor.fetchall()
        
        # Expenses by category (ORIGINAL - from old expenses table)
        cursor.execute('''
            SELECT category, SUM(amount) as total_amount, COUNT(*) as count
            FROM expenses
            WHERE date >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY category
            ORDER BY total_amount DESC
        ''')
        expense_categories = cursor.fetchall()
        
        # ====== NEW ADDITIONS FOR ADMIN ANALYTICS FROM TRANSACTIONS TABLE ======
        cursor.execute('''
            SELECT category, SUM(amount) as total_amount, COUNT(*) as count
            FROM transactions
            WHERE type='expense' AND created_at >= NOW() - INTERVAL '30 days'
            GROUP BY category
            ORDER BY total_amount DESC
        ''')
        tx_categories = cursor.fetchall()
        
        # Determine which data to send based on whether transactions table has data
        final_categories = tx_categories if tx_categories else expense_categories
        # =======================================================================
        
        cursor.close()
        release_db(conn)
        
        return jsonify({
            'user_growth': [dict(u) for u in user_growth],
            'expense_categories': [dict(e) for e in final_categories]
        }), 200
    except Exception as e:
        print(f"Error fetching analytics: {e}")
        return jsonify({'error': str(e)}), 500

# ==================== UTILITY FUNCTIONS ====================

def log_analytics(user_id, event_type, event_data=None):
    """Log user analytics"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO analytics (user_id, event_type, event_data)
            VALUES (%s, %s, %s)
        ''', (user_id, event_type, event_data))
        
        conn.commit()
        cursor.close()
        release_db(conn)
    except Exception as e:
        print(f"Error logging analytics: {e}")

def log_admin_action(admin_id, action, target_user_id=None, details=None):
    """Log admin actions"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        ip_address = request.remote_addr if request else None
        
        cursor.execute('''
            INSERT INTO admin_logs 
            (admin_id, action, target_user_id, details, ip_address)
            VALUES (%s, %s, %s, %s, %s)
        ''', (admin_id, action, target_user_id, details, ip_address))
        
        conn.commit()
        cursor.close()
        release_db(conn)
    except Exception as e:
        print(f"Error logging admin action: {e}")

# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    """404 error handler"""
    return render_template('error.html', code=404, 
                         message='Page not found'), 404

@app.errorhandler(500)
def internal_error(error):
    """500 error handler"""
    return render_template('error.html', code=500, 
                         message='Internal server error'), 500

# ==================== CLI COMMANDS ====================

@app.shell_context_processor
def make_shell_context():
    return {'db': get_db, 'init_db': init_db}

# ==================== GOOGLE OAUTH INTEGRATION ====================

@app.route('/login/google')
def social_login_google():
    """Redirect to Google OAuth for authentication"""
    if request.host.startswith('127.0.0.1'):
        return redirect(request.url.replace('127.0.0.1', 'localhost'))

    from urllib.parse import urlencode
    client_id = os.getenv('GOOGLE_CLIENT_ID', '').strip()
    if not client_id or client_id.startswith('YOUR_'):
        flash('Google OAuth is not configured. Please add credentials to .env', 'error')
        return redirect(url_for('login'))

    state = secrets.token_hex(16)
    session['oauth_state'] = state
    session.modified = True 
    redirect_uri = url_for('google_callback', _external=True)

    if '127.0.0.1' in redirect_uri:
        redirect_uri = redirect_uri.replace('127.0.0.1', 'localhost')
    if not request.host.startswith('localhost') and not request.host.startswith('127.0.0.1'):
        redirect_uri = redirect_uri.replace('http://', 'https://')

    params = urlencode({
        'response_type': 'code',
        'client_id': client_id,
        'redirect_uri': redirect_uri,
        'scope': 'openid email profile',
        'state': state,
        'access_type': 'online',
        'prompt': 'select_account', # ADDED FOR ACCOUNT SELECTION/REMOVAL
    })
    return redirect(f"https://accounts.google.com/o/oauth2/v2/auth?{params}")


@app.route('/login/google/callback')
def google_callback():
    """Handle Google OAuth callback"""
    if request.host.startswith('127.0.0.1'):
        return redirect(request.url.replace('127.0.0.1', 'localhost'))

    import requests as req

    code = request.args.get('code')
    state = request.args.get('state')
    error = request.args.get('error')

    if error:
        flash(f'Google login cancelled: {error}', 'error')
        return redirect(url_for('login'))

    if not code or state != session.get('oauth_state'):
        flash('Google OAuth security validation failed. Please try again.', 'error')
        return redirect(url_for('login'))

    redirect_uri = url_for('google_callback', _external=True)
    if '127.0.0.1' in redirect_uri:
        redirect_uri = redirect_uri.replace('127.0.0.1', 'localhost')
    if not request.host.startswith('localhost') and not request.host.startswith('127.0.0.1'):
        redirect_uri = redirect_uri.replace('http://', 'https://')

    try:
        token_resp = req.post('https://oauth2.googleapis.com/token', data={
            'code': code,
            'client_id': os.getenv('GOOGLE_CLIENT_ID', '').strip(),
            'client_secret': os.getenv('GOOGLE_CLIENT_SECRET', '').strip(),
            'redirect_uri': redirect_uri,
            'grant_type': 'authorization_code',
        }).json()

        if 'error' in token_resp:
            flash(f"Google authentication failed: {token_resp.get('error_description', token_resp.get('error'))}", 'error')
            return redirect(url_for('login'))

        user_info = req.get('https://www.googleapis.com/oauth2/v3/userinfo', headers={
            'Authorization': f"Bearer {token_resp.get('access_token')}"
        }).json()

        email = user_info.get('email')
        name = user_info.get('given_name', 'Google')
        last_name = user_info.get('family_name', 'User')
        avatar_url = user_info.get('picture')

        if not email:
            flash('Could not retrieve email from Google. Please ensure your account has a verified email.', 'error')
            return redirect(url_for('login'))

        conn = get_db()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute('SELECT * FROM users WHERE email = %s', (email.strip().lower(),))
        user = cursor.fetchone()

        if not user:
            password_hash = generate_password_hash(secrets.token_urlsafe(16), method='pbkdf2:sha256')
            mfa_secret = secrets.token_urlsafe(32)
            cursor.execute('''
                INSERT INTO users (email, name, last_name, password_hash, mfa_secret, is_verified, mfa_enabled, avatar_url)
                VALUES (%s, %s, %s, %s, %s, TRUE, TRUE, %s)
                RETURNING *
            ''', (email.strip().lower(), name, last_name, password_hash, mfa_secret, avatar_url))
            user = cursor.fetchone()
            conn.commit()
        else:
            if avatar_url:
                cursor.execute('UPDATE users SET avatar_url = %s WHERE id = %s', (avatar_url, user['id']))
                conn.commit()

        cursor.execute('UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = %s', (user['id'],))
        conn.commit()
        cursor.close()
        release_db(conn)

        session['user_id'] = user['id']
        session['email'] = user['email']
        session['is_admin'] = user['is_admin']

        log_analytics(user['id'], 'user_social_login_google')
        flash("Successfully signed in via Google!", 'success')
        return redirect(url_for('admin' if user['is_admin'] else 'dashboard'))

    except Exception as e:
        print(f"Google OAuth callback error: {e}")
        flash('Google authentication failed. Please try again.', 'error')
        return redirect(url_for('login'))


# ==================== NEW FINORAXPENSE DASHBOARD ROUTES ====================

@app.route('/dashboard')
@login_required
def dashboard():
    """FR-01 to FR-09: User Dashboard for Financial Tracking"""
    user = get_current_user()
    conn = get_db()
    if not conn:
        flash('Database connection failed.', 'error')
        return redirect(url_for('login'))
        
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        user_id = session['user_id']
        
        # Ensure transactions table exists for both Income & Expense (FR-04)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                type VARCHAR(10) NOT NULL,
                amount DECIMAL(12, 2) NOT NULL,
                category VARCHAR(50) DEFAULT 'Uncategorized',
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

        # Calculate Totals
        cursor.execute('''
            SELECT 
                COALESCE(SUM(CASE WHEN type='income' THEN amount ELSE 0 END), 0) as total_income,
                COALESCE(SUM(CASE WHEN type='expense' THEN amount ELSE 0 END), 0) as total_expense
            FROM transactions WHERE user_id = %s
        ''', (user_id,))
        
        summary = cursor.fetchone()
        total_income = float(summary['total_income'])
        total_expense = float(summary['total_expense'])
        total_balance = total_income - total_expense
        
        # Get Recent Transactions (FR-05) - For Dashboard Home
        cursor.execute('''
            SELECT * FROM transactions 
            WHERE user_id = %s 
            ORDER BY created_at DESC LIMIT 6
        ''', (user_id,))
        recent_transactions = cursor.fetchall()

        # Get All Transactions - For Transactions Tab
        cursor.execute('''
            SELECT * FROM transactions 
            WHERE user_id = %s 
            ORDER BY created_at DESC
        ''', (user_id,))
        all_transactions = cursor.fetchall()

        # Fetch Category Stats for Analytics Tab
        cursor.execute('''
            SELECT category, SUM(amount) as total 
            FROM transactions 
            WHERE user_id = %s AND type='expense' 
            GROUP BY category
        ''', (user_id,))
        category_stats = cursor.fetchall()
        pie_labels = json.dumps([row['category'] for row in category_stats]) if category_stats else json.dumps(['No Data'])
        pie_data = json.dumps([float(row['total']) for row in category_stats]) if category_stats else json.dumps([0])
        
        # FR-09: Real-time Chart Analytics Data (All transaction dates, no limit)
        cursor.execute('''
            SELECT DATE(created_at) as date,
                   COALESCE(SUM(CASE WHEN type='income' THEN amount ELSE 0 END), 0) as income,
                   COALESCE(SUM(CASE WHEN type='expense' THEN amount ELSE 0 END), 0) as expense
            FROM transactions
            WHERE user_id = %s
            GROUP BY DATE(created_at)
            ORDER BY DATE(created_at) ASC
        ''', (user_id,))
        
        stats = cursor.fetchall()
        
        chart_labels = json.dumps([row['date'].strftime('%b %d') for row in stats]) if stats else json.dumps(['No Data'])
        chart_income = json.dumps([float(row['income']) for row in stats]) if stats else json.dumps([0])
        chart_expense = json.dumps([float(row['expense']) for row in stats]) if stats else json.dumps([0])

        # FR-07: Real-time Budget Data
        cursor.execute('''
            SELECT id, category, limit_amount FROM budgets 
            WHERE user_id = %s AND EXTRACT(MONTH FROM current_month) = EXTRACT(MONTH FROM CURRENT_DATE)
        ''', (user_id,))
        budget_row = cursor.fetchone()
        
        budget_limit = float(budget_row['limit_amount']) if budget_row else 50000.0
        
        budget_percentage = 0
        if budget_limit > 0:
            budget_percentage = min(int((total_expense / budget_limit) * 100), 100)

        # FR-08: Savings Goals
        cursor.execute('''
            SELECT * FROM financial_goals WHERE user_id = %s ORDER BY created_at DESC
        ''', (user_id,))
        savings_goals = cursor.fetchall()
        
        # AI Insights Generation (Supervised Learning Simulation based on User Data)
        ai_insights = []
        if total_expense > total_income and total_income > 0:
            ai_insights.append({'type': 'danger', 'icon': 'fa-triangle-exclamation', 'title': 'Critical Warning', 'message': "Your expenses have exceeded your income. We highly recommend cutting down on unnecessary spending immediately."})
        elif budget_percentage >= 90:
            ai_insights.append({'type': 'warning', 'icon': 'fa-bell', 'title': 'Budget Alert', 'message': f"You have used {budget_percentage}% of your monthly budget limit. Please be cautious with your upcoming expenses."})
        elif budget_percentage >= 70:
            ai_insights.append({'type': 'info', 'icon': 'fa-circle-info', 'title': 'Budget Notice', 'message': "A significant portion of your budget has already been spent. Please plan your expenses carefully for the rest of the month."})
        else:
            ai_insights.append({'type': 'success', 'icon': 'fa-circle-check', 'title': 'Great Job', 'message': "Your expenses are perfectly within your income and budget limits. Keep up the great work!"})

        if category_stats:
            top_category = max(category_stats, key=lambda x: float(x['total']))
            ai_insights.append({'type': 'primary', 'icon': 'fa-magnifying-glass-chart', 'title': 'Trend Analysis', 'message': f"Your highest spending is in the '{top_category['category']}' category (৳{top_category['total']}). Try finding cost-effective alternatives here to boost your savings."})
            
        cursor.close()
        release_db(conn)
        
        return render_template('user.html', 
                               user=user,
                               total_income=total_income,
                               total_expense=total_expense,
                               total_balance=total_balance,
                               transactions=recent_transactions,
                               all_transactions=all_transactions,
                               chart_labels=chart_labels,
                               chart_income=chart_income,
                               chart_expense=chart_expense,
                               budget_limit=budget_limit,
                               budget_percentage=budget_percentage,
                               savings_goals=savings_goals,
                               pie_labels=pie_labels,
                               pie_data=pie_data,
                               ai_insights=ai_insights)
                               
    except Exception as e:
        print(f"Dashboard Error: {e}")
        release_db(conn)
        flash("An error occurred while loading the dashboard.", 'error')
        return redirect(url_for('home'))


@app.route('/api/transaction', methods=['POST'])
@login_required
def add_transaction():
    """FR-04: User can add income and expense records"""
    conn = get_db()
    try:
        data = request.form
        tx_type = data.get('type')
        amount = float(data.get('amount', 0))
        description = data.get('description', '').strip()
        
        if amount <= 0:
            flash('Amount must be greater than 0', 'error')
            return redirect(url_for('dashboard'))

        # FR-06: Claude AI Smart Categorization (Anthropic API-powered)
        category = ai_categorize(description)

        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO transactions (user_id, type, amount, description, category) 
            VALUES (%s, %s, %s, %s, %s)
        ''', (session['user_id'], tx_type, amount, description, category))
        
        conn.commit()
        cursor.close()
        release_db(conn)
        flash('Transaction added successfully!', 'success')
        
    except Exception as e:
        print(f"Transaction Error: {e}")
        if conn:
            release_db(conn)
        flash('Failed to add transaction.', 'error')
        
    return redirect(url_for('dashboard'))

@app.route('/api/transaction/<int:tx_id>/update', methods=['POST'])
@login_required
def update_transaction(tx_id):
    """User can edit and update existing transactions"""
    conn = get_db()
    try:
        data = request.form
        tx_type = data.get('type')
        amount = float(data.get('amount', 0))
        description = data.get('description', '').strip()
        
        if amount <= 0:
            flash('Amount must be greater than 0', 'error')
            return redirect(url_for('dashboard'))

        # Claude AI Smart Categorization Re-run (Anthropic API-powered)
        category = ai_categorize(description)

        cursor = conn.cursor()
        cursor.execute('''
            UPDATE transactions 
            SET type = %s, amount = %s, description = %s, category = %s, created_at = CURRENT_TIMESTAMP 
            WHERE id = %s AND user_id = %s
        ''', (tx_type, amount, description, category, tx_id, session['user_id']))
        
        conn.commit()
        cursor.close()
        release_db(conn)
        flash('Transaction updated successfully!', 'success')
        
    except Exception as e:
        print(f"Transaction Update Error: {e}")
        if conn: release_db(conn)
        flash('Failed to update transaction.', 'error')
        
    return redirect(url_for('dashboard'))

@app.route('/api/transaction/<int:tx_id>/delete', methods=['POST'])
@login_required
def delete_transaction(tx_id):
    """FR-05: User can delete transaction data"""
    conn = get_db()
    try:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM transactions WHERE id = %s AND user_id = %s', (tx_id, session['user_id']))
        conn.commit()
        cursor.close()
        release_db(conn)
        flash('Transaction deleted successfully!', 'success')
    except Exception as e:
        print(f"Transaction Delete Error: {e}")
        release_db(conn)
        flash('Failed to delete transaction.', 'error')
        
    return redirect(url_for('dashboard'))


@app.route('/api/chart-data')
@login_required
def get_chart_data():
    """Real-time chart data API for Income vs Expense graph"""
    conn = get_db()
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute('''
            SELECT DATE(created_at) AS date,
                   COALESCE(SUM(CASE WHEN type='income' THEN amount ELSE 0 END), 0) AS income,
                   COALESCE(SUM(CASE WHEN type='expense' THEN amount ELSE 0 END), 0) AS expense
            FROM transactions
            WHERE user_id = %s
            GROUP BY DATE(created_at)
            ORDER BY DATE(created_at) ASC
        ''', (session['user_id'],))
        rows = cursor.fetchall()
        cursor.close()
        release_db(conn)

        if rows:
            labels  = [row['date'].strftime('%b %d') for row in rows]
            income  = [float(row['income'])  for row in rows]
            expense = [float(row['expense']) for row in rows]
        else:
            labels  = ['No Data']
            income  = [0]
            expense = [0]

        return jsonify({'labels': labels, 'income': income, 'expense': expense})

    except Exception as e:
        print(f"Chart data API error: {e}")
        if conn:
            release_db(conn)
        return jsonify({'labels': ['Error'], 'income': [0], 'expense': [0]})


@app.route('/api/budget/update', methods=['POST'])
@login_required
def update_budget():
    """FR-07: User can set/update their monthly budget limit"""
    conn = get_db()
    try:
        limit_amount = float(request.form.get('limit_amount', 0))
        if limit_amount <= 0:
            flash('Budget must be greater than 0', 'error')
            return redirect(url_for('dashboard'))

        cursor = conn.cursor()
        # Upsert logic for current month budget
        cursor.execute('''
            INSERT INTO budgets (user_id, category, limit_amount, current_month)
            VALUES (%s, 'Overall', %s, CURRENT_DATE)
            ON CONFLICT (user_id, category, current_month)
            DO UPDATE SET limit_amount = EXCLUDED.limit_amount, updated_at = CURRENT_TIMESTAMP
        ''', (session['user_id'], limit_amount))
        
        conn.commit()
        cursor.close()
        release_db(conn)
        flash('Budget Limit updated successfully!', 'success')
    except Exception as e:
        print(f"Budget Error: {e}")
        release_db(conn)
        flash('Failed to update budget.', 'error')
        
    return redirect(url_for('dashboard'))


@app.route('/api/goal/add', methods=['POST'])
@login_required
def add_goal():
    """FR-08: User can add financial savings goals"""
    conn = get_db()
    try:
        goal_name = request.form.get('goal_name', '').strip()
        target_amount = float(request.form.get('target_amount', 0))
        
        if target_amount <= 0 or not goal_name:
            flash('Invalid goal inputs', 'error')
            return redirect(url_for('dashboard'))

        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO financial_goals (user_id, goal_name, target_amount, current_amount, category)
            VALUES (%s, %s, %s, 0, 'Savings')
        ''', (session['user_id'], goal_name, target_amount))
        
        conn.commit()
        cursor.close()
        release_db(conn)
        flash('Savings goal added successfully!', 'success')
    except Exception as e:
        print(f"Goal Error: {e}")
        release_db(conn)
        flash('Failed to add goal.', 'error')
        
    return redirect(url_for('dashboard'))

@app.route('/api/goal/<int:goal_id>/add_funds', methods=['POST'])
@login_required
def add_goal_funds(goal_id):
    """User can add funds to a specific savings goal from their main balance"""
    conn = get_db()
    try:
        amount = float(request.form.get('amount', 0))
        if amount <= 0:
            flash('Amount must be greater than 0', 'error')
            return redirect(url_for('dashboard'))

        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Check if goal exists and belongs to user
        cursor.execute('SELECT * FROM financial_goals WHERE id = %s AND user_id = %s', (goal_id, session['user_id']))
        goal = cursor.fetchone()
        if not goal:
            flash('Goal not found', 'error')
            return redirect(url_for('dashboard'))

        # Update the goal's current amount
        cursor.execute('''
            UPDATE financial_goals 
            SET current_amount = current_amount + %s, updated_at = CURRENT_TIMESTAMP 
            WHERE id = %s
        ''', (amount, goal_id))
        
        # Add a record to transactions to deduct from main balance (Expense)
        cursor.execute('''
            INSERT INTO transactions (user_id, type, amount, description, category) 
            VALUES (%s, 'expense', %s, %s, 'Savings Goal')
        ''', (session['user_id'], amount, f"Fund Transfer to Goal: {goal['goal_name']}"))
        
        conn.commit()
        cursor.close()
        release_db(conn)
        flash(f'Successfully transferred ৳{amount} to {goal["goal_name"]}!', 'success')
    except Exception as e:
        print(f"Goal Funding Error: {e}")
        if conn: release_db(conn)
        flash('Failed to transfer funds to goal.', 'error')
        
    return redirect(url_for('dashboard'))


# ==================== MAIN ====================

if __name__ == '__main__':
    # Initialize database
    init_db()
    
    # Run Flask app
    app.run(
        host=os.getenv('FLASK_HOST', '0.0.0.0'),
        port=int(os.getenv('FLASK_PORT', 5000)),
        debug=os.getenv('FLASK_ENV', 'production') == 'development'
    )