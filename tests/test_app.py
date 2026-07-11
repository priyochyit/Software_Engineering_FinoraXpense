import pytest
import os
import json
import io
from unittest.mock import patch, MagicMock
from flask import session
import numpy as np
from datetime import datetime

# Added internal_error to imports
from app import app, allowed_file, init_db, ai_categorize, train_ai_model_realtime, internal_error


## ==================== TESTING CORE FIXTURES ====================

@pytest.fixture
def client():
    """Configures the Flask runtime instance explicitly for testing environments."""
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads', 'avatars')
    with app.test_client() as client:
        with app.app_context():
            yield client

@pytest.fixture(autouse=True)
def mock_db():
    """Globally intercepts database operations with strict parameter-aware routing."""
    with patch('app.get_db') as mock_get, patch('app.release_db') as mock_rel:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        
        context = {'query': '', 'params': ()}
        
        def mock_execute(query, params=None):
            context['query'] = query.lower()
            context['params'] = params or ()
            
        def mock_fetchone():
            q = context['query']
            p = context['params']
            p_strs = [str(x).lower() for x in p]
            now = datetime.now()
            
            # 1. Exact Structural Match for Dashboard Calculations
            if 'coalesce(sum(case when type=\'income\'' in q:
                return {'total_income': 15000.00, 'total_expense': 5000.00}
                
            # 2. Admin Dashboard & Totals
            if 'count(*)' in q or 'sum(amount)' in q:
                return {'total': 100}
                
            # 3. Registration Collisions
            if 'select id from users where email =' in q:
                if any('dup@t.com' in s for s in p_strs):
                    return (1,)
                return None
                
            if 'name, last_name from users where nid_number =' in q:
                if any('1234567890' in s for s in p_strs):
                    return ("Original", "User")
                return None
                
            # 4. Login Paths
            if "select * from users where email =" in q:
                if any(x in p_strs for x in ['missing@t.com', 'unknown@test.com']):
                    return None
                is_admin = any('admin@t.com' in s for s in p_strs)
                return {
                    'id': 2 if is_admin else 50,
                    'email': 'admin@t.com' if is_admin else 'u@t.com',
                    'name': 'Admin' if is_admin else 'John',
                    'password_hash': 'pbkdf2:sha256:250000$good_hash',
                    'is_admin': is_admin
                }
                
            # 5. Session Authorization & Profile
            if "select is_admin from users where id =" in q:
                return {'is_admin': session.get('is_admin', False)}
                
            if "select id, email, name, last_name, phone" in q:
                is_admin = session.get('is_admin', False)
                return {
                    'id': session.get('user_id', 50),
                    'email': 'admin@t.com' if is_admin else 'u@t.com',
                    'name': 'Admin' if is_admin else 'John',
                    'last_name': 'Doe', 'phone': '12345', 'date_of_birth': None,
                    'gender': 'Male', 'division': 'Dhaka', 'district': 'Dhaka',
                    'nid_number': '1234567890', 'is_admin': is_admin,
                    'is_verified': True, 'avatar_url': None, 'created_at': now
                }
                
            # 6. Financial Tools & Budgets
            if 'from budgets' in q:
                return {'id': 1, 'category': 'Overall', 'limit_amount': 5000.00}
                
            if 'from financial_goals' in q:
                if session.get('fail_goal_lookup'):
                    return None
                return {'id': 2, 'goal_name': 'EID Tour', 'target_amount': 5000.00, 'current_amount': 1000.00}
                
            # 7. Insertions Returning Data
            if 'insert into users' in q: return (101,)
            if 'insert into transactions' in q: return (1,)
            
            return None
            
        def mock_fetchall():
            q = context['query']
            now = datetime.now()
            
            # Chart API - CRITICAL FIX: Returning actual datetime objects
            if 'group by date(created_at)' in q:
                return [{'date': now, 'income': 500, 'expense': 200}]
            
            # Analytics Categories
            if 'group by category' in q:
                return [{'category': 'Food & Dining', 'total': 1500.00}, {'category': 'Entertainment', 'total': 1200.00}]
            if 'from expenses' in q:
                return [{'category': 'Food & Dining', 'total_amount': 1500.00, 'count': 10}]
            
            # Transaction Lists
            if 'from transactions' in q:
                return [{'id': 1, 'type': 'expense', 'amount': 100, 'category': 'Food', 'description': 'Lunch', 'created_at': now}]
            
            # Admin Goals & Analytics
            if 'from financial_goals' in q:
                return [{'id': 1, 'goal_name': 'Laptop', 'target_amount': 1000, 'current_amount': 500}]
            if 'date_trunc' in q:
                return [{'date': now, 'count': 5}]
            
            # User Management Lists
            if 'select id, email, name, last_name, is_verified' in q:
                return [{'id': 1, 'name': 'Test', 'email': 't@t.com', 'is_verified': True, 'created_at': now, 'is_admin': False, 'phone':'1', 'nid_number':'1', 'last_login': now}]
                
            return []
            
        mock_cursor.execute.side_effect = mock_execute
        mock_cursor.fetchone.side_effect = mock_fetchone
        mock_cursor.fetchall.side_effect = mock_fetchall
        mock_conn.cursor.return_value = mock_cursor
        mock_get.return_value = mock_conn
        yield mock_cursor


## ==================== DB UTILITIES & APPLICATION ERROR PATHS ====================

def test_database_initialization_success(client):
    status = init_db()
    assert status is True

def test_database_initialization_failure(client):
    with patch('app.get_db', return_value=None):
        status = init_db()
        assert status is False

def test_error_handler_404(client):
    response = client.get('/this-route-does-not-exist')
    assert response.status_code == 404
    assert b'Page not found' in response.data

def test_error_handler_500(client):
    """FIXED: Uses test_request_context to properly evaluate template renders without crashing test runner."""
    with app.test_request_context('/'):
        response_html, status_code = internal_error(Exception("Test Failure"))
        assert status_code == 500
        assert "Internal server error" in response_html


## ==================== USER REGISTRATION CONDITIONAL BRANCHES ====================

def test_registration_get_page(client):
    response = client.get('/register')
    assert response.status_code == 200

def test_registration_validation_missing_fields(client):
    response = client.post('/register', data={'name': '', 'email': '', 'password': ''}, follow_redirects=True)
    assert b'All required fields must be filled' in response.data

def test_registration_validation_password_length(client):
    response = client.post('/register', data={'name': 'User', 'email': 'u@t.com', 'password': '123', 'terms': 'on'}, follow_redirects=True)
    assert b'Password must be at least 8 characters' in response.data

def test_registration_validation_email_regex(client):
    response = client.post('/register', data={'name': 'User', 'email': 'bad_email', 'password': 'password123', 'terms': 'on'}, follow_redirects=True)
    assert b'Invalid email format' in response.data

def test_registration_validation_terms_checkbox(client):
    response = client.post('/register', data={'name': 'User', 'email': 'u@t.com', 'password': 'password123'}, follow_redirects=True)
    assert b'You must agree to the terms' in response.data

def test_registration_validation_nid_format(client):
    response = client.post('/register', data={'name': 'User', 'email': 'u@t.com', 'password': 'password123', 'nid_number': '12345', 'terms': 'on'}, follow_redirects=True)
    assert b'NID Number must be exactly 10 or 17 digits' in response.data

def test_registration_admin_role_invalid_secret(client):
    with patch('os.getenv', return_value="EXPECTED_SECRET"):
        response = client.post('/register', data={'name': 'Admin', 'email': 'a@t.com', 'password': 'password123', 'role': 'admin', 'admin_secret': 'WRONG'}, follow_redirects=True)
        assert b'Invalid Admin Secret Key! Registration failed.' in response.data

def test_registration_duplicate_email(client):
    response = client.post('/register', data={'name': 'User', 'email': 'dup@t.com', 'password': 'password123', 'terms': 'on'}, follow_redirects=True)
    assert b'Email already registered' in response.data

def test_registration_duplicate_nid(client):
    response = client.post('/register', data={'name': 'User', 'email': 'u@t.com', 'password': 'password123', 'nid_number': '1234567890', 'terms': 'on'}, follow_redirects=True)
    assert b'already registered' in response.data

def test_registration_execution_success(client):
    response = client.post('/register', data={'name': 'Valid', 'email': 'v@t.com', 'password': 'password123', 'terms': 'on'}, follow_redirects=True)
    assert b'Account created successfully! Please login.' in response.data


## ==================== USER LOGIN CONDITIONAL BRANCHES ====================

def test_login_get_page(client):
    response = client.get('/login')
    assert response.status_code == 200

def test_login_missing_parameters(client):
    response = client.post('/login', data={'email': '', 'password': ''}, follow_redirects=True)
    assert b'Email and password are required' in response.data

def test_login_user_not_found(client):
    response = client.post('/login', data={'email': 'missing@t.com', 'password': 'password'}, follow_redirects=True)
    assert b'Invalid email or password' in response.data

def test_login_incorrect_password(client):
    with patch('app.verify_password', return_value=False):
        response = client.post('/login', data={'email': 'u@t.com', 'password': 'password'}, follow_redirects=True)
        assert b'Invalid email or password' in response.data

def test_login_role_mismatch_admin_access_denied(client):
    with patch('app.verify_password', return_value=True):
        response = client.post('/login', data={'email': 'u@t.com', 'password': 'password', 'role': 'admin'}, follow_redirects=True)
        assert b'Access Denied' in response.data

def test_login_role_mismatch_user_access_denied(client):
    with patch('app.verify_password', return_value=True):
        response = client.post('/login', data={'email': 'admin@t.com', 'password': 'password', 'role': 'user'}, follow_redirects=True)
        assert b'Administrator role' in response.data

def test_login_success_standard_user(client):
    with patch('app.verify_password', return_value=True):
        response = client.post('/login', data={'email': 'u@t.com', 'password': 'password', 'role': 'user'}, follow_redirects=True)
        assert b'Welcome back' in response.data

def test_logout_session_purge(client):
    """FIXED: Direct evaluation of session state confirms accurate teardown without template unreliability."""
    with client.session_transaction() as sess:
        sess['user_id'] = 50
    response = client.get('/logout', follow_redirects=False)
    assert response.status_code == 302
    assert 'user_id' not in session


## ==================== SECURITY WRAPPERS & ACCESS ENFORCEMENT ====================

def test_login_required_decorator_interception(client):
    response = client.get('/dashboard')
    assert response.status_code == 302

def test_admin_required_decorator_non_admin(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 10
        sess['is_admin'] = False
    response = client.get('/admin', follow_redirects=True)
    assert b'Admin access required' in response.data


## ==================== PROFILE ASSETS MANAGEMENT PATHWAYS ====================

def test_profile_update_text_only(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 1
    response = client.post('/api/profile/update', data={'name': 'NewName', 'last_name': 'NewLastName'}, follow_redirects=True)
    assert b'Profile settings successfully updated!' in response.data

def test_profile_update_invalid_extension(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 1
    payload = {'name': 'Name', 'profile_picture': (io.BytesIO(b"dummy"), 'malicious.exe')}
    response = client.post('/api/profile/update', data=payload, follow_redirects=True)
    assert b'Invalid file format' in response.data

def test_profile_update_file_size_exceeded(client):
    """FIXED: The route's except block successfully catches the 413 error and issues a 302 redirect."""
    with client.session_transaction() as sess:
        sess['user_id'] = 1
    payload = {'name': 'Name', 'profile_picture': (io.BytesIO(b"C" * (3 * 1024 * 1024)), 'large.png')}
    response = client.post('/api/profile/update', data=payload, follow_redirects=False)
    # Flawlessly handles its own error by redirecting back
    assert response.status_code == 302

def test_profile_update_file_success(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 1
    payload = {'name': 'Name', 'profile_picture': (io.BytesIO(b"Valid picture"), 'avatar.png')}
    with patch('werkzeug.datastructures.FileStorage.save'):
        response = client.post('/api/profile/update', data=payload, follow_redirects=True)
        assert b'Profile settings successfully updated!' in response.data


## ==================== MACHINE LEARNING DASHBOARD PREDICTIONS ====================

def test_dashboard_ml_status_0_critical_stress(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 1
    with patch('app.finance_model') as mock_model:
        mock_model.predict.return_value = [0]
        response = client.get('/dashboard')
        assert b'AI Warning: Critical Stress' in response.data

def test_dashboard_ml_status_1_moderate(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 1
    with patch('app.finance_model') as mock_model:
        mock_model.predict.return_value = [1]
        response = client.get('/dashboard')
        assert b'AI Prediction: Moderate Status' in response.data

def test_dashboard_ml_status_2_excellent(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 1
    with patch('app.finance_model') as mock_model:
        mock_model.predict.return_value = [2]
        response = client.get('/dashboard')
        assert b'AI Health Check: Excellent' in response.data

def test_dashboard_ml_status_3_food_spending(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 1
    with patch('app.finance_model') as mock_model:
        mock_model.predict.return_value = [3]
        response = client.get('/dashboard')
        assert b'AI Action Plan: Reduce Food Spend' in response.data

def test_dashboard_ml_status_4_entertainment(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 1
    with patch('app.finance_model') as mock_model:
        mock_model.predict.return_value = [4]
        response = client.get('/dashboard')
        assert b'AI Action Plan: Cut Entertainment' in response.data

def test_dashboard_ml_status_5_shopping(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 1
    with patch('app.finance_model') as mock_model:
        mock_model.predict.return_value = [5]
        response = client.get('/dashboard')
        assert b'AI Action Plan: Shopping Alert' in response.data

def test_dashboard_ml_missing_model(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 1
    with patch('app.finance_model', None):
        response = client.get('/dashboard')
        assert b'file is missing' in response.data


## ==================== FINANCIAL TRANSACTION MANAGEMENT API ====================

def test_transaction_positivity_constraint(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 1
    response = client.post('/api/transaction', data={'type': 'expense', 'amount': '-10', 'description': 'Bad'}, follow_redirects=True)
    assert b'Amount must be greater than 0' in response.data

def test_transaction_addition_success(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 1
    with patch('app.ai_categorize', return_value="Food"):
        response = client.post('/api/transaction', data={'type': 'expense', 'amount': '250.50', 'description': 'Burgers'}, follow_redirects=True)
        assert b'Transaction added successfully!' in response.data

def test_transaction_update_invalid_amount(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 1
    response = client.post('/api/transaction/5/update', data={'type': 'income', 'amount': '0', 'description': 'Salary'}, follow_redirects=True)
    assert b'Amount must be greater than 0' in response.data

def test_transaction_update_success(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 1
    response = client.post('/api/transaction/5/update', data={'type': 'income', 'amount': '5000', 'description': 'Freelance'}, follow_redirects=True)
    assert b'Transaction updated successfully!' in response.data

def test_transaction_deletion_success(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 1
    response = client.post('/api/transaction/12/delete', follow_redirects=True)
    assert b'Transaction deleted successfully!' in response.data


## ==================== FINANCIAL TRACKING MODALITY ENDPOINTS ====================

def test_budget_update_invalid_amount(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 1
    response = client.post('/api/budget/update', data={'limit_amount': '-50'}, follow_redirects=True)
    assert b'Budget must be greater than 0' in response.data

def test_budget_update_success(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 1
    response = client.post('/api/budget/update', data={'limit_amount': '15000'}, follow_redirects=True)
    assert b'Budget Limit updated successfully!' in response.data

def test_goal_addition_invalid_inputs(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 1
    response = client.post('/api/goal/add', data={'goal_name': '', 'target_amount': '0'}, follow_redirects=True)
    assert b'Invalid goal inputs' in response.data

def test_goal_addition_success(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 1
    response = client.post('/api/goal/add', data={'goal_name': 'New Laptop', 'target_amount': '85000'}, follow_redirects=True)
    assert b'Savings goal added successfully!' in response.data

def test_goal_funding_not_found(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 1
        sess['fail_goal_lookup'] = True
    response = client.post('/api/goal/99/add_funds', data={'amount': '500'}, follow_redirects=True)
    assert b'Goal not found' in response.data

def test_goal_funding_success(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 1
        sess['fail_goal_lookup'] = False
    response = client.post('/api/goal/2/add_funds', data={'amount': '2500'}, follow_redirects=True)
    assert b'Successfully transferred' in response.data


## ==================== ADMINISTRATIVE COMMAND MANAGEMENT API ====================

def test_admin_view_rendering_success(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 2
        sess['is_admin'] = True
    response = client.get('/admin')
    assert response.status_code == 200

def test_admin_api_list_users(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 2
        sess['is_admin'] = True
    response = client.get('/api/admin/users')
    assert response.status_code == 200

def test_admin_api_verify_user_success(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 2
        sess['is_admin'] = True
    response = client.post('/api/admin/users/5/verify')
    assert response.status_code == 200

def test_admin_api_delete_user_self_rejection(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 2
        sess['is_admin'] = True
    response = client.post('/api/admin/users/2/delete')
    assert response.status_code == 400
    assert b'Cannot delete your own account' in response.data

def test_admin_api_delete_user_success(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 2
        sess['is_admin'] = True
    response = client.post('/api/admin/users/15/delete')
    assert response.status_code == 200


## ==================== GOOGLE THIRD-PARTY AUTHENTICATION ====================

def test_google_social_login_redirection(client):
    with patch('os.getenv', return_value="CLIENT_ID_STRING"):
        response = client.get('/login/google')
        assert response.status_code == 302
        assert 'accounts.google.com' in response.headers['Location']

def test_google_callback_cancellation_path(client):
    response = client.get('/login/google/callback?error=access_denied', follow_redirects=True)
    assert b'Google login cancelled' in response.data

def test_google_callback_security_state_mismatch(client):
    with client.session_transaction() as sess:
        sess['oauth_state'] = 'ORIGINAL_TOKEN'
    response = client.get('/login/google/callback?code=abc&state=MALICIOUS', follow_redirects=True)
    assert b'Google OAuth security validation failed.' in response.data


## ==================== MACHINE LEARNING HELPERS DIRECT UNIT TESTS ====================

def test_ai_categorize_empty_or_unloaded():
    with patch('app.nlp_model', None):
        category = ai_categorize("")
        assert category == "General"

def test_ai_continuous_learning_error_capture():
    with patch('app.nlp_model') as mock_model:
        mock_model.partial_fit.side_effect = Exception("Partial Fit Exception")
        train_ai_model_realtime("Rice bought", "Food & Dining")

def test_api_chart_data_success(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 1
    response = client.get('/api/chart-data')
    assert response.status_code == 200