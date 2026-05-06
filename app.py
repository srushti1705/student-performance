"""
AI-Based Student Academic Monitoring System
Institutional Backend API
"""

from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'institutional-monitoring-secure-key-2024'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///monitoring_system.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

# ========================
# DATABASE MODELS
# ========================

class User(UserMixin, db.Model):
    """User model for authentication"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    user_type = db.Column(db.String(20), default='teacher')  # teacher, admin
    
    def set_password(self, password):
        self.password = generate_password_hash(password, method='pbkdf2:sha256')
    
    def check_password(self, password):
        return check_password_hash(self.password, password)

class Student(db.Model):
    """Student academic profile"""
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(20), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    gpa = db.Column(db.Float, default=3.0)
    attendance_rate = db.Column(db.Float, default=85.0)
    
    # Lifestyle metrics
    study_hours = db.Column(db.Float, default=5.0)
    sleep_hours = db.Column(db.Float, default=7.0)
    social_hours = db.Column(db.Float, default=2.0)
    extracurricular_hours = db.Column(db.Float, default=1.0)
    physical_activity_hours = db.Column(db.Float, default=1.0)
    stress_level = db.Column(db.String(20), default='Moderate')  # Low, Moderate, High
    
    # AI Predictions
    risk_category = db.Column(db.String(20), default='Low Risk')  # Low, Moderate, High Risk
    risk_score = db.Column(db.Float, default=0.0)
    last_assessment = db.Column(db.DateTime, default=datetime.now)
    
    # Flags
    flagged = db.Column(db.Boolean, default=False)
    intervention_active = db.Column(db.Boolean, default=False)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# Initialize database
with app.app_context():
    db.create_all()

# ========================
# MODEL LOADING
# ========================

model = None
scaler = None
feature_metadata = None
risk_thresholds = None

def load_ml_resources():
    """Load ML model, scaler, and metadata"""
    global model, scaler, feature_metadata, risk_thresholds
    
    try:
        if os.path.exists('model.h5'):
            model = tf.keras.models.load_model('model.h5')
            print("[INFO] Model loaded successfully")
        
        if os.path.exists('scaler.pkl'):
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
                print("[INFO] Scaler loaded successfully")
        
        if os.path.exists('model_metadata.json'):
            with open('model_metadata.json', 'r') as f:
                feature_metadata = json.load(f)
                print("[INFO] Metadata loaded successfully")
        
        if os.path.exists('risk_thresholds.json'):
            with open('risk_thresholds.json', 'r') as f:
                risk_thresholds = json.load(f)
                print("[INFO] Risk thresholds loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load ML resources: {e}")

load_ml_resources()

# ========================
# AI PREDICTION ENGINE
# ========================

def calculate_risk_score(study_hours, sleep_hours, social_hours, extracurricular_hours, 
                         physical_activity_hours, stress_level):
    """Calculate academic risk score"""
    
    if model is None or scaler is None or feature_metadata is None:
        return None, "Model not loaded"
    
    try:
        # Map stress level to score
        stress_map = {'Low': 20, 'Moderate': 50, 'High': 90}
        stress_score = stress_map.get(stress_level, 50)
        
        # Calculate derived features
        study_consistency = min(100, (study_hours / 8) * 100)
        sleep_quality = min(100, (sleep_hours / 8) * 100)
        lifestyle_balance = (
            study_consistency * 0.3 +
            sleep_quality * 0.3 +
            (100 - min(100, social_hours * 12.5)) * 0.2 +
            min(100, physical_activity_hours * 16.7) * 0.2
        )
        screen_time_risk = min(100, social_hours * 10)
        productivity_index = (
            (study_hours / 12) * 0.4 +
            (sleep_hours / 8) * 0.3 +
            ((8 - social_hours) / 8) * 0.3
        ) * 100
        
        academic_consistency = 50.0  # Placeholder - would be calculated from GPA history
        
        # Prepare input for model
        features = np.array([[
            study_hours,
            sleep_hours,
            social_hours,
            extracurricular_hours,
            physical_activity_hours,
            stress_score,
            study_consistency,
            sleep_quality,
            lifestyle_balance,
            screen_time_risk,
            productivity_index
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Get predictions
        predictions = model.predict(features_scaled, verbose=0)
        probabilities = predictions[0]
        
        # Calculate risk score (0-100)
        risk_score = (
            probabilities[0] * 0 +      # Low risk = 0 points
            probabilities[1] * 50 +     # Moderate risk = 50 points
            probabilities[2] * 100      # High risk = 100 points
        )
        
        # Determine risk category
        if risk_score <= 30:
            risk_category = 'Low Risk'
        elif risk_score <= 70:
            risk_category = 'Moderate Risk'
        else:
            risk_category = 'High Risk'
        
        return {
            'risk_score': float(risk_score),
            'risk_category': risk_category,
            'probabilities': {
                'low_risk': float(probabilities[0]),
                'moderate_risk': float(probabilities[1]),
                'high_risk': float(probabilities[2])
            }
        }, None
        
    except Exception as e:
        return None, str(e)

def generate_insights_and_interventions(study_hours, sleep_hours, social_hours, 
                                       physical_activity_hours, stress_level, risk_score):
    """Generate AI insights and personalized interventions"""
    
    insights = []
    interventions = []
    weak_areas = []
    
    # Analyze each metric
    if study_hours < 4:
        weak_areas.append('Insufficient study hours')
        interventions.append({
            'priority': 'High',
            'action': 'Increase study hours to 6-8 per day',
            'impact': 'Increase of 2-3 hours daily can improve GPA significantly'
        })
        insights.append(f'Study hours are low ({study_hours:.1f} hrs/day). Recommendation: increase to 6-8 hours.')
    
    if sleep_hours < 6 or sleep_hours > 9:
        weak_areas.append('Irregular sleep pattern')
        target_sleep = 7 if sleep_hours < 6 else 8
        interventions.append({
            'priority': 'High',
            'action': f'Maintain sleep schedule of {target_sleep} hours daily',
            'impact': 'Proper sleep improves cognitive function and academic performance'
        })
        insights.append(f'Sleep hours are irregular ({sleep_hours:.1f} hrs/day). Optimal: 7-8 hours.')
    
    if social_hours > 4:
        weak_areas.append('High social/screen time')
        interventions.append({
            'priority': 'Medium',
            'action': f'Reduce social media/screen time from {social_hours:.1f} to <2 hours',
            'impact': 'Reducing screen time can improve focus and study efficiency'
        })
        insights.append(f'Social/screen time is high ({social_hours:.1f} hrs/day). Reduce to <2 hours.')
    
    if physical_activity_hours < 1:
        weak_areas.append('Low physical activity')
        interventions.append({
            'priority': 'Medium',
            'action': 'Engage in 1+ hour of physical activity daily',
            'impact': 'Physical activity improves mental health and academic focus'
        })
        insights.append('Physical activity is low. Exercise 1+ hour daily.')
    
    if stress_level == 'High':
        weak_areas.append('High stress levels')
        interventions.append({
            'priority': 'High',
            'action': 'Schedule mentor counseling and stress management sessions',
            'impact': 'Professional counseling can reduce stress by 40-50%'
        })
        insights.append('Stress level is HIGH. Recommend immediate counseling support.')
    
    return {
        'insights': insights,
        'weak_areas': weak_areas,
        'interventions': interventions
    }

# ========================
# ROUTES
# ========================

@app.route('/')
@login_required
def dashboard():
    """Main institutional dashboard"""
    
    try:
        # Get statistics
        total_students = Student.query.count()
        high_risk_students = Student.query.filter(Student.risk_category == 'High Risk').count()
        moderate_risk_students = Student.query.filter(Student.risk_category == 'Moderate Risk').count()
        low_risk_students = Student.query.filter(Student.risk_category == 'Low Risk').count()
        
        avg_gpa = db.session.query(db.func.avg(Student.gpa)).scalar() or 0
        avg_attendance = db.session.query(db.func.avg(Student.attendance_rate)).scalar() or 0
        avg_risk_score = db.session.query(db.func.avg(Student.risk_score)).scalar() or 0
        
        flagged_count = Student.query.filter(Student.flagged == True).count()
        
        # Get recent assessments
        recent_students = Student.query.order_by(Student.last_assessment.desc()).limit(10).all()
        
        dashboard_data = {
            'total_students': total_students,
            'high_risk_count': high_risk_students,
            'moderate_risk_count': moderate_risk_students,
            'low_risk_count': low_risk_students,
            'flagged_count': flagged_count,
            'avg_gpa': round(avg_gpa, 2),
            'avg_attendance': round(avg_attendance, 2),
            'avg_risk_score': round(avg_risk_score, 2),
            'risk_distribution': {
                'Low Risk': low_risk_students,
                'Moderate Risk': moderate_risk_students,
                'High Risk': high_risk_students
            }
        }
        
        return render_template('dashboard.html', data=dashboard_data, students=recent_students)
    
    except Exception as e:
        print(f"[ERROR] Dashboard error: {e}")
        return render_template('dashboard.html', data={}, students=[])

@app.route('/students')
@login_required
def students_list():
    """List all students with filtering"""
    
    risk_filter = request.args.get('risk', 'all')
    search = request.args.get('search', '')
    
    query = Student.query
    
    if risk_filter != 'all':
        query = query.filter(Student.risk_category == risk_filter)
    
    if search:
        query = query.filter(
            (Student.name.ilike(f'%{search}%')) |
            (Student.student_id.ilike(f'%{search}%'))
        )
    
    students = query.all()
    
    return render_template('students_list.html', students=students, risk_filter=risk_filter, search=search)

@app.route('/student/<int:student_id>')
@login_required
def student_profile(student_id):
    """Individual student profile and analysis"""
    
    student = Student.query.get_or_404(student_id)
    
    # Generate insights
    analysis = generate_insights_and_interventions(
        student.study_hours,
        student.sleep_hours,
        student.social_hours,
        student.physical_activity_hours,
        student.stress_level,
        student.risk_score
    )
    
    return render_template('student_profile.html', student=student, analysis=analysis)

@app.route('/api/assess', methods=['POST'])
@login_required
def assess_student():
    """API endpoint to assess student and calculate risk"""
    
    try:
        data = request.get_json()
        
        # Extract metrics
        study_hours = float(data.get('study_hours', 5))
        sleep_hours = float(data.get('sleep_hours', 7))
        social_hours = float(data.get('social_hours', 2))
        extracurricular_hours = float(data.get('extracurricular_hours', 1))
        physical_activity_hours = float(data.get('physical_activity_hours', 1))
        stress_level = data.get('stress_level', 'Moderate')
        gpa = float(data.get('gpa', 3.0))
        attendance_rate = float(data.get('attendance_rate', 85.0))
        
        # Calculate risk score
        result, error = calculate_risk_score(
            study_hours, sleep_hours, social_hours,
            extracurricular_hours, physical_activity_hours, stress_level
        )
        
        if error:
            return jsonify({'error': error}), 400
        
        # Generate insights
        analysis = generate_insights_and_interventions(
            study_hours, sleep_hours, social_hours,
            physical_activity_hours, stress_level,
            result['risk_score']
        )
        
        return jsonify({
            'risk_score': result['risk_score'],
            'risk_category': result['risk_category'],
            'probabilities': result['probabilities'],
            'analysis': analysis
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/what-if', methods=['POST'])
@login_required
def what_if_analysis():
    """What-if analysis for risk score"""
    
    try:
        data = request.get_json()
        
        results = {}
        scenarios = data.get('scenarios', {})
        
        for scenario_name, metrics in scenarios.items():
            result, _ = calculate_risk_score(
                metrics.get('study_hours', 5),
                metrics.get('sleep_hours', 7),
                metrics.get('social_hours', 2),
                metrics.get('extracurricular_hours', 1),
                metrics.get('physical_activity_hours', 1),
                metrics.get('stress_level', 'Moderate')
            )
            results[scenario_name] = result
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard-stats', methods=['GET'])
@login_required
def dashboard_stats():
    """Get dashboard statistics as JSON"""
    
    try:
        total_students = Student.query.count()
        high_risk = Student.query.filter(Student.risk_category == 'High Risk').count()
        moderate_risk = Student.query.filter(Student.risk_category == 'Moderate Risk').count()
        low_risk = Student.query.filter(Student.risk_category == 'Low Risk').count()
        
        avg_gpa = db.session.query(db.func.avg(Student.gpa)).scalar() or 0
        avg_attendance = db.session.query(db.func.avg(Student.attendance_rate)).scalar() or 0
        avg_risk_score = db.session.query(db.func.avg(Student.risk_score)).scalar() or 0
        
        return jsonify({
            'total_students': total_students,
            'risk_distribution': {
                'low': low_risk,
                'moderate': moderate_risk,
                'high': high_risk
            },
            'metrics': {
                'avg_gpa': round(avg_gpa, 2),
                'avg_attendance': round(avg_attendance, 2),
                'avg_risk_score': round(avg_risk_score, 2)
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('dashboard'))
        
        flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """User registration"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user_type = request.form.get('user_type', 'teacher')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return redirect(url_for('signup'))
        
        new_user = User(username=username, user_type=user_type)
        new_user.set_password(password)
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account created successfully! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    return redirect(url_for('login'))

# ========================
# ERROR HANDLERS
# ========================

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    db.session.rollback()
    return render_template('500.html'), 500

# ========================
# MAIN
# ========================

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return redirect(url_for('index'))
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if model is None or scaler is None:
        load_resources()
        if model is None or scaler is None:
            return jsonify({'error': 'Model or scaler not found. Please train the model first.'}), 500

    try:
        data = request.json
        
        # Validate input ranges
        study_hours = float(data.get('study_hours', 0))
        extracurricular_hours = float(data.get('extracurricular_hours', 0))
        sleep_hours = float(data.get('sleep_hours', 0))
        social_hours = float(data.get('social_hours', 0))
        physical_activity = float(data.get('physical_activity', 0))
        stress_level_str = data.get('stress_level', 'Moderate')
        
        # Individual ranges are allowed up to 24, we rely on the total hours check.
        
        # Check total hours don't exceed 24
        total_hours = study_hours + extracurricular_hours + sleep_hours + social_hours + physical_activity
        if total_hours > 24:
            return jsonify({'error': f'Total daily hours ({total_hours:.1f}) exceeds 24 hours. Please adjust your inputs.'}), 400
        
        if stress_level_str not in ['Low', 'Moderate', 'High']:
            return jsonify({'error': 'Stress level must be Low, Moderate, or High'}), 400
        
        # Map stress level text to numeric values
        stress_map = {'Low': 0, 'Moderate': 1, 'High': 2}
        stress_level = stress_map.get(stress_level_str, 1)
        
        # Features matching the training data order
        features = [
            study_hours,           # Study_Hours_Per_Day
            extracurricular_hours, # Extracurricular_Hours_Per_Day
            sleep_hours,           # Sleep_Hours_Per_Day
            social_hours,          # Social_Hours_Per_Day
            physical_activity,     # Physical_Activity_Hours_Per_Day
            float(stress_level)    # Stress_Level (encoded)
        ]
        
        # Reshape for scaling and prediction
        features_array = np.array([features])
        features_scaled = scaler.transform(features_array)
        
        # Predict
        prediction_probs = model.predict(features_scaled)
        category_index = int(np.argmax(prediction_probs[0]))
        confidence = float(np.max(prediction_probs[0]))
        
        # Rule-based correction (Hybrid model)
        if study_hours < 2 and extracurricular_hours > 4:
            category_index = 2
            confidence = max(0.85, confidence)
        
        categories = ['High Performer 🟢', 'Average Performer 🟡', 'At-Risk Student 🔴']
        category = categories[category_index]
        
        # Calculate a mock score for the UI (0-100)
        # Higher index (At-Risk) should generally result in lower score
        if category_index == 0:
            score = 85 + (confidence * 15)
        elif category_index == 1:
            score = 65 + (confidence * 20)
        else:
            score = 40 + (confidence * 25)
        
        score = min(100, max(0, score))

        # Intelligence Features: Explanation & Recommendations
        explanation = []
        recommendations = []
        
        # Logic for explanation based on lifestyle factors
        if study_hours < 5:
            explanation.append("Your study hours are lower than optimal for strong performance.")
            recommendations.append("Aim for 5-7 hours of daily study time.")
        
        if sleep_hours < 7:
            explanation.append("Insufficient sleep can negatively impact academic performance.")
            recommendations.append("Target 7-8 hours of sleep daily for better cognitive function.")
            
        if stress_level_str == 'High':
            explanation.append("High stress levels can interfere with learning and retention.")
            recommendations.append("Practice stress management techniques like meditation or exercise.")
        
        if physical_activity < 1:
            explanation.append("Limited physical activity may reduce overall well-being.")
            recommendations.append("Include at least 1 hour of physical activity daily.")

        if not explanation:
            explanation.append("Your lifestyle habits are well-balanced for academic success.")
            recommendations.append("Maintain your current healthy routine.")

        # What-if: Aiming for HIGH PERFORMANCE (Category 0)
        target_scenario = "Target: High Performer Achievement"
        required_improvements = []
        
        if category_index != 0:
            if study_hours < 6:
                required_improvements.append("Increase Study Hours to 6+")
            if sleep_hours < 8:
                required_improvements.append("Increase Sleep to 8 hours")
            if stress_level_str == 'High':
                required_improvements.append("Reduce Stress to Moderate level")
            if physical_activity < 2:
                required_improvements.append("Increase Physical Activity to 2+ hours")
        else:
            required_improvements.append("You are already on track for High Performance! Maintain these standards.")

        # Chart Data: Current lifestyle metrics (normalized to 0-100)
        # Map stress level to inverse score (Low=85, Moderate=60, High=35)
        stress_score_map = {'Low': 85, 'Moderate': 60, 'High': 35}
        stress_score = stress_score_map.get(stress_level_str, 60)
        
        current_metrics = [
            min(100, (study_hours / 12) * 100),
            min(100, (sleep_hours / 12) * 100),
            min(100, (social_hours / 8) * 100),
            min(100, (extracurricular_hours / 8) * 100),
            min(100, (physical_activity / 6) * 100),
            stress_score  # Inverse stress (lower stress = higher score)
        ]
        
        ideal_metrics = ideal_metrics_global  # Use data-driven ideal lifestyle for High Performer

        # ===== PERSONALIZED RECOMMENDATIONS (ML-BASED) =====
        personalized_recs = []
        
        # Analyze gaps and create ranked recommendations
        gaps = {
            'study': max(0, 6 - study_hours),
            'sleep': max(0, 8 - sleep_hours),
            'exercise': max(0, 2 - physical_activity),
            'stress': 2 if stress_level_str == 'High' else (1 if stress_level_str == 'Moderate' else 0),
            'extracurricular': max(0, 3 - extracurricular_hours)
        }
        
        # Create recommendations ranked by impact
        recs_list = []
        
        if gaps['stress'] > 0:
            recs_list.append({
                'priority': 1,
                'impact': gaps['stress'],
                'action': 'Reduce Stress Levels',
                'reason': 'High stress significantly impacts cognitive function and academic performance.',
                'impact_text': 'Potential score improvement: +8-12 points'
            })
        
        if gaps['sleep'] > 0:
            recs_list.append({
                'priority': 2,
                'impact': gaps['sleep'],
                'action': f'Increase Sleep to {8} Hours',
                'reason': f'You\'re currently sleeping {sleep_hours}h/day. More sleep improves focus and retention.',
                'impact_text': f'Potential score improvement: +{5 + int(gaps["sleep"])} points'
            })
        
        if gaps['study'] > 0:
            recs_list.append({
                'priority': 3,
                'impact': gaps['study'],
                'action': f'Boost Study Time to {6} Hours/Day',
                'reason': f'Current study time is {study_hours}h. Increasing study hours strengthens performance.',
                'impact_text': f'Potential score improvement: +{3 + int(gaps["study"])} points'
            })
        
        if gaps['exercise'] > 0:
            recs_list.append({
                'priority': 4,
                'impact': gaps['exercise'],
                'action': f'Add {2} Hours of Physical Activity',
                'reason': 'Exercise improves mental health, focus, and overall well-being.',
                'impact_text': 'Potential score improvement: +4-6 points'
            })
        
        if gaps['extracurricular'] > 0 and category_index == 0:
            recs_list.append({
                'priority': 5,
                'impact': gaps['extracurricular'],
                'action': 'Expand Extracurricular Activities',
                'reason': 'You have room to explore more interests while maintaining performance.',
                'impact_text': 'Helps develop well-rounded profile'
            })
        
        # Sort by impact (descending) and limit to top 5
        recs_list.sort(key=lambda x: x['impact'], reverse=True)
        personalized_recs = recs_list[:5]
        
        # If no gaps, provide maintenance recommendations
        if not personalized_recs:
            personalized_recs = [{
                'action': 'Maintain Your Current Lifestyle',
                'reason': 'Your lifestyle is well-optimized for academic success.',
                'impact': 'Keep this routine for consistent performance'
            }]
        
        # Generate personalized intro text
        if category_index == 0:
            intro_text = f"🌟 Excellent! You're performing as a High Performer. To maintain this level, focus on these priorities:"
        elif category_index == 1:
            intro_text = f"📈 You're doing average. Here's how to reach High Performance status:"
        else:
            intro_text = f"⚠️ Your performance indicates at-risk status. These changes can make a real difference:"
        
        # Convert to format expected by frontend
        personalized_recommendations = []
        for i, rec in enumerate(personalized_recs):
            personalized_recommendations.append({
                'action': rec.get('action', 'Action item'),
                'reason': rec.get('reason', 'Impact on performance'),
                'impact': rec.get('impact_text', 'Positive impact')
            })

        return jsonify({
            'score': round(score, 1),
            'category': category,
            'confidence': round(confidence, 4),
            'explanation': explanation,
            'suggestions': recommendations,
            'personalized_intro': intro_text,
            'personalized_recommendations': personalized_recommendations,
            'what_if': {
                'scenario': target_scenario,
                'requirements': required_improvements
            },
            'chart_data': {
                'current': current_metrics,
                'ideal': ideal_metrics,
                'labels': ['Study Hours', 'Sleep Hours', 'Social Hours', 'Extracurricular', 'Physical Activity', 'Low Stress']
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
