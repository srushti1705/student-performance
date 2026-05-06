"""
Generate Sample Student Data for AcademiX Dashboard
"""

import sys
sys.path.insert(0, '/app')

from app import app, db, Student
import random
from datetime import datetime, timedelta

def generate_sample_data():
    """Generate 50 sample students with realistic data"""
    
    with app.app_context():
        # Clear existing students
        Student.query.delete()
        db.session.commit()
        
        names = [
            "Alice Johnson", "Bob Smith", "Charlie Brown", "Diana Prince", "Eve Williams",
            "Frank Miller", "Grace Lee", "Henry Davis", "Ivy Chen", "Jack Wilson",
            "Karen White", "Liam Martinez", "Mia Taylor", "Noah Anderson", "Olivia Thomas",
            "Peter Jackson", "Quinn Adams", "Rachel Green", "Samuel Johnson", "Tina Turner",
            "Uma Patel", "Victor Hugo", "Wendy Peters", "Xavier Lopez", "Yara Khan",
            "Zachary Scott", "Abigail Rose", "Benjamin Hart", "Chloe King", "Daniel Ford",
            "Elena Garcia", "Ethan Moore", "Fiona Bell", "George Walker", "Hannah Young",
            "Isaac Newton", "Julia Roberts", "Kevin Hart", "Lily Evans", "Mason Stone",
            "Natalie Wood", "Oliver Stone", "Piper Chapman", "Quinn Jones", "Ryan Cooper",
            "Sophia Turner", "Thomas Hill", "Ursula Bloom", "Victoria Secret", "William Hill"
        ]
        
        stress_levels = ["Low", "Moderate", "High"]
        
        students = []
        for i, name in enumerate(names):
            # Randomly distribute risk categories
            risk_rand = random.random()
            if risk_rand < 0.4:
                risk_category = "Low Risk"
                gpa = random.uniform(3.5, 4.0)
                study_hours = random.uniform(6, 8)
                risk_score = random.uniform(0, 30)
            elif risk_rand < 0.75:
                risk_category = "Moderate Risk"
                gpa = random.uniform(2.8, 3.4)
                study_hours = random.uniform(4, 6)
                risk_score = random.uniform(31, 70)
            else:
                risk_category = "High Risk"
                gpa = random.uniform(1.5, 2.7)
                study_hours = random.uniform(2, 4)
                risk_score = random.uniform(71, 100)
            
            student = Student(
                student_id=f"STU{10000 + i}",
                name=name,
                email=f"student{i}@university.edu",
                gpa=round(gpa, 2),
                attendance_rate=random.uniform(60, 100),
                study_hours=round(study_hours, 1),
                sleep_hours=round(random.uniform(5, 9), 1),
                social_hours=round(random.uniform(1, 5), 1),
                extracurricular_hours=round(random.uniform(0, 3), 1),
                physical_activity_hours=round(random.uniform(0, 2.5), 1),
                stress_level=random.choice(stress_levels),
                risk_category=risk_category,
                risk_score=round(risk_score, 2),
                last_assessment=datetime.now() - timedelta(days=random.randint(0, 30)),
                flagged=risk_score > 75,
                intervention_active=70 < risk_score < 75
            )
            students.append(student)
        
        # Add all students to database
        for student in students:
            db.session.add(student)
        
        db.session.commit()
        print(f"✅ Generated {len(students)} sample students")
        print(f"   - Low Risk: {sum(1 for s in students if s.risk_category == 'Low Risk')}")
        print(f"   - Moderate Risk: {sum(1 for s in students if s.risk_category == 'Moderate Risk')}")
        print(f"   - High Risk: {sum(1 for s in students if s.risk_category == 'High Risk')}")

if __name__ == '__main__':
    generate_sample_data()
