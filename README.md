<<<<<<< HEAD
# AI Student Performance Predictor

An AI-powered web application that uses Deep Learning to predict student academic performance and identify potential risk levels based on study habits, attendance, and lifestyle factors.

---

## 🛠️ Technology Stack

This project leverages a modern stack combining Machine Learning, Web Development, and Data Science:

### 🧠 Backend & Machine Learning
- **Python**: The core programming language.
- **Flask**: A lightweight WSGI web application framework used to serve the API and frontend.
- **TensorFlow / Keras**: Used to build and train the Artificial Neural Network (ANN) model.
- **Scikit-learn**: Used for data preprocessing (StandardScaler) and dataset splitting.
- **Pandas**: Essential library for data manipulation and analysis of `data.csv`.
- **NumPy**: Used for efficient numerical operations and array handling.
- **Pickle / H5Py**: Used for serializing and saving the trained model and data scalers.

### 🎨 Frontend & Visualization
- **HTML5**: Semantic structure for the web interface.
- **CSS3 (Vanilla)**: Custom styling using modern techniques like Glassmorphism, Flexbox, and Grid.
- **JavaScript (ES6+)**: Handles asynchronous API calls (Fetch API) and UI interactions.
- **Chart.js**: A powerful library used to render dynamic bar charts for performance benchmarking.
- **Google Fonts (Outfit)**: Premium typography for a professional look.

---

## 🚀 How to Run (Easy Method)

I have created a one-click launcher for Windows:

1. **Double-click `run.bat`** in the project folder.
2. It will automatically activate the virtual environment, install dependencies, and start the server.
3. Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

---

## 💻 Manual Setup (Developers)

If you prefer to run it manually:

### 1. Activate Environment
```powershell
.\.venv\Scripts\activate
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Run Application
```powershell
python app.py
```

---

## 📁 Project Structure

- **`app.py`**: Flask server & API endpoints.
- **`train_model.py`**: Model training script.
- **`data.csv`**: Raw dataset.
- **`requirements.txt`**: List of all Python dependencies.
- **`run.bat`**: One-click launcher for Windows.
- **`templates/`**: HTML view.
- **`static/`**: CSS and JS assets.
- **`model.h5`**: Trained Neural Network model.
- **`scaler.pkl`**: Saved feature scaler.

## 🧠 Features
- **Deep Learning Prediction**: Uses a Neural Network to classify students into 'High Performer', 'Average', or 'At-Risk'.
- **Interactive Dashboard**: Modern glassmorphism UI with real-time performance estimation.
- **What-If Analysis**: Provides specific, actionable steps to reach 'High Performer' status.
- **Visual Analytics**: Dynamic bar charts comparing current student metrics against success benchmarks.
- **AI Recommendations**: Personalized suggestions based on data like study hours, sleep, and screen time.
=======
# student-performance
>>>>>>> d5a724b59d403060f9341188ab371007fc886f8f
