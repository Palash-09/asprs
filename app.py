from flask import Flask, request, jsonify, render_template, send_file
import joblib
import numpy as np
import pandas as pd
import io
import os

app = Flask(__name__)

# ── Load all models & scalers ──────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), 'models')

kt_model      = joblib.load(os.path.join(BASE, 'kt_next_semester_model.pkl'))
kt_scaler     = joblib.load(os.path.join(BASE, 'kt_next_semester_scaler.pkl'))

dropout_model  = joblib.load(os.path.join(BASE, 'dropout_model.pkl'))
dropout_scaler = joblib.load(os.path.join(BASE, 'dropout_scaler.pkl'))

weak_model     = joblib.load(os.path.join(BASE, 'weak_subject_model.pkl'))
weak_scaler    = joblib.load(os.path.join(BASE, 'weak_subject_scaler.pkl'))

placement_model  = joblib.load(os.path.join(BASE, 'placement.pkl'))
placement_scaler = joblib.load(os.path.join(BASE, 'placement_scaler.pkl'))

# ── Pages ──────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/bulk')
def bulk_page():
    return render_template('bulk.html')

# ── API: KT Predictor (single) ─────────────────────────────────────────────
@app.route('/api/kt', methods=['POST'])
def predict_kt():
    try:
        data = request.json
        features = np.array([[
            float(data['attendance']),
            float(data['sem_marks_avg']),
            float(data['ut_marks_avg']),
            float(data['internal_marks_avg']),
            float(data['subject_count']),
            float(data['current_kts'])
        ]])
        scaled = kt_scaler.transform(features)
        prediction = int(kt_model.predict(scaled)[0])
        proba = kt_model.predict_proba(scaled)[0]
        return jsonify({
            'prediction': 'KT Likely' if prediction == 1 else 'No KT',
            'kt_probability': round(float(proba[1]) * 100, 1),
            'safe_probability': round(float(proba[0]) * 100, 1),
            'raw': prediction
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── API: Dropout Risk (single) ─────────────────────────────────────────────
@app.route('/api/dropout', methods=['POST'])
def predict_dropout():
    try:
        data = request.json
        features = np.array([[
            float(data['cgpa']),
            float(data['total_kts']),
            float(data['attendance']),
            float(data['avg_grade_score']),
            float(data['live_kt']),
            float(data['kt_attempt_level'])
        ]])
        scaled = dropout_scaler.transform(features)
        prediction = dropout_model.predict(scaled)[0]
        proba = dropout_model.predict_proba(scaled)[0]
        classes = dropout_model.classes_
        proba_dict = {cls: round(float(p) * 100, 1) for cls, p in zip(classes, proba)}
        return jsonify({'prediction': prediction, 'probabilities': proba_dict})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── API: Weak Subject (single) ─────────────────────────────────────────────
@app.route('/api/weak-subject', methods=['POST'])
def predict_weak_subject():
    try:
        data = request.json
        features = np.array([[
            float(data['ut_marks']),
            float(data['semester_marks']),
            float(data['internal_marks']),
            float(data['practical_marks']),
            float(data['attendance'])
        ]])
        scaled = weak_scaler.transform(features)
        prediction = weak_model.predict(scaled)[0]
        proba = weak_model.predict_proba(scaled)[0]
        classes = weak_model.classes_
        proba_dict = {cls: round(float(p) * 100, 1) for cls, p in zip(classes, proba)}
        grade_labels = {
            'A': 'Excellent — No intervention needed',
            'B': 'Good — Minor improvement suggested',
            'C': 'Average — Needs focused study',
            'D': 'Weak — Immediate intervention required'
        }
        return jsonify({
            'grade': prediction,
            'label': grade_labels.get(prediction, prediction),
            'probabilities': proba_dict
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── API: Placement (single) ────────────────────────────────────────────────
@app.route('/api/placement', methods=['POST'])
def predict_placement():
    try:
        data = request.json
        features = np.array([[
            float(data['cgpa']),
            float(data['projects']),
            float(data['internships']),
            float(data['communication_skills']),
            float(data['professor_review']),
            float(data['attendance']),
            float(data['total_kts'])
        ]])
        scaled = placement_scaler.transform(features)
        prediction = placement_model.predict(scaled)[0]
        proba = placement_model.predict_proba(scaled)[0]
        classes = placement_model.classes_
        proba_dict = {cls: round(float(p) * 100, 1) for cls, p in zip(classes, proba)}
        return jsonify({
            'prediction': prediction,
            'label': f'{prediction} Placement Chance',
            'probabilities': proba_dict
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ══════════════════════════════════════════════════════════════════════════════
# BULK PREDICTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def run_kt_bulk(df):
    cols = ['Sem1_Attendance_%','Sem1_Semester_Marks_Avg','Sem1_UT_Marks_Avg',
            'Sem1_Internal_Marks_Avg','Sem1_Subject_Count','Sem1_KTs']
    X = df[cols].astype(float).values
    scaled = kt_scaler.transform(X)
    preds  = kt_model.predict(scaled)
    probas = kt_model.predict_proba(scaled)
    df = df.copy()
    df['KT_Prediction']    = ['KT Likely' if p==1 else 'No KT' for p in preds]
    df['KT_Probability_%'] = [round(float(p[1])*100,1) for p in probas]
    df['Safe_Probability_%'] = [round(float(p[0])*100,1) for p in probas]
    return df

def run_dropout_bulk(df):
    cols = ['CGPA','Total_KTs','Attendance_%','Average_Grade_Score','Live_KT','KT_Attempt_Level']
    X = df[cols].astype(float).values
    scaled = dropout_scaler.transform(X)
    preds  = dropout_model.predict(scaled)
    probas = dropout_model.predict_proba(scaled)
    classes = dropout_model.classes_
    df = df.copy()
    df['Dropout_Risk'] = preds
    for i,cls in enumerate(classes):
        df[f'Prob_{cls}_%'] = [round(float(p[i])*100,1) for p in probas]
    return df

def run_placement_bulk(df):
    cols = ['CGPA','Projects','Physical_Internships','Communication_Skills',
            'Professor_Review','Attendance_%','Total_KTs']
    X = df[cols].astype(float).values
    scaled = placement_scaler.transform(X)
    preds  = placement_model.predict(scaled)
    probas = placement_model.predict_proba(scaled)
    classes = placement_model.classes_
    df = df.copy()
    df['Placement_Chance'] = preds
    for i,cls in enumerate(classes):
        df[f'Prob_{cls}_%'] = [round(float(p[i])*100,1) for p in probas]
    return df

def run_weak_bulk(df):
    cols = ['UT_Marks','Semester_Marks','Internal_Marks','Practical_Marks','Attendance_%']
    X = df[cols].astype(float).values
    scaled = weak_scaler.transform(X)
    preds  = weak_model.predict(scaled)
    probas = weak_model.predict_proba(scaled)
    classes = weak_model.classes_
    grade_labels = {'A':'Excellent','B':'Good','C':'Average','D':'Weak'}
    df = df.copy()
    df['Subject_Grade'] = preds
    df['Grade_Label']   = [grade_labels.get(p,p) for p in preds]
    for i,cls in enumerate(classes):
        df[f'Prob_Grade_{cls}_%'] = [round(float(p[i])*100,1) for p in probas]
    return df

RUNNERS = {'kt':run_kt_bulk,'dropout':run_dropout_bulk,
           'placement':run_placement_bulk,'weak':run_weak_bulk}

# ── Bulk preview (JSON for on-screen table) ────────────────────────────────
@app.route('/api/bulk/preview', methods=['POST'])
def bulk_preview():
    try:
        model = request.form.get('model')
        file  = request.files.get('file')
        if not file:
            return jsonify({'error':'No file uploaded'}), 400
        if model not in RUNNERS:
            return jsonify({'error':'Unknown model'}), 400
        df = pd.read_excel(file)
        result = RUNNERS[model](df)
        result = result.where(pd.notnull(result), None)
        return jsonify({
            'columns': list(result.columns),
            'rows':    result.values.tolist(),
            'total':   len(result)
        })
    except KeyError as e:
        return jsonify({'error': f'Missing column in Excel: {e}. Download the template to see the correct format.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── Bulk download Excel ────────────────────────────────────────────────────
@app.route('/api/bulk/download', methods=['POST'])
def bulk_download():
    try:
        model = request.form.get('model')
        file  = request.files.get('file')
        if not file or model not in RUNNERS:
            return jsonify({'error':'Invalid request'}), 400
        df     = pd.read_excel(file)
        result = RUNNERS[model](df)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            result.to_excel(writer, sheet_name='Predictions', index=False)
            ws = writer.sheets['Predictions']
            for col in ws.columns:
                max_len = max((len(str(c.value)) if c.value else 0) for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max_len+4, 40)
        output.seek(0)
        return send_file(output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True, download_name=f'{model}_predictions.xlsx')
    except KeyError as e:
        return jsonify({'error': f'Missing column: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── Template download ──────────────────────────────────────────────────────
@app.route('/api/template/<model>')
def download_template(model):
    templates = {
        'kt': {
            'cols': ['Student_Name','Roll_No','Sem1_Attendance_%','Sem1_Semester_Marks_Avg',
                     'Sem1_UT_Marks_Avg','Sem1_Internal_Marks_Avg','Sem1_Subject_Count','Sem1_KTs'],
            'rows': [['John Doe','CS101',82.5,61.0,15.0,19.0,6,0],
                     ['Jane Smith','CS102',55.0,38.0,9.0,12.0,6,2]]
        },
        'dropout': {
            'cols': ['Student_Name','Roll_No','CGPA','Total_KTs','Attendance_%',
                     'Average_Grade_Score','Live_KT','KT_Attempt_Level'],
            'rows': [['John Doe','CS101',7.2,1,78.0,6.8,0,1],
                     ['Jane Smith','CS102',5.1,4,52.0,4.9,2,2]]
        },
        'placement': {
            'cols': ['Student_Name','Roll_No','CGPA','Projects','Physical_Internships',
                     'Communication_Skills','Professor_Review','Attendance_%','Total_KTs'],
            'rows': [['John Doe','CS101',8.1,4,2,8,7,88.0,0],
                     ['Jane Smith','CS102',6.2,1,0,5,6,70.0,3]]
        },
        'weak': {
            'cols': ['Student_Name','Roll_No','Subject_Name','UT_Marks','Semester_Marks',
                     'Internal_Marks','Practical_Marks','Attendance_%'],
            'rows': [
                     ['John Doe','CS101','Data Structures',16.0,62.0,20.0,23.0,85.0],
                     ['John Doe','CS101','DBMS',12.0,48.0,14.0,16.0,75.0],
                     ['John Doe','CS101','Mathematics',7.0,32.0,9.0,10.0,58.0],
                     ['John Doe','CS101','Networks',14.0,58.0,18.0,20.0,80.0],
                     ['John Doe','CS101','Operating Systems',10.0,42.0,12.0,13.0,68.0],
                     ['John Doe','CS101','Python',17.0,70.0,22.0,24.0,90.0],
                     ['Jane Smith','CS102','Data Structures',6.0,28.0,8.0,9.0,52.0],
                     ['Jane Smith','CS102','DBMS',13.0,55.0,16.0,18.0,78.0],
                     ['Jane Smith','CS102','Mathematics',9.0,38.0,11.0,12.0,62.0],
                     ['Jane Smith','CS102','Networks',4.0,22.0,6.0,7.0,45.0],
                     ['Jane Smith','CS102','Operating Systems',11.0,44.0,13.0,15.0,70.0],
                     ['Jane Smith','CS102','Python',15.0,60.0,19.0,21.0,82.0]]
        }
    }
    if model not in templates:
        return jsonify({'error':'Unknown model'}), 400
    t  = templates[model]
    df = pd.DataFrame(t['rows'], columns=t['cols'])
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Student_Data', index=False)
        ws = writer.sheets['Student_Data']
        for col in ws.columns:
            max_len = max((len(str(c.value)) if c.value else 0) for c in col)
            ws.column_dimensions[col[0].column_letter].width = min(max_len+4, 40)
    output.seek(0)
    return send_file(output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True, download_name=f'{model}_template.xlsx')

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
