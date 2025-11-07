import os
import cv2
import numpy as np
import joblib
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException
from PIL import Image, ImageOps, JpegImagePlugin
import pillow_heif
from datetime import datetime
import pandas as pd

# --- 초기 설정 ---
app = Flask(__name__, template_folder='templates')
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# --- 최종 AI 전문가 팀 전체 로딩 ---
try:
    doctor_model = joblib.load("doctor_classification_model.joblib")
    prophet_model = joblib.load("prophet_regression_model.joblib")
    nurse_ph_model = joblib.load("nurse_pH_model.joblib")
    nurse_alc_model = joblib.load("nurse_도수_model.joblib")
    nurse_brix_model = joblib.load("nurse_Brix_model.joblib")
    label_encoder = joblib.load("label_encoder_3stages.joblib")
    print("[알림] 모든 AI 전문가 모델을 성공적으로 불러왔습니다.")
except FileNotFoundError as e:
    print(f"[오류] 모델 파일을 찾을 수 없습니다: {e}")
    doctor_model = None

# ★★★★★ 생략 없는 완전한 함수 1: 요술봉 Lab 추출 ★★★★★
def extract_lab_with_floodfill(image, seed_point):
    img_copy = image.copy()
    h, w = img_copy.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    tolerance = 3
    loDiff, upDiff = (tolerance,) * 3, (tolerance,) * 3

    if not (0 <= seed_point[0] < w and 0 <= seed_point[1] < h):
        raise ValueError(f"전달된 좌표 {seed_point}가 이미지 크기 ({w}x{h})를 벗어났습니다.")

    try:
        cv2.floodFill(img_copy, mask, seed_point, (0, 0, 0), loDiff, upDiff)
    except Exception as e:
        mask.fill(0)

    mask = mask[1:-1, 1:-1]
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    if np.sum(mask) == 0:
        pixel = lab_image[seed_point[1], seed_point[0]]
        a_val, b_val = pixel[1], pixel[2]
    else:
        if len(mask.shape) > 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, a_val, b_val, _ = cv2.mean(lab_image, mask=mask)
    
    a_std = a_val - 128
    b_std = b_val - 128
    return a_std, b_std

# ★★★★★ 생략 없는 완전한 함수 2: 안정적인 이미지 로딩 ★★★★★
def load_image_safely(filepath):
    JpegImagePlugin.LOAD_TRUNCATED_IMAGES = True
    file_extension = os.path.splitext(filepath)[1].lower()
    
    if file_extension in ['.heic', '.heif']:
        heif_file = pillow_heif.read_heif(filepath, convert_hdr_to_8bit=True)
        image_pil = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw")
    else:
        image_pil = Image.open(filepath)

    # Check for animated images (specifically WebP, as GIF is already blocked by ALLOWED_EXTENSIONS)
    if file_extension == '.webp' and getattr(image_pil, 'is_animated', False):
        raise ValueError("애니메이션 WebP 파일은 지원되지 않습니다.")

    image_pil = ImageOps.exif_transpose(image_pil)
    
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')
        
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

@app.route('/')
def index():
    return render_template('index.html')

@app.errorhandler(Exception)
def handle_exception(e):
    # handle HTTP errors
    if isinstance(e, HTTPException):
        return jsonify({"error": e.description}), e.code
    # handle non-HTTP exceptions
    return jsonify({"error": "서버 내부 오류가 발생했습니다."}), 500

ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.heic', '.heif', '.bmp', '.tif', '.tiff', '.webp'}

@app.route('/predict', methods=['POST'])
def predict():
    if doctor_model is None:
        return jsonify({"error": "모델이 로드되지 않았습니다."}), 500
    
    file = request.files.get('file')
    start_date_str = request.form.get('start_date')
    if not file or not start_date_str or file.filename == '':
        return jsonify({"error": "이미지 파일과 발효 시작 시간을 모두 입력해주세요."}), 400

    filename = secure_filename(file.filename)
    file_ext = os.path.splitext(filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"지원하지 않는 파일 형식입니다. 지원하는 형식: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        file.save(filepath)
        original_image = load_image_safely(filepath)

        # --- 1. 정보 수집 ---
        schiff_x = int(float(request.form.get('schiff_x')))
        schiff_y = int(float(request.form.get('schiff_y')))
        litmus_x = int(float(request.form.get('litmus_x')))
        litmus_y = int(float(request.form.get('litmus_y')))

        # 사용자가 클릭한 좌표가 이미지 경계를 벗어나는 경우를 방지하기 위해 값을 보정합니다.
        h, w = original_image.shape[:2]
        schiff_x = max(0, min(schiff_x, w - 1))
        schiff_y = max(0, min(schiff_y, h - 1))
        litmus_x = max(0, min(litmus_x, w - 1))
        litmus_y = max(0, min(litmus_y, h - 1))
        
        schiff_a, schiff_b = extract_lab_with_floodfill(original_image, (schiff_x, schiff_y))
        litmus_a, litmus_b = extract_lab_with_floodfill(original_image, (litmus_x, litmus_y))
        elapsed_hours = (datetime.now() - datetime.fromisoformat(start_date_str)).total_seconds() / 3600

        # Validate elapsed_hours to prevent model errors with extreme values
        MAX_ELAPSED_HOURS = 30 * 24 # 30 days
        if elapsed_hours < 0:
            return jsonify({"error": "미래의 날짜는 선택할 수 없습니다."}), 400
        if elapsed_hours > MAX_ELAPSED_HOURS:
            return jsonify({"error": "선택하신 날짜가 너무 오래되었습니다. 최근 한 달 이내의 날짜를 선택해주세요."}), 400

        # --- 2. AI 간호사: 활력 징후 예측 ---
        nurse_input_df = pd.DataFrame([[schiff_a, schiff_b, litmus_a, litmus_b, elapsed_hours]], 
                                      columns=['시프_a*', '시프_b*', '리트머스_a*', '리트머스_b*', '경과시간'])
        
        predicted_ph = nurse_ph_model.predict(nurse_input_df)[0]
        predicted_alc = nurse_alc_model.predict(nurse_input_df)[0]
        predicted_brix = nurse_brix_model.predict(nurse_input_df)[0]

        # --- 3. AI 예언가: '적정 발효까지' 남은 시간 예측 ---
        prophet_input_df = pd.DataFrame([[schiff_a, schiff_b, litmus_a, litmus_b, elapsed_hours]],
                                        columns=['시프_a*', '시프_b*', '리트머스_a*', '리트머스_b*', '경과시간'])
        predicted_time = prophet_model.predict(prophet_input_df)[0]

        # --- 4. AI 의사: 최종 상태 진단 ---
        doctor_input_df = pd.DataFrame([[schiff_a, schiff_b, litmus_a, litmus_b, elapsed_hours,
                                         predicted_ph, predicted_alc, predicted_brix]],
                                       columns=['시프_a*', '시프_b*', '리트머스_a*', '리트머스_b*', '경과시간',
                                                '예상_pH', '예상_도수', '예상_Brix'])
        prediction_encoded = doctor_model.predict(doctor_input_df)
        prediction_text = label_encoder.inverse_transform(prediction_encoded)[0]
        
        # --- 5. 최종 결과 종합 및 전송 ---
        final_state_text = prediction_text # '초기', '중기', '후기'
        remaining_time_to_optimal = predicted_time # 예: 100.5 또는 -80.0
        optimal_period_text = ""

        # 규칙 1: '후기'일 때 (가장 우선 순위)
        if final_state_text == '후기':
            optimal_period_text = "산패가 시작되었습니다."

        # 규칙 2: '후기'가 아니고, 아직 '적정 발효' 시점이 오지 않았을 때
        elif remaining_time_to_optimal > 1:
            optimal_period_text = f"약 {remaining_time_to_optimal:.0f} 시간"
        
        # 규칙 3: '후기'가 아니고, '적정 발효' 시점이 거의 다 됐거나 이미 지났을 때
        else:
            optimal_period_text = "최적의 음용 기간입니다."

        return jsonify({
            "prediction_state": prediction_text,
            "prediction_time_to_optimal": optimal_period_text,
            "estimated_ph": f"{predicted_ph:.2f}",
            "estimated_alcohol": f"{predicted_alc:.1f}°",
            "estimated_brix": f"{predicted_brix:.1f} Brix"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    app.run(debug=True)