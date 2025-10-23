import os
import cv2
import numpy as np
import joblib
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image 
import pillow_heif   

# --- 초기 설정 ---
app = Flask(__name__, template_folder='templates')
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# --- AI 모델 로딩 ---
try:
    # 모델 파일 이름을 최종 버전으로 수정
    model = joblib.load("makgeolli_final.joblib") 
    label_encoder = joblib.load("label_encoder.joblib")
    print("[알림] AI 모델을 성공적으로 불러왔습니다.")
except FileNotFoundError as e:
    print(f"[오류] 모델 파일을 찾을 수 없습니다: {e}")
    model = None

# --- 핵심 기능 함수 (이전과 동일) ---
def extract_lab_with_floodfill(image, seed_point):
    img_copy = image.copy()
    h, w = img_copy.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    tolerance = 3
     # 민감도를 약간 높여서 더 넓은 영역을 잡도록 조정
    loDiff, upDiff = (tolerance,) * 3, (tolerance,) * 3
    cv2.floodFill(img_copy, mask, seed_point, (0, 0, 0), loDiff, upDiff)
    mask = mask[1:-1, 1:-1]
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    if np.sum(mask) == 0:
        pixel = lab_image[seed_point[1], seed_point[0]]
        l_val, a_val, b_val = pixel[0], pixel[1], pixel[2]
    else:
        l_val, a_val, b_val, _ = cv2.mean(lab_image, mask=mask)
    a_std = a_val - 128
    b_std = b_val - 128
    return a_std, b_std

# --- 웹페이지 라우팅 ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None: return jsonify({"error": "모델이 로드되지 않았습니다."}), 500
    if 'file' not in request.files: return jsonify({"error": "파일이 없습니다."}), 400
    
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "파일이 선택되지 않았습니다."}), 400

    try:
        schiff_x = int(float(request.form.get('schiff_x')))
        schiff_y = int(float(request.form.get('schiff_y')))
        litmus_x = int(float(request.form.get('litmus_x')))
        litmus_y = int(float(request.form.get('litmus_y')))
    except (TypeError, ValueError):
        return jsonify({"error": "유효하지 않은 좌표 값입니다."}), 400

    if file:
        filename = secure_filename(file.filename)
        if not os.path.exists(app.config['UPLOAD_FOLDER']): os.makedirs(app.config['UPLOAD_FOLDER'])
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # ★★★★★ 이미지 로딩 부분 전면 수정 ★★★★★
            file_extension = os.path.splitext(filename)[1].lower()

            if file_extension in ['.heic', '.heif']:
                # 1. HEIC/HEIF 파일 처리
                heif_file = pillow_heif.read_heif(filepath)
                image_pil = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw")
            else:
                # 2. 다른 모든 파일은 Pillow로 먼저 열기
                image_pil = Image.open(filepath)

            # 3. EXIF 회전 정보 자동 적용
            #   Pillow의 ExifTags를 사용해 회전 정보를 읽고 이미지를 바로 세움
            from PIL import ImageOps
            image_pil = ImageOps.exif_transpose(image_pil)

            # 4. Pillow 이미지를 OpenCV 형식(numpy array)으로 변환
            #   RGB(Pillow) -> BGR(OpenCV) 순서로 변경
            original_image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            
            # 5. 투명도 채널(4채널) 처리
            if original_image.shape[2] == 4:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGRA2BGR)

            schiff_a, schiff_b = extract_lab_with_floodfill(original_image, (schiff_x, schiff_y))
            litmus_a, litmus_b = extract_lab_with_floodfill(original_image, (litmus_x, litmus_y))

            lab_values = [schiff_a, schiff_b, litmus_a, litmus_b]
            prediction_encoded = model.predict([lab_values])
            prediction_text = label_encoder.inverse_transform(prediction_encoded)[0]
            
            return jsonify({"prediction": prediction_text})

        except Exception as e:
            print(f"[서버 처리 오류] {e}")
            return jsonify({"error": "이미지 분석 중 오류가 발생했습니다."}), 500
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    return jsonify({"error": "알 수 없는 오류가 발생했습니다."}), 500

if __name__ == '__main__':
    app.run(debug=True)