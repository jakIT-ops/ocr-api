import cv2
import pytesseract
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)
@app.route('/api/ocr', methods=['POST'])
def ocr():

    if 'image' not in request.files:
        return jsonify({'error': 'file oldsongvi'}), 400

    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Recognize key points on A4 paper
    #sift = cv2.xfeatures2d.SIFT_create()
    # kp = sift.detect(thresh, None)

    # Draw keypoints
    # img_kp = cv2.drawKeypoints(image, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow('Keypoints', img_kp)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Find A4 paper using key points and crop the image
    #MIN_MATCH_COUNT = 10
    #a4_template = cv2.imread('a4_template.png', 0)
    #kp_template, des_template = sift.detectAndCompute(a4_template, None)
    #bf = cv2.BFMatcher()
   # matches = bf.knnMatch(des_template, des_template, k=2)
    #good_matches = []
    #for m, n in matches:
     #   if m.distance < 0.75 * n.distance:
      #      good_matches.append(m)
   # if len(good_matches) > MIN_MATCH_COUNT:
    #    src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
     #   dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    #    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    #    h, w = a4_template.shape
    #    corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    #    dst = cv2.perspectiveTransform(corners, M)
    #    x, y, w, h = cv2.boundingRect(dst)
    #    image = image[y:y+h, x:x+w]
    #    gray = gray[y:y+h, x:x+w]
    #    thresh = thresh[y:y+h, x:x+w]
   # else:
   #     return jsonify({'error': 'A4 paper not found'}), 400

    config = ('-l mon — oem 3 — psm 3')
    text = pytesseract.image_to_string(thresh, lang="mon")
    
    if not text:
        return "Амжилтгүй."

    return text
    #return jsonify({'text': text})


if __name__ == '__main__':
    app.run()

