# tesseract OCR  API

### tesseract сан суулгах
```bash
sudo apt install tesseract-ocr

pip install pytesseract
```

### api request явуулах

```curl
curl -X POST -F "image=@/path/to/your/image.jpg" http://localhost:5000/api/ocr
```
