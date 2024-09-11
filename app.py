from flask import Flask, send_file, abort
import os

app = Flask(__name__)

# 設置存放PDF文件的目錄
PDF_DIRECTORY = "data/"  # 請替換為您的實際PDF目錄路徑

@app.route('/get-pdf/<filename>', methods=['GET'])
def get_pdf(filename):
    # 檢查文件名是否安全
    if not filename.endswith('.pdf') or '..' in filename:
        abort(400, description="Invalid filename")
    
    file_path = os.path.join(PDF_DIRECTORY, filename)
    
    # 檢查文件是否存在
    if not os.path.isfile(file_path):
        abort(404, description="PDF not found")
    
    try:
        return send_file(file_path, mimetype='application/pdf')
    except Exception as e:
        app.logger.error(f"Error sending file: {e}")
        abort(500, description="Internal server error")

@app.route('/list-pdfs', methods=['GET'])
def list_pdfs():
    try:
        pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.endswith('.pdf')]
        return {"pdf_files": pdf_files}
    except Exception as e:
        app.logger.error(f"Error listing PDFs: {e}")
        abort(500, description="Internal server error")

if __name__ == '__main__':
    app.run(debug=True, port=5000)