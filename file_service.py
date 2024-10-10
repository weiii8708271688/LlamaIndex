import base64
import os
from pathlib import Path
from typing import Dict, Any
import mimetypes

class PrivateFileService:
    PRIVATE_STORE_PATH = "output/uploaded"

    @staticmethod
    def process_file(file_name: str, base64_content: str) -> Dict[str, Any]:
        file_data = base64.b64decode(base64_content.split(",", 1)[1])
        
        # 存儲文件
        os.makedirs(PrivateFileService.PRIVATE_STORE_PATH, exist_ok=True)
        file_path = Path(os.path.join(PrivateFileService.PRIVATE_STORE_PATH, file_name))
        
        with open(file_path, "wb") as f:
            f.write(file_data)
        
        # 獲取文件大小
        file_size = os.path.getsize(file_path)
        
        # 獲取文件類型
        file_type = mimetypes.guess_type(file_name)[0]
        extension = os.path.splitext(file_name)[1][1:].lower()
        
        # 如果無法猜測文件類型，使用擴展名作為後備
        if file_type is None:
            file_type = extension
        
        # 構建符合前端期望的響應
        return {
            "filename": file_name,
            "filesize": file_size,
            "filetype": extension,
            "content": {
                "type": "ref",
                "value": [str(file_path)]
            }
        }