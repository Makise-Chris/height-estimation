<h4>Thêm file unet.h5 từ https://drive.google.com/drive/folders/1yewztICaSS3QUEgmGIk4L5gp9n6NSI2Q?usp=sharing vào Project</h4>
<h4>Các bước thực nghiệm:
    <p>
    1. Chụp ảnh bằng file calibration_images.py
    (Ảnh bàn cờ bấm "s", ảnh body bấm "o")
    </p>
    <p>
    2. Thực hiện Calibration với ảnh bàn cờ bằng file stereo_calibration.py, tham số lưu vào file stereoMap.xml
    </p>
    <p>
    3. Thực hiện Rectification ảnh body bằng file rectification.py
    </p>
    <p>
    4. Thực hiện Background Subtraction ảnh body bằng file bg-subtraction.py
    </p>
    <p>
    5. Thực hiện Triangulation để khôi phục thông tin 3D của điểm đỉnh đầu bằng file triangulation.py
    </p>
</h4>

