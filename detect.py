import os
import cv2
from ultralytics import YOLO

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)
    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = predict(chosen_model, img, classes, conf=conf)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img,
                          (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])),
                          (255, 0, 0), rectangle_thickness)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
    return img, results

# 初始化模型
model = YOLO("ghostn.pt")

# 输入输出路径
input_folder = "E:\课题组调研\yolov11_rk3588/ultralytics-8.3.31/ultralytics-8.3.31/test_pictures"          # 替换为你的图片文件夹路径
output_folder = "E:\课题组调研\yolov11_rk3588/ultralytics-8.3.31/ultralytics-8.3.31\output_pictures"         # 保存检测结果图像的路径
os.makedirs(output_folder, exist_ok=True)

# 支持的图像格式
image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

# 遍历文件夹中的图片
for filename in os.listdir(input_folder):
    if any(filename.lower().endswith(ext) for ext in image_extensions):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"无法读取图像: {filename}")
            continue

        result_img, _ = predict_and_detect(model, image, classes=[], conf=0.5)

        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, result_img)
        print(f"检测结果已保存: {save_path}")
