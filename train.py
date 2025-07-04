from ultralytics import YOLO

# 加载模型
model = YOLO('drones_ghost.yaml')

# 训练模型
results = model.train(data='drones.yaml', epochs=500, imgsz=640, batch=16)

# 在ultralytics/nn/modules/conv.py and block.py 中激活函数修改 nn.SiLU() 或 nn.ReLU()
