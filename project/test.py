import cv2

print("🎥 检查可用摄像头索引...")
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ 摄像头索引 {i} 可用")
        cap.release()
    else:
        print(f"❌ 摄像头索引 {i} 无法打开")
