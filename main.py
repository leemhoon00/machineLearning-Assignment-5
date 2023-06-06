import cv2
import torch

# 모델 가중치 파일 경로
model_weights = 'my_model.pt'

# Yolov5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_weights)
model.conf = 0.5  # 객체 감지 임계값 (threshold)

# 카메라 연결
cap = cv2.VideoCapture(0)  # 0은 기본 카메라를 의미합니다.

while True:
    ret, frame = cap.read()  # 프레임 읽기

    # 프레임을 모델에 전달하여 객체 감지 수행
    results = model(frame)
    # 결과 시각화
    # results.show()


    for detection in results.xyxy[0]:
        xmin, ymin, xmax, ymax, conf, cls = detection.tolist()
        label = f'{model.names[int(cls)]} {conf:.2f}'
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 화면에 출력
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) == ord('q'):  # 'q' 키를 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()
