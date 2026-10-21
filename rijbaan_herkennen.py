import cv2
import numpy as np
from rfdetr import RFDETRSmall
from pathlib import Path

image = cv2.imread('test19.jpg')
weights = Path("weights.pt").resolve()

model = RFDETRSmall(pretrain_weights=str(weights), num_classes=8)

if image is None:
    print("Kan de afbeelding niet vinden.")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

detections = model.predict(image, threshold=0.5)
class_names = ["_","auto_achter","auto_links","auto_rechts","auto_voor","auto_onbekend","fietser","huis","voetganger"] # model start bij index 1
color_palette = [(0, 0, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 165, 255), (128, 128, 128), (0, 128, 128)]

# Teken de detecties op een kopie van de afbeelding
output_image = image.copy()

for xyxy, class_id, confidence in zip(detections.xyxy, detections.class_id, detections.confidence):
    x1, y1, x2, y2 = map(int, xyxy)
    cid = int(class_id)
    class_name = class_names[cid]
    color = color_palette[cid]

    # Teken het kader in de bijbehorende kleur, naam en zekerheid
    cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
    label = f"{class_name} ({confidence:.2f})"
    cv2.putText(output_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Canny edges
edges = cv2.Canny(blurred, 100, 200)

# Toon het resultaat op je scherm
cv2.imshow('object herkenning', output_image)
cv2.imshow('edge detection', edges)

# Wacht tot een toets wordt ingedrukt en sluit alle vensters
cv2.waitKey(0)
cv2.destroyAllWindows()