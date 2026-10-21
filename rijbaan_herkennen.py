import cv2
import numpy as np
from rfdetr import RFDETRSmall
from pathlib import Path

# Instellingen
image_file = "test1.jpg"
model_file = "weights.pt"
lower_green = np.array([35, 40, 40])    # herkennen gras
upper_green = np.array([85, 255, 255])
lower_blue = np.array([90, 50, 50])     # herkennen lucht
upper_blue = np.array([130, 255, 255])

def setup():
    img = cv2.imread(image_file)
    if img is None:
        print("Kan de afbeelding niet vinden.")
        exit()
    weights = Path(model_file).resolve()
    loaded_model = RFDETRSmall(pretrain_weights=str(weights), num_classes=8)
    return img, loaded_model

def edge_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred_img, 100, 200)
    return edges

def computer_vision(img, loaded_model):
    detections = loaded_model.predict(img, threshold=0.5)
    class_names = ["_", "auto_achter", "auto_links", "auto_rechts", "auto_voor", "auto_onbekend", "fietser", "huis", "voetganger"]  # model start bij index 1
    color_palette = [(0, 0, 0),(255, 0, 0),(0, 0, 255),(0, 255, 255),(255, 0, 255),(255, 255, 0),(0, 165, 255),(128, 128, 128),(0, 128, 128)]

    # Teken de detecties op een kopie van de afbeelding
    output_image = img.copy()

    for xyxy, class_id, confidence in zip(detections.xyxy, detections.class_id, detections.confidence):
        x1, y1, x2, y2 = map(int, xyxy)
        cid = int(class_id)
        class_name = class_names[cid]
        color = color_palette[cid]

        # Teken het kader in de bijbehorende kleur, naam en zekerheid
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} ({confidence:.2f})"
        cv2.putText(output_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return output_image, detections


def clean_image_for_edges(img, detections):
    """Maakt de afbeelding schoon door objecten, lucht en gras zwart te maken."""
    masked_image = img.copy()

    # 1. Maak alle gedetecteerde objecten (auto's, huizen, etc.) zwart
    for xyxy in detections.xyxy:
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(masked_image, (x1, y1), (x2, y2), (0, 0, 0), -1)

    # 2. Filter lucht en gras weg via HSV-kleurruimte
    hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    background_mask = cv2.bitwise_or(mask_green, mask_blue)
    masked_image[background_mask > 0] = [0, 0, 0]

    return masked_image

def main():
    print("hallo")
    image, model = setup()
    image_vision, detections = computer_vision(image, model)
    cleaned_image = clean_image_for_edges(image, detections)
    image_edge = edge_detection(cleaned_image)

    # Toon het resultaat op je scherm
    cv2.imshow('object herkenning', image_vision)
    cv2.imshow('edge detection', image_edge)
    cv2.imshow('cleaned_image', cleaned_image)

    # Wacht tot een toets wordt ingedrukt en sluit alle vensters
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()