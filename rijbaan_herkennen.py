import cv2
import numpy as np
from rfdetr import RFDETRSmall
from pathlib import Path

# Instellingen
image_file = "test2.jpg"
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

def cv_box(img, detections):
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

    return output_image

def cv_predict(img, loaded_model):
    detections = loaded_model.predict(img, threshold=0.5)

    return detections

def threshold_mask(img):
    masked_image = img.copy()
    hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    background_mask = cv2.bitwise_or(mask_green, mask_blue)

    return background_mask


def box_mask(img, detections):
    box_mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255

    for xyxy in detections.xyxy:
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(box_mask, (x1, y1), (x2, y2), 0, -1)

    return box_mask


def bridge_gaps(edges):
    """Verbindt gaten in de randdetectie met behulp van morfologische sluiting (Closing)."""
    # Maak een kernel (matrix) van bijv. 5x5 pixels om gaten te overbruggen
    kernel = np.ones((7, 7), np.uint8)

    # MORPH_CLOSE voert eerst een dilation (uitzetten) en dan een erosion (inkrimpen) uit,
    # wat perfect is om gaten in lijnen op te vullen.
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    return closed_edges


def detect_major_lines(closed_edges, original_img):
    """Detecteert grote rechte lijnen en tekent ze strak over de afbeelding."""
    line_image = original_img.copy()

    # HoughLinesP parameters:
    # rho en theta: resolutie van de accumulator
    # threshold: hoeveel stemmen een lijn nodig heeft om gedetecteerd te worden
    # minLineLength: minimale lengte van een lijn om mee te tellen
    # maxLineGap: de maximale afstand tussen punten om ze toch als één doorlopende lijn te zien (belangrijk voor gaten!)
    lines = cv2.HoughLinesP(closed_edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=50, maxLineGap=30)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Teken de gedetecteerde hoofdlijnen in het felgroen
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return line_image

def main():
    print("hallo")
    image, model = setup()
    detections = cv_predict(image, model)
    image_vision = cv_box(image, detections)
    image_edge = edge_detection(image)
    mask1 = box_mask(image, detections)
    mask2 = threshold_mask(image)
    image_edge[mask1 == 0] = [0]
    image_edge[mask2 > 0] = [0]

    #Gaten in de randen opvullen
    closed_edges = bridge_gaps(image_edge)
    #Grote uitlijningen (rechte lijnen) detecteren en tekenen
    line_result = detect_major_lines(closed_edges, image)

    # Toon het resultaat op je scherm
    cv2.imshow('image_no_gaps', line_result)
    cv2.imshow('cleaned_image', image_edge)

    # Wacht tot een toets wordt ingedrukt en sluit alle vensters
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()