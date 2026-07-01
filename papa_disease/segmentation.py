"""Segmentacion experimental de hojas mediante HSV y Watershed."""


def apply_watershed(image_bgr):
    """Devuelve mascara, marcadores, contornos y hoja aislada.

    La entrada debe ser una imagen OpenCV BGR. Esta transformacion no se aplica
    automaticamente al clasificador porque el modelo original no fue entrenado
    con imagenes segmentadas.
    """
    import cv2
    import numpy as np

    if image_bgr is None or getattr(image_bgr, "ndim", 0) != 3:
        raise ValueError("Se esperaba una imagen BGR valida")

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([25, 40, 40]), np.array([95, 255, 255]))
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    sure_background = cv2.dilate(mask, kernel, iterations=3)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    threshold = 0.5 * distance.max()
    _, sure_foreground = cv2.threshold(distance, threshold, 255, 0)
    sure_foreground = np.uint8(sure_foreground)
    unknown = cv2.subtract(sure_background, sure_foreground)

    _, markers = cv2.connectedComponents(sure_foreground)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image_bgr.copy(), markers)

    contours = image_bgr.copy()
    contours[markers == -1] = [0, 0, 255]
    isolated = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    return {
        "mask": mask,
        "markers": markers,
        "contours": contours,
        "isolated": isolated,
    }

