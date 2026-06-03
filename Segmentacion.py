import cv2
import numpy as np

def aplicar_watershed(ruta_imagen):
    # 1. Cargar imagen
    img = cv2.imread(ruta_imagen)
    assert img is not None, "Error: No se encontró la imagen."
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Binarización de Otsu (Filtro espacial)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Operaciones Morfológicas para eliminar ruido (Opening)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 4. Encontrar el área de fondo seguro (Dilation)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 5. Encontrar el área de primer plano seguro (Distance Transform)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # 6. Encontrar la región desconocida (Bordes colindantes)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 7. Etiquetado de marcadores
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1 # El fondo debe ser 1, no 0
    markers[unknown == 255] = 0 # Marcar región desconocida con 0

    # 8. Aplicar Watershed
    markers = cv2.watershed(img, markers)
    
    # Dibujar los contornos en rojo sobre la imagen original
    img[markers == -1] = [0, 0, 255]

    # Mostrar resultados
    cv2.imshow('Procesamiento Watershed', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Reemplazar con la ruta de una imagen de prueba local
    aplicar_watershed('papa_prueba.jpg')