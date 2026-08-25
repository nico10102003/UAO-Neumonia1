"""Funciones para el preprocesamiento de imágenes."""

import cv2
import numpy as np


def preprocess(array):
    """Preprocesa una imagen para ingresarla al modelo.

    Redimensiona la imagen a 512x512 píxeles, la convierte
    a escala de grises, aplica CLAHE, normaliza los valores
    entre 0 y 1 y agrega las dimensiones necesarias para
    formar un lote de una imagen.

    Args:
        array: Imagen de entrada en formato NumPy.

    Returns:
        np.ndarray: Imagen preprocesada con forma
        (1, 512, 512, 1).

    """
    array = cv2.resize(array, (512, 512))
    array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(4, 4)
    )

    array = clahe.apply(array)
    array = array.astype(np.float32) / 255.0
    array = np.expand_dims(array, axis=-1)
    array = np.expand_dims(array, axis=0)

    return array