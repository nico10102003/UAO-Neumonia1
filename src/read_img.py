"""Funciones para lectura de imágenes DICOM y convencionales."""

import cv2
import numpy as np
import pydicom as dicom
from PIL import Image


def read_dicom_file(path):
    """Lee una imagen DICOM y prepara sus datos para el sistema.

    Args:
        path: Ruta del archivo DICOM.

    Returns:
        tuple: Arreglo NumPy de la imagen procesada y objeto PIL
        utilizado para su visualización.

    """
    img = dicom.dcmread(path)
    img_array = img.pixel_array
    img2show = Image.fromarray(img_array)

    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)

    img_rgb = cv2.cvtColor(
        img2,
        cv2.COLOR_GRAY2RGB
    )

    return img_rgb, img2show


def read_jpg_file(path):
    """Lee una imagen convencional y prepara sus datos.

    Args:
        path: Ruta del archivo de imagen.

    Returns:
        tuple: Arreglo NumPy de la imagen procesada y objeto PIL
        utilizado para su visualización.

    Raises:
        ValueError: Si la imagen no puede ser leída.

    """
    img = cv2.imread(path)

    if img is None:
        raise ValueError(
            f"No se pudo leer la imagen: {path}"
        )

    img_array = np.asarray(img)
    img2show = Image.fromarray(img_array)

    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)

    return img2, img2show