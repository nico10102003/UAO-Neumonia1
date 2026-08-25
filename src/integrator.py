"""Integración entre preprocesamiento, modelo y Grad-CAM."""

import numpy as np

from src.grad_cam import grad_cam
from src.load_model import model_fun
from src.preprocess_img import preprocess


def predict(array):
    """Realiza la predicción y genera el mapa de calor.

    Args:
        array: Imagen de entrada como numpy.ndarray.

    Returns:
        tuple: Clase predicha, probabilidad de la predicción
        y mapa de calor Grad-CAM.

    """
    batch_array_img = preprocess(array)

    model = model_fun()

    predictions = model(
        batch_array_img,
        training=False
    )

    predictions = np.asarray(predictions)

    prediction = np.argmax(predictions[0])
    proba = np.max(predictions[0]) * 100

    labels = {
        0: "bacteriana",
        1: "normal",
        2: "viral"
    }

    label = labels.get(
        prediction,
        "desconocida"
    )

    heatmap = grad_cam(array)

    return label, proba, heatmap