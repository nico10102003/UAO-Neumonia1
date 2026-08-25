"""Carga del modelo de detección de neumonía."""

import tensorflow as tf


def model_fun():
    """Carga el modelo entrenado para realizar inferencias.

    Returns:
        tensorflow.keras.Model: Modelo de clasificación cargado
        desde el archivo conv_MLP_84.h5.

    """
    model = tf.keras.models.load_model(
        "conv_MLP_84.h5",
        compile=False
    )

    return model