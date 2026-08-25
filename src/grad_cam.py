"""Generación del mapa de calor Grad-CAM."""

import cv2
import numpy as np
import tensorflow as tf

from src.load_model import model_fun
from src.preprocess_img import preprocess


def grad_cam(array):
    """Genera un mapa de calor Grad-CAM para una imagen.

    Args:
        array: Imagen original como numpy.ndarray.

    Returns:
        numpy.ndarray: Imagen RGB con el mapa de calor superpuesto.

    """
    img = preprocess(array)
    model = model_fun()

    # Obtener la última capa convolucional.
    last_conv_layer = model.get_layer("conv10_thisone")

    # Crear un modelo que devuelva las activaciones de la última
    # capa convolucional y las predicciones.
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            last_conv_layer.output,
            model.outputs[0],
        ],
    )

    # Obtener activaciones y predicciones.
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(
            img,
            training=False,
        )

        predicted_class = tf.argmax(
            predictions[0]
        )

        class_channel = predictions[0][predicted_class]

    # Gradientes respecto a la última capa convolucional.
    grads = tape.gradient(
        class_channel,
        conv_outputs,
    )

    # Promedio de los gradientes por canal.
    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2),
    )

    # Activaciones de la última capa convolucional.
    conv_outputs = conv_outputs[0]

    # Aplicar importancia de cada canal.
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Aplicar ReLU.
    heatmap = tf.maximum(
        heatmap,
        0,
    )

    # Normalizar.
    max_value = tf.reduce_max(heatmap)

    if max_value > 0:
        heatmap /= max_value

    # Convertir a NumPy.
    heatmap = heatmap.numpy()

    # Redimensionar.
    heatmap = cv2.resize(
        heatmap,
        (512, 512),
    )

    # Convertir a 0-255.
    heatmap = np.uint8(
        255 * heatmap
    )

    # Aplicar mapa de colores.
    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET,
    )

    # Imagen original.
    img2 = cv2.resize(
        array,
        (512, 512),
    )

    # Superponer mapa de calor.
    hif = 0.8
    transparency = heatmap * hif
    transparency = transparency.astype(
        np.uint8
    )

    superimposed_img = cv2.add(
        transparency,
        img2,
    )

    superimposed_img = superimposed_img.astype(
        np.uint8
    )

    # OpenCV utiliza BGR; Tkinter/PIL necesita RGB.
    return superimposed_img[:, :, ::-1]