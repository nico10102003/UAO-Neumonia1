"""Pruebas unitarias para el sistema de detección de neumonía."""

from unittest.mock import patch

import numpy as np
from PIL import Image

from src.integrator import predict
from src.read_img import read_jpg_file


def test_read_jpg_file(tmp_path):
    """Verifica la lectura y conversión de una imagen JPG."""
    image = Image.new("RGB", (100, 100), "white")
    image_path = tmp_path / "test.jpg"
    image.save(image_path)

    img_array, img2show = read_jpg_file(str(image_path))

    assert isinstance(img_array, np.ndarray)
    assert isinstance(img2show, Image.Image)
    assert img_array.shape == (100, 100, 3)


@patch("src.integrator.grad_cam")
@patch("src.integrator.model_fun")
@patch("src.integrator.preprocess")
def test_predict(mock_preprocess, mock_model_fun, mock_grad_cam):
    """Verifica la lógica de predicción utilizando componentes simulados."""
    array = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    mock_preprocess.return_value = np.zeros(
        (1, 512, 512, 1),
        dtype=np.float32,
    )

    def mock_model(batch, training=False):
        """Simula la salida del modelo de clasificación."""
        return np.array(
            [[0.10, 0.90, 0.00]],
            dtype=np.float32,
        )

    mock_model_fun.return_value = mock_model

    mock_grad_cam.return_value = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    label, proba, heatmap = predict(array)

    assert label == "normal"
    assert proba == 90.0
    assert isinstance(heatmap, np.ndarray)

    mock_preprocess.assert_called_once_with(array)
    mock_model_fun.assert_called_once()
    mock_grad_cam.assert_called_once_with(array)