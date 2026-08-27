"""Pruebas unitarias para el sistema de detección de neumonía."""

from unittest.mock import patch

import numpy as np
from PIL import Image

from src.integrator import predict
from src.read_img import read_jpg_file


def crear_imagen(alto=100, ancho=100):
    """Crea una imagen RGB de prueba."""
    return np.zeros(
        (alto, ancho, 3),
        dtype=np.uint8,
    )


def configurar_mocks(
    mock_preprocess,
    mock_model_fun,
    mock_grad_cam,
):
    """Configura los mocks utilizados por predict."""
    mock_preprocess.return_value = np.zeros(
        (1, 512, 512, 1),
        dtype=np.float32,
    )

    mock_model_fun.return_value = (
        lambda batch, training=False: np.array(
            [[0.10, 0.90, 0.00]],
            dtype=np.float32,
        )
    )

    mock_grad_cam.return_value = np.zeros(
        (512, 512, 3),
        dtype=np.uint8,
    )


def test_read_jpg_file(tmp_path):
    """Verifica la lectura y conversión de una imagen JPG."""
    image = Image.new("RGB", (100, 100), "white")
    image_path = tmp_path / "test.jpg"
    image.save(image_path)

    img_array, img2show = read_jpg_file(
        str(image_path)
    )

    assert isinstance(img_array, np.ndarray)
    assert isinstance(img2show, Image.Image)
    assert img_array.shape == (100, 100, 3)


@patch("src.integrator.grad_cam")
@patch("src.integrator.model_fun")
@patch("src.integrator.preprocess")
def test_predict(
    mock_preprocess,
    mock_model_fun,
    mock_grad_cam,
):
    """Verifica la lógica básica de predicción."""
    imagen = crear_imagen()

    configurar_mocks(
        mock_preprocess,
        mock_model_fun,
        mock_grad_cam,
    )

    label, proba, heatmap = predict(imagen)

    assert label == "normal"
    assert proba == 90.0
    assert isinstance(heatmap, np.ndarray)

    mock_preprocess.assert_called_once_with(imagen)
    mock_model_fun.assert_called_once()
    mock_grad_cam.assert_called_once_with(imagen)


@patch("src.integrator.grad_cam")
@patch("src.integrator.model_fun")
@patch("src.integrator.preprocess")
def test_predict_retorna_tupla(
    mock_preprocess,
    mock_model_fun,
    mock_grad_cam,
):
    """Verifica que predict retorne una tupla."""
    imagen = crear_imagen()

    configurar_mocks(
        mock_preprocess,
        mock_model_fun,
        mock_grad_cam,
    )

    resultado = predict(imagen)

    assert isinstance(resultado, tuple)


@patch("src.integrator.grad_cam")
@patch("src.integrator.model_fun")
@patch("src.integrator.preprocess")
def test_predict_clase_bacteriana(
    mock_preprocess,
    mock_model_fun,
    mock_grad_cam,
):
    """Verifica la clasificación bacteriana."""
    imagen = crear_imagen()

    configurar_mocks(
        mock_preprocess,
        mock_model_fun,
        mock_grad_cam,
    )

    mock_model_fun.return_value = (
        lambda batch, training=False: np.array(
            [[0.90, 0.05, 0.05]],
            dtype=np.float32,
        )
    )

    label, proba, _ = predict(imagen)

    assert label == "bacteriana"
    assert proba == 90.0


@patch("src.integrator.grad_cam")
@patch("src.integrator.model_fun")
@patch("src.integrator.preprocess")
def test_predict_clase_normal(
    mock_preprocess,
    mock_model_fun,
    mock_grad_cam,
):
    """Verifica la clasificación normal."""
    imagen = crear_imagen()

    configurar_mocks(
        mock_preprocess,
        mock_model_fun,
        mock_grad_cam,
    )

    mock_model_fun.return_value = (
        lambda batch, training=False: np.array(
            [[0.05, 0.90, 0.05]],
            dtype=np.float32,
        )
    )

    label, proba, _ = predict(imagen)

    assert label == "normal"
    assert proba == 90.0


@patch("src.integrator.grad_cam")
@patch("src.integrator.model_fun")
@patch("src.integrator.preprocess")
def test_predict_clase_viral(
    mock_preprocess,
    mock_model_fun,
    mock_grad_cam,
):
    """Verifica la clasificación viral."""
    imagen = crear_imagen()

    configurar_mocks(
        mock_preprocess,
        mock_model_fun,
        mock_grad_cam,
    )

    mock_model_fun.return_value = (
        lambda batch, training=False: np.array(
            [[0.05, 0.05, 0.90]],
            dtype=np.float32,
        )
    )

    label, proba, _ = predict(imagen)

    assert label == "viral"
    assert proba == 90.0


@patch("src.integrator.grad_cam")
@patch("src.integrator.model_fun")
@patch("src.integrator.preprocess")
def test_predict_probabilidad_cero(
    mock_preprocess,
    mock_model_fun,
    mock_grad_cam,
):
    """Verifica una probabilidad de cero."""
    imagen = crear_imagen()

    configurar_mocks(
        mock_preprocess,
        mock_model_fun,
        mock_grad_cam,
    )

    mock_model_fun.return_value = (
        lambda batch, training=False: np.array(
            [[0.00, 1.00, 0.00]],
            dtype=np.float32,
        )
    )

    label, proba, _ = predict(imagen)

    assert label == "normal"
    assert proba == 100.0


@patch("src.integrator.grad_cam")
@patch("src.integrator.model_fun")
@patch("src.integrator.preprocess")
def test_predict_probabilidad_maxima(
    mock_preprocess,
    mock_model_fun,
    mock_grad_cam,
):
    """Verifica una probabilidad máxima."""
    imagen = crear_imagen()

    configurar_mocks(
        mock_preprocess,
        mock_model_fun,
        mock_grad_cam,
    )

    mock_model_fun.return_value = (
        lambda batch, training=False: np.array(
            [[1.00, 0.00, 0.00]],
            dtype=np.float32,
        )
    )

    label, proba, _ = predict(imagen)

    assert label == "bacteriana"
    assert proba == 100.0


@patch("src.integrator.grad_cam")
@patch("src.integrator.model_fun")
@patch("src.integrator.preprocess")
def test_predict_probabilidad_decimal(
    mock_preprocess,
    mock_model_fun,
    mock_grad_cam,
):
    """Verifica una probabilidad decimal."""
    imagen = crear_imagen()

    configurar_mocks(
        mock_preprocess,
        mock_model_fun,
        mock_grad_cam,
    )

    mock_model_fun.return_value = (
        lambda batch, training=False: np.array(
            [[0.73, 0.20, 0.07]],
            dtype=np.float32,
        )
    )

    label, proba, _ = predict(imagen)

    assert label == "bacteriana"
    assert 72.0 <= proba <= 74.0


@patch("src.integrator.grad_cam")
@patch("src.integrator.model_fun")
@patch("src.integrator.preprocess")
def test_predict_probabilidad_rango(
    mock_preprocess,
    mock_model_fun,
    mock_grad_cam,
):
    """Verifica que la probabilidad esté entre 0 y 100."""
    imagen = crear_imagen()

    configurar_mocks(
        mock_preprocess,
        mock_model_fun,
        mock_grad_cam,
    )

    label, proba, _ = predict(imagen)

    assert 0.0 <= proba <= 100.0
    assert isinstance(label, str)


@patch("src.integrator.grad_cam")
@patch("src.integrator.model_fun")
@patch("src.integrator.preprocess")
def test_predict_heatmap_tipo_numpy(
    mock_preprocess,
    mock_model_fun,
    mock_grad_cam,
):
    """Verifica que el heatmap sea un NumPy array."""
    imagen = crear_imagen()

    configurar_mocks(
        mock_preprocess,
        mock_model_fun,
        mock_grad_cam,
    )

    _, _, heatmap = predict(imagen)

    assert isinstance(heatmap, np.ndarray)


@patch("src.integrator.grad_cam")
@patch("src.integrator.model_fun")
@patch("src.integrator.preprocess")
def test_predict_heatmap_dimensiones(
    mock_preprocess,
    mock_model_fun,
    mock_grad_cam,
):
    """Verifica las dimensiones del heatmap."""
    imagen = crear_imagen()

    configurar_mocks(
        mock_preprocess,
        mock_model_fun,
        mock_grad_cam,
    )

    _, _, heatmap = predict(imagen)

    assert heatmap.shape == (512, 512, 3)


@patch("src.integrator.grad_cam")
@patch("src.integrator.model_fun")
@patch("src.integrator.preprocess")
def test_predict_heatmap_uint8(
    mock_preprocess,
    mock_model_fun,
    mock_grad_cam,
):
    """Verifica que el heatmap sea uint8."""
    imagen = crear_imagen()

    configurar_mocks(
        mock_preprocess,
        mock_model_fun,
        mock_grad_cam,
    )

    _, _, heatmap = predict(imagen)

    assert heatmap.dtype == np.uint8


@patch("src.integrator.grad_cam")
@patch("src.integrator.model_fun")
@patch("src.integrator.preprocess")
def test_predict_heatmap_no_vacio(
    mock_preprocess,
    mock_model_fun,
    mock_grad_cam,
):
    """Verifica que el heatmap no esté vacío."""
    imagen = crear_imagen()

    configurar_mocks(
        mock_preprocess,
        mock_model_fun,
        mock_grad_cam,
    )

    _, _, heatmap = predict(imagen)

    assert heatmap.size > 0


@patch("src.integrator.grad_cam")
@patch("src.integrator.model_fun")
@patch("src.integrator.preprocess")
def test_predict_heatmap_no_nan(
    mock_preprocess,
    mock_model_fun,
    mock_grad_cam,
):
    """Verifica que el heatmap no contenga NaN."""
    imagen = crear_imagen()

    configurar_mocks(
        mock_preprocess,
        mock_model_fun,
        mock_grad_cam,
    )

    _, _, heatmap = predict(imagen)

    assert not np.isnan(heatmap).any()


@patch("src.integrator.grad_cam")
@patch("src.integrator.model_fun")
@patch("src.integrator.preprocess")
def test_predict_llama_preprocess(
    mock_preprocess,
    mock_model_fun,
    mock_grad_cam,
):
    """Verifica que predict llame a preprocess."""
    imagen = crear_imagen()

    configurar_mocks(
        mock_preprocess,
        mock_model_fun,
        mock_grad_cam,
    )

    predict(imagen)

    mock_preprocess.assert_called_once_with(imagen)


@patch("src.integrator.grad_cam")
@patch("src.integrator.model_fun")
@patch("src.integrator.preprocess")
def test_predict_llama_model(
    mock_preprocess,
    mock_model_fun,
    mock_grad_cam,
):
    """Verifica que predict cargue el modelo."""
    imagen = crear_imagen()

    configurar_mocks(
        mock_preprocess,
        mock_model_fun,
        mock_grad_cam,
    )

    predict(imagen)

    mock_model_fun.assert_called_once()


@patch("src.integrator.grad_cam")
@patch("src.integrator.model_fun")
@patch("src.integrator.preprocess")
def test_predict_llama_grad_cam(
    mock_preprocess,
    mock_model_fun,
    mock_grad_cam,
):
    """Verifica que predict llame a Grad-CAM."""
    imagen = crear_imagen()

    configurar_mocks(
        mock_preprocess,
        mock_model_fun,
        mock_grad_cam,
    )

    predict(imagen)

    mock_grad_cam.assert_called_once_with(imagen)


@patch("src.integrator.grad_cam")
@patch("src.integrator.model_fun")
@patch("src.integrator.preprocess")
def test_predict_imagen_pequena(
    mock_preprocess,
    mock_model_fun,
    mock_grad_cam,
):
    """Verifica predict con una imagen pequeña."""
    imagen = crear_imagen(20, 20)

    configurar_mocks(
        mock_preprocess,
        mock_model_fun,
        mock_grad_cam,
    )

    label, proba, heatmap = predict(imagen)

    assert isinstance(label, str)
    assert 0 <= proba <= 100
    assert heatmap.shape == (512, 512, 3)


@patch("src.integrator.grad_cam")
@patch("src.integrator.model_fun")
@patch("src.integrator.preprocess")
def test_predict_imagen_grande(
    mock_preprocess,
    mock_model_fun,
    mock_grad_cam,
):
    """Verifica predict con una imagen grande."""
    imagen = crear_imagen(1000, 1000)

    configurar_mocks(
        mock_preprocess,
        mock_model_fun,
        mock_grad_cam,
    )

    label, proba, heatmap = predict(imagen)

    assert isinstance(label, str)
    assert 0 <= proba <= 100
    assert heatmap.shape == (512, 512, 3)


@patch("src.integrator.grad_cam")
@patch("src.integrator.model_fun")
@patch("src.integrator.preprocess")
def test_predict_imagen_rectangular(
    mock_preprocess,
    mock_model_fun,
    mock_grad_cam,
):
    """Verifica predict con una imagen rectangular."""
    imagen = crear_imagen(100, 200)

    configurar_mocks(
        mock_preprocess,
        mock_model_fun,
        mock_grad_cam,
    )

    label, proba, heatmap = predict(imagen)

    assert isinstance(label, str)
    assert 0 <= proba <= 100
    assert heatmap.shape == (512, 512, 3)


@patch("src.integrator.grad_cam")
@patch("src.integrator.model_fun")
@patch("src.integrator.preprocess")
def test_predict_imagen_blanca(
    mock_preprocess,
    mock_model_fun,
    mock_grad_cam,
):
    """Verifica predict con una imagen completamente blanca."""
    imagen = np.full(
        (100, 100, 3),
        255,
        dtype=np.uint8,
    )

    configurar_mocks(
        mock_preprocess,
        mock_model_fun,
        mock_grad_cam,
    )

    label, proba, heatmap = predict(imagen)

    assert isinstance(label, str)
    assert 0 <= proba <= 100
    assert isinstance(heatmap, np.ndarray)
    assert heatmap.dtype == np.uint8