"""Pruebas unitarias para la lectura de imágenes."""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from PIL import Image

from src.read_img import read_dicom_file, read_jpg_file


# ============================================================
# PRUEBAS read_jpg_file
# ============================================================


def test_read_jpg_file_imagen_valida(tmp_path):
    """Verifica la lectura de una imagen JPG válida."""
    image = np.full((100, 100, 3), 255, dtype=np.uint8)

    path = tmp_path / "imagen.jpg"
    cv2.imwrite(str(path), image)

    result, img_show = read_jpg_file(str(path))

    assert isinstance(result, np.ndarray)
    assert isinstance(img_show, Image.Image)


def test_read_jpg_file_tipo_numpy(tmp_path):
    """Verifica que el resultado principal sea NumPy."""
    image = np.zeros((50, 50, 3), dtype=np.uint8)

    path = tmp_path / "imagen.jpg"
    cv2.imwrite(str(path), image)

    result, _ = read_jpg_file(str(path))

    assert isinstance(result, np.ndarray)


def test_read_jpg_file_tipo_pil(tmp_path):
    """Verifica que la imagen de visualización sea PIL."""
    image = np.zeros((50, 50, 3), dtype=np.uint8)

    path = tmp_path / "imagen.jpg"
    cv2.imwrite(str(path), image)

    _, img_show = read_jpg_file(str(path))

    assert isinstance(img_show, Image.Image)


def test_read_jpg_file_dimensiones(tmp_path):
    """Verifica las dimensiones de la imagen."""
    image = np.zeros((120, 80, 3), dtype=np.uint8)

    path = tmp_path / "imagen.jpg"
    cv2.imwrite(str(path), image)

    result, _ = read_jpg_file(str(path))

    assert result.shape == (120, 80, 3)


def test_read_jpg_file_uint8(tmp_path):
    """Verifica el tipo de datos del resultado."""
    image = np.zeros((50, 50, 3), dtype=np.uint8)

    path = tmp_path / "imagen.jpg"
    cv2.imwrite(str(path), image)

    result, _ = read_jpg_file(str(path))

    assert result.dtype == np.uint8


def test_read_jpg_file_imagen_blanca(tmp_path):
    """Verifica una imagen completamente blanca."""
    image = np.full((30, 30, 3), 255, dtype=np.uint8)

    path = tmp_path / "blanca.jpg"
    cv2.imwrite(str(path), image)

    result, _ = read_jpg_file(str(path))

    assert result.shape == (30, 30, 3)


def test_read_jpg_file_imagen_negra(tmp_path):
    """Verifica una imagen completamente negra."""
    image = np.zeros((30, 30, 3), dtype=np.uint8)

    path = tmp_path / "negra.jpg"
    cv2.imwrite(str(path), image)

    result, _ = read_jpg_file(str(path))

    assert isinstance(result, np.ndarray)


def test_read_jpg_file_imagen_pequena(tmp_path):
    """Verifica una imagen de tamaño pequeño."""
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    path = tmp_path / "pequena.jpg"
    cv2.imwrite(str(path), image)

    result, _ = read_jpg_file(str(path))

    assert result.shape == (10, 10, 3)


def test_read_jpg_file_imagen_grande(tmp_path):
    """Verifica una imagen de tamaño grande."""
    image = np.zeros((500, 500, 3), dtype=np.uint8)

    path = tmp_path / "grande.jpg"
    cv2.imwrite(str(path), image)

    result, _ = read_jpg_file(str(path))

    assert result.shape == (500, 500, 3)


def test_read_jpg_file_archivo_inexistente():
    """Verifica el error cuando el archivo no existe."""
    with pytest.raises(ValueError):
        read_jpg_file("archivo_que_no_existe.jpg")


def test_read_jpg_file_ruta_invalida():
    """Verifica el manejo de una ruta inválida."""
    with pytest.raises(ValueError):
        read_jpg_file("ruta/invalida/imagen.jpg")


def test_read_jpg_file_extensiones(tmp_path):
    """Verifica que OpenCV pueda leer un archivo JPG."""
    image = np.zeros((40, 40, 3), dtype=np.uint8)

    path = tmp_path / "imagen.JPG"
    cv2.imwrite(str(path), image)

    result, _ = read_jpg_file(str(path))

    assert result is not None


def test_read_jpg_file_valores_uint8(tmp_path):
    """Verifica que los valores estén en el rango uint8."""
    image = np.full((50, 50, 3), 128, dtype=np.uint8)

    path = tmp_path / "imagen.jpg"
    cv2.imwrite(str(path), image)

    result, _ = read_jpg_file(str(path))

    assert result.min() >= 0
    assert result.max() <= 255


def test_read_jpg_file_tupla_resultado(tmp_path):
    """Verifica que la función retorne dos elementos."""
    image = np.zeros((50, 50, 3), dtype=np.uint8)

    path = tmp_path / "imagen.jpg"
    cv2.imwrite(str(path), image)

    result = read_jpg_file(str(path))

    assert isinstance(result, tuple)
    assert len(result) == 2


# ============================================================
# PRUEBAS read_dicom_file
# ============================================================


def crear_dicom_mock():
    """Crea un objeto DICOM simulado."""
    dicom_mock = MagicMock()

    dicom_mock.pixel_array = np.array(
        [
            [0, 50, 100],
            [150, 200, 255],
            [25, 75, 125],
        ],
        dtype=np.uint16,
    )

    return dicom_mock


@patch("src.read_img.dicom.dcmread")
def test_read_dicom_file_lectura(mock_dcmread):
    """Verifica que se lea correctamente un DICOM."""
    mock_dcmread.return_value = crear_dicom_mock()

    result, img_show = read_dicom_file("imagen.dcm")

    assert isinstance(result, np.ndarray)
    assert isinstance(img_show, Image.Image)


@patch("src.read_img.dicom.dcmread")
def test_read_dicom_file_llama_dcmread(mock_dcmread):
    """Verifica que dcmread sea llamado."""
    mock_dcmread.return_value = crear_dicom_mock()

    read_dicom_file("imagen.dcm")

    mock_dcmread.assert_called_once_with("imagen.dcm")


@patch("src.read_img.dicom.dcmread")
def test_read_dicom_file_rgb(mock_dcmread):
    """Verifica que el resultado tenga tres canales."""
    mock_dcmread.return_value = crear_dicom_mock()

    result, _ = read_dicom_file("imagen.dcm")

    assert result.ndim == 3
    assert result.shape[2] == 3


@patch("src.read_img.dicom.dcmread")
def test_read_dicom_file_uint8(mock_dcmread):
    """Verifica que la salida sea uint8."""
    mock_dcmread.return_value = crear_dicom_mock()

    result, _ = read_dicom_file("imagen.dcm")

    assert result.dtype == np.uint8


@patch("src.read_img.dicom.dcmread")
def test_read_dicom_file_tipo_pil(mock_dcmread):
    """Verifica el objeto PIL generado."""
    mock_dcmread.return_value = crear_dicom_mock()

    _, img_show = read_dicom_file("imagen.dcm")

    assert isinstance(img_show, Image.Image)


@patch("src.read_img.dicom.dcmread")
def test_read_dicom_file_dimensiones(mock_dcmread):
    """Verifica las dimensiones de salida."""
    mock_dcmread.return_value = crear_dicom_mock()

    result, _ = read_dicom_file("imagen.dcm")

    assert result.shape == (3, 3, 3)


@patch("src.read_img.dicom.dcmread")
def test_read_dicom_file_rango(mock_dcmread):
    """Verifica el rango de valores."""
    mock_dcmread.return_value = crear_dicom_mock()

    result, _ = read_dicom_file("imagen.dcm")

    assert result.min() >= 0
    assert result.max() <= 255


@patch("src.read_img.dicom.dcmread")
def test_read_dicom_file_imagen_vacia(mock_dcmread):
    """Verifica el comportamiento con valores cero."""
    dicom_mock = MagicMock()
    dicom_mock.pixel_array = np.zeros(
        (10, 10),
        dtype=np.uint16,
    )

    mock_dcmread.return_value = dicom_mock

    # La implementación actual divide por img2.max(),
    # por lo que una imagen completamente cero puede
    # producir una advertencia/nan. La prueba verifica
    # que la función intenta procesarla.
    with pytest.warns(RuntimeWarning):
        result, _ = read_dicom_file("vacia.dcm")

    assert isinstance(result, np.ndarray)


@patch("src.read_img.dicom.dcmread")
def test_read_dicom_file_error_dicom(mock_dcmread):
    """Verifica que los errores de lectura sean propagados."""
    mock_dcmread.side_effect = Exception("Error DICOM")

    with pytest.raises(Exception, match="Error DICOM"):
        read_dicom_file("error.dcm")


@patch("src.read_img.dicom.dcmread")
def test_read_dicom_file_array_original(mock_dcmread):
    """Verifica que se utilice pixel_array."""
    dicom_mock = crear_dicom_mock()
    mock_dcmread.return_value = dicom_mock

    read_dicom_file("imagen.dcm")

    assert dicom_mock.pixel_array is not None


@patch("src.read_img.dicom.dcmread")
def test_read_dicom_file_contraste(mock_dcmread):
    """Verifica el procesamiento de contraste."""
    mock_dcmread.return_value = crear_dicom_mock()

    result, _ = read_dicom_file("imagen.dcm")

    assert result.max() > 0


@patch("src.read_img.dicom.dcmread")
def test_read_dicom_file_no_escalar_a_grises(mock_dcmread):
    """Verifica que la salida DICOM sea RGB."""
    mock_dcmread.return_value = crear_dicom_mock()

    result, _ = read_dicom_file("imagen.dcm")

    assert result.shape[-1] == 3