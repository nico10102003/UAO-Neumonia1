"""Pruebas unitarias para el preprocesamiento de imágenes."""

import cv2
import numpy as np
import pytest

from src.preprocess_img import preprocess


# ============================================================
# PRUEBAS BÁSICAS
# ============================================================


def test_preprocess_retorna_numpy():
    """Verifica que preprocess retorne un arreglo NumPy."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    result = preprocess(image)

    assert isinstance(result, np.ndarray)


def test_preprocess_dimension():
    """Verifica las dimensiones finales."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    result = preprocess(image)

    assert result.shape == (1, 512, 512, 1)


def test_preprocess_tamano_512():
    """Verifica que la imagen tenga tamaño 512x512."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    result = preprocess(image)

    assert result.shape[1] == 512
    assert result.shape[2] == 512


def test_preprocess_un_canal():
    """Verifica que la imagen final tenga un solo canal."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    result = preprocess(image)

    assert result.shape[3] == 1


def test_preprocess_un_batch():
    """Verifica que exista un único elemento en el batch."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    result = preprocess(image)

    assert result.shape[0] == 1


def test_preprocess_float32():
    """Verifica que el resultado sea float32."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    result = preprocess(image)

    assert result.dtype == np.float32


def test_preprocess_valores_minimo():
    """Verifica que los valores no sean menores que cero."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    result = preprocess(image)

    assert result.min() >= 0.0


def test_preprocess_valores_maximo():
    """Verifica que los valores no superen uno."""
    image = np.full(
        (100, 100, 3),
        255,
        dtype=np.uint8,
    )

    result = preprocess(image)

    assert result.max() <= 1.0


def test_preprocess_normalizacion():
    """Verifica que los valores estén normalizados."""
    image = np.full(
        (100, 100, 3),
        128,
        dtype=np.uint8,
    )

    result = preprocess(image)

    assert result.min() >= 0
    assert result.max() <= 1


# ============================================================
# DIFERENTES TAMAÑOS
# ============================================================


def test_preprocess_imagen_pequena():
    """Verifica una imagen pequeña."""
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    result = preprocess(image)

    assert result.shape == (1, 512, 512, 1)


def test_preprocess_imagen_mediana():
    """Verifica una imagen de tamaño mediano."""
    image = np.zeros((256, 256, 3), dtype=np.uint8)

    result = preprocess(image)

    assert result.shape == (1, 512, 512, 1)


def test_preprocess_imagen_grande():
    """Verifica una imagen grande."""
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)

    result = preprocess(image)

    assert result.shape == (1, 512, 512, 1)


def test_preprocess_imagen_rectangular():
    """Verifica una imagen rectangular."""
    image = np.zeros((200, 400, 3), dtype=np.uint8)

    result = preprocess(image)

    assert result.shape == (1, 512, 512, 1)


# ============================================================
# DIFERENTES VALORES
# ============================================================


def test_preprocess_imagen_negra():
    """Verifica una imagen completamente negra."""
    image = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    result = preprocess(image)

    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 512, 512, 1)


def test_preprocess_imagen_blanca():
    """Verifica una imagen completamente blanca."""
    image = np.full(
        (100, 100, 3),
        255,
        dtype=np.uint8,
    )

    result = preprocess(image)

    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 512, 512, 1)


def test_preprocess_imagen_gris():
    """Verifica una imagen de intensidad intermedia."""
    image = np.full(
        (100, 100, 3),
        128,
        dtype=np.uint8,
    )

    result = preprocess(image)

    assert result.shape == (1, 512, 512, 1)


def test_preprocess_imagen_aleatoria():
    """Verifica una imagen con valores aleatorios."""
    rng = np.random.default_rng(42)

    image = rng.integers(
        0,
        256,
        size=(100, 100, 3),
        dtype=np.uint8,
    )

    result = preprocess(image)

    assert result.shape == (1, 512, 512, 1)


def test_preprocess_no_contiene_nan():
    """Verifica que el resultado no contenga NaN."""
    rng = np.random.default_rng(42)

    image = rng.integers(
        0,
        256,
        size=(100, 100, 3),
        dtype=np.uint8,
    )

    result = preprocess(image)

    assert not np.isnan(result).any()


def test_preprocess_no_contiene_inf():
    """Verifica que el resultado no contenga infinitos."""
    rng = np.random.default_rng(42)

    image = rng.integers(
        0,
        256,
        size=(100, 100, 3),
        dtype=np.uint8,
    )

    result = preprocess(image)

    assert not np.isinf(result).any()


# ============================================================
# COMPORTAMIENTO DE OPENCV
# ============================================================


def test_preprocess_convierte_a_grises():
    """Verifica indirectamente la conversión a escala de grises."""
    image = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    image[:, :, 0] = 255

    result = preprocess(image)

    assert result.shape[-1] == 1


def test_preprocess_redimensionamiento():
    """Verifica que la salida siempre sea 512x512."""
    image = np.zeros(
        (75, 125, 3),
        dtype=np.uint8,
    )

    result = preprocess(image)

    assert result.shape[1:3] == (512, 512)


def test_preprocess_reproducible():
    """Verifica que la misma entrada produzca la misma salida."""
    rng = np.random.default_rng(123)

    image = rng.integers(
        0,
        256,
        size=(100, 100, 3),
        dtype=np.uint8,
    )

    result1 = preprocess(image)
    result2 = preprocess(image)

    assert np.array_equal(result1, result2)


def test_preprocess_copia_independiente():
    """Verifica que el resultado sea un arreglo independiente."""
    image = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    result = preprocess(image)

    assert result is not image