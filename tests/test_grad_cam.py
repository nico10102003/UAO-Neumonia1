"""Pruebas unitarias para la generación de Grad-CAM."""

from unittest.mock import patch

import numpy as np
import tensorflow as tf

from src.grad_cam import grad_cam


def crear_imagen(alto=100, ancho=100):
    """Crea una imagen RGB de prueba."""
    return np.zeros(
        (alto, ancho, 3),
        dtype=np.uint8,
    )


def crear_modelo_prueba():
    """Crea un modelo TensorFlow pequeño para probar Grad-CAM."""
    inputs = tf.keras.Input(
        shape=(512, 512, 1)
    )

    x = tf.keras.layers.Conv2D(
        8,
        (3, 3),
        activation="relu",
        name="conv10_thisone",
    )(inputs)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    outputs = tf.keras.layers.Dense(
        3,
        activation="softmax",
    )(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
    )


def ejecutar_grad_cam():
    """Ejecuta Grad-CAM utilizando un modelo de prueba real."""
    imagen = crear_imagen()

    modelo = crear_modelo_prueba()

    with (
        patch(
            "src.grad_cam.preprocess"
        ) as mock_preprocess,
        patch(
            "src.grad_cam.model_fun",
            return_value=modelo,
        ),
    ):
        mock_preprocess.return_value = np.zeros(
            (1, 512, 512, 1),
            dtype=np.float32,
        )

        resultado = grad_cam(imagen)

    return resultado


def test_grad_cam_retorna_numpy():
    """Verifica que Grad-CAM retorne un arreglo NumPy."""
    resultado = ejecutar_grad_cam()

    assert isinstance(resultado, np.ndarray)


def test_grad_cam_retorna_tres_canales():
    """Verifica que la salida tenga tres canales."""
    resultado = ejecutar_grad_cam()

    assert resultado.shape[2] == 3


def test_grad_cam_tamano_alto():
    """Verifica que la salida tenga 512 píxeles de alto."""
    resultado = ejecutar_grad_cam()

    assert resultado.shape[0] == 512


def test_grad_cam_tamano_ancho():
    """Verifica que la salida tenga 512 píxeles de ancho."""
    resultado = ejecutar_grad_cam()

    assert resultado.shape[1] == 512


def test_grad_cam_dimensiones_completas():
    """Verifica las dimensiones completas del resultado."""
    resultado = ejecutar_grad_cam()

    assert resultado.shape == (512, 512, 3)


def test_grad_cam_tipo_uint8():
    """Verifica que el resultado sea uint8."""
    resultado = ejecutar_grad_cam()

    assert resultado.dtype == np.uint8


def test_grad_cam_valor_minimo():
    """Verifica que el resultado tenga valores válidos mínimos."""
    resultado = ejecutar_grad_cam()

    assert resultado.min() >= 0


def test_grad_cam_valor_maximo():
    """Verifica que el resultado tenga valores válidos máximos."""
    resultado = ejecutar_grad_cam()

    assert resultado.max() <= 255


def test_grad_cam_no_contiene_nan():
    """Verifica que el resultado no contenga NaN."""
    resultado = ejecutar_grad_cam()

    assert not np.isnan(resultado).any()


def test_grad_cam_no_contiene_inf():
    """Verifica que el resultado no contenga infinitos."""
    resultado = ejecutar_grad_cam()

    assert not np.isinf(resultado).any()


def test_grad_cam_llama_preprocess():
    """Verifica que se ejecute el preprocesamiento."""
    imagen = crear_imagen()

    modelo = crear_modelo_prueba()

    with (
        patch(
            "src.grad_cam.preprocess"
        ) as mock_preprocess,
        patch(
            "src.grad_cam.model_fun",
            return_value=modelo,
        ),
    ):
        mock_preprocess.return_value = np.zeros(
            (1, 512, 512, 1),
            dtype=np.float32,
        )

        grad_cam(imagen)

        mock_preprocess.assert_called_once_with(
            imagen
        )


def test_grad_cam_llama_model_fun():
    """Verifica que se cargue el modelo."""
    imagen = crear_imagen()

    modelo = crear_modelo_prueba()

    with (
        patch(
            "src.grad_cam.preprocess"
        ) as mock_preprocess,
        patch(
            "src.grad_cam.model_fun",
            return_value=modelo,
        ) as mock_model_fun,
    ):
        mock_preprocess.return_value = np.zeros(
            (1, 512, 512, 1),
            dtype=np.float32,
        )

        grad_cam(imagen)

        mock_model_fun.assert_called_once()


def test_grad_cam_busca_capa_conv():
    """Verifica que se busque la capa convolucional correcta."""
    imagen = crear_imagen()

    modelo = crear_modelo_prueba()

    with (
        patch(
            "src.grad_cam.preprocess"
        ) as mock_preprocess,
        patch(
            "src.grad_cam.model_fun",
            return_value=modelo,
        ),
    ):
        mock_preprocess.return_value = np.zeros(
            (1, 512, 512, 1),
            dtype=np.float32,
        )

        with patch.object(
            modelo,
            "get_layer",
            wraps=modelo.get_layer,
        ) as mock_get_layer:

            grad_cam(imagen)

            mock_get_layer.assert_called_once_with(
                "conv10_thisone"
            )


def test_grad_cam_crea_modelo_gradiente():
    """Verifica que se cree el modelo para Grad-CAM."""
    imagen = crear_imagen()

    modelo = crear_modelo_prueba()

    with (
        patch(
            "src.grad_cam.preprocess"
        ) as mock_preprocess,
        patch(
            "src.grad_cam.model_fun",
            return_value=modelo,
        ),
        patch(
            "src.grad_cam.tf.keras.models.Model",
            wraps=tf.keras.models.Model,
        ) as mock_model,
    ):
        mock_preprocess.return_value = np.zeros(
            (1, 512, 512, 1),
            dtype=np.float32,
        )

        grad_cam(imagen)

        assert mock_model.called


def test_grad_cam_imagen_pequena():
    """Verifica una imagen de entrada pequeña."""
    imagen = crear_imagen(20, 20)

    modelo = crear_modelo_prueba()

    with (
        patch(
            "src.grad_cam.preprocess",
            return_value=np.zeros(
                (1, 512, 512, 1),
                dtype=np.float32,
            ),
        ),
        patch(
            "src.grad_cam.model_fun",
            return_value=modelo,
        ),
    ):
        resultado = grad_cam(imagen)

        assert resultado.shape == (512, 512, 3)


def test_grad_cam_imagen_grande():
    """Verifica una imagen de entrada grande."""
    imagen = crear_imagen(1000, 1000)

    modelo = crear_modelo_prueba()

    with (
        patch(
            "src.grad_cam.preprocess",
            return_value=np.zeros(
                (1, 512, 512, 1),
                dtype=np.float32,
            ),
        ),
        patch(
            "src.grad_cam.model_fun",
            return_value=modelo,
        ),
    ):
        resultado = grad_cam(imagen)

        assert resultado.shape == (512, 512, 3)


def test_grad_cam_imagen_rectangular():
    """Verifica una imagen rectangular."""
    imagen = crear_imagen(100, 200)

    modelo = crear_modelo_prueba()

    with (
        patch(
            "src.grad_cam.preprocess",
            return_value=np.zeros(
                (1, 512, 512, 1),
                dtype=np.float32,
            ),
        ),
        patch(
            "src.grad_cam.model_fun",
            return_value=modelo,
        ),
    ):
        resultado = grad_cam(imagen)

        assert resultado.shape == (512, 512, 3)


def test_grad_cam_imagen_blanca():
    """Verifica una imagen completamente blanca."""
    imagen = np.full(
        (100, 100, 3),
        255,
        dtype=np.uint8,
    )

    modelo = crear_modelo_prueba()

    with (
        patch(
            "src.grad_cam.preprocess",
            return_value=np.zeros(
                (1, 512, 512, 1),
                dtype=np.float32,
            ),
        ),
        patch(
            "src.grad_cam.model_fun",
            return_value=modelo,
        ),
    ):
        resultado = grad_cam(imagen)

        assert resultado.shape == (512, 512, 3)
        assert resultado.dtype == np.uint8


def test_grad_cam_imagen_aleatoria():
    """Verifica una imagen con valores aleatorios."""
    rng = np.random.default_rng(42)

    imagen = rng.integers(
        0,
        256,
        size=(100, 100, 3),
        dtype=np.uint8,
    )

    modelo = crear_modelo_prueba()

    with (
        patch(
            "src.grad_cam.preprocess",
            return_value=np.zeros(
                (1, 512, 512, 1),
                dtype=np.float32,
            ),
        ),
        patch(
            "src.grad_cam.model_fun",
            return_value=modelo,
        ),
    ):
        resultado = grad_cam(imagen)

        assert resultado.shape == (512, 512, 3)
        assert resultado.dtype == np.uint8


def test_grad_cam_salida_rgb():
    """Verifica que la salida tenga formato RGB."""
    resultado = ejecutar_grad_cam()

    assert resultado.ndim == 3
    assert resultado.shape[-1] == 3


def test_grad_cam_resultado_no_vacio():
    """Verifica que el resultado contenga datos."""
    resultado = ejecutar_grad_cam()

    assert resultado.size > 0


def test_grad_cam_resultado_contiguo():
    """Verifica que exista información válida en el resultado."""
    resultado = ejecutar_grad_cam()

    assert resultado.nbytes > 0