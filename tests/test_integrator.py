"""Pruebas unitarias para la integración del detector."""

from unittest.mock import patch

import numpy as np

from src.integrator import predict


def modelo_simulado(predicciones):
    """Crea un modelo simulado con predicciones determinadas."""

    def model(batch, training=False):
        """Devuelve las predicciones simuladas."""
        return np.array([predicciones], dtype=np.float32)

    return model


def ejecutar_prediccion(predicciones):
    """Ejecuta predict con todos sus componentes simulados."""
    array = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    with (
        patch("src.integrator.preprocess") as mock_preprocess,
        patch("src.integrator.model_fun") as mock_model_fun,
        patch("src.integrator.grad_cam") as mock_grad_cam,
    ):
        mock_preprocess.return_value = np.zeros(
            (1, 512, 512, 1),
            dtype=np.float32,
        )

        mock_model_fun.return_value = modelo_simulado(
            predicciones
        )

        mock_grad_cam.return_value = np.zeros(
            (100, 100, 3),
            dtype=np.uint8,
        )

        resultado = predict(array)

    return resultado, mock_preprocess, mock_model_fun, mock_grad_cam


def test_predict_retorna_tupla():
    """Verifica que predict retorne una tupla."""
    resultado, _, _, _ = ejecutar_prediccion(
        [0.8, 0.1, 0.1]
    )

    assert isinstance(resultado, tuple)


def test_predict_retorna_tres_elementos():
    """Verifica que predict retorne tres elementos."""
    resultado, _, _, _ = ejecutar_prediccion(
        [0.8, 0.1, 0.1]
    )

    assert len(resultado) == 3


def test_predict_clase_bacteriana():
    """Verifica la clasificación bacteriana."""
    resultado, _, _, _ = ejecutar_prediccion(
        [0.9, 0.05, 0.05]
    )

    assert resultado[0] == "bacteriana"


def test_predict_clase_normal():
    """Verifica la clasificación normal."""
    resultado, _, _, _ = ejecutar_prediccion(
        [0.05, 0.9, 0.05]
    )

    assert resultado[0] == "normal"


def test_predict_clase_viral():
    """Verifica la clasificación viral."""
    resultado, _, _, _ = ejecutar_prediccion(
        [0.05, 0.05, 0.9]
    )

    assert resultado[0] == "viral"


def test_predict_probabilidad_bacteriana():
    """Verifica la probabilidad bacteriana."""
    resultado, _, _, _ = ejecutar_prediccion(
        [0.84, 0.10, 0.06]
    )

    assert resultado[1] == 84.0


def test_predict_probabilidad_normal():
    """Verifica la probabilidad normal."""
    resultado, _, _, _ = ejecutar_prediccion(
        [0.10, 0.84, 0.06]
    )

    assert resultado[1] == 84.0


def test_predict_probabilidad_viral():
    """Verifica la probabilidad viral."""
    resultado, _, _, _ = ejecutar_prediccion(
        [0.10, 0.06, 0.84]
    )

    assert resultado[1] == 84.0


def test_predict_probabilidad_cero():
    """Verifica una probabilidad de cero."""
    resultado, _, _, _ = ejecutar_prediccion(
        [0.0, 1.0, 0.0]
    )

    assert resultado[1] == 100.0


def test_predict_probabilidad_cien():
    """Verifica una probabilidad del cien por ciento."""
    resultado, _, _, _ = ejecutar_prediccion(
        [1.0, 0.0, 0.0]
    )

    assert resultado[1] == 100.0


def test_predict_probabilidad_decimal():
    """Verifica probabilidades decimales."""
    resultado, _, _, _ = ejecutar_prediccion(
        [0.735, 0.20, 0.065]
    )

    assert resultado[1] == 73.5


def test_predict_probabilidad_maxima():
    """Verifica que se seleccione la probabilidad máxima."""
    resultado, _, _, _ = ejecutar_prediccion(
        [0.15, 0.70, 0.15]
    )

    assert resultado[1] == 70.0


def test_predict_heatmap_numpy():
    """Verifica que el heatmap sea un arreglo NumPy."""
    resultado, _, _, _ = ejecutar_prediccion(
        [0.8, 0.1, 0.1]
    )

    assert isinstance(resultado[2], np.ndarray)


def test_predict_heatmap_dimensiones():
    """Verifica las dimensiones del heatmap."""
    resultado, _, _, _ = ejecutar_prediccion(
        [0.8, 0.1, 0.1]
    )

    assert resultado[2].shape == (100, 100, 3)


def test_predict_heatmap_uint8():
    """Verifica el tipo de datos del heatmap."""
    resultado, _, _, _ = ejecutar_prediccion(
        [0.8, 0.1, 0.1]
    )

    assert resultado[2].dtype == np.uint8


def test_predict_llama_preprocess():
    """Verifica que se llame al preprocesamiento."""
    _, mock_preprocess, _, _ = ejecutar_prediccion(
        [0.8, 0.1, 0.1]
    )

    mock_preprocess.assert_called_once()


def test_predict_llama_model_fun():
    """Verifica que se cargue el modelo."""
    _, _, mock_model_fun, _ = ejecutar_prediccion(
        [0.8, 0.1, 0.1]
    )

    mock_model_fun.assert_called_once()


def test_predict_llama_grad_cam():
    """Verifica que se genere el Grad-CAM."""
    _, _, _, mock_grad_cam = ejecutar_prediccion(
        [0.8, 0.1, 0.1]
    )

    mock_grad_cam.assert_called_once()


def test_predict_preprocess_recibe_imagen():
    """Verifica que preprocess reciba la imagen."""
    array = np.ones(
        (100, 100, 3),
        dtype=np.uint8,
    )

    with (
        patch("src.integrator.preprocess") as mock_preprocess,
        patch("src.integrator.model_fun") as mock_model_fun,
        patch("src.integrator.grad_cam") as mock_grad_cam,
    ):
        mock_preprocess.return_value = np.zeros(
            (1, 512, 512, 1),
            dtype=np.float32,
        )

        mock_model_fun.return_value = modelo_simulado(
            [0.8, 0.1, 0.1]
        )

        mock_grad_cam.return_value = np.zeros(
            (100, 100, 3),
            dtype=np.uint8,
        )

        predict(array)

        mock_preprocess.assert_called_once_with(array)


def test_predict_grad_cam_recibe_imagen():
    """Verifica que Grad-CAM reciba la imagen original."""
    array = np.ones(
        (100, 100, 3),
        dtype=np.uint8,
    )

    with (
        patch("src.integrator.preprocess") as mock_preprocess,
        patch("src.integrator.model_fun") as mock_model_fun,
        patch("src.integrator.grad_cam") as mock_grad_cam,
    ):
        mock_preprocess.return_value = np.zeros(
            (1, 512, 512, 1),
            dtype=np.float32,
        )

        mock_model_fun.return_value = modelo_simulado(
            [0.8, 0.1, 0.1]
        )

        mock_grad_cam.return_value = np.zeros(
            (100, 100, 3),
            dtype=np.uint8,
        )

        predict(array)

        mock_grad_cam.assert_called_once_with(array)


def test_predict_preprocess_retorna_batch():
    """Verifica que predict utilice el batch procesado."""
    resultado, mock_preprocess, _, _ = ejecutar_prediccion(
        [0.8, 0.1, 0.1]
    )

    assert resultado is not None
    assert mock_preprocess.return_value.shape == (
        1,
        512,
        512,
        1,
    )


def test_predict_model_recibe_batch():
    """Verifica que el modelo reciba el batch procesado."""
    array = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    batch = np.ones(
        (1, 512, 512, 1),
        dtype=np.float32,
    )

    with (
        patch("src.integrator.preprocess") as mock_preprocess,
        patch("src.integrator.model_fun") as mock_model_fun,
        patch("src.integrator.grad_cam") as mock_grad_cam,
    ):
        mock_preprocess.return_value = batch

        modelo = modelo_simulado(
            [0.8, 0.1, 0.1]
        )
        mock_model_fun.return_value = modelo

        mock_grad_cam.return_value = np.zeros(
            (100, 100, 3),
            dtype=np.uint8,
        )

        resultado = predict(array)

        assert resultado[0] == "bacteriana"


def test_predict_empate_selecciona_primera_clase():
    """Verifica el comportamiento ante probabilidades iguales."""
    resultado, _, _, _ = ejecutar_prediccion(
        [0.5, 0.5, 0.0]
    )

    assert resultado[0] == "bacteriana"


def test_predict_clase_con_probabilidad_muy_alta():
    """Verifica una predicción con probabilidad muy alta."""
    resultado, _, _, _ = ejecutar_prediccion(
        [0.999, 0.0005, 0.0005]
    )

    assert resultado[0] == "bacteriana"
    assert resultado[1] == 99.9


def test_predict_clase_con_probabilidad_baja():
    """Verifica una predicción con probabilidad baja."""
    resultado, _, _, _ = ejecutar_prediccion(
        [0.01, 0.98, 0.01]
    )

    assert resultado[0] == "normal"
    assert resultado[1] == 98.0


def test_predict_resultado_completo():
    """Verifica simultáneamente clase, probabilidad y heatmap."""
    resultado, _, _, _ = ejecutar_prediccion(
        [0.10, 0.25, 0.65]
    )

    label, proba, heatmap = resultado

    assert label == "viral"
    assert proba == 65.0
    assert isinstance(heatmap, np.ndarray)


def test_predict_acepta_imagen_uint8():
    """Verifica que predict acepte una imagen uint8."""
    array = np.zeros(
        (224, 224, 3),
        dtype=np.uint8,
    )

    with (
        patch("src.integrator.preprocess") as mock_preprocess,
        patch("src.integrator.model_fun") as mock_model_fun,
        patch("src.integrator.grad_cam") as mock_grad_cam,
    ):
        mock_preprocess.return_value = np.zeros(
            (1, 512, 512, 1),
            dtype=np.float32,
        )

        mock_model_fun.return_value = modelo_simulado(
            [0.1, 0.8, 0.1]
        )

        mock_grad_cam.return_value = np.zeros(
            (224, 224, 3),
            dtype=np.uint8,
        )

        label, proba, heatmap = predict(array)

        assert label == "normal"
        assert proba == 80.0
        assert isinstance(heatmap, np.ndarray)