## Hola! Bienvenido a la herramienta para la detección rápida de neumonía

Hola y bienvenido a la herramienta para la detección de neumonía mediante Deep Learning aplicado al procesamiento de imágenes radiográficas de tórax.

El sistema permite procesar imágenes radiográficas en formato DICOM, JPG, JPEG y PNG, clasificándolas en tres categorías:

1. Neumonía Bacteriana

2. Sin Neumonía

3. Neumonía Viral

Aplicación de una técnica de explicación llamada Grad-CAM para resaltar con un mapa de calor las regiones relevantes de la imagen de entrada.

---

## Requerimientos del sistema

El sistema permite:

- Cargar una imagen radiográfica.
- Mostrar la imagen cargada.
- Ejecutar el modelo de Deep Learning.
- Mostrar la clase predicha.
- Mostrar la probabilidad asociada a la predicción.
- Generar un mapa de calor mediante Grad-CAM.
- Guardar el resultado de la predicción.
- Generar un reporte PDF con la información de la predicción.

---

## Tecnologías utilizadas

El proyecto utiliza las siguientes tecnologías:

- Python 3.13
- UV
- TensorFlow / Keras
- OpenCV
- Pillow
- Pydicom
- NumPy
- Pandas
- Matplotlib
- Tkinter
- Pytest
- Ruff
- Docker
- Git y GitHub

---

## Instalación de UV

Consulte la documentación oficial:

https://docs.astral.sh/uv/

---

## Clonar el proyecto

git clone <URL_DEL_REPOSITORIO> cd UAO-Neumonia

---

## Verificar Python 3.13

uv python list

La versión del proyecto está definida en el archivo:

.python-version

---

## Instalar las dependencias

uv sync

Para instalar también las dependencias utilizadas durante el desarrollo y las pruebas:

uv sync --dev

---

## Ejecución de la aplicación

Una vez configurado el entorno, ejecutar:

uv run python detector_neumonia.py

La aplicación abrirá la interfaz gráfica desarrollada con Tkinter.

---

## Uso de la interfaz gráfica

Cargar una imagen
Ingrese la identificación del paciente en el campo correspondiente y presione el botón:

Cargar Imagen
Seleccione una radiografía desde el explorador de archivos.

El sistema acepta los siguientes formatos:

- .dcm
- .jpg
- .jpeg
- .png

Una vez seleccionada, la imagen será mostrada en la interfaz.
Realizar la predicción

Presione el botón:

Predecir

El sistema realizará el procesamiento de la imagen y mostrará:

- La clase predicha.
- La probabilidad de la predicción.
- El mapa de calor Grad-CAM.
- Guardar el resultado

Presione el botón:
Guardar

para registrar la identificación del paciente, la clase predicha y la probabilidad en el historial del sistema.

Generar el reporte PDF
Presione:

PDF

para generar un reporte que contiene la predicción, la probabilidad y el mapa de calor Grad-CAM.

Limpiar la interfaz

Presione:
Borrar

para eliminar los datos de la predicción actual y permitir cargar una nueva imagen.

---

## Imágenes de prueba

El proyecto incluye imágenes DICOM de prueba dentro de:

PruebaImagenes/DICOM/

Actualmente se incluyen ejemplos correspondientes a imágenes normales y virales.

Estas imágenes permiten verificar la carga y el procesamiento de radiografías sin necesidad de utilizar archivos externos.

---

## Arquitectura MVC

La solución está organizada siguiendo el patrón de arquitectura Modelo-Vista-Controlador (MVC).

La separación de responsabilidades permite mantener el código modular, facilitar las pruebas y desacoplar la interfaz de la lógica de procesamiento y del modelo de Deep Learning.

MODELO:

El Modelo se encuentra en:

src/load_model.py
src/grad_cam.py
load_model.py

Es responsable de cargar el modelo entrenado: conv_MLP_84.h5

El modelo se carga utilizando TensorFlow/Keras y se utiliza para realizar la inferencia.

grad_cam.py es responsable de generar el mapa de calor Grad-CAM.

Este módulo obtiene las activaciones de la última capa convolucional de interés y calcula los gradientes asociados a la clase predicha para generar una representación visual de las regiones relevantes.

CONTROLADOR

El Controlador se encuentra en:

src/read_img.py
src/preprocess_img.py
src/integrator.py
read_img.py

Se encarga de leer las imágenes de entrada.

Permite procesar imágenes DICOM y convencionales, convirtiéndolas en arreglos NumPy y preparando una representación para su visualización en la interfaz.

preprocess_img.py realiza el preprocesamiento requerido antes de enviar la imagen al modelo.

Las operaciones realizadas son:

- Redimensionamiento a 512 x 512 píxeles.
- Conversión a escala de grises.
- Aplicación de CLAHE para mejorar el contraste.
- Normalización de los valores de píxel al rango 0-1.
- Conversión de la imagen al formato de batch utilizado por el modelo.

integrator.py integra el procesamiento de la imagen, la inferencia del modelo y la generación del mapa de calor.

La función principal de este módulo retorna:

- La clase predicha.
- La probabilidad.
- El mapa de calor generado mediante Grad-CAM.

VISTA / CLIENTE

La Vista se encuentra en:

detector_neumonia.py

Este módulo implementa la interfaz gráfica utilizando Tkinter.

La interfaz permite al usuario:

- Ingresar la identificación del paciente.
- Cargar una imagen.
- Ejecutar la predicción.
- Visualizar la radiografía.
- Visualizar el mapa Grad-CAM.
- Guardar los resultados.
- Generar el reporte PDF.
- Limpiar la información.

---

## Acerca del Modelo

La red neuronal convolucional implementada (CNN) es basada en el modelo implementado por F. Pasa, V.Golkov, F. Pfeifer, D. Cremers & D. Pfeifer
en su artículo Efcient Deep Network Architectures for Fast Chest X-Ray Tuberculosis Screening and Visualization.

Está compuesta por 5 bloques convolucionales, cada uno contiene 3 convoluciones; dos secuenciales y una conexión 'skip' que evita el desvanecimiento del gradiente a medida que se avanza en profundidad.
Con 16, 32, 48, 64 y 80 filtros de 3x3 para cada bloque respectivamente.

Después de cada bloque convolucional se encuentra una capa de max pooling y después de la última una capa de Average Pooling seguida por tres capas fully-connected (Dense) de 1024, 1024 y 3 neuronas respectivamente.

Para regularizar el modelo utilizamos 3 capas de Dropout al 20%; dos en los bloques 4 y 5 conv y otra después de la 1ra capa Dense.

---

## Acerca de Grad-CAM

Es una técnica utilizada para resaltar las regiones de una imagen que son importantes para la clasificación. Un mapeo de activaciones de clase para una categoría en particular indica las regiones de imagen relevantes utilizadas por la CNN para identificar esa categoría.

Grad-CAM realiza el cálculo del gradiente de la salida correspondiente a la clase a visualizar con respecto a las neuronas de una cierta capa de la CNN. Esto permite tener información de la importancia de cada neurona en el proceso de decisión de esa clase en particular. Una vez obtenidos estos pesos, se realiza una combinación lineal entre el mapa de activaciones de la capa y los pesos, de esta manera, se captura la importancia del mapa de activaciones para la clase en particular y se ve reflejado en la imagen de entrada como un mapa de calor con intensidades más altas en aquellas regiones relevantes para la red con las que clasificó la imagen en cierta categoría.

---

## Pruebas unitarias

El proyecto contiene dos pruebas unitarias implementadas mediante pytest.

Las pruebas se encuentran en:

tests/test_detector.py

Las funciones seleccionadas para las pruebas son:

test_read_jpg_file
test_predict

Ejecutar las pruebas:
uv run pytest -v

El resultado esperado es:

tests/test_detector.py::test_read_jpg_file PASSED
tests/test_detector.py::test_predict PASSED

============================== 2 passed ==============================

Las pruebas permiten verificar el funcionamiento de la lectura de imágenes y la lógica de integración de la predicción.

---

## Análisis de calidad del código

El proyecto utiliza Ruff para realizar verificaciones de estilo y calidad del código.

Ejecutar:

uv run ruff check .

El objetivo es mantener el código organizado y alineado con buenas prácticas de Python y PEP 8.

---

## Docker

El proyecto incluye un Dockerfile funcional para construir el entorno de ejecución.

Construcción de la imagen desde la raíz del proyecto:

docker build -t neumonia-base .

Verificar Python

docker run --rm neumonia-base python --version

La imagen debe utilizar Python 3.13.

Ejecutar las pruebas dentro de Docker
docker run --rm neumonia-base uv run pytest -v

---

## Ejecución de Tkinter con Docker y WSLg

Cuando se utiliza Docker Desktop con WSL2 y WSLg, la interfaz Tkinter requiere acceso a la pantalla del sistema.

En el entorno utilizado para el proyecto, la aplicación puede ejecutarse mediante:

docker run --rm -it \
  -e DISPLAY=$DISPLAY \
  -e WAYLAND_DISPLAY=$WAYLAND_DISPLAY \
  -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /mnt/wslg:/mnt/wslg \
  neumonia-base

La configuración puede variar dependiendo del entorno local de Docker Desktop y WSLg.

---

## Control de versiones

El proyecto utiliza Git para el control de versiones y GitHub como plataforma de alojamiento del repositorio.

El archivo .gitignore excluye archivos temporales, cachés, entornos virtuales, archivos generados por la aplicación y el modelo:

conv_MLP_84.h5

El archivo .dockerignore evita enviar al contexto de Docker archivos innecesarios como entornos virtuales, cachés y el directorio .git.

---

## Buenas prácticas aplicadas

Durante el desarrollo se consideran las siguientes prácticas:

- Arquitectura MVC.
- Separación de responsabilidades.
- Uso de Python 3.13.
- Gestión de dependencias mediante UV.
- Uso de Docker.
- Pruebas unitarias con pytest.
- Análisis de código con Ruff.
- Uso de .gitignore y .dockerignore.
- Documentación mediante README.
- Organización modular del código.
- Uso de docstrings para documentar las funciones.

---

## Archivo requirements.txt

El proyecto conserva un archivo requirements.txt generado mediante UV para facilitar la compatibilidad con herramientas que utilizan este formato.

La gestión oficial de dependencias del proyecto se realiza mediante:

pyproject.toml
uv.lock

Por lo tanto, la instalación y ejecución recomendadas utilizan UV:

uv sync
uv run python detector_neumonia.py
uv run pytest -v

---

## Proyecto original

El proyecto base fue desarrollado originalmente por:

Isabella Torres Revelo
https://github.com/isa-tr

Nicolas Diaz Salazar
https://github.com/nicolasdiazsalazar

La implementación actual corresponde a la adaptación y finalización académica del proyecto de acuerdo con los requerimientos establecidos para el curso.

Autor / Equipo

Proyecto académico desarrollado para la Universidad Autónoma de Occidente (UAO).

Repositorio:

<URL_DEL_REPOSITORIO>