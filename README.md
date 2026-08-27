## Hola! Bienvenido a la herramienta para la detección rápida de neumonía

Hola y bienvenido a esta herramienta desarrollada para la detección y clasificación de neumonía mediante técnicas de Deep Learning aplicadas al procesamiento de imágenes radiográficas de tórax.

El sistema permite procesar imágenes médicas en formatos DICOM, JPG, JPEG y PNG, realizando un proceso de lectura y preprocesamiento para posteriormente analizar la imagen mediante un modelo de aprendizaje profundo. El modelo clasifica las radiografías en tres categorías:

1. Neumonía Bacteriana
2. Sin Neumonía
3. Neumonía Viral

Como complemento a la predicción, la aplicación incorpora la técnica de Grad-CAM, que genera un mapa de calor para resaltar las regiones de la radiografía que tuvieron mayor influencia en la decisión del modelo. Esto permite obtener una representación visual de la predicción y facilita la interpretación de los resultados.

El proyecto está organizado de manera modular, incluyendo componentes para la lectura y procesamiento de imágenes, carga del modelo, clasificación y generación de mapas de explicación, además de pruebas unitarias para verificar el correcto funcionamiento de sus principales componentes.

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
El proyecto utiliza las siguientes tecnologías y herramientas:

* **Python 3.13** — Lenguaje principal de programación.
* **UV** — Gestión de dependencias y ejecución del entorno de Python.
* **TensorFlow / Keras** — Desarrollo y ejecución del modelo de Deep Learning.
* **Grad-CAM** — Técnica de explicabilidad para visualizar las regiones relevantes de las radiografías.
* **OpenCV** — Lectura y procesamiento de imágenes.
* **Pillow (PIL)** — Manipulación y conversión de imágenes.
* **Pydicom** — Lectura y procesamiento de imágenes médicas en formato DICOM.
* **NumPy** — Procesamiento y manipulación de arreglos numéricos e imágenes.
* **Pandas** — Manejo y análisis de datos.
* **Matplotlib** — Generación y visualización de gráficos.
* **Tkinter** — Desarrollo de la interfaz gráfica de la aplicación.
* **Pytest** — Automatización de pruebas unitarias.
* **Ruff** — Análisis estático, linting y validación del código Python.
* **Docker** — Contenerización y configuración reproducible del entorno de ejecución.
* **Git** — Control de versiones del código fuente.
* **GitHub** — Gestión y almacenamiento del repositorio.
* **Visual Studio Code** — Entorno de desarrollo utilizado para la implementación y mantenimiento del proyecto.

---

## Instalación de UV

Consulte la documentación oficial:

[Documentación oficial de UV](https://docs.astral.sh/uv/?utm_source=chatgpt.com)

---

## Clonar el proyecto

Clonar el repositorio y acceder a la carpeta del proyecto:

```bash
git clone <URL_DEL_REPOSITORIO>
cd UAO-Neumonia1
```

---

## Verificar Python 3.13

Verificar las versiones de Python disponibles mediante:

```bash
uv python list
```

La versión utilizada por el proyecto está definida en el archivo:

```text
.python-version
```

---

## Crear y sincronizar el entorno

Para configurar el entorno virtual e instalar las dependencias definidas por el proyecto:

```bash
uv sync
```

Para instalar también las dependencias utilizadas durante el desarrollo y las pruebas:

```bash
uv sync --dev
```

UV se encarga de administrar el entorno virtual y las dependencias necesarias para ejecutar el proyecto.

---

## Ejecución de la aplicación

Una vez configurado el entorno, ejecutar:

```bash
uv run python detector_neumonia.py
```

La aplicación abrirá la interfaz gráfica desarrollada con **Tkinter**.

---

## Uso de la interfaz gráfica

### Cargar una imagen

Ingrese la identificación del paciente en el campo correspondiente y presione el botón:

**Cargar Imagen**

Seleccione una radiografía desde el explorador de archivos.

El sistema acepta los siguientes formatos:

* `.dcm`
* `.jpg`
* `.jpeg`
* `.png`

Una vez seleccionada, la imagen será mostrada en la interfaz.

### Realizar la predicción

Presione el botón:

**Predecir**

El sistema realizará el procesamiento de la imagen mediante el modelo de Deep Learning y mostrará:

* La clase predicha.
* La probabilidad de la predicción.
* El mapa de calor generado mediante **Grad-CAM**.

Las categorías de clasificación son:

* **Neumonía Bacteriana**
* **Sin Neumonía**
* **Neumonía Viral**

### Guardar el resultado

Presione el botón:

**Guardar**

para registrar la identificación del paciente, la clase predicha y la probabilidad en el historial del sistema.

### Generar el reporte PDF

Presione:

**PDF**

para generar un reporte que contiene información de la predicción, la probabilidad obtenida y el mapa de calor Grad-CAM.

### Limpiar la interfaz

Presione:

**Borrar**

para eliminar los datos de la predicción actual y permitir cargar una nueva imagen.

---

## Procesamiento de las imágenes

Antes de realizar la predicción, las imágenes pasan por una etapa de preprocesamiento para adaptarlas a las características requeridas por el modelo.

El proceso incluye:

1. Lectura de la imagen.
2. Conversión a escala de grises cuando corresponde.
3. Redimensionamiento de la imagen.
4. Normalización de los valores de los píxeles.
5. Preparación de la dimensión de entrada del modelo.
6. Envío de la imagen procesada al modelo de Deep Learning.

Este proceso permite mantener un formato de entrada consistente independientemente del tamaño original de la radiografía.

---

## Clasificación mediante Deep Learning

El sistema utiliza un modelo de **Deep Learning basado en TensorFlow/Keras** para analizar las imágenes radiográficas.

El modelo recibe la imagen preprocesada y genera las probabilidades correspondientes a las tres categorías de clasificación:

```text
Bacteriana
Normal
Viral
```

La clase con mayor probabilidad es seleccionada como resultado de la predicción.

---

## Explicabilidad mediante Grad-CAM

Además de generar una clasificación, el sistema utiliza **Grad-CAM (Gradient-weighted Class Activation Mapping)** para proporcionar una explicación visual de la predicción.

Grad-CAM genera un mapa de calor que permite identificar las regiones de la radiografía que tuvieron mayor influencia en la decisión del modelo.

Esto permite complementar la predicción numérica con una representación visual de las zonas relevantes de la imagen.

---

## Pruebas unitarias

El proyecto cuenta con un conjunto de pruebas unitarias automatizadas utilizando **Pytest**.

Las pruebas se encuentran organizadas según los principales componentes del sistema:

```text
tests/
├── test_detector.py
├── test_grad_cam.py
├── test_integrator.py
├── test_preprocess.py
└── test_read_img.py
```

Estas pruebas permiten verificar de forma independiente:

* Lectura de imágenes JPG y DICOM.
* Preprocesamiento de imágenes.
* Clasificación y probabilidades.
* Integración entre los diferentes componentes.
* Generación de mapas Grad-CAM.
* Diferentes tamaños y tipos de imágenes.
* Tipos y dimensiones de los datos de salida.
* Manejo de diferentes escenarios de entrada.

Para ejecutar todas las pruebas:

```bash
uv run pytest -v
```

---

## Imágenes de prueba

El proyecto incluye imágenes DICOM de prueba dentro de:

```text
PruebaImagenes/DICOM/
```

Estas imágenes permiten verificar la carga y procesamiento de radiografías sin necesidad de utilizar archivos externos.

---

## Estructura general del proyecto

La organización del proyecto separa las diferentes responsabilidades del sistema:

```text
UAO-Neumonia1/
│
├── detector_neumonia.py
├── src/
│   ├── grad_cam.py
│   ├── integrator.py
│   ├── load_model.py
│   ├── preprocess_img.py
│   └── read_img.py
│
├── tests/
│   ├── test_detector.py
│   ├── test_grad_cam.py
│   ├── test_integrator.py
│   ├── test_preprocess.py
│   └── test_read_img.py
│
├── PruebaImagenes/
│   └── DICOM/
│
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

Esta estructura permite separar la interfaz principal, los componentes de procesamiento y las pruebas automatizadas, facilitando el mantenimiento y la validación del proyecto.

---

## Control de versiones

El código fuente se administra mediante **Git** y se encuentra alojado en **GitHub**.

## Los cambios realizados durante el desarrollo se registran mediante commits y pueden integrarse al repositorio mediante **Pull Requests**, permitiendo mantener un historial de modificaciones y facilitar la colaboración entre los integrantes del proyecto.

## Ejecución completa del proyecto

De forma resumida, el flujo de ejecución del sistema es:

```text
Imagen radiográfica
        ↓
Lectura de imagen
        ↓
Preprocesamiento
        ↓
Modelo de Deep Learning
        ↓
Clasificación
        ↓
Probabilidad
        ↓
Grad-CAM
        ↓
Mapa de calor
        ↓
Resultado en la interfaz
        ↓
Guardar / Generar PDF
```

De esta manera, el proyecto integra procesamiento de imágenes médicas, Deep Learning, explicabilidad mediante Grad-CAM, interfaz gráfica y pruebas automatizadas en una única aplicación.


## Arquitectura MVC

La solución está organizada siguiendo una estructura basada en el patrón de arquitectura Modelo-Vista-Controlador (MVC). Esta separación de responsabilidades permite mantener el código modular, facilitar las pruebas unitarias y desacoplar la interfaz gráfica de los procesos de lectura, procesamiento de imágenes y clasificación mediante Deep Learning.

La arquitectura se divide en los siguientes componentes:

- **Modelo:** gestión del modelo de Deep Learning y generación de explicaciones mediante Grad-CAM.
- **Controlador:** lectura, procesamiento e integración de las imágenes con el modelo.
- **Vista / Cliente:** interfaz gráfica mediante la cual el usuario interactúa con el sistema.

---

### MODELO

La lógica relacionada con el modelo de Deep Learning se encuentra principalmente en:

```text
src/load_model.py
src/grad_cam.py

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

El proyecto cuenta con una suite de **120 pruebas unitarias** implementadas mediante **Pytest**, cuyo objetivo es verificar el correcto funcionamiento de los diferentes componentes que hacen parte del sistema de detección de neumonía.

Las pruebas permiten comprobar de manera independiente el comportamiento de los módulos relacionados con la lectura de imágenes, el preprocesamiento, la integración del modelo, la generación de mapas de calor Grad-CAM y las funciones principales de predicción.

Las pruebas se encuentran organizadas dentro de la carpeta:

```text
tests/

Las funciones seleccionadas para las pruebas son:

test_detector.py
test_grad_cam.py
test_integrator.py
test_preprocess.py
test_read_img.pyt

Ejecutar las pruebas:
uv run pytest -v

El resultado esperado es:

tests/test_read_img.py::test_read_jpg_file_tipo_numpy
tests/test_read_img.py::test_read_jpg_file_tipo_pil
tests/test_read_img.py::test_read_jpg_file_dimensiones
tests/test_read_img.py::test_read_jpg_file_uint8
tests/test_read_img.py::test_read_jpg_file_imagen_negra
tests/test_read_img.py::test_read_jpg_file_imagen_pequena
tests/test_read_img.py::test_read_jpg_file_imagen_grande
tests/test_read_img.py::test_read_jpg_file_extensiones
tests/test_read_img.py::test_read_jpg_file_tupla_resultado
  C:\Users\Nicolas\Documents\ESPECIALIZACION IA\SEMESTRE 2\DESSARROLLO DE PROYECTOS DE IA\PROYECTO1NEUMONIA\UAO-Neumonia1\src\read_img.py:62: RuntimeWarning: invalid value encountered in cast
    img2 = np.uint8(img2)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
============================================ 120 passed ==============================================================

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

## Proyecto

El proyecto base fue desarrollado por:

Nicolas Bolaños
https://github.com/nico10102003

Davinson Tulande
https://github.com/DTulandeM

Jensen Tamayo


Nicolas Pulido
https://github.com/nicolaspulido-crypto

La implementación actual corresponde a la adaptación y finalización académica del proyecto de acuerdo con los requerimientos establecidos para el curso.

Autor / Equipo

Proyecto académico desarrollado para la Universidad Autónoma de Occidente (UAO).

Repositorio:

https://github.com/nicolaspulido-crypto/UAO-Neumonia1