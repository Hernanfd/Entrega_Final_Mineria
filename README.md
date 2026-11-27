
# Esperanto NLP — Clasificación multietiqueta de textos

Proyecto desarrollado para la práctica de **Minería de Textos de CUNEF (2026/2027)**.

El objetivo es desarrollar un sistema propio de procesamiento del lenguaje natural para Esperanto y clasificar textos en tres categorías:

- Ciberseguridad.
- Inteligencia Artificial.
- Programación.

La clasificación es multietiqueta, por lo que un mismo párrafo puede pertenecer a varias categorías simultáneamente.

## Funcionalidades

- Extracción de párrafos mediante web scraping.
- Construcción y procesamiento del dataset.
- Tokenización y normalización Unicode.
- Lematización y etiquetado gramatical (POS-tagging).
- Análisis morfológico y gestión de stopwords.
- Objetos `Doc` y `Token` con una API similar a spaCy.
- Vectorización mediante TF-IDF.
- Clasificación multietiqueta mediante LinearSVC y MLP.
- Validación cruzada y evaluación de modelos.
- Clasificación de páginas web en Esperanto.

## Instalación

Se necesita Python 3.10 o superior. El proyecto se ha desarrollado utilizando Python 3.12.

Clonar el repositorio:

```bash
git clone https://github.com/Hernanfd/Entrega_Final_Mineria.git
cd Entrega_Final_Mineria
```

Crear un entorno virtual:

```bash
python -m venv .venv
```

Activarlo en Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instalar el paquete y sus dependencias:

```bash
python -m pip install -e ".[dev]"
```

## Estructura del proyecto

```text
Entrega_Final_Mineria/
├── data/
│   ├── enlaces/
│   ├── originales/
│   ├── procesados/
│   ├── modelos/
│   └── resultados/
├── src/
│   └── esperanto_nlp/
│       ├── recursos/
│       ├── __init__.py
│       ├── libreria.py
│       ├── extraccion_web.py
│       ├── crear_dataset.py
│       ├── procesar_dataset.py
│       ├── modelo.py
│       └── main.py
├── tests/
├── pipeline_datos.ipynb
├── pipeline_modelos.ipynb
├── pyproject.toml
└── README.md
```

Las carpetas de datos procesados, modelos y resultados se generan durante la ejecución.

## Ejecución del proyecto

### 1. Pipeline de datos

Abrir `pipeline_datos.ipynb`, seleccionar el entorno virtual como kernel y ejecutar todas las celdas.

Este notebook incluye:

- Demostración de la extracción web.
- Análisis y visualización del dataset.
- Demostración de nuestra librería NLP.
- Procesamiento de los párrafos.
- Verificación de la integridad de los datos.

El corpus original está disponible en el repositorio, por lo que no es necesario descargar nuevamente todos los artículos.

El archivo generado es:

`data/procesados/dataset_multietiqueta_procesado.csv`

### 2. Pipeline de modelos

Abrir `pipeline_modelos.ipynb` y ejecutar todas las celdas después de completar el pipeline de datos.

Este notebook incluye:

- Separación de entrenamiento y prueba por URL.
- Vectorización TF-IDF.
- Entrenamiento de LinearSVC y MLP.
- Validación cruzada.
- Evaluación mediante métricas multietiqueta.
- Análisis de errores.
- Guardado de modelos.
- Clasificación de páginas web nuevas.

### 3. Clasificación de páginas web

Una vez entrenados los modelos, se puede clasificar una página desde la terminal:

```bash
python -m esperanto_nlp.main "https://eo.wikipedia.org/wiki/Artefarita_inteligenteco"
```

También se puede seleccionar explícitamente LinearSVC:

```bash
python -m esperanto_nlp.main "https://eo.wikipedia.org/wiki/Artefarita_inteligenteco" --modelo LinearSVC
```

Los resultados se guardan en:

`data/resultados/clasificacion_web.csv`

## Dataset

El corpus contiene 1.659 párrafos procedentes de 120 artículos.

| Indicador | Cantidad |
|---|---:|
| Párrafos con etiquetas propuestas completas | 1.578 |
| Párrafos con varias etiquetas | 108 |
| Pendientes de revisión | 59 |
| Sin categoría | 22 |
| Párrafos de entrenamiento | 1.215 |
| Párrafos de prueba | 363 |

Las categorías no son excluyentes. Las anotaciones pendientes y los registros sin categoría se excluyen del entrenamiento inicial.

## Resultados de clasificación

Se comparan dos modelos entrenados utilizando TF-IDF:

| Métrica | LinearSVC | MLP |
|---|---:|---:|
| F1 macro en validación | 0,8823 | 0,8887 |
| F1 macro en prueba | 0,8992 | 0,9172 |
| F1 micro en prueba | 0,9025 | 0,9179 |
| Coincidencia exacta | 82,37 % | 84,57 % |
| Hamming loss | 0,0670 | 0,0579 |

La MLP fue seleccionada por obtener el mayor F1 macro medio en validación cruzada.

Estos resultados corresponden a una ejecución previa y se recalculan al ejecutar el notebook. La evaluación es preliminar, puesto que las etiquetas son propuestas asistidas y no están completamente verificadas.

## Ejemplo de uso de la librería NLP

```python
from esperanto_nlp import EsperantoNLP, preparar_texto

nlp = EsperantoNLP()

texto = "La programistoj analizas datumojn kaj protektas komputilajn retojn."

doc = nlp(texto)

for token in doc:
    print(token.text, token.lemma_, token.pos_)

print(preparar_texto(texto, nlp))
```

Resultado del procesamiento:

```text
['programisto', 'analizi', 'datumo', 'protekti', 'komputila', 'reto']
```

## Pruebas

El proyecto incluye pruebas automatizadas para verificar el funcionamiento de la librería NLP y la construcción del dataset.

Para ejecutarlas:

```bash
python -m pytest -v
```

En las comprobaciones anteriores se superaron 85 pruebas correspondientes a NLP y dataset.

## Recursos utilizados

- [Guía de Esperanto](https://jparisu.github.io/nlp-esperantilo/es/guide/esperanto/)
- [Vocabulario y stopwords](https://github.com/jparisu/nlp-esperantilo/blob/main/resources/esperanto/vortoj.json)
- [EITCA Academy en Esperanto](https://eo.eitca.org/)
- [Wikipedia en Esperanto](https://eo.wikipedia.org/)

Los recursos lingüísticos se incluyen en la carpeta `recursos`, junto con sus licencias correspondientes.

## Limitaciones

- La librería NLP está basada en reglas y no resuelve todas las ambigüedades lingüísticas.
- Algunas anotaciones requieren revisión manual.
- Determinados párrafos necesitan el contexto del artículo para clasificarse correctamente.
- TF-IDF no captura directamente el orden de las palabras ni relaciones semánticas complejas.
- La procedencia del corpus limita la evaluación de la generalización a otras fuentes.
- Las puntuaciones de la MLP no son necesariamente probabilidades calibradas y las de LinearSVC son márgenes de decisión.
