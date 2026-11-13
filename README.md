# Esperanto NLP

Librería de Python desarrollada para la práctica de **Minería de Textos de CUNEF (2026/2027)**.

El objetivo es crear nuestro propio modelo de procesamiento de lenguaje natural (NLP) para Esperanto, aprovechando las reglas gramaticales del idioma, sin utilizar modelos como spaCy o NLTK.

## Funcionalidades

La librería permite:

- Tokenizar textos y normalizar caracteres Unicode.
- Lematizar palabras y reconocer su categoría gramatical (POS-tagging).
- Identificar características morfológicas como número, caso y tiempo verbal.
- Detectar y modificar stop-words.
- Procesar textos mediante objetos `Doc` y `Token`, con una estructura similar a spaCy.

## Instalación

Se necesita Python 3.10 o superior.

Para instalar el proyecto localmente desde su carpeta principal:

```bash
pip install -e ".[dev]"
```

También se podrá instalar directamente desde GitHub una vez publicada y validada la versión definitiva.

## Ejemplo de uso

```python
from esperanto_nlp import EsperantoNLP, preparar_texto

nlp = EsperantoNLP()

texto = "La programistoj analizas datumojn kaj protektas komputilajn retojn."

doc = nlp(texto)

for token in doc:
    print(token.text, token.lemma_, token.pos_)

print(preparar_texto(texto, nlp))
```

El texto procesado devuelve:

```text
['programisto', 'analizi', 'datumo', 'protekti', 'komputila', 'reto']
```

## Pruebas

Hemos creado tests para comprobar la tokenización, lematización, etiquetado gramatical, stop-words y funcionamiento de los objetos `Doc` y `Token`.

Se ejecutan con:

```bash
python -m pytest tests/test_nlp.py -v
```

En la última ejecución se superaron **81 pruebas**.

## Recursos utilizados

Para desarrollar las reglas gramaticales y el vocabulario hemos utilizado la documentación proporcionada por el profesor:

- [Guía de Esperanto](https://jparisu.github.io/nlp-esperantilo/es/guide/esperanto/)
- [Vocabulario y stop-words](https://github.com/jparisu/nlp-esperantilo/blob/main/resources/esperanto/vortoj.json)

El vocabulario se incluye en la carpeta `recursos`, junto con las licencias correspondientes.

## Limitaciones

Al ser un modelo basado en reglas, no siempre puede resolver palabras ambiguas según el contexto. Tampoco contempla todas las construcciones gramaticales del Esperanto, como los tiempos verbales compuestos.

La idea es utilizar esta librería para procesar nuestro dataset y, posteriormente, entrenar un modelo que clasifique textos por temática.