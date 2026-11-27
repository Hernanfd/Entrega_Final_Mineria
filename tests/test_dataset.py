import csv
import re
import unicodedata
from pathlib import Path

import pytest
from openpyxl import load_workbook

from esperanto_nlp.crear_dataset import limpiar_bloque
from esperanto_nlp.extraccion_web import extraer_parrafos_html


DATOS = Path(__file__).resolve().parents[1] / "data" / "originales"


@pytest.fixture(scope="module")
def dataset():
    ruta = DATOS / "dataset_revision.csv"

    with ruta.open(encoding="utf-8", newline="") as archivo:
        lector = csv.DictReader(archivo, delimiter=";", strict=True)
        assert lector.fieldnames == ["text", "class"]
        filas = list(lector)

    assert filas, "El dataset está vacío."
    return filas


def test_extraccion_del_articulo():
    html = """
    <html><body>
        <header><p>Menú de navegación</p></header>

        <div class="itemBody kl-blog-post-body">
            <p>La <strong>programistoj</strong> analizas datumojn.</p>
            <p>La reto estas sekura.</p>
        </div>

        <footer><p>Aviso de cookies</p></footer>
    </body></html>
    """

    assert extraer_parrafos_html(html.encode("utf-8")) == [
        "La programistoj analizas datumojn.",
        "La reto estas sekura.",
    ]


def test_formato_y_categorias(dataset):
    assert {fila["class"] for fila in dataset} == {
        "Ciberseguridad", "IA", "Programacion"
    }

    for numero, fila in enumerate(dataset, start=1):
        assert set(fila) == {"text", "class"}, (
            f"Fila {numero} mal formada."
        )

        texto = fila["text"]

        assert texto and texto.strip(), (
            f"Texto vacío en la fila {numero}."
        )
        assert ";" not in texto, (
            f"Punto y coma en la fila {numero}."
        )
        assert not re.search(r"https?://|www\.", texto, re.IGNORECASE), (
            f"URL dentro del texto en la fila {numero}."
        )


def test_sin_textos_duplicados(dataset):
    textos = [
        unicodedata.normalize(
            "NFC", " ".join(fila["text"].split())
        ).casefold()
        for fila in dataset
    ]

    assert len(textos) == len(set(textos)), "Hay textos repetidos."


def test_correspondencia_con_el_excel(dataset):
    ruta = DATOS / "origen_revision.csv"

    with ruta.open(encoding="utf-8", newline="") as archivo:
        lector = csv.DictReader(archivo, delimiter=";", strict=True)

        assert lector.fieldnames == [
            "fila_dataset", "class", "url", "subcategoria", "celda"
        ]

        origenes = list(lector)

    assert len(dataset) == len(origenes), (
        "Los CSV tienen distinto tamaño."
    )

    libro = load_workbook(DATOS / "parrafos.xlsx", data_only=True)

    try:
        for numero, (fila, origen) in enumerate(
            zip(dataset, origenes), start=1
        ):
            assert origen["fila_dataset"] == str(numero)
            assert origen["class"] == fila["class"]

            hoja = libro[origen["class"]]
            celda = hoja[origen["celda"]]

            assert celda.row >= 5 and celda.column >= 2
            assert hoja.cell(4, celda.column).value == "OK"
            assert hoja.cell(2, celda.column).value == origen["url"]
            assert hoja.cell(3, celda.column).value == origen["subcategoria"]

            esperado = limpiar_bloque(celda.value)

            assert esperado is not None
            assert esperado.replace(";", ",") == fila["text"], (
                f"El texto de la fila {numero} no coincide con su origen."
            )

    finally:
        libro.close()