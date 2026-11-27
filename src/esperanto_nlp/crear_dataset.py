# Construcción del dataset
import argparse
import re
import csv
import time
from pathlib import Path

import requests
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .extraccion_web import extraer_parrafos


CATEGORIAS = ("Ciberseguridad", "IA", "Programacion")


def limpiar_bloque(texto):
    # Conservamos el nombre visible de los enlaces Markdown.
    texto = re.sub(
        r"\[([^\]]+)\]\(https?://[^)\s]+\)",
        r"\1",
        texto,
        flags=re.IGNORECASE,
    )

    texto = re.sub(
        r"""https?://[^\s<>"']+""",
        "",
        texto,
        flags=re.IGNORECASE,
    )

    texto = " ".join(texto.split())
    texto_comparacion = texto.casefold()

    prefijos_navegacion = (
        "vidu pliajn demandojn kaj respondojn",
        "rigardu pliajn demandojn kaj respondojn",
    )

    etiquetas_aisladas = {
        "ekzemple:",
        "resuma paragrafo",
        "pitono:",
        "c:",
        'ekzemple: – bilda url: "" paĝadreso: ""',
        'ekzemple: – paĝadreso: "" bilda url: ""',
        'ekzemple: – bilda url: ""',
        'ekzemple: – paĝadreso: ""',
    }

    if texto_comparacion.startswith(prefijos_navegacion):
        return None

    if texto_comparacion in etiquetas_aisladas:
        return None

    fragmentos_incompletos = {
        "la publika ŝlosilo estas , kaj la privata ŝlosilo estas .",
        "la originala mesaĝo estas reakirita de .",
        "kie indikas kunligon.",
        "konsideru enigan vektoron .",
        "- lineara modelo: - profunda modelo (dnn): , "
        "kie estas nelinearaj aktivigaj funkcioj.",
        "if , kaj , tiam por studento kiu studis 4 horojn "
        "kaj dormis 6 horojn:",
    }

    if texto_comparacion in fragmentos_incompletos:
        return None
    
    return texto or None


def exportar_csv(
    entrada="data/originales/parrafos.xlsx",
    salida="data/originales",
):
    salida = Path(salida)
    ruta_revision = Path(entrada).with_name("subtitulos_revisados.csv")

    with ruta_revision.open(encoding="utf-8-sig", newline="") as archivo:
        descartes = {
            (fila["class"], fila["text"])
            for fila in csv.DictReader(archivo, delimiter=";")
        }
    libro = load_workbook(entrada, read_only=True, data_only=True)
    registros = []

    try:
        for categoria in CATEGORIAS:
            filas = list(libro[categoria].iter_rows(values_only=True))

            for columna in range(1, len(filas[0])):
                if filas[3][columna] != "OK":
                    continue

                url = filas[1][columna]
                subcategoria = filas[2][columna]

                for numero_fila, fila in enumerate(filas[4:], start=5):
                    valor = fila[columna]
                    if not isinstance(valor, str):
                        continue

                    texto = limpiar_bloque(valor)
                    if not texto:
                        continue

                    if (categoria, texto.replace(";", ",")) in descartes:
                        continue

                    registros.append({
                        "fila_dataset": len(registros) + 1,
                        "text": texto.replace(";", ","),
                        "class": categoria,
                        "url": url,
                        "subcategoria": subcategoria,
                        "celda": f"{get_column_letter(columna + 1)}{numero_fila}",
                    })
    finally:
        libro.close()

    if not registros:
        raise ValueError("No se encontraron textos para exportar.")

    salida.mkdir(parents=True, exist_ok=True)
    archivos = {
        "dataset_revision.csv": ["text", "class"],
        "origen_revision.csv": [
            "fila_dataset", "class", "url", "subcategoria", "celda"
        ],
    }

    for nombre, campos in archivos.items():
        ruta = salida / nombre
        with ruta.open("w", encoding="utf-8", newline="") as archivo:
            escritor = csv.DictWriter(
                archivo, fieldnames=campos, delimiter=";", extrasaction="ignore"
            )
            escritor.writeheader()
            escritor.writerows(registros)
        print(f"Creado: {ruta.resolve()}")

    print(f"\nTotal de textos: {len(registros)}")
    for categoria in CATEGORIAS:
        cantidad = sum(r["class"] == categoria for r in registros)
        print(f"{categoria}: {cantidad}")


def leer_enlaces(ruta):
    libro = load_workbook(ruta, data_only=True)
    enlaces = {}

    try:
        for categoria in CATEGORIAS:
            hoja = libro[categoria]
            enlaces[categoria] = []

            # A contiene la subcategoría; B:K contienen los enlaces.
            for fila in hoja.iter_rows(min_row=2, max_col=11):
                subcategoria = str(fila[0].value or "").strip()

                for celda in fila[1:]:
                    valor = celda.value
                    if celda.hyperlink and celda.hyperlink.target:
                        valor = celda.hyperlink.target

                    if not isinstance(valor, str):
                        continue

                    url = valor.strip()
                    if url.startswith(("https://", "http://")):
                        enlaces[categoria].append((subcategoria, url))

            if not enlaces[categoria]:
                raise ValueError(f"No hay enlaces en la hoja {categoria}.")
    finally:
        libro.close()

    return enlaces


def escribir_celda(hoja, fila, columna, valor):
    if isinstance(valor, str):
        # Excel no admite estos caracteres de control.
        valor = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", valor)
        if len(valor) > 32767:
            raise ValueError("El texto supera el tamaño de una celda Excel.")

    celda = hoja.cell(row=fila, column=columna, value=valor)
    if isinstance(valor, str):
        celda.data_type = "s"
    celda.alignment = Alignment(vertical="top", wrap_text=True)


def crear_excel(entrada, salida, prueba=False):
    entrada, salida = Path(entrada), Path(salida)
    if entrada.resolve() == salida.resolve():
        raise ValueError("La entrada y la salida deben ser archivos distintos.")

    enlaces = leer_enlaces(entrada)
    if prueba:
        enlaces = {categoria: lista[:1] for categoria, lista in enlaces.items()}

    salida.parent.mkdir(parents=True, exist_ok=True)
    libro = Workbook()
    libro.remove(libro.active)
    for categoria in enlaces:
        libro.create_sheet(categoria)

    resumen = libro.create_sheet("Resumen")
    resumen.append([
        "Categoría", "Enlace", "Subcategoría", "URL original",
        "URL final", "Párrafos", "Estado", "Detalle"
    ])

    total = sum(len(lista) for lista in enlaces.values())
    procesados = correctos = 0

    with requests.Session() as sesion:
        sesion.headers.update({"User-Agent": "EsperantoNLP-CUNEF/0.1"})

        for categoria, lista in enlaces.items():
            hoja = libro[categoria]
            for fila, etiqueta in enumerate(
                ["Contenido", "URL", "Subcategoría", "Estado"], start=1
            ):
                escribir_celda(hoja, fila, 1, etiqueta)

            for columna, (subcategoria, url) in enumerate(lista, start=2):
                nombre = f"Link {columna - 1}"
                procesados += 1
                print(f"[{procesados}/{total}] {categoria} - {nombre}")

                parrafos, url_final = [], ""
                estado, detalle = "OK", ""

                try:
                    url_final, parrafos = extraer_parrafos(url, sesion)
                    if any(len(p) > 32767 for p in parrafos):
                        raise ValueError(
                            "Algún párrafo supera 32767 caracteres. "
                            "No se ha exportado esta página para evitar recortarlo."
                        )
                    correctos += 1
                except (requests.RequestException, ValueError) as error:
                    parrafos = []
                    estado, detalle = "ERROR", str(error)

                for fila, valor in enumerate(
                    [nombre, url, subcategoria, estado], start=1
                ):
                    escribir_celda(hoja, fila, columna, valor)

                for numero, texto in enumerate(parrafos, start=1):
                    escribir_celda(hoja, numero + 4, 1, f"Párrafo {numero}")
                    escribir_celda(hoja, numero + 4, columna, texto)

                fila_resumen = resumen.max_row + 1
                valores = [
                    categoria, nombre, subcategoria, url, url_final,
                    len(parrafos), estado, detalle
                ]
                for numero, valor in enumerate(valores, start=1):
                    escribir_celda(resumen, fila_resumen, numero, valor)

                print(f"  {len(parrafos)} párrafos" if estado == "OK"
                      else f"  ERROR: {detalle}")
                time.sleep(1)

            hoja.freeze_panes = "B5"
            hoja.column_dimensions["A"].width = 20
            for columna in range(2, hoja.max_column + 1):
                hoja.column_dimensions[get_column_letter(columna)].width = 55

    resumen.freeze_panes = "A2"
    resumen.auto_filter.ref = resumen.dimensions
    for columna, ancho in enumerate([20, 12, 28, 55, 55, 12, 12, 60], start=1):
        resumen.column_dimensions[get_column_letter(columna)].width = ancho

    for hoja in libro:
        for celda in hoja[1]:
            celda.font = Font(bold=True, color="FFFFFF")
            celda.fill = PatternFill("solid", fgColor="25476A")
            celda.alignment = Alignment(wrap_text=True, vertical="top")

    libro.save(salida)
    libro.close()
    print(f"\nExcel guardado en: {salida.resolve()}")
    print(f"Enlaces con párrafos: {correctos}/{total}.")
    print(f"Enlaces con errores: {total - correctos}. Consulta la hoja Resumen.")
    return salida


def main():
    parser = argparse.ArgumentParser(description="Extraer párrafos a Excel.")
    parser.add_argument("--entrada", default="data/enlaces/BBDD_Links_completo.xlsx")
    parser.add_argument("--salida")
    parser.add_argument("--prueba", action="store_true")
    args = parser.parse_args()

    salida = args.salida or (
        "data/originales/parrafos_prueba.xlsx" if args.prueba
        else "data/originales/parrafos.xlsx"
    )
    crear_excel(args.entrada, salida, prueba=args.prueba)


if __name__ == "__main__":
    main()