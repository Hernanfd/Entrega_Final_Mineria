# Coordinación del flujo completo

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
import requests
from bs4 import BeautifulSoup

from esperanto_nlp.libreria import EsperantoNLP, preparar_texto


ETIQUETAS = ("Ciberseguridad", "IA", "Programacion")


def extraer_parrafos(url):
    respuesta = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20,
    )
    respuesta.raise_for_status()

    soup = BeautifulSoup(respuesta.text, "html.parser")

    # Primero intentamos el contenedor principal de EITCA.
    contenedor = soup.select_one("div.itemBody.kl-blog-post-body")

    if contenedor is None:
        contenedor = (
            soup.select_one("article")
            or soup.select_one("main")
        )

    if contenedor is None:
        raise ValueError(
            "No se encontró el contenido principal de la página."
        )

    for elemento in contenedor.select(
        "script, style, nav, footer, aside"
    ):
        elemento.decompose()

    parrafos = [
        p.get_text(" ", strip=True)
        for p in contenedor.select("p")
    ]

    if not parrafos:
        raise ValueError(
            "No se encontraron párrafos útiles en la página."
        )

    return parrafos


def cargar_modelo(nombre="MLP"):
    rutas = {
        "MLP": Path("data/modelos/mlp.joblib"),
        "LinearSVC": Path("data/modelos/linearsvc.joblib"),
    }

    if nombre not in rutas:
        raise ValueError(f"Modelo no reconocido: {nombre}")

    ruta = rutas[nombre]

    if not ruta.is_file():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecuta primero modelo.py."
        )

    # Cargar únicamente modelos generados por nuestro proyecto.
    modelo = joblib.load(ruta)

    return modelo


def clasificar_pagina(url, nombre_modelo="MLP"):
    parrafos = extraer_parrafos(url)

    nlp = EsperantoNLP(conservar_negaciones=True)

    textos_procesados = [
        " ".join(
            preparar_texto(
                parrafo,
                nlp,
                quitar_numeros=True,
            )
        )
        for parrafo in parrafos
    ]

    modelo = cargar_modelo(nombre_modelo)

    indices_validos = [
        i for i, texto in enumerate(textos_procesados)
        if texto.strip()
    ]

    if not indices_validos:
        raise ValueError(
            "Todos los párrafos han quedado vacíos tras el NLP."
        )

    validos = [textos_procesados[i] for i in indices_validos]

    predicciones = modelo.predict(validos)

    if nombre_modelo == "MLP":
        puntuaciones = modelo.predict_proba(validos)
    else:
        puntuaciones = modelo.decision_function(validos)

    predicciones = np.asarray(predicciones)
    puntuaciones = np.asarray(puntuaciones)


    resultados = []

    posiciones_validas = {
        indice: posicion
        for posicion, indice in enumerate(indices_validos)
    }

    for indice, parrafo in enumerate(parrafos):
        fila = {
            "parrafo": indice + 1,
            "texto_original": parrafo,
            "texto_procesado": textos_procesados[indice],
        }

        if indice in posiciones_validas:
            posicion = posiciones_validas[indice]

            etiquetas = [
                ETIQUETAS[j]
                for j, valor in enumerate(predicciones[posicion])
                if valor == 1
            ]

            fila["categorias"] = (
                ", ".join(etiquetas) or "Sin etiqueta"
            )
            fila["estado"] = "Clasificado"

            for j, categoria in enumerate(ETIQUETAS):
                fila[f"pred_{categoria}"] = int(
                    predicciones[posicion, j]
                )
                fila[f"score_{categoria}"] = round(
                    float(puntuaciones[posicion, j]), 4
                )

        else:
            fila["categorias"] = "No clasificable"
            fila["estado"] = "Texto vacío tras NLP"

            for categoria in ETIQUETAS:
                fila[f"pred_{categoria}"] = ""
                fila[f"score_{categoria}"] = ""

        resultados.append(fila)

    return resultados



def guardar_resultados(resultados, nombre_modelo):
    salida = Path("data/resultados/clasificacion_web.csv")
    salida.parent.mkdir(parents=True, exist_ok=True)

    with salida.open(
        "w", encoding="utf-8-sig", newline=""
    ) as archivo:
        escritor = csv.DictWriter(
            archivo,
            fieldnames=list(resultados[0].keys()),
            delimiter=";",
        )

        escritor.writeheader()
        escritor.writerows(resultados)

    print(f"\nModelo utilizado: {nombre_modelo}")
    print(f"Párrafos clasificados: {len(resultados)}")
    print(f"Resultados guardados: {salida}")

    for categoria in ETIQUETAS:
        cantidad = sum(
            fila[f"pred_{categoria}"] == 1
            for fila in resultados
        )
        print(f"{categoria}: {cantidad} párrafos")

    sin_etiqueta = sum(
        fila["categorias"] == "Sin etiqueta"
        for fila in resultados
    )
    print(f"Sin etiqueta: {sin_etiqueta}")

    print("\nCLASIFICACIÓN DE TODOS LOS PÁRRAFOS")

    for fila in resultados:
        print(f"\n{'=' * 70}")
        print(f"PÁRRAFO {fila['parrafo']}")
        print(f"{'=' * 70}")

        print(f"\nTexto original:\n{fila['texto_original']}")

        print(f"\nCategorías: {fila['categorias']}")

        print("\nPuntuaciones:")

        for categoria in ETIQUETAS:
            puntuacion = fila[f"score_{categoria}"]
            prediccion = fila[f"pred_{categoria}"]

            if puntuacion == "":
                print(f"  {categoria}: No clasificable")
            else:
                print(
                    f"  {categoria}: {puntuacion:.4f} "
                    f"(predicción: {prediccion})"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clasificador web de textos en Esperanto"
    )

    parser.add_argument(
        "url",
        help="URL de la página en Esperanto",
    )

    parser.add_argument(
        "--modelo",
        choices=["MLP", "LinearSVC"],
        default="MLP",
    )

    argumentos = parser.parse_args()

    resultados = clasificar_pagina(
        argumentos.url,
        argumentos.modelo,
    )

    guardar_resultados(resultados, argumentos.modelo)