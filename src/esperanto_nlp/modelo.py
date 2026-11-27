import csv
import warnings
import joblib
from pathlib import Path
from collections import Counter
from random import Random
from statistics import mean, pstdev
from time import perf_counter

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    jaccard_score,
    multilabel_confusion_matrix,
)
from sklearn.model_selection import GroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from .procesar_dataset import ETIQUETAS, leer_dataset_multietiqueta


def matriz_etiquetas(datos):
    return np.asarray(
        [[fila[c] for c in ETIQUETAS] for fila in datos],
        dtype=int,
    )


def resumir_datos(nombre, datos):
    print(f"\n{nombre}")
    print(f"Textos: {len(datos)}")
    print(f"URLs: {len({f['url'] for f in datos})}")

    for categoria in ETIQUETAS:
        positivos = sum(f[categoria] for f in datos)
        print(f"Positivos de {categoria}: {positivos}")

    multiples = sum(
        sum(f[c] for c in ETIQUETAS) > 1 for f in datos
    )
    print(f"Textos con varias etiquetas: {multiples}")


def cargar_datos_modelo(
    ruta="data/procesados/dataset_multietiqueta_procesado.csv",
):
    _, filas = leer_dataset_multietiqueta(ruta)

    datos = []
    excluidas = Counter()

    for fila in filas:
        if fila["estado_revision"] != "propuesto":
            excluidas[fila["estado_revision"]] += 1
            continue

        if not fila["text"].strip():
            excluidas["texto_vacio"] += 1
            continue

        dato = fila.copy()
        dato["fila_dataset"] = int(fila["fila_dataset"])

        for categoria in ETIQUETAS:
            dato[categoria] = int(fila[categoria])

        datos.append(dato)

    if not datos:
        raise ValueError("No quedan textos para entrenar.")

    print(f"Filas leídas: {len(filas)}")
    print(f"Pendientes: {excluidas['revisar']}")
    print(f"Sin categoría: {excluidas['sin_categoria']}")
    print(f"Propuestas vacías: {excluidas['texto_vacio']}")

    resumir_datos("Datos disponibles", datos)

    return datos


def comprobar_cobertura(datos, nombre):
    if not datos:
        raise ValueError(f"{nombre}: conjunto vacío.")

    etiquetas = matriz_etiquetas(datos)

    for i, categoria in enumerate(ETIQUETAS):
        positivos = int(etiquetas[:, i].sum())

        if positivos == 0 or positivos == len(datos):
            raise ValueError(
                f"{nombre}: faltan ejemplos positivos o negativos "
                f"de {categoria}."
            )


def separar_datos(datos, proporcion_prueba=0.2, semilla=42):
    if not datos or not 0 < proporcion_prueba < 1:
        raise ValueError("Parámetros de separación incorrectos.")

    urls_por_origen = {}
    origen_por_url = {}

    for fila in datos:
        url = fila["url"]
        origen = fila["clase_original"]

        if url in origen_por_url and origen_por_url[url] != origen:
            raise ValueError(f"URL con origen contradictorio: {url}")

        origen_por_url[url] = origen
        urls_por_origen.setdefault(origen, set()).add(url)

    generador = Random(semilla)
    urls_prueba = set()

    for origen in sorted(urls_por_origen):
        urls = sorted(urls_por_origen[origen])

        if len(urls) < 2:
            raise ValueError(
                f"Faltan URLs de la categoría original {origen}."
            )

        cantidad = round(len(urls) * proporcion_prueba)
        cantidad = max(1, min(cantidad, len(urls) - 1))

        urls_prueba.update(
            generador.sample(urls, cantidad)
        )

    entrenamiento = [
        f for f in datos if f["url"] not in urls_prueba
    ]
    prueba = [
        f for f in datos if f["url"] in urls_prueba
    ]

    comprobar_cobertura(entrenamiento, "Entrenamiento")
    comprobar_cobertura(prueba, "Prueba")

    urls_train = {f["url"] for f in entrenamiento}
    urls_test = {f["url"] for f in prueba}

    if urls_train & urls_test:
        raise ValueError("Hay URLs compartidas.")

    resumir_datos("Entrenamiento", entrenamiento)
    resumir_datos("Prueba", prueba)

    print(f"\nURLs compartidas: {len(urls_train & urls_test)}")

    return entrenamiento, prueba


def crear_particiones(entrenamiento):
    textos = [f["text"] for f in entrenamiento]
    urls = [f["url"] for f in entrenamiento]

    validacion = GroupKFold(n_splits=3)
    particiones = list(validacion.split(textos, groups=urls))

    for numero, (indices_train, indices_val) in enumerate(
        particiones, start=1
    ):
        train = [entrenamiento[i] for i in indices_train]
        val = [entrenamiento[i] for i in indices_val]

        comprobar_cobertura(
            train, f"Entrenamiento de partición {numero}"
        )
        comprobar_cobertura(
            val, f"Validación de partición {numero}"
        )

        if {f["url"] for f in train} & {f["url"] for f in val}:
            raise ValueError("Hay URLs compartidas en validación.")

    return particiones


def crear_modelos():
    clasificadores = {
        "LinearSVC": OneVsRestClassifier(
            LinearSVC(random_state=42, max_iter=5000),
            n_jobs=1,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64,),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            max_iter=300,
            early_stopping=False,
            random_state=42,
        ),
    }

    return {
        nombre: Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=False,
                token_pattern=r"(?u)\b\w+\b",
                min_df=2,
            )),
            ("clasificador", clasificador),
        ])
        for nombre, clasificador in clasificadores.items()
    }


def ajustar_modelo(modelo, textos, etiquetas):
    inicio = perf_counter()

    with threadpool_limits(limits=1):
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter("always", ConvergenceWarning)
            modelo.fit(textos, etiquetas)

    segundos = perf_counter() - inicio

    convergencia = any(
        issubclass(a.category, ConvergenceWarning)
        for a in avisos
    )

    for aviso in avisos:
        print(
            f"Aviso ({aviso.category.__name__}): {aviso.message}"
        )

    return segundos, convergencia


def calcular_metricas(reales, predicciones):
    return {
        "f1_macro": f1_score(
            reales, predicciones,
            average="macro", zero_division=0,
        ),
        "f1_micro": f1_score(
            reales, predicciones,
            average="micro", zero_division=0,
        ),
        "f1_samples": f1_score(
            reales, predicciones,
            average="samples", zero_division=0,
        ),
        "coincidencia_exacta": accuracy_score(
            reales, predicciones
        ),
        "hamming_loss": hamming_loss(
            reales, predicciones
        ),
        "jaccard_samples": jaccard_score(
            reales, predicciones,
            average="samples", zero_division=0,
        ),
    }


def comparar_modelos(entrenamiento):
    textos = [f["text"] for f in entrenamiento]
    etiquetas = matriz_etiquetas(entrenamiento)

    particiones = crear_particiones(entrenamiento)
    modelos = crear_modelos()
    resultados = {}

    for nombre, modelo in modelos.items():
        metricas_cv = []
        segundos_cv = 0
        avisos_cv = 0

        print(f"\nValidación de {nombre}", flush=True)

        for numero, (indices_train, indices_val) in enumerate(
            particiones, start=1
        ):
            modelo_fold = clone(modelo)

            segundos, aviso = ajustar_modelo(
                modelo_fold,
                [textos[i] for i in indices_train],
                etiquetas[indices_train],
            )

            predicciones = modelo_fold.predict(
                [textos[i] for i in indices_val]
            )

            metricas = calcular_metricas(
                etiquetas[indices_val], predicciones
            )

            metricas_cv.append(metricas)
            segundos_cv += segundos
            avisos_cv += int(aviso)

            print(
                f"Partición {numero}/3: "
                f"F1 macro = {metricas['f1_macro']:.4f} | "
                f"F1 micro = {metricas['f1_micro']:.4f}",
                flush=True,
            )

        resultados[nombre] = {
            f"{clave}_media": mean(
                m[clave] for m in metricas_cv
            )
            for clave in metricas_cv[0]
        }

        resultados[nombre].update({
            "f1_macro_desv": pstdev(
                m["f1_macro"] for m in metricas_cv
            ),
            "segundos_ajuste_cv": segundos_cv,
            "avisos_cv": avisos_cv,
        })

    print("\nCOMPARACIÓN EN VALIDACIÓN")

    for nombre, r in resultados.items():
        print(
            f"{nombre}: "
            f"F1 macro = {r['f1_macro_media']:.4f} "
            f"± {r['f1_macro_desv']:.4f} | "
            f"F1 micro = {r['f1_micro_media']:.4f} | "
            f"Coincidencia exacta = "
            f"{r['coincidencia_exacta_media']:.4f} | "
            f"Tiempo = {r['segundos_ajuste_cv']:.2f} s"
        )

    elegido = max(
        resultados,
        key=lambda n: resultados[n]["f1_macro_media"],
    )

    print(f"\nSeleccionado: {elegido}")

    for nombre, modelo in modelos.items():
        print(f"Entrenando {nombre} con todos los datos...")

        segundos, aviso = ajustar_modelo(
            modelo, textos, etiquetas
        )

        print(
            f"Tiempo: {segundos:.2f} s | "
            f"Aviso de convergencia: {aviso}"
        )

    return modelos, resultados, elegido



def exportar_errores(
    modelos,
    prueba,
    ruta_salida="data/resultados/analisis_errores.csv",
):
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    textos = [fila["text"] for fila in prueba]
    etiquetas_reales = matriz_etiquetas(prueba)

    predicciones_modelos = {
        nombre: modelo.predict(textos)
        for nombre, modelo in modelos.items()
    }

    resultados = []

    for i, fila in enumerate(prueba):
        resultado = {
            "fila_dataset": fila["fila_dataset"],
            "url": fila["url"],
            "texto_original": fila["text_original"],
            "texto_procesado": fila["text"],
        }

        for j, categoria in enumerate(ETIQUETAS):
            resultado[f"real_{categoria}"] = int(
                etiquetas_reales[i, j]
            )

        for nombre, predicciones in predicciones_modelos.items():
            for j, categoria in enumerate(ETIQUETAS):
                resultado[f"{nombre}_{categoria}"] = int(
                    predicciones[i, j]
                )

            resultado[f"error_{nombre}"] = int(
                not np.array_equal(
                    etiquetas_reales[i],
                    predicciones[i],
                )
            )

            resultado[f"sin_etiqueta_{nombre}"] = int(
                predicciones[i].sum() == 0
            )

        resultado["error_ambos"] = int(
            all(
                resultado[f"error_{nombre}"] == 1
                for nombre in modelos
            )
        )

        resultados.append(resultado)

    campos = list(resultados[0].keys())

    with ruta_salida.open(
        "w", encoding="utf-8-sig", newline=""
    ) as archivo:
        escritor = csv.DictWriter(
            archivo,
            fieldnames=campos,
            delimiter=";",
        )
        escritor.writeheader()
        escritor.writerows(resultados)

    print(f"\nAnálisis guardado en: {ruta_salida}")
    print(f"Párrafos analizados: {len(resultados)}")

    for nombre in modelos:
        errores = sum(
            fila[f"error_{nombre}"]
            for fila in resultados
        )
        sin_etiqueta = sum(
            fila[f"sin_etiqueta_{nombre}"]
            for fila in resultados
        )

        print(f"\n{nombre}")
        print(f"Párrafos con errores: {errores}")
        print(f"Predicciones sin etiqueta: {sin_etiqueta}")

    errores_ambos = sum(
        fila["error_ambos"] for fila in resultados
    )

    print(
        f"\nPárrafos incorrectos en ambos modelos: "
        f"{errores_ambos}"
    )

    return resultados


def guardar_modelos(
    modelos,
    elegido,
    carpeta="data/modelos",
):
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)

    for nombre, modelo in modelos.items():
        ruta = carpeta / f"{nombre.lower()}.joblib"
        joblib.dump(modelo, ruta)

        print(f"Modelo guardado: {ruta}")

    ruta_elegido = carpeta / "modelo_seleccionado.txt"
    ruta_elegido.write_text(elegido, encoding="utf-8")

    print(f"Modelo seleccionado: {elegido}")


def evaluar_modelos(modelos, prueba):
    textos = [f["text"] for f in prueba]
    etiquetas = matriz_etiquetas(prueba)
    resultados = {}

    for nombre, modelo in modelos.items():
        predicciones = modelo.predict(textos)
        metricas = calcular_metricas(etiquetas, predicciones)

        matrices = multilabel_confusion_matrix(
            etiquetas, predicciones
        )

        print(f"\nPRUEBA FINAL — {nombre}")

        for clave, valor in metricas.items():
            print(f"{clave}: {valor:.4f}")

        print(classification_report(
            etiquetas,
            predicciones,
            target_names=list(ETIQUETAS),
            digits=4,
            zero_division=0,
        ))

        print("Matrices de confusión por categoría:")

        for categoria, matriz in zip(ETIQUETAS, matrices):
            print(categoria)
            print(matriz)

        sin_etiquetas = int(
            np.sum(predicciones.sum(axis=1) == 0)
        )

        print(
            "Predicciones sin etiqueta:", sin_etiquetas
        )

        resultados[nombre] = {
            **metricas,
            "matrices_confusion": matrices.tolist(),
            "predicciones_sin_etiquetas": sin_etiquetas,
        }

    return resultados


if __name__ == "__main__":
    datos = cargar_datos_modelo()

    entrenamiento, prueba = separar_datos(datos)

    modelos, resultados_cv, elegido = comparar_modelos(
        entrenamiento
    )

    resultados_prueba = evaluar_modelos(modelos, prueba)

    exportar_errores(modelos, prueba)

    guardar_modelos(modelos, elegido)

    modelo = modelos[elegido]