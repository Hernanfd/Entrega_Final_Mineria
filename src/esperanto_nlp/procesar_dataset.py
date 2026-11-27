import csv
from collections import Counter
from pathlib import Path

from .libreria import EsperantoNLP, preparar_texto


ETIQUETAS = ("Ciberseguridad", "IA", "Programacion")
ESTADOS = {"propuesto", "revisar", "sin_categoria"}


def leer_dataset_multietiqueta(ruta):
    with Path(ruta).open(encoding="utf-8-sig", newline="") as archivo:
        lector = csv.DictReader(archivo, delimiter=";", strict=True)
        campos = lector.fieldnames or []

        obligatorios = {
            "fila_dataset", "text", *ETIQUETAS,
            "estado_revision", "clase_original", "url",
        }

        faltantes = obligatorios - set(campos)

        if faltantes:
            raise ValueError(f"Faltan columnas: {sorted(faltantes)}")

        if len(campos) != len(set(campos)):
            raise ValueError("Hay nombres de columnas duplicados.")

        filas = list(lector)

    if not filas:
        raise ValueError("El dataset está vacío.")

    identificadores = set()

    for numero, fila in enumerate(filas, start=1):
        if None in fila or any(v is None for v in fila.values()):
            raise ValueError(
                f"La fila {numero} tiene un número incorrecto de campos."
            )

        identificador = int(fila["fila_dataset"])

        if identificador < 1 or identificador in identificadores:
            raise ValueError(f"Identificador no válido: {identificador}")

        identificadores.add(identificador)

        estado = fila["estado_revision"]
        valores = [fila[c] for c in ETIQUETAS]

        if estado not in ESTADOS:
            raise ValueError(f"Estado no válido: {estado}")

        if any(v not in {"0", "1", ""} for v in valores):
            raise ValueError(f"Etiquetas incorrectas: {identificador}")

        if estado == "propuesto" and (
            "" in valores or "1" not in valores
        ):
            raise ValueError(
                f"Propuesta incompleta: {identificador}"
            )

        if estado == "sin_categoria" and valores != ["0", "0", "0"]:
            raise ValueError(
                f"Etiquetas incorrectas: {identificador}"
            )

        if not fila["url"].strip():
            raise ValueError(f"Falta URL: {identificador}")

        if fila["clase_original"] not in ETIQUETAS:
            raise ValueError(
                f"Clase original incorrecta: {identificador}"
            )

    return campos, filas


def procesar_dataset(
    entrada="data/originales/dataset_multietiqueta.csv",
    salida="data/procesados/dataset_multietiqueta_procesado.csv",
):
    entrada = Path(entrada)
    salida = Path(salida)

    if entrada.resolve() == salida.resolve():
        raise ValueError(
            "La entrada y la salida deben ser diferentes."
        )

    campos, filas = leer_dataset_multietiqueta(entrada)

    if "text_original" in campos:
        raise ValueError("Utiliza el CSV original.")

    campos_salida = campos.copy()
    campos_salida.insert(
        campos_salida.index("text") + 1, "text_original"
    )

    nlp = EsperantoNLP(conservar_negaciones=True)

    procesados = []
    vacios = []

    for fila in filas:
        nueva = fila.copy()

        nueva["text_original"] = fila["text"]
        nueva["text"] = " ".join(
            preparar_texto(
                fila["text"], nlp, quitar_numeros=True
            )
        )

        procesados.append(nueva)

        if not nueva["text"].strip():
            vacios.append(int(fila["fila_dataset"]))

    salida.parent.mkdir(parents=True, exist_ok=True)

    with salida.open("w", encoding="utf-8-sig", newline="") as archivo:
        escritor = csv.DictWriter(
            archivo,
            fieldnames=campos_salida,
            delimiter=";",
        )
        escritor.writeheader()
        escritor.writerows(procesados)

    estados = Counter(
        fila["estado_revision"] for fila in procesados
    )

    utilizables = [
        fila for fila in procesados
        if fila["estado_revision"] == "propuesto"
        and fila["text"].strip()
    ]

    multiples = sum(
        sum(fila[c] == "1" for c in ETIQUETAS) > 1
        for fila in utilizables
    )

    print(f"Archivo guardado: {salida.resolve()}")
    print(f"Textos procesados: {len(procesados)}")
    print(f"Propuestas completas: {estados['propuesto']}")
    print(f"Pendientes: {estados['revisar']}")
    print(f"Sin categoría: {estados['sin_categoria']}")
    print(f"Textos vacíos: {len(vacios)}")
    print(f"Identificadores vacíos: {vacios}")
    print(f"Textos utilizables: {len(utilizables)}")
    print(f"Textos con varias etiquetas: {multiples}")

    for categoria in ETIQUETAS:
        positivos = sum(
            fila[categoria] == "1" for fila in utilizables
        )
        print(f"Positivos de {categoria}: {positivos}")

    return procesados


if __name__ == "__main__":
    procesar_dataset()