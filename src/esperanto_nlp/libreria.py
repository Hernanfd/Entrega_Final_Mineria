"""
Modelo NLP basado en reglas para Esperanto.

Guía de referencia proporcionada por el profesor:
https://jparisu.github.io/nlp-esperantilo/es/guide/esperanto/

Recurso de vocabulario para incorporar en los siguientes bloques:
https://github.com/jparisu/nlp-esperantilo/blob/main/resources/esperanto/vortoj.json
"""

import json
import re
import unicodedata
from importlib.resources import files
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

# Patron de palabras con soporte para caracteres Unicode
PALABRA = r"(?:[^\W\d_][\u0300-\u036f]*)+"

# Números enteros o decimal, con . o ,
NUMERO = r"\d+(?:[.,]\d+)?"

PATRON_TOKEN = re.compile(rf"{NUMERO}|{PALABRA}|[^\s]")


def normalizar(texto: str, *, minusculas: bool = True) -> str:
    """
    Pasa el texto a NFC y se puede mantener o cambiar a minúsculas. Mantiene los diacríticos
    """
    if not isinstance(texto, str):
        raise TypeError("El texto debe ser una cadena de caracteres (str)")

    if minusculas:
        texto = texto.lower()

    return unicodedata.normalize("NFC", texto)


def tokenizar(texto: str) -> list[str]:
    """
    Separa el texto en palabras, números y signos
    """
    texto_normalizado = normalizar(texto, minusculas=False)

    return PATRON_TOKEN.findall(texto_normalizado)


def cargar_vocabulario() -> dict[str, dict]:
    """
    Carga vortoj.json y devuelve {palabra_normalizada: entrada}. Lanza error si el formato no es válido
    """
    recurso = files("esperanto_nlp").joinpath("recursos", "vortoj.json")

    with recurso.open("r", encoding="utf-8") as archivo:
        datos = json.load(archivo)

    if not isinstance(datos, dict) or datos.get("lingvo") != "eo":
        raise ValueError("El recurso no tiene el formato de Esperanto esperado")

    entradas = datos.get("vortoj")
    categorias = datos.get("kategorioj")

    if not isinstance(entradas, list) or not entradas:
        raise ValueError("El campo 'vortoj' debe contener una lista no vacía")

    if not isinstance(categorias, dict):
        raise ValueError("Falta el diccionario de categorías 'kategorioj'")

    vocabulario = {}

    for entrada in entradas:
        if not isinstance(entrada, dict):
            raise ValueError("Cada entrada del vocabulario debe ser un diccionario")

        palabra = entrada.get("vorto")
        categoria = entrada.get("kategorio")

        if not isinstance(palabra, str) or not palabra.strip():
            raise ValueError("Hay una entrada sin una palabra válida")

        if not isinstance(categoria, str) or categoria not in categorias:
            raise ValueError(f"Categoría inválida en la palabra: {palabra}")

        if not isinstance(entrada.get("ignorinda"), bool):
            raise ValueError(f"Falta un valor booleano 'ignorinda' en: {palabra}")

        palabra = normalizar(palabra)

        if palabra in vocabulario:
            raise ValueError(f"Palabra duplicada en el recurso: {palabra}")

        vocabulario[palabra] = entrada.copy()

    return vocabulario


def obtener_stopwords(vocabulario: dict[str, dict]) -> set[str]:
    """
    Devuelve el conjunto de palabras con ignorinda=true (marcadas para ignorar)
    """
    return {palabra for palabra, ficha in vocabulario.items() if ficha["ignorinda"] is True}

# Reglas y etiquetas

PRONOMBRES = tuple("mi vi li ŝi ĝi ni ili oni si ci".split())
PREFIJOS_CORRELATIVOS = ("ki", "ti", "i", "ĉi", "neni")
FINALES_CORRELATIVOS = ("u", "o", "a", "es", "e", "am", "al", "el", "om")

CORRELATIVOS = {
    prefijo + final
    for prefijo in PREFIJOS_CORRELATIVOS
    for final in FINALES_CORRELATIVOS
}

POS_CATEGORIAS = {
    "artikolo": "DET",
    "prepozicio": "ADP",
    "konjunkcio": "CCONJ",
    "pronomo": "PRON",
    "adverbo": "ADV",
    "partikulo": "PART",
    "nombro": "NUM",
    "interjekcio": "INTJ",
}

SUBORDINANTES = {"ke", "se", "ĉar", "kvankam", "kvazaŭ", "ol"}

VERBOS = {
    "as": {"VerbForm": "Fin", "Mood": "Ind", "Tense": "Pres"},
    "is": {"VerbForm": "Fin", "Mood": "Ind", "Tense": "Past"},
    "os": {"VerbForm": "Fin", "Mood": "Ind", "Tense": "Fut"},
    "us": {"VerbForm": "Fin", "Mood": "Cnd"},
    "u": {"VerbForm": "Fin", "Mood": "Imp"},
    "i": {"VerbForm": "Inf"},
}

PATRON_LETRAS_EO = re.compile(r"[abcĉdefgĝhĥijĵklmnoprsŝtuŭvz]+")
PATRON_NOMINAL = re.compile(r"(.+)([oa])(j?)(n?)")

_UNIDAD = r"(?:unu|du|tri|kvar|kvin|ses|sep|ok|naŭ)"
_GRUPO_NUMERAL = rf"(?:{_UNIDAD}?cent)?(?:{_UNIDAD}?dek)?{_UNIDAD}?"
PATRON_CARDINAL = re.compile(rf"(?:nul|(?:{_GRUPO_NUMERAL}mil)?{_GRUPO_NUMERAL})")


def _es_cardinal(palabra: str) -> bool:
    return (bool(palabra) and PATRON_CARDINAL.fullmatch(palabra) is not None)


def _rasgos_nominales(sufijo: str) -> dict[str, str]:
    return {"Number": "Plur" if "j" in sufijo else "Sing", "Case": "Acc" if "n" in sufijo else "Nom",}


# Vocabulario y formas flexionadas

def construir_lexico(vocabulario: dict[str, dict]) -> dict:
    """
    Crea análisis directos para palabras funcionales y sus variantes
    """
    lexico = {}

    for palabra, ficha in vocabulario.items():
        categoria = ficha["kategorio"]
        pos = POS_CATEGORIAS.get(categoria)

        # Millón y millardo son sustantivos según las notas del recurso.
        if categoria == "nombro" and palabra.endswith("o"):
            continue

        if categoria == "ĝentilaĵo" and palabra == "adiaŭ":
            pos = "INTJ"

        if pos is None:
            continue

        if palabra in SUBORDINANTES:
            pos = "SCONJ"

        rasgos = {"NumType": "Card"} if pos == "NUM" else {}
        lexico[palabra] = (palabra, pos, rasgos)

    for pronombre in PRONOMBRES:
        lexico[pronombre] = (pronombre, "PRON", {"Case": "Nom"})
        lexico[pronombre + "n"] = (pronombre, "PRON", {"Case": "Acc"})

        posesivo = pronombre + "a"

        for sufijo in ("", "j", "n", "jn"):
            rasgos = _rasgos_nominales(sufijo)
            rasgos["Poss"] = "Yes"
            lexico[posesivo + sufijo] = (posesivo, "ADJ", rasgos)

    for prefijo in PREFIJOS_CORRELATIVOS:
        for final in FINALES_CORRELATIVOS:
            lema = prefijo + final

            if final in ("u", "a"):
                sufijos = ("", "j", "n", "jn")
            elif final in ("o", "e"):
                sufijos = ("", "n")
            else:
                sufijos = ("",)

            pos = {
                "u": "PRON",
                "o": "PRON",
                "a": "ADJ",
                "es": "DET",
            }.get(final, "ADV")

            for sufijo in sufijos:
                rasgos = {}

                if final in ("u", "a"):
                    rasgos = _rasgos_nominales(sufijo)
                elif final == "o":
                    rasgos["Case"] = "Acc" if sufijo else "Nom"
                elif final == "e" and sufijo == "n":
                    rasgos["Case"] = "Acc"

                lexico[lema + sufijo] = (lema, pos, rasgos)

    return lexico


def analizar_token(texto: str, lexico: dict) -> tuple[str, str, dict[str, str]]:
    """
    Devuelve lema, POS y rasgos; conserva los afijos derivativos
    """
    palabra = normalizar(texto)

    if not palabra:
        return palabra, "X", {}

    if re.fullmatch(NUMERO, palabra):
        return palabra, "NUM", {"NumType": "Card","NumForm": "Digit",}

    if all(unicodedata.category(c).startswith("P") for c in palabra):
        return palabra, "PUNCT", {}

    if all(unicodedata.category(c).startswith("S")for c in palabra):
        return palabra, "SYM", {}

    if palabra in lexico:
        lema, pos, rasgos = lexico[palabra]
        return lema, pos, rasgos.copy()

    if not PATRON_LETRAS_EO.fullmatch(palabra):
        return palabra, "X", {}

    if _es_cardinal(palabra):
        return palabra, "NUM", {"NumType": "Card","NumForm": "Word",}

    coincidencia = PATRON_NOMINAL.fullmatch(palabra)

    if coincidencia:
        raiz, final, plural, acusativo = coincidencia.groups()
        lema = raiz + final

        if lema in CORRELATIVOS:
            return palabra, "X", {}  # Flexión no válida, como «tioj».

        rasgos = _rasgos_nominales(plural + acusativo)

        if final == "a" and _es_cardinal(raiz):
            rasgos["NumType"] = "Ord"
        elif lema in {"miliono", "miliardo"}:
            rasgos["NumType"] = "Card"

        return lema, "NOUN" if final == "o" else "ADJ", rasgos

    for final, rasgos in VERBOS.items():
        if palabra.endswith(final) and len(palabra) > len(final):
            return palabra[:-len(final)] + "i", "VERB", rasgos.copy()

    coincidencia = re.fullmatch(r"(.+)e(n?)", palabra)

    if coincidencia:
        raiz, direccion = coincidencia.groups()
        rasgos = {"Case": "Acc"} if direccion else {}
        return raiz + "e", "ADV", rasgos

    return palabra, "X", {}


# Objetos del documento

@dataclass
class Token:
    text: str
    lemma_: str
    pos_: str
    is_stop: bool
    morph: dict[str, str]
    idx: int
    i: int

    @property
    def norm_(self) -> str:
        return normalizar(self.text)

    @property
    def is_alpha(self) -> bool:
        return self.norm_.isalpha()

    @property
    def is_punct(self) -> bool:
        return self.pos_ == "PUNCT"

    @property
    def like_num(self) -> bool:
        return (self.pos_ == "NUM" or self.morph.get("NumType") in {"Card", "Ord"})


class Doc:
    """
    Conserva el texto original y sus tokens, sin eliminar palabras
    """
    def __init__(self, texto: str, tokens: Iterable[Token]):
        self.text = texto
        self.tokens = tuple(tokens)

    def __iter__(self) -> Iterator[Token]:
        return iter(self.tokens)

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, indice: int | slice) -> Token | tuple[Token, ...]:
        return self.tokens[indice]

# Modelo integrado

class EsperantoNLP:
    """
    Modelo por reglas. Una lista personalizada sustituye las stop-words base
    """
    def __init__(self, stop_words: Iterable[str] | None = None, *, conservar_negaciones: bool = False):
        self.vocabulario = cargar_vocabulario()
        self.lexico = construir_lexico(self.vocabulario)

        if stop_words is None:
            base = obtener_stopwords(self.vocabulario)
            base.update(CORRELATIVOS)
            base.update(p + "a" for p in PRONOMBRES if p in base)
        else:
            base = stop_words

        self.stop_words = self._lemas_stopwords(base)

        if conservar_negaciones:
            negativas = {"ne", "nek", "sen"}
            negativas.update("neni" + final for final in FINALES_CORRELATIVOS)
            self.stop_words.difference_update(negativas)

    def _lemas_stopwords(self, palabras: Iterable[str]) -> set[str]:
        if isinstance(palabras, str):
            palabras = [palabras]

        lemas = set()

        for palabra in palabras:
            palabra = normalizar(palabra).strip()

            if not palabra or re.search(r"\s", palabra):
                raise ValueError("Cada stop-word debe ser una palabra no vacía")

            lema = analizar_token(palabra, self.lexico)[0]
            lemas.add(lema)

        return lemas

    def add_stopwords(self, palabras: Iterable[str]) -> None:
        """
        Añade lemas: también afecta a sus formas flexionadas
        """
        self.stop_words.update(self._lemas_stopwords(palabras))

    def remove_stopwords(self, palabras: Iterable[str]) -> None:
        """
        Retira lemas de las stop-words para los análisis posteriores
        """
        self.stop_words.difference_update(self._lemas_stopwords(palabras))

    def __call__(self, texto: str) -> Doc:
        if not isinstance(texto, str):
            raise TypeError("El texto debe ser una cadena de caracteres (str)")

        tokens = []

        # Posiciones sobre el original; normalización dentro del análisis.
        for i, coincidencia in enumerate(PATRON_TOKEN.finditer(texto)):
            original = coincidencia.group(0)
            lema, pos, rasgos = analizar_token(original, self.lexico)

            tokens.append(
                Token(
                    text=original,
                    lemma_=lema,
                    pos_=pos,
                    is_stop=lema in self.stop_words,
                    morph=rasgos,
                    idx=coincidencia.start(),
                    i=i
                    ))

        return Doc(texto, tokens)

    def pipe(self, textos: Iterable[str]) -> Iterator[Doc]:
        """
        Procesa una secuencia de textos reutilizando el mismo modelo
        """
        if isinstance(textos, str):
            raise TypeError("pipe espera una secuencia de textos, no un solo str")

        for texto in textos:
            yield self(texto)



def preparar_texto(texto: str, modelo: EsperantoNLP | None = None, *, quitar_numeros: bool = True) -> list[str]:
    """
    Devuelve lemas útiles, manteniendo su orden y sus repeticiones
    """
    if modelo is None:
        modelo = EsperantoNLP()

    lemas = []

    for token in modelo(texto):
        if token.like_num:
            if not quitar_numeros:
                lemas.append(token.lemma_)
            continue

        if token.is_alpha and not token.is_stop:
            lemas.append(token.lemma_)

    return lemas