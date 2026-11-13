import unicodedata

import pytest

from esperanto_nlp import (
    Doc,
    EsperantoNLP,
    Token,
    cargar_vocabulario,
    normalizar,
    obtener_stopwords,
    preparar_texto,
    tokenizar,
)


@pytest.fixture
def nlp():
    return EsperantoNLP()


@pytest.mark.parametrize("palabra,lema,pos", [
    ("komputilojn", "komputilo", "NOUN"),
    ("libroj", "libro", "NOUN"),
    ("sekurajn", "sekura", "ADJ"),
    ("programis", "programi", "VERB"),
    ("lernos", "lerni", "VERB"),
    ("farus", "fari", "VERB"),
    ("legu", "legi", "VERB"),
    ("legi", "legi", "VERB"),
    ("estas", "esti", "VERB"),
    ("havis", "havi", "VERB"),
    ("min", "mi", "PRON"),
    ("ilin", "ili", "PRON"),
    ("siajn", "sia", "ADJ"),
    ("miajn", "mia", "ADJ"),
    ("kiujn", "kiu", "PRON"),
    ("kion", "kio", "PRON"),
    ("kiaj", "kia", "ADJ"),
    ("kies", "kies", "DET"),
    ("kien", "kie", "ADV"),
    ("neniam", "neniam", "ADV"),
    ("hejmen", "hejme", "ADV"),
    ("rapide", "rapide", "ADV"),
    ("supren", "supre", "ADV"),
    ("ĝis", "ĝis", "ADP"),
    ("plu", "plu", "ADV"),
    ("tri", "tri", "NUM"),
    ("dudek", "dudek", "NUM"),
    ("ducent", "ducent", "NUM"),
    ("dumil", "dumil", "NUM"),
    ("duan", "dua", "ADJ"),
    ("dudekaj", "dudeka", "ADJ"),
    ("milionojn", "miliono", "NOUN"),
    ("miliardoj", "miliardo", "NOUN"),
    ("ajn", "ajn", "PART"),
    ("jen", "jen", "PART"),
    ("nun", "nun", "ADV"),
    ("la", "la", "DET"),
    ("mi", "mi", "PRON"),
    ("ĉu", "ĉu", "PART"),
    ("kaj", "kaj", "CCONJ"),
    ("ke", "ke", "SCONJ"),
    ("adiaŭ", "adiaŭ", "INTJ"),
    ("bonvolu", "bonvoli", "VERB"),
    ("feliĉajn", "feliĉa", "ADJ"),
    ("legantajn", "leganta", "ADJ"),
    ("legitajn", "legita", "ADJ"),
    ("plantojn", "planto", "NOUN"),
    ("malbonajn", "malbona", "ADJ"),
    ("instruistinojn", "instruistino", "NOUN"),
    ("3,14", "3,14", "NUM"),
    ("HTML", "html", "X"),
    ("xeno", "xeno", "X"),
    ("tioj", "tioj", "X"),
    ("?", "?", "PUNCT"),
    ("€", "€", "SYM"),
])
def test_lema_pos(nlp, palabra, lema, pos):
    doc = nlp(palabra)

    assert len(doc) == 1
    assert (doc[0].lemma_, doc[0].pos_) == (lema, pos)


@pytest.mark.parametrize("texto", ["", " ", "\t\n"])
def test_vacio(nlp, texto):
    assert tokenizar(texto) == []
    assert len(nlp(texto)) == 0
    assert preparar_texto(texto, nlp) == []


def test_tokenizacion():
    assert tokenizar("La komputilo kostas 25,50 eŭrojn.") == ["La", "komputilo", "kostas", "25,50", "eŭrojn", "."]


def test_normalizacion():
    assert normalizar("ĈU ĜI FUNKCIAS?") == "ĉu ĝi funkcias?"
    assert normalizar("Ĉu", minusculas=False) == "Ĉu"


def test_unicode_y_posiciones(nlp):
    # NFD descompone ĉ en c + diacrítico: comprobamos que lemas y posiciones siguen siendo correctos
    texto = unicodedata.normalize("NFD", "  Ĉu ĝi funkcias?  ")
    doc = nlp(texto)

    assert doc.text == texto
    assert [t.lemma_ for t in doc] == ["ĉu", "ĝi", "funkcii", "?"]

    assert [
        normalizar(t.text, minusculas=False)
        for t in doc
    ] == tokenizar(texto)

    for indice, token in enumerate(doc):
        assert token.i == indice
        assert (texto[token.idx:token.idx + len(token.text)] == token.text)


def test_recurso():
    vocabulario = cargar_vocabulario()
    stopwords = obtener_stopwords(vocabulario)

    assert vocabulario["al"]["kategorio"] == "prepozicio"
    assert "al" in stopwords
    assert "bona" not in stopwords


def test_rasgos(nlp):
    assert nlp("retojn")[0].morph == {"Number": "Plur","Case": "Acc",}

    assert nlp("miajn")[0].morph["Poss"] == "Yes"
    assert nlp("legis")[0].morph["Tense"] == "Past"
    assert nlp("duan")[0].morph["NumType"] == "Ord"


@pytest.mark.parametrize("palabra",["min", "miajn", "estas", "kiujn", "supren"],)
def test_stopwords_flexionadas(nlp, palabra):
    assert nlp(palabra)[0].is_stop


def test_negaciones():
    predeterminado = EsperantoNLP()
    conservar = EsperantoNLP(conservar_negaciones=True)

    for palabra in ("ne", "nek", "sen", "neniam", "neniun"):
        assert predeterminado(palabra)[0].is_stop
        assert not conservar(palabra)[0].is_stop


def test_stopwords_personalizadas():
    nlp = EsperantoNLP(stop_words=[])

    assert not nlp("la")[0].is_stop

    nlp.add_stopwords(["KOMPUTILOJN"])

    assert nlp("komputilo")[0].is_stop
    assert nlp("komputiloj")[0].is_stop

    nlp.remove_stopwords("komputilon")

    assert not nlp("komputilojn")[0].is_stop

def test_pipe_y_doc(nlp):
    docs = list(nlp.pipe(["La reto.", "La programo."]))

    assert len(docs) == 2
    assert isinstance(docs[0], Doc)
    assert isinstance(docs[0][0], Token)
    assert len(docs[0][:2]) == 2
    assert docs[0][-1].is_punct


def test_preparar(nlp):
    texto = "La programistoj protektas la komputilajn retojn."

    assert preparar_texto(texto, nlp) == ["programisto", "protekti", "komputila", "reto"]

    assert preparar_texto("reto retoj retojn", nlp) == ["reto", "reto", "reto"]

    assert preparar_texto("du 2 retoj", nlp, quitar_numeros=False) == ["du", "2", "reto"]

    assert preparar_texto("du 2 retoj", nlp) == ["reto"]
    assert preparar_texto("HTML retoj", nlp) == ["html", "reto"]


@pytest.mark.parametrize("valor", [None, 12, ["reto"], {}])
def test_tipo_incorrecto(nlp, valor):
    with pytest.raises(TypeError):
        nlp(valor)

    with pytest.raises(TypeError):
        tokenizar(valor)


def test_pipe_incorrecto(nlp):
    with pytest.raises(TypeError):
        list(nlp.pipe("La reto."))


@pytest.mark.parametrize("valor", ["", "   ", "du vortoj"])
def test_stopword_invalida(nlp, valor):
    with pytest.raises(ValueError):
        nlp.add_stopwords([valor])

# Incluimos un test nuestro, para verificar que funciona
def test_frase_propia(nlp):
    texto = "La programistoj analizas datumojn kaj protektas komputilajn retojn."

    assert preparar_texto(texto, nlp) == ["programisto","analizi","datumo","protekti","komputila","reto"]