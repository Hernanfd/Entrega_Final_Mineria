# Descarga y extracción de párrafos
import re
import unicodedata

import requests
from bs4 import BeautifulSoup


SELECTORES = (
    "div.itemBody.kl-blog-post-body",
    "article .entry-content",
    "article .post-content",
    "[itemprop='articleBody']",
    ".entry-content",
    ".post-content",
    ".elementor-widget-theme-post-content",
    "main article",
    "article",
)


def extraer_parrafos_html(html: bytes) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")

    # Quitamos elementos que no forman parte del texto del artículo.
    for elemento in soup.select(
        "script, style, noscript, nav, header, footer, aside, form, "
        "button, .comments, #comments, .related-posts, .sharedaddy"
    ):
        elemento.decompose()

    for selector in SELECTORES:
        contenedor = soup.select_one(selector)
        if contenedor is None:
            continue

        parrafos = []
        for etiqueta in contenedor.find_all("p"):
            texto = etiqueta.get_text(" ", strip=True)
            texto = unicodedata.normalize("NFC", texto)
            texto = re.sub(r"\s+", " ", texto).strip()

            if texto:
                parrafos.append(texto)

        if parrafos:
            return parrafos

    raise ValueError(
        "No se encontraron párrafos en el cuerpo del artículo. "
        "Hay que revisar la página y sus selectores."
    )


def extraer_parrafos(
    url: str, sesion: requests.Session
) -> tuple[str, list[str]]:
    respuesta = sesion.get(url, timeout=(10, 30))
    respuesta.raise_for_status()
    
    html = BeautifulSoup(respuesta.content, "html.parser")
    print("Título:", html.title.get_text(strip=True) if html.title else "Sin título")
    print("Número de párrafos en el HTML:", len(html.find_all("p")))

    with open("pagina_prueba.html", "wb") as archivo:
        archivo.write(respuesta.content)

    tipo = respuesta.headers.get("Content-Type", "").lower()
    if tipo and "html" not in tipo:
        raise ValueError(f"La URL no devuelve HTML: {tipo}")

    parrafos = extraer_parrafos_html(respuesta.content)
    return respuesta.url, parrafos