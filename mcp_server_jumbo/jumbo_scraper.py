"""Scraper de Jumbo.cl usando Playwright async.

Estrategia:
  1. Intercepta las respuestas JSON de la API de búsqueda de Jumbo
  2. Extrae la URL del primer producto del resultado
  3. Intercepta la respuesta JSON de la página del producto
  4. Parsea nutrición + calcula sellos
"""

import json
import logging
import re
from typing import Optional
from urllib.parse import quote_plus

import requests as _requests
from bs4 import BeautifulSoup
from playwright.async_api import TimeoutError as PWTimeout
from playwright.async_api import async_playwright

_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "es-CL,es;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

logger = logging.getLogger(__name__)

JUMBO_BASE = "https://www.jumbo.cl"

_NUTRIENTES_MAP = {
    # Energía
    "energía":                          "calorias_kcal",
    "energia":                          "calorias_kcal",
    "calorías":                         "calorias_kcal",
    "calorias":                         "calorias_kcal",
    "valor energético":                  "calorias_kcal",
    "valor energetico":                  "calorias_kcal",
    # Sodio
    "sodio":                            "sodio_mg",
    # Azúcares
    "azúcares":                         "azucares_g",
    "azucares":                         "azucares_g",
    "azúcares totales":                  "azucares_g",
    "azucares totales":                  "azucares_g",
    # Grasas
    "grasas saturadas":                  "grasas_sat_g",
    "ácidos grasos saturados":           "grasas_sat_g",
    "acidos grasos saturados":           "grasas_sat_g",
    "grasas totales":                    "grasas_totales_g",
    "lípidos totales":                   "grasas_totales_g",
    "lipidos totales":                   "grasas_totales_g",
    "grasas monoinsaturadas":            "grasas_mono_g",
    "ácidos grasos monoinsaturados":     "grasas_mono_g",
    "acidos grasos monoinsaturados":     "grasas_mono_g",
    "grasas poliinsaturadas":            "grasas_poli_g",
    "ácidos grasos poliinsaturados":     "grasas_poli_g",
    "acidos grasos poliinsaturados":     "grasas_poli_g",
    "grasas trans":                      "grasas_trans_g",
    "ácidos grasos trans":               "grasas_trans_g",
    "acidos grasos trans":               "grasas_trans_g",
    # Colesterol
    "colesterol":                        "colesterol_mg",
    # Proteínas
    "proteínas":                         "proteinas_g",
    "proteinas":                         "proteinas_g",
    # Carbohidratos
    "hidratos de carbono":               "carbohidratos_g",
    "hidratos de carbono disponibles":   "carbohidratos_g",
    "carbohidratos":                     "carbohidratos_g",
    # Fibra
    "fibra":                             "fibra_g",
    "fibra dietética":                   "fibra_g",
    "fibra dietetica":                   "fibra_g",
    "fibra alimentaria":                 "fibra_g",
    "fibra dietaria":                    "fibra_g",
    "fibra soluble":                     "fibra_g",
}

_LIQUIDO_KEYWORDS = {
    "bebida", "jugo", "leche", "agua", "yogur", "yogurt",
    "néctar", "nectar", "refresco", "zumo", "licor", "vino",
    "cerveza", "té", "te", "café", "cafe", "infusión",
}

# URLs de API que Jumbo usa internamente
_SEARCH_URL_PATTERNS = [
    "bff.jumbo.cl/catalog/plp",
    "intelligent-search",
    "product_search",
    "graphql",
    "_search",
]

_PRODUCT_URL_PATTERNS = [
    "bff.jumbo.cl/catalog/pdp",
    "bff.jumbo.cl/product",
    "product?",
    "/products/",
    "productById",
]


def _parsear_tabla_texto(full_text: str) -> dict:
    """Parsea la tabla nutricional desde texto plano cuando no hay <table> HTML.

    Jumbo renderiza la tabla como líneas:
        Por cada 100g/ml
        Por cada 1 porción
        Energía (kCal)
        60
        72
        Proteínas (g)
        ...
    """
    nutricion: dict = {}

    # Encontrar inicio después de los encabezados de columna
    inicio = -1
    for marker in ("Por cada 100g/ml", "Por cada 100 g/ml", "Por cada 100g", "por cada 100g"):
        idx = full_text.find(marker)
        if idx >= 0:
            inicio = idx + len(marker)
            break
    if inicio < 0:
        return nutricion

    # Saltar la 2.ª línea de cabecera ("Por cada 1 porción" o similar)
    seccion = full_text[inicio:inicio + 3000]
    m_hdr2 = re.search(r"Por cada [^\n]+\n", seccion)
    if m_hdr2:
        seccion = seccion[m_hdr2.end():]

    lines = [l.strip() for l in seccion.split("\n")]

    _STOP_WORDS = {
        "características", "tipo de producto", "ingredientes",
        "puede contener", "alérgenos", "alergenos", "información adicional",
    }

    i = 0
    while i < len(lines):
        nombre_raw = lines[i]
        # Debe tener letras
        if not nombre_raw or not re.search(r"[a-zA-ZáéíóúÁÉÍÓÚñÑ]", nombre_raw):
            i += 1
            continue
        # Parar en notas al pie o nuevas secciones
        if nombre_raw.startswith("*") or nombre_raw.lower() in _STOP_WORDS:
            break
        # El siguiente debe ser un número (valor por 100g)
        val_str = lines[i + 1] if i + 1 < len(lines) else ""
        num = _extraer_numero(val_str)
        if num is None:
            i += 1
            continue
        # Normalizar: quitar "(unidad)" al final y pasar a minúsculas
        nombre_limpio = re.sub(r"\s*\([^)]*\)\s*$", "", nombre_raw).strip().lower()
        clave = _NUTRIENTES_MAP.get(nombre_limpio)
        if clave:
            if clave not in nutricion:
                nutricion[clave] = num
        else:
            # Guardar igualmente con clave normalizada (snake_case)
            clave_extra = re.sub(r"\s+", "_", nombre_limpio)
            if clave_extra not in nutricion:
                nutricion[clave_extra] = num
        i += 3  # nombre + valor_100g + valor_porción
    return nutricion


def _extraer_numero(texto: str) -> Optional[float]:
    texto = texto.replace(",", ".")
    m = re.search(r"[\d]+\.?\d*", texto)
    return float(m.group()) if m else None


def _es_liquido(nombre: str, categoria: str) -> bool:
    txt = (nombre + " " + categoria).lower()
    return any(k in txt for k in _LIQUIDO_KEYWORDS)


def _parsear_specs(specs_groups: list) -> tuple[dict, str]:
    """Extrae nutrición e ingredientes de specificationGroups de VTEX."""
    nutricion: dict = {}
    ingredientes = ""
    for grupo in specs_groups or []:
        for spec in grupo.get("specifications") or []:
            nombre = spec.get("name", "").strip().lower()
            valores = spec.get("values") or []
            valor   = valores[0] if valores else ""
            clave = _NUTRIENTES_MAP.get(nombre)
            if clave:
                num = _extraer_numero(valor)
                if num is not None:
                    nutricion[clave] = num
            if "ingrediente" in nombre:
                ingredientes = valor
    return nutricion, ingredientes


def _extraer_producto_de_search(data: dict) -> Optional[str]:
    """Intenta sacar la URL/slug del primer producto de una respuesta de búsqueda."""
    # Formato VTEX Intelligent Search
    products = (
        data.get("products")
        or data.get("data", {}).get("productSearch", {}).get("products")
        or data.get("data", {}).get("search", {}).get("products")
        or []
    )
    if not products:
        # Buscar recursivamente listas que parezcan productos
        def _buscar(obj):
            if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                if "linkText" in obj[0] or "slug" in obj[0] or "link" in obj[0]:
                    return obj
            if isinstance(obj, dict):
                for v in obj.values():
                    r = _buscar(v)
                    if r:
                        return r
            return None
        products = _buscar(data) or []

    if not products:
        return None

    prod = products[0]
    slug = prod.get("linkText") or prod.get("slug") or prod.get("link", "")
    if slug and not slug.startswith("http"):
        return f"{JUMBO_BASE}/{slug}/p"
    return slug or None


def _extraer_datos_producto(data: dict, url: str) -> Optional[dict]:
    """Intenta extraer datos desde __NEXT_DATA__ (VTEX legacy)."""
    product = (
        data.get("product")
        or data.get("data", {}).get("product")
        or data.get("props", {}).get("pageProps", {}).get("product")
    )
    if not product:
        def _find_product(obj):
            if isinstance(obj, dict):
                if "productName" in obj and "specificationGroups" in obj:
                    return obj
                for v in obj.values():
                    r = _find_product(v)
                    if r:
                        return r
            return None
        product = _find_product(data)

    if not product:
        return None

    nombre    = product.get("productName", "")
    marca     = product.get("brand", "")
    categorias = product.get("categories") or [""]
    categoria  = categorias[0] if categorias else ""
    nutricion, ingredientes = _parsear_specs(product.get("specificationGroups"))

    return {
        "nombre": nombre, "marca": marca, "categoria": categoria,
        "nutricion": nutricion, "ingredientes": ingredientes,
        "es_liquido": _es_liquido(nombre, categoria), "url": url,
    }


def _scrape_producto_html(product_url: str) -> Optional[dict]:
    """Descarga la página del producto con requests y parsea con BeautifulSoup."""
    try:
        resp = _requests.get(product_url, headers=_HTTP_HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        logger.error("Error descargando página del producto: %s", e)
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    # ── 1. Intentar __NEXT_DATA__ (VTEX / Next.js) ───────────────────────────
    next_script = soup.find("script", {"id": "__NEXT_DATA__"})
    if next_script and next_script.string:
        try:
            next_data = json.loads(next_script.string)
            resultado = _extraer_datos_producto(next_data, product_url)
            if resultado and resultado.get("nutricion"):
                logger.info("__NEXT_DATA__ OK — nombre='%s' nutricion=%s",
                            resultado["nombre"], list(resultado["nutricion"].keys()))
                return resultado
            if resultado:
                logger.info("__NEXT_DATA__ encontrado pero sin nutrición, intentando HTML…")
        except Exception as e:
            logger.warning("Error parseando __NEXT_DATA__: %s", e)

    # ── 2. Fallback: parsear HTML directamente ────────────────────────────────
    nombre = ""
    h1 = soup.find("h1")
    if h1:
        nombre = h1.get_text(strip=True)

    # Marca
    marca = ""
    for tag in soup.find_all(["a", "span", "p"], class_=re.compile(r"brand", re.I)):
        txt = tag.get_text(strip=True)
        if txt and txt != nombre:
            marca = txt
            break

    full_text = soup.get_text(separator="\n")

    # Tabla nutricional — primero intenta <table>, luego parser de texto
    nutricion: dict = {}
    for table in soup.find_all("table"):
        headers = [th.get_text(strip=True).lower() for th in table.find_all(["th", "td"])[:5]]
        col_100g = next(
            (i for i, t in enumerate(headers) if "100g" in t or "100 g" in t),
            1,
        )
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) <= col_100g:
                continue
            raw_name   = cells[0].get_text(strip=True).lower()
            clean_name = re.sub(r"\s*\([^)]*\)", "", raw_name).strip()
            clave = _NUTRIENTES_MAP.get(clean_name) or _NUTRIENTES_MAP.get(raw_name)
            if clave:
                num = _extraer_numero(cells[col_100g].get_text(strip=True))
                if num is not None:
                    nutricion[clave] = num
        if nutricion:
            break

    # Fallback: parsear la tabla desde el texto plano (Jumbo usa divs, no <table>)
    if not nutricion:
        nutricion = _parsear_tabla_texto(full_text)

    # Ingredientes — busca la sección y toma el texto hasta la siguiente sección
    ingredientes = ""
    m = re.search(
        r"Ingredientes\s*[:\n]+([\s\S]+?)(?=\n{2,}|Puede contener|Tabla nutricional|Información nutricional|Alérgenos|Alergenos|Características|$)",
        full_text,
        re.IGNORECASE,
    )
    if m:
        raw = m.group(1).strip()
        # Limpiar si quedó el encabezado repetido
        raw = re.sub(r"^[Ii]ngredientes\s*[:\n]*", "", raw).strip()
        # Colapsar saltos de línea internos a espacio
        ingredientes = re.sub(r"\n+", " ", raw).strip()

    logger.info("HTML parseado — nombre='%s' nutricion=%s ingredientes=%s",
                nombre, list(nutricion.keys()), bool(ingredientes))

    if not nombre and not nutricion:
        return None

    return {
        "nombre":       nombre,
        "marca":        marca,
        "categoria":    "",
        "nutricion":    nutricion,
        "ingredientes": ingredientes,
        "es_liquido":   _es_liquido(nombre, ""),
        "url":          product_url,
    }


async def buscar_producto(nombre: str) -> Optional[dict]:
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        ctx = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            locale="es-CL",
        )
        page = await ctx.new_page()
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4,5]});
            Object.defineProperty(navigator, 'languages', {get: () => ['es-CL','es','en']});
            window.chrome = {runtime: {}};
            Object.defineProperty(navigator, 'permissions', {
                get: () => ({ query: () => Promise.resolve({ state: 'granted' }) })
            });
        """)

        # ── Interceptar respuestas JSON ──────────────────────────────────────
        json_responses: list[tuple[str, dict]] = []

        async def on_response(response):
            ct = response.headers.get("content-type", "")
            if "json" not in ct:
                return
            url_r = response.url
            try:
                data = await response.json()
                json_responses.append((url_r, data))
                logger.debug("JSON capturado: %s", url_r)
            except Exception:
                pass

        page.on("response", on_response)

        try:
            # 1. Página de búsqueda
            slug = nombre.strip().lower().replace(" ", "-")
            search_url = f"{JUMBO_BASE}/busqueda?ft={quote_plus(slug)}"
            logger.info("Buscando: %s", search_url)
            await page.goto(search_url, timeout=30_000, wait_until="load")
            await page.evaluate("window.scrollBy(0, 600)")
            await page.wait_for_timeout(2_500)

            # Screenshot de debug
            await page.screenshot(path="/tmp/jumbo_search_debug.png", full_page=False)
            logger.info("Screenshot guardado en /tmp/jumbo_search_debug.png")
            logger.info("APIs capturadas en búsqueda: %s", [u for u, _ in json_responses])

            # 2. Buscar URL del primer producto en las respuestas capturadas
            product_url = None
            for url_r, data in json_responses:
                if any(p in url_r for p in _SEARCH_URL_PATTERNS):
                    logger.info("Analizando API de búsqueda: %s — claves: %s", url_r, list(data.keys()) if isinstance(data, dict) else type(data).__name__)
                    product_url = _extraer_producto_de_search(data)
                    if product_url:
                        logger.info("Producto en API: %s → %s", url_r, product_url)
                        break

            # Fallback: buscar href en HTML
            if not product_url:
                logger.info("API no dio resultado, intentando HTML...")
                el = await page.query_selector("a[href$='/p']")
                if el:
                    href = await el.get_attribute("href") or ""
                    product_url = href if href.startswith("http") else f"{JUMBO_BASE}{href}"

            if not product_url:
                logger.warning("Sin resultados para '%s'", nombre)
                await browser.close()
                return None

            # 3. Scrapear página del producto con requests (sin Playwright)
            await browser.close()
            logger.info("Scrapeando producto con requests: %s", product_url)
            resultado = _scrape_producto_html(product_url)
            if resultado and resultado.get("nombre"):
                return resultado

            logger.warning("Sin datos en página del producto: %s", product_url)
            await browser.close()
            return None

        except PWTimeout:
            logger.warning("Timeout buscando '%s'", nombre)
            await browser.close()
            return None
        except Exception as e:
            logger.error("Error scraping '%s': %s", nombre, e, exc_info=True)
            try:
                await browser.close()
            except Exception:
                pass
            return None
