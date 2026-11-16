# Diseño de jab

`prototipo/` es la referencia visual y de interacción de la app (issue #13). Se revisa y se aprueba aquí antes de llevarlo a `app/`.

```powershell
python design/prototipo/server.py   # http://localhost:8700, usa el motor real
```

## Sistema

Todo valor sale de los tokens de `prototipo/styles.css`; los componentes no usan colores ni medidas sueltas.

- **Paleta:** fondo casi blanco, tinta casi negra del mismo tono y un solo acento. Tres candidatas: ciruela (predeterminada), oro viejo y oliva. Todo par de texto cumple WCAG AA (≥ 4,5:1).
- **Tipografía:** Schibsted Grotesk. Cifras tabulares solo en rango y tiempo; nada en monoespaciada.
- **Forma:** radio interior = radio exterior − relleno (barra 28 px, chip 22 px; menú 18 px, opción 10 px). Esquinas continuas donde el navegador las admite.
- **Movimiento:** salida rápida (ease-out) de 120–260 ms; menús de 180 ms que nacen de su botón y salen un 20 % más rápido; filas escalonadas 30 ms (máximo 8); pulsación a 0,97; sin rebotes. `prefers-reduced-motion` lo desactiva todo.

## Reglas de pantalla

- Una sola barra. Modelo, top-n y Porter viven en el menú del chip (también Ctrl/⌘ K, teclas 1–4).
- FiQA no tiene títulos: la primera frase de cada documento hace de titular; el resto, dos líneas.
- El id del documento se muestra discreto ("Doc. 501384") porque la issue lo exige.
- Mientras llega una búsqueda nueva, los resultados anteriores se atenúan y una línea fina indica progreso; la primera búsqueda muestra esqueletos.
- Esc solo cierra menús; nunca borra lo escrito.
