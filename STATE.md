# Estado del proyecto

Actualizado: miércoles 11 de noviembre de 2026.

## Ubicación y alcance

La planificación está integrada en `main` (PR #19). La implementación de #1 está en curso en la rama `chore/01-estructura-entorno-ci`, sin commit ni PR todavía. El plazo interno es el martes 17 de noviembre; del 18 al 23 queda contingencia.

## Avance de #1 (estructura, entorno y CI)

- **Probado localmente** (`.venv` con Python 3.12.10 sobre Windows): `python -m pytest` pasa 1 prueba, `python -m ruff check .` sin errores, `import ris` funciona y `pip check` no encuentra conflictos.
- **Escrito, sin ejecutar**: `.github/workflows/ci.yml` (lint y tests en cada `pull_request`, Ubuntu, Python 3.12). Las versiones de las acciones existen, pero el CI no se ha corrido; la instalación con torch CPU en Linux solo se confirmará en el PR.
- Creados: `pyproject.toml`, `requirements.txt` con versiones exactas, `src/ris/`, `tests/`, `app/`, `scripts/`, `notebooks/`, `reports/`, `docs/informe/` y el comando de instalación en `readme.md`.
- **Pendiente**: commit (archivo por archivo), push, PR con `Closes #1`, CI en verde y merge. Tras el merge, ajustar "Comandos previstos" de AGENTS.md, que sigue diciendo que se habilitan al cerrar #1.

## GitHub configurado y verificado por API el 9 de noviembre de 2026

- Ruleset `proteger-main` (id 26888439), activo en la rama por defecto: reglas `deletion`, `non_fast_forward`, `update` y `pull_request`; exige 1 aprobación, CODEOWNER, descarta aprobaciones con commits nuevos y requiere resolver hilos. El rol admin puede omitir requisitos solo mediante bypass `pull_request`; no hay push directo.
- Borrado automático de ramas al hacer merge activado.
- Seis milestones M0–M5 creados con las fechas del plan; 12 etiquetas creadas (9 de área y 3 `resp:*`).
- Las 18 issues #1–#18 están creadas con la numeración del plan. Todas asignadas: Vladimirjon (10), Ismael Freire @BrandonFreire (#3, #6, #10, #14) y Jorge Torres @Almejandro04 (#4, #12, #15, #17).

## Decisiones

Python 3.12, `src/ris/`, FiQA inglés, cuatro recuperadores propios/FAISS, evaluación P@10/R@10 y MAP@1000, Streamlit en HF Spaces CPU básico y artefactos fuera de Git; detalle en [PLAN](docs/PLAN.md).

## Próximos pasos

1. #1 (Vladimirjon): publicar la rama y abrir el PR; es lo único que falta. Bloquea a todos y ya pasó la fecha interna del 10 de noviembre.
2. En cuanto #1 esté en `main`: #2 (Vladimirjon) y #3 (Ismael) en paralelo; #10 (Ismael) puede empezar con los contratos del plan.
3. Jorge prepara el esqueleto del informe (#17) y el notebook de #4, que ejecuta al tener #2 y #3.

## Aprendido y pendiente de confirmar

- El equipo tenía solo Python 3.14; #1 exige 3.12 (`requires-python = ">=3.12,<3.13"`). Se instaló 3.12 en paralelo con winget y el `.venv` se crea con `py -3.12`.

- Los qrels usan ids `int64`, mientras que los ids del corpus son `str`; #2 debe normalizarlos coherentemente.
- Las 648 consultas test únicas siguen pendientes de confirmar en #2; no confundirlas con las 1 706 filas de qrels.
