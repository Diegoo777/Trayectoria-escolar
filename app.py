import os
import uuid

from flask import Flask, jsonify, render_template, request, session

from analysis import run_analysis, analyze_cohort

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

ALLOWED = {"xlsx", "xls", "csv"}

# Server-side data cache: token → {pob_unico, resumen, ciclos, df_T}
_cache: dict = {}


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    mat_file = request.files.get("materias_file")

    if not mat_file or mat_file.filename == "":
        return render_template("index.html", error="Debes subir el archivo de materias.")

    if not _allowed(mat_file.filename):
        return render_template(
            "index.html", error="Formato no soportado. Usa .xlsx, .xls o .csv"
        )

    tit_data = None
    tit_file = request.files.get("titulaciones_file")
    if tit_file and tit_file.filename and _allowed(tit_file.filename):
        tit_data = (tit_file.read(), tit_file.filename)

    try:
        ctx, cache = run_analysis(mat_file.read(), mat_file.filename, tit_data)

        token = str(uuid.uuid4())
        session["token"] = token
        _cache[token] = cache

        return render_template("dashboard.html", **ctx)

    except ValueError as e:
        return render_template("index.html", error=str(e))
    except Exception as e:
        return render_template("index.html", error=f"Error al procesar el archivo: {e}")


@app.route("/api/cohort")
def api_cohort():
    token = session.get("token")
    if not token or token not in _cache:
        return jsonify({"error": "Sesión expirada. Sube el archivo nuevamente."}), 400

    ciclo = request.args.get("ciclo", "")
    cache = _cache[token]

    if ciclo not in cache["ciclos"]:
        return jsonify({"error": "Generación no encontrada."}), 400

    result = analyze_cohort(
        ciclo,
        cache["pob_unico"],
        cache["resumen"],
        cache["ciclos"],
        cache.get("df_T"),
    )
    return jsonify(result)


@app.route("/demo")
def demo():
    """Load dashboard using local sample files (only works when running locally)."""
    base = os.path.join(os.path.dirname(__file__), "..")
    mat_path = os.path.join(base, "BaseDatosMaterias.xlsx")
    tit_path = os.path.join(base, "Titulaciones.xlsx")
    if not os.path.exists(mat_path):
        return "Archivo BaseDatosMaterias.xlsx no encontrado.", 404
    with open(mat_path, "rb") as f:
        mat_data = f.read()
    tit_data = None
    if os.path.exists(tit_path):
        with open(tit_path, "rb") as f:
            tit_data = (f.read(), "Titulaciones.xlsx")
    try:
        ctx, cache = run_analysis(mat_data, "BaseDatosMaterias.xlsx", tit_data)
        token = str(uuid.uuid4())
        session["token"] = token
        _cache[token] = cache
        return render_template("dashboard.html", **ctx)
    except Exception as e:
        return f"Error: {e}", 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)
