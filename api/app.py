import os
import sys
import json
import sqlite3
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, jsonify, request
from models.engine import RecommendationEngine

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                       "data", "recommendation.db")

print("[API] Loading engine — please wait...")
engine = RecommendationEngine(DB_PATH)
print("[API] Engine ready ✓")

app = Flask(__name__)


def ok(data):
    return jsonify({"status": "ok", "data": data,
                    "timestamp": datetime.utcnow().isoformat()})

def err(msg, code=400):
    return jsonify({"status": "error", "message": msg}), code


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return ok({
        "service":      "Recommendation API",
        "products":     len(engine.products),
        "users":        len(engine.users),
        "interactions": len(engine.interactions),
    })


@app.route("/recommend/<user_id>")
def recommend(user_id):
    top_n  = min(int(request.args.get("n", 10)), 50)
    method = request.args.get("method", "hybrid")
    save   = request.args.get("save", "true").lower() == "true"

    if method not in ("hybrid", "collaborative", "content", "popular"):
        return err("method must be: hybrid | collaborative | content | popular")

    if user_id not in engine.users["user_id"].values:
        return err(f"user_id '{user_id}' not found", 404)

    recs = engine.recommend(user_id, top_n=top_n, method=method)

    if save:
        engine.save_recommendations(user_id, recs, method)

    return ok({
        "user_id":         user_id,
        "method":          method,
        "top_n":           top_n,
        "recommendations": json.loads(recs.to_json(orient="records")),
    })


@app.route("/similar/<product_id>")
def similar(product_id):
    top_n = min(int(request.args.get("n", 10)), 50)

    if product_id not in engine.products["product_id"].values:
        return err(f"product_id '{product_id}' not found", 404)

    sims = engine.recommend_similar(product_id, top_n=top_n)
    src  = engine.products[engine.products["product_id"] == product_id].iloc[0]

    return ok({
        "source_product": {
            "product_id": src["product_id"],
            "name":       src["name"],
            "category":   src["category"],
        },
        "similar_products": json.loads(sims.to_json(orient="records")),
    })


@app.route("/popular")
def popular():
    top_n    = min(int(request.args.get("n", 10)), 50)
    category = request.args.get("category", None)

    recs = engine._popular_fallback(top_n * 3)
    if category:
        recs = recs[recs["category"].str.lower() == category.lower()]
    recs = recs.head(top_n)

    return ok({
        "category": category,
        "top_n":    top_n,
        "items":    json.loads(recs.to_json(orient="records")),
    })


@app.route("/feedback", methods=["POST"])
def feedback():
    body = request.get_json(force=True, silent=True)
    if not body:
        return err("JSON body required")

    required = ["user_id", "product_id", "event_type"]
    missing  = [k for k in required if k not in body]
    if missing:
        return err(f"Missing fields: {missing}")

    valid = {"view", "add_to_cart", "purchase", "wishlist"}
    if body["event_type"] not in valid:
        return err(f"event_type must be one of {valid}")

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO interactions (user_id, product_id, event_type, rating, timestamp) "
        "VALUES (?, ?, ?, ?, ?)",
        (body["user_id"], body["product_id"], body["event_type"],
         body.get("rating", 3.0), datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

    return ok({"message": "Feedback recorded", "data": body})


@app.route("/stats")
def stats():
    conn       = sqlite3.connect(DB_PATH)
    saved_recs = conn.execute("SELECT COUNT(*) FROM recommendations").fetchone()[0]
    conn.close()

    return ok({
        "total_products":        len(engine.products),
        "total_users":           len(engine.users),
        "total_interactions":    len(engine.interactions),
        "saved_recommendations": saved_recs,
        "products_by_category":  engine.products["category"].value_counts().to_dict(),
        "avg_rating":            round(float(engine.products["rating"].mean()), 2),
    })


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)