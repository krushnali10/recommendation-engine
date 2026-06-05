import sqlite3
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings("ignore")


class RecommendationEngine:
    def __init__(self, db_path):
        self.db_path  = db_path
        self.products = None
        self.users    = None
        self.interactions = None

        self.user_item_matrix   = None
        self.user_similarity    = None
        self.item_similarity_cf = None
        self.item_similarity_cb = None
        self.tfidf_matrix       = None
        self.product_index      = None
        self.product_rev        = None
        self.user_index         = None

        self._load_data()
        self._feature_engineering()
        self._build_models()

    # ── Load from SQLite ──────────────────────────────────────────────────────
    def _load_data(self):
        conn = sqlite3.connect(self.db_path)
        self.products     = pd.read_sql("SELECT * FROM products",     conn)
        self.users        = pd.read_sql("SELECT * FROM users",        conn)
        self.interactions = pd.read_sql("SELECT * FROM interactions", conn)
        conn.close()
        print(f"[Engine] Loaded {len(self.products):,} products, "
              f"{len(self.users):,} users, "
              f"{len(self.interactions):,} interactions")

    # ── Feature Engineering ───────────────────────────────────────────────────
    def _feature_engineering(self):
        ix = self.interactions

        # 1. Event weighting
        weight_map = {"purchase": 5.0, "add_to_cart": 3.0,
                      "wishlist":  2.0, "view": 1.0}
        ix["event_weight"] = ix["event_type"].map(weight_map).fillna(1.0)

        # 2. Recency boost
        ix["timestamp"] = pd.to_datetime(ix["timestamp"], format="mixed")
        ref_date  = ix["timestamp"].max()
        days_ago  = (ref_date - ix["timestamp"]).dt.days
        ix["recency_boost"] = np.where(days_ago <= 30,  1.5,
                              np.where(days_ago <= 90,  1.2, 1.0))

        # 3. Composite score
        ix["composite_score"] = ix["rating"] * ix["event_weight"] * ix["recency_boost"]

        # 4. Popularity score per product
        pop = ix.groupby("product_id")["composite_score"].sum()
        pop = np.log1p(pop)
        scaler = MinMaxScaler()
        pop_scaled = scaler.fit_transform(pop.values.reshape(-1, 1)).flatten()
        self.products["popularity_score"] = (
            self.products["product_id"]
                .map(dict(zip(pop.index, pop_scaled)))
                .fillna(0)
        )

        # 5. Quality score
        q = self.products["rating"] * np.log1p(self.products["num_reviews"])
        self.products["quality_score"] = (
            scaler.fit_transform(q.values.reshape(-1, 1)).flatten()
        )

        # 6. Content features for TF-IDF
        self.products["content_features"] = (
            self.products["category"].fillna("") + " " +
            self.products["subcategory"].fillna("") + " " +
            self.products["brand"].fillna("") + " " +
            self.products["tags"].fillna("")
        )

        self.interactions = ix
        print("[Engine] Feature engineering complete ✓")

    # ── Build Models ──────────────────────────────────────────────────────────
    def _build_models(self):
        self._build_collaborative_filter()
        self._build_content_based_filter()
        print("[Engine] All models ready ✓")

    def _build_collaborative_filter(self):
        ix = self.interactions
        agg = (ix.groupby(["user_id", "product_id"])["composite_score"]
                 .max().reset_index())

        users    = sorted(agg["user_id"].unique())
        products = sorted(agg["product_id"].unique())
        self.user_index    = {u: i for i, u in enumerate(users)}
        self.product_index = {p: i for i, p in enumerate(products)}
        self.product_rev   = {i: p for p, i in self.product_index.items()}

        rows = agg["user_id"].map(self.user_index)
        cols = agg["product_id"].map(self.product_index)
        vals = agg["composite_score"].values

        sparse = csr_matrix((vals, (rows, cols)),
                            shape=(len(users), len(products)))
        self.user_item_matrix   = sparse
        self.user_similarity    = cosine_similarity(sparse, dense_output=False)
        self.item_similarity_cf = cosine_similarity(sparse.T, dense_output=False)
        print("[Engine] Collaborative filtering built ✓")

    def _build_content_based_filter(self):
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000)
        self.tfidf_matrix       = vectorizer.fit_transform(self.products["content_features"])
        self.item_similarity_cb = cosine_similarity(self.tfidf_matrix, dense_output=False)
        print("[Engine] Content-based filtering built ✓")

    # ── Recommendation Methods ────────────────────────────────────────────────
    def _cf_user_based(self, user_id, top_n=20):
        if user_id not in self.user_index:
            return pd.Series(dtype=float)
        u_idx   = self.user_index[user_id]
        sim_row = np.array(self.user_similarity[u_idx].todense()).flatten()
        sim_row[u_idx] = 0
        top_k   = 30
        similar = np.argsort(sim_row)[::-1][:top_k]
        weights = sim_row[similar]
        if weights.sum() == 0:
            return pd.Series(dtype=float)
        subset  = self.user_item_matrix[similar]
        weighted = (subset.T.multiply(weights)).T
        scores  = np.array(weighted.sum(axis=0)).flatten() / (weights.sum() + 1e-9)
        already = self.user_item_matrix[u_idx].nonzero()[1]
        scores[already] = 0
        return pd.Series(scores, index=list(self.product_rev.values()))

    def _cf_item_based(self, user_id, top_n=20):
        if user_id not in self.user_index:
            return pd.Series(dtype=float)
        u_idx  = self.user_index[user_id]
        user_v = np.array(self.user_item_matrix[u_idx].todense()).flatten()
        interacted = np.where(user_v > 0)[0]
        if len(interacted) == 0:
            return pd.Series(dtype=float)
        scores = np.zeros(self.user_item_matrix.shape[1])
        for i_idx in interacted:
            sim_row = np.array(self.item_similarity_cf[i_idx].todense()).flatten()
            scores += sim_row * user_v[i_idx]
        scores[interacted] = 0
        return pd.Series(scores, index=list(self.product_rev.values()))

    def _cb_scores(self, user_id, top_n=20):
        ix = self.interactions[self.interactions["user_id"] == user_id]
        if ix.empty:
            return pd.Series(dtype=float)
        all_pids = self.products["product_id"].tolist()
        prod_row = {pid: idx for idx, pid in enumerate(all_pids)}
        interacted_idxs = [prod_row[p] for p in ix["product_id"].unique() if p in prod_row]
        if not interacted_idxs:
            return pd.Series(dtype=float)
        profile  = np.array(self.tfidf_matrix[interacted_idxs].mean(axis=0))
        cb_scores = cosine_similarity(profile, self.tfidf_matrix)[0]
        for i in interacted_idxs:
            cb_scores[i] = 0
        return pd.Series(cb_scores, index=all_pids)

    # ── Hybrid Recommender ────────────────────────────────────────────────────
    def recommend(self, user_id, top_n=10, method="hybrid"):
        if method == "popular":
            return self._popular_fallback(top_n)

        all_pids    = self.products["product_id"].tolist()
        scores_cf_u = self._cf_user_based(user_id, top_n)
        scores_cf_i = self._cf_item_based(user_id, top_n)
        scores_cb   = self._cb_scores(user_id, top_n)

        def align(s):
            s  = s.reindex(all_pids, fill_value=0.0)
            mn, mx = s.min(), s.max()
            return (s - mn) / (mx - mn + 1e-9)

        s_cf_u = align(scores_cf_u)
        s_cf_i = align(scores_cf_i)
        s_cb   = align(scores_cb)

        if method == "collaborative":
            blended = 0.5 * s_cf_u + 0.5 * s_cf_i
        elif method == "content":
            blended = s_cb
        else:
            blended = 0.35 * s_cf_u + 0.30 * s_cf_i + 0.25 * s_cb

        pop_boost  = pd.Series(self.products["popularity_score"].values, index=all_pids)
        qual_boost = pd.Series(self.products["quality_score"].values,    index=all_pids)
        final = blended + 0.07 * pop_boost + 0.03 * qual_boost

        top_ids = final.nlargest(top_n).index.tolist()
        result  = self.products[self.products["product_id"].isin(top_ids)].copy()
        result["recommendation_score"] = result["product_id"].map(final).round(4)
        result["method"] = method
        return result.sort_values("recommendation_score", ascending=False)[
            ["product_id","name","category","brand","price",
             "rating","recommendation_score","method"]
        ]

    def recommend_similar(self, product_id, top_n=10):
        all_pids = self.products["product_id"].tolist()
        if product_id not in all_pids:
            return pd.DataFrame()
        idx  = all_pids.index(product_id)
        sims = np.array(self.item_similarity_cb[idx].todense()).flatten()
        sims[idx] = 0
        top  = np.argsort(sims)[::-1][:top_n]
        result = self.products.iloc[top].copy()
        result["similarity_score"] = sims[top].round(4)
        return result[["product_id","name","category","brand","price","rating","similarity_score"]]

    def _popular_fallback(self, top_n=10):
        result = (self.products
                  .sort_values(["popularity_score","rating"], ascending=False)
                  .head(top_n).copy())
        result["recommendation_score"] = result["popularity_score"].round(4)
        result["method"] = "popular"
        return result[["product_id","name","category","brand","price",
                        "rating","recommendation_score","method"]]

    def save_recommendations(self, user_id, recs, method):
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM recommendations WHERE user_id=?", (user_id,))
        rows = [(user_id, str(r["product_id"]), float(r["recommendation_score"]), method)
                for _, r in recs.iterrows()]
        conn.executemany(
            "INSERT INTO recommendations (user_id, product_id, score, method) VALUES (?,?,?,?)",
            rows
        )
        conn.commit()
        conn.close()


if __name__ == "__main__":
    import os
    db = os.path.join(os.path.dirname(__file__), "..", "data", "recommendation.db")
    engine = RecommendationEngine(db)

    # Quick demo with first user
    sample_user = engine.interactions["user_id"].iloc[0]
    print(f"\nTop-5 recommendations for user: {sample_user}")
    recs = engine.recommend(sample_user, top_n=5, method="hybrid")
    print(recs.to_string(index=False))

    print("\nTop-5 popular items:")
    print(engine._popular_fallback(5).to_string(index=False))