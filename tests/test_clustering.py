import numpy as np

from jlucid.leida.clustering import (
    assign_states,
    cluster_k,
    dunn_index_centroid,
    kmeans_grid,
    pool_v1,
    select_k,
)


def test_dunn_index_separates():
    rng = np.random.default_rng(1)
    c1 = rng.normal(0, 0.1, size=(200, 10)) + np.array([1.0] * 10)
    c2 = rng.normal(0, 0.1, size=(200, 10)) - np.array([1.0] * 10)
    X = np.vstack([c1, c2])
    labels = np.r_[np.zeros(200, int), np.ones(200, int)]
    centroids = np.array([X[labels == 0].mean(0), X[labels == 1].mean(0)])
    dunn = dunn_index_centroid(X, labels, centroids)
    assert dunn > 1.0


def test_pool_v1_index():
    d = {"a": np.zeros((5, 3)), "b": np.ones((7, 3))}
    X, idx = pool_v1(d, ["a", "b"])
    assert X.shape == (12, 3)
    assert idx == {"a": (0, 5), "b": (5, 12)}


def test_cluster_k_recovers_two_clusters():
    rng = np.random.default_rng(2)
    X = np.vstack(
        [
            rng.normal(0, 0.05, size=(500, 20)) + np.array([1.0] * 20),
            rng.normal(0, 0.05, size=(500, 20)) - np.array([1.0] * 20),
        ]
    )
    res = cluster_k(X, 2, n_init=10, seed=42, eval_subsample=1000)
    counts = np.bincount(res.labels)
    assert len(counts) == 2 and counts.min() > 400


def test_select_k_prefers_high_dunn():
    results = {k: type("R", (), {"inertia": 0.0, "silhouette": 0.1, "dunn": k * 0.1,
                                "stability": 0.9})() for k in range(3, 21)}
    chosen, table = select_k(results, anchor=5)
    assert chosen == 20
    assert len(table) == 18


def test_assign_states_nearest_centroid():
    V1 = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
    centers = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = assign_states(V1, centers)
    assert labels.tolist() == [0, 1, 0]


def test_kmeans_grid_keys():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 6))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    results = kmeans_grid(X, k_range=range(3, 5), n_init=3, seed=42,
                          stability_runs=2, eval_subsample=300)
    assert set(results) == {3, 4}
    assert results[3].labels.shape == (300,)
