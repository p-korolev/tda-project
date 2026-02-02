import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import kmapper as km
import networkx as nx

from typing import List
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# Data Generators

def gen_bvn(n: int, rho: float, seed: int | None = None) -> np.ndarray:
    """
    Bivariate normal with correlation rho, mean 0, variances 1.
    Returns array shape (n, 2).
    """
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, rho], [rho, 1.0]], dtype=float)
    mean = np.array([0.0, 0.0], dtype=float)
    X = rng.multivariate_normal(mean, cov, size=n)
    return X


def gen_nonlinear_x2(n: int, sigma: float = 0.05, seed: int | None = None) -> np.ndarray:
    """
    Control: nonlinear dependence with potentially low Pearson correlation.
    x ~ Uniform[-1,1], y = x^2 + sigma * N(0,1).
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=n)
    y = x**2 + sigma * rng.normal(0.0, 1.0, size=n)
    return np.column_stack([x, y])


def gen_multiregime_diagonal(n: int, K: int = 1, noise: float = 0.05,
                            spacing: float = 8.0, seed: int | None = None) -> np.ndarray:
    """
    Control: K separated regimes along the diagonal direction.
    Creates K clusters centered around (c, c) with small diagonal noise so rho ~ 1,
    but beta0 can be ~K if regimes stay disconnected in Mapper.
    """
    rng = np.random.default_rng(seed)
    counts = np.full(K, n // K, dtype=int)
    counts[: n % K] += 1

    centers = spacing * (np.arange(K) - (K - 1) / 2.0)
    pts = []
    for k, ck in enumerate(centers):
        m = counts[k]
        # x centered at ck
        x = ck + rng.normal(0.0, 1.0, size=m)
        # y ~ x + small noise
        y = x + noise * rng.normal(0.0, 1.0, size=m)
        pts.append(np.column_stack([x, y]))
    return np.vstack(pts)


# Mapper/Betti Computation

def lens_abs_diff(X: np.ndarray) -> np.ndarray:
    """
    Lens f(x,y)=|x-y|. KeplerMapper expects lens as shape (n, m).
    We'll return shape (n, 1).
    """
    return np.abs(X[:, 0] - X[:, 1]).reshape(-1, 1)


def build_mapper_graph(
    X: np.ndarray,
    n_cubes: int = 15,
    perc_overlap: float = 0.4,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
    normalize_z: bool = True,
) -> dict:
    """
    Build Mapper graph using KeplerMapper and return the raw mapper graph dict.
    """
    X_use = X.copy()
    if normalize_z:
        X_use = StandardScaler().fit_transform(X_use)

    mapper = km.KeplerMapper(verbose=0)
    lens = lens_abs_diff(X_use)

    # cluster method
    clusterer = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)

    graph = mapper.map(
        lens,
        X_use,
        cover=km.Cover(n_cubes=n_cubes, perc_overlap=perc_overlap),
        clusterer=clusterer
    )
    return graph


def betti_from_mapper_graph(graph: dict) -> tuple[int, int, int, int]:
    """
    Compute (beta0, beta1, |V|, |E|) for the undirected 1-skeleton of the Mapper nerve.
    KeplerMapper graph dict typically has:
      - graph["nodes"]: dict mapping node_id -> list of point indices
      - graph["links"]: dict mapping node_id -> list/set of neighbor node_ids
    """
    nodes = list(graph.get("nodes", {}).keys())
    links = graph.get("links", {})

    G = nx.Graph()
    G.add_nodes_from(nodes)

    # Add edges
    for u, nbrs in links.items():
        for v in nbrs:
            G.add_edge(u, v)

    V = G.number_of_nodes()
    E = G.number_of_edges()
    beta0 = nx.number_connected_components(G)
    beta1 = E - V + beta0  # cycle rank for graphs

    return beta0, beta1, V, E


# Monte-Carlo simul sweep

def sample_corr_abs(X: np.ndarray) -> float:
    """Absolute sample Pearson correlation between columns."""
    x = X[:, 0]
    y = X[:, 1]
    r = np.corrcoef(x, y)[0, 1]
    return float(abs(r))


def run_sweep(
    rhos_abs: list[float],
    n: int = 800,
    R: int = 40,
    seed: int = 123,
    n_cubes: int = 15,
    perc_overlap: float = 0.4,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
    normalize_z: bool = True,
    use_negative: bool = False,
    ) -> pd.DataFrame:
    """
    Sweeps |rho| values. For each value, runs R trials, builds Mapper graphs and records
    bettis + counts. Returns df with means/stds.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for r_abs in rhos_abs:
        if r_abs==1:
            for j in range(R):
                X = gen_bvn(n=n, rho=rho)
                graph = build_mapper_graph(
                    X,
                    n_cubes = 2,
                    perc_overlap = 0.3,
                    dbscan_eps = dbscan_eps,
                    dbscan_min_samples = dbscan_min_samples,
                    normalize_z = normalize_z,
                )
                beta0, beta1, V, E = betti_from_mapper_graph(graph)
                rows.append({
                "rho_target": rho,
                "rho_abs_target": r_abs,
                "rho_abs_sample": sample_corr_abs(X),
                "beta0": beta0,
                "beta1": beta1,
                "V": V,
                "E": E
                })
            break
        for j in range(R):
            # choose sign if (use_negative)
            rho = -r_abs if (use_negative and (j % 2 == 1)) else r_abs

            X = gen_bvn(n=n, rho=rho, seed=int(rng.integers(0, 2**32 - 1)))

            graph = build_mapper_graph(
                X,
                n_cubes=n_cubes,
                perc_overlap=perc_overlap,
                dbscan_eps=dbscan_eps,
                dbscan_min_samples=dbscan_min_samples,
                normalize_z=normalize_z,
            )
            beta0, beta1, V, E = betti_from_mapper_graph(graph)

            rows.append({
                "rho_target": rho,
                "rho_abs_target": r_abs,
                "rho_abs_sample": sample_corr_abs(X),
                "beta0": beta0,
                "beta1": beta1,
                "V": V,
                "E": E
            })

    df = pd.DataFrame(rows)

    # summaries by target corr.
    g = df.groupby("rho_abs_target", as_index=False)
    out = g.agg(
        rho_abs_sample_mean=("rho_abs_sample", "mean"),

        beta0_mean=("beta0", "mean"),
        beta0_std=("beta0", "std"),
        beta0_min=("beta0", "min"),
        beta0_max=("beta0", "max"),
        beta0_mode=("beta0", lambda x: x.mode().iloc[0]),

        beta1_mean=("beta1", "mean"),
        beta1_std=("beta1", "std"),
        beta1_min=("beta1", "min"),
        beta1_max=("beta1", "max"),

        V_mean=("V", "mean"),
        E_mean=("E", "mean"),
    )
    return out, df


#Config/Controls

def run_control_nonlinear_x2(
    n: int = 800,
    R: int = 40,
    sigma: float = 0.05,
    seed: int = 999,
    **mapper_kwargs
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(R):
        X = gen_nonlinear_x2(n=n, sigma=sigma, seed=int(rng.integers(0, 2**32 - 1)))
        graph = build_mapper_graph(X, **mapper_kwargs)
        beta0, beta1, V, E = betti_from_mapper_graph(graph)
        rows.append({
            "rho_abs_sample": sample_corr_abs(X),
            "beta0": beta0,
            "beta1": beta1,
            "V": V,
            "E": E
        })
    return pd.DataFrame(rows)


def run_control_multiregime(
    n: int = 800,
    R: int = 40,
    K: int = 4,
    noise: float = 0.05,
    spacing: float = 8.0,
    seed: int = 2026,
    **mapper_kwargs
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(R):
        X = gen_multiregime_diagonal(n=n, K=K, noise=noise, spacing=spacing,
                                     seed=int(rng.integers(0, 2**32 - 1)))
        graph = build_mapper_graph(X, **mapper_kwargs)
        beta0, beta1, V, E = betti_from_mapper_graph(graph)
        rows.append({
            "rho_abs_sample": sample_corr_abs(X),
            "beta0": beta0,
            "beta1": beta1,
            "V": V,
            "E": E
        })
    return pd.DataFrame(rows)


# plotting

def plot_betti_vs_rho(summary: pd.DataFrame, title_suffix: str = "") -> None:
    x = summary["rho_abs_target"].to_numpy()

    plt.figure()
    plt.errorbar(x, summary["beta0_mean"], yerr=summary["beta0_std"], fmt="o-")
    plt.xlabel(r"$|\rho|$")
    plt.ylabel(r"$\mathbb{E}[\beta_0]$")
    plt.title(r"Expected $\beta_0$ vs $|\rho|$" + title_suffix)
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.errorbar(x, summary["beta1_mean"], yerr=summary["beta1_std"], fmt="o-")
    plt.xlabel(r"$|\rho|$")
    plt.ylabel(r"$\mathbb{E}[\beta_1]$")
    plt.title(r"Expected $\beta_1$ vs $|\rho|$" + title_suffix)
    plt.grid(True)
    plt.show()

def plot_betti_vs_rho_minmax(summary: pd.DataFrame, title_suffix: str = "") -> None:
    x = summary["rho_abs_target"].to_numpy()

    yupper = summary["beta0_mean"] + summary["beta0_std"]
    ylower = summary["beta0_mean"] - summary["beta0_std"]
    plt.figure()
    plt.plot(x, summary["beta0_mean"], color='steelblue', lw=1.5, marker='o', ms=3, label=r"Mean $\beta_0$")
    #plt.plot(x, ylower, color='grey', lw=1, label=r"error $\beta_0$")
    #plt.plot(x, yupper, color='grey', lw=1)
    plt.fill_between(x, ylower, yupper, color='grey', alpha=0.3, label=r"Error band ($\pm \sigma$)")
    plt.legend()
    plt.xlabel(r"$|\rho|$")
    plt.ylabel(r"$\beta_0$")
    #plt.title(r"Betti $\beta_0$ value by correlation $|\rho|$" + title_suffix)
    plt.grid()
    plt.show()


    #yupperb1 = summary["beta1_mean"] + summary["beta1_std"]
    #ylowerb1 = summary["beta1_mean"] - summary["beta1_std"]
    #plt.figure()
    #plt.plot(x, summary["beta1_max"], color='seagreen', lw=1.5, marker='o', ms=4, label=r"max $\beta_1$")
    #plt.fill_between(x, ylowerb1, yupperb1, color='grey', alpha=0.3, label="Error band")
    #plt.legend()
    #plt.xlabel(r"$|\rho|$")
    #plt.ylabel(r"Betti value $\beta_1$")
    #plt.title(r"Maximum cycles observed through $|\rho|$" + title_suffix)
    #plt.grid()
    #plt.show()

def plot_cycles_vs_rho(summary: pd.DataFrame) -> None:
    x = summary["rho_abs_target"].to_numpy()
    
    yupperb1 = summary["beta1_mean"] + summary["beta1_std"]
    ylowerb1 = summary["beta1_mean"] - summary["beta1_std"]

    plt.figure()
    plt.plot(x, summary["beta1_mean"], color='seagreen', lw=1.5, marker='o', ms=4, label=r"Mean $\beta_1$")
    plt.fill_between(x, ylowerb1, yupperb1, color='grey', alpha=0.3, label="Error band")
    plt.legend()
    plt.xlabel(r"$|\rho|$")
    plt.ylabel(r"$\beta_1$")
    plt.grid()
    plt.show()

def plot_betti_vs_rho_mode(summary: pd.DataFrame) -> None:
    x = summary["rho_abs_target"].to_numpy()
    plt.figure()
    plt.plot(x, summary["beta0_mode"], color='steelblue', lw=1.5, marker='o', ms=3, label=r'Mode $\beta_0$')
    plt.legend()
    plt.xlabel(r"$|\rho|$")
    plt.ylabel(r"$\beta_0$")
    plt.grid()
    plt.show()

def plot_summaries_by_cover(summaries: List[pd.DataFrame], covers_used=List[int]) -> None:
    '''
    Plot multiple E[\beta_0] vs |\rho| lines with sweeps using different amount of covers (n_cubes param in build_mapper_graph)
    
    :param summaries: List of summaries of type df
    :param covers_used: List of cube param passed into sweep function
    '''
    if len(summaries)!=len(covers_used):
        return IndexError(f"length of summaries list is {len(summaries)}, while length of covers_used list is {len(covers_used)}")
    plt.figure()
    c_idx = 0
    for summary in summaries:
        x = summary["rho_abs_target"].to_numpy()
        current_cover_amt = covers_used[c_idx]
        plt.plot(x, summary["beta0_mean"], label=r"Mean $\beta_0$ (covers=" + f"{current_cover_amt})")
        c_idx += 1
    plt.axhline(y=1.0, color='black', linestyle='--', label=r"$\beta_0 = 1$")
    plt.legend()
    plt.xlabel(r"$|\rho|$")
    plt.ylabel(r"$\beta_0$")
    plt.ylim(bottom=0)
    plt.grid()
    plt.show()

def plot_summaries_by_overlap(summaries: List[pd.DataFrame], overlaps_used=List[float]) -> None:
    '''
    Plot multiple E[\beta_0] vs |\rho| lines with sweeps using different overlap values (overlap param in build_mapper_graph).

    Overlaps O in overlaps_used must be between 0 and 1.
    
    :param summaries: List of summaries of type df
    :param covers_used: List of cube param passed into sweep function
    '''
    if len(summaries)!=len(overlaps_used):
        return IndexError(f"length of summaries list is {len(summaries)}, while length of covers_used list is {len(overlaps_used)}")
    plt.figure()
    o_idx = 0
    for summary in summaries:
        x = summary["rho_abs_target"].to_numpy()
        current_overlap = overlaps_used[o_idx]
        plt.plot(x, summary["beta0_mean"], label=r"Mean $\beta_0$ ($\alpha$=" + f"{current_overlap})")
        o_idx += 1
    plt.axhline(y=1.0, color='black', linestyle='--', label=r"$\beta_0 = 1$")
    plt.legend()
    plt.xlabel(r"$|\rho|$")
    plt.ylabel(r"$\beta_0$")
    plt.ylim(bottom=0)
    plt.grid()
    plt.show()