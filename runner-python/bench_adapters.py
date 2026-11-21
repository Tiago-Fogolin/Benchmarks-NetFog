import os
import time


def timed(fn):
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    return result, elapsed


def resolve_thread_count(threads_spec):
    if threads_spec == 1:
        return 1
    return int(os.environ.get("OMP_NUM_THREADS", "16"))


def generate_graph(n, avg_degree, kind="er", seed=42):
    import networkx as nx

    if kind == "er":
        p = avg_degree / (n - 1) if n > 1 else 0.0
        g = nx.gnp_random_graph(n, p, seed=seed)
    elif kind == "ba":
        m = max(1, round(avg_degree / 2))
        g = nx.barabasi_albert_graph(n, m, seed=seed)
    elif kind == "ws":
        k = max(2, round(avg_degree))
        if k % 2 == 1:
            k += 1
        g = nx.watts_strogatz_graph(n, k, 0.1, seed=seed)
    else:
        raise ValueError(f"tipo de grafo desconhecido: {kind}")

    edges = list(g.edges())

    degrees = [0] * n
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1
    start_node = max(range(n), key=lambda i: degrees[i]) if n > 0 else 0

    return n, edges, start_node



def build_netfog(n, edges, variant):
    import netfog

    structure = getattr(netfog.GraphStructureType, variant["structure"])
    mode = variant["mode"]
    g = netfog.Graph(structure=structure)

    if mode == "id":
        def _build():
            g.add_nodes(n)
            g.add_edges_from_ids([(u, v, 1.0, False) for u, v in edges])
        _, elapsed = timed(_build)
    else:
        labels = [str(i) for i in range(n)]

        def _build():
            g.add_nodes_from(labels)
            g.add_edges_from([(str(u), str(v), 1.0, False) for u, v in edges])
        _, elapsed = timed(_build)

    return g, elapsed, getattr(g, "build_time_ms", None)


def kernels_netfog(variant, start_node=0):
    mode = variant["mode"]

    def node_ref(i):
        return i if mode == "id" else str(i)

    def k_bfs(g):
        return timed(lambda: g.bfs(node_ref(start_node)))

    def k_dfs(g):
        return timed(lambda: g.dfs(node_ref(start_node)))

    def k_dijkstra(g):
        return timed(lambda: g.dijkstra(node_ref(start_node)))

    def k_connected_components(g):
        return timed(lambda: g.connected_components_count())

    def k_pagerank(g):
        return timed(lambda: g.pagerank())

    def k_triangle_count(g):
        return timed(lambda: g.triangle_count())

    def k_betweenness(g):
        return timed(lambda: g.betweenness_centrality())

    def k_closeness(g):
        return timed(lambda: g.closeness_centrality_all())

    def k_clustering(g):
        return timed(lambda: g.transitivity())

    def k_kcore(g):
        return timed(lambda: g.k_core_decomposition())

    return {
        "bfs": k_bfs,
        "dfs": k_dfs,
        "dijkstra": k_dijkstra,
        "connected_components": k_connected_components,
        "pagerank": k_pagerank,
        "triangle_count": k_triangle_count,
        "betweenness_centrality": k_betweenness,
        "closeness_centrality": k_closeness,
        "clustering_coefficient": k_clustering,
        "k_core_decomposition": k_kcore,
    }



def build_igraph(n, edges, variant):
    import igraph as ig

    def _build():
        return ig.Graph(n, edges=edges, directed=False)
    g, elapsed = timed(_build)
    return g, elapsed, None


def _invalidate_igraph_cache(g):
    getattr(g, "__invalidate_cache")()


def kernels_igraph(variant, start_node=0):
    def k_bfs(g):
        _invalidate_igraph_cache(g)
        return timed(lambda: g.bfs(start_node)[0])

    def k_dfs(g):
        _invalidate_igraph_cache(g)
        return timed(lambda: g.dfs(start_node)[0])

    def k_dijkstra(g):
        _invalidate_igraph_cache(g)
        return timed(lambda: g.distances(source=[start_node])[0])

    def k_connected_components(g):
        _invalidate_igraph_cache(g)
        return timed(lambda: len(g.connected_components()))

    def k_pagerank(g):
        _invalidate_igraph_cache(g)
        return timed(lambda: g.pagerank())

    def k_triangle_count(g):
        _invalidate_igraph_cache(g)
        return timed(lambda: len(g.list_triangles()))

    def k_betweenness(g):
        _invalidate_igraph_cache(g)
        return timed(lambda: g.betweenness())

    def k_closeness(g):
        _invalidate_igraph_cache(g)
        return timed(lambda: g.closeness())

    def k_clustering(g):
        _invalidate_igraph_cache(g)
        return timed(lambda: g.transitivity_undirected())

    def k_kcore(g):
        _invalidate_igraph_cache(g)
        return timed(lambda: g.coreness())

    return {
        "bfs": k_bfs,
        "dfs": k_dfs,
        "dijkstra": k_dijkstra,
        "connected_components": k_connected_components,
        "pagerank": k_pagerank,
        "triangle_count": k_triangle_count,
        "betweenness_centrality": k_betweenness,
        "closeness_centrality": k_closeness,
        "clustering_coefficient": k_clustering,
        "k_core_decomposition": k_kcore,
    }



def build_networkx(n, edges, variant):
    import networkx as nx

    def _build():
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from(edges)
        return g
    g, elapsed = timed(_build)
    return g, elapsed, None


def kernels_networkx(variant, start_node=0):
    import networkx as nx

    def k_bfs(g):
        return timed(lambda: list(nx.bfs_edges(g, start_node)))

    def k_dfs(g):
        return timed(lambda: list(nx.dfs_edges(g, start_node)))

    def k_dijkstra(g):
        return timed(lambda: nx.single_source_dijkstra_path_length(g, start_node))

    def k_connected_components(g):
        return timed(lambda: nx.number_connected_components(g))

    def k_pagerank(g):
        return timed(lambda: nx.pagerank(g))

    def k_triangle_count(g):
        return timed(lambda: sum(nx.triangles(g).values()) // 3)

    def k_betweenness(g):
        return timed(lambda: nx.betweenness_centrality(g))

    def k_closeness(g):
        return timed(lambda: nx.closeness_centrality(g))

    def k_clustering(g):
        return timed(lambda: nx.transitivity(g))

    def k_kcore(g):
        return timed(lambda: nx.core_number(g))

    return {
        "bfs": k_bfs,
        "dfs": k_dfs,
        "dijkstra": k_dijkstra,
        "connected_components": k_connected_components,
        "pagerank": k_pagerank,
        "triangle_count": k_triangle_count,
        "betweenness_centrality": k_betweenness,
        "closeness_centrality": k_closeness,
        "clustering_coefficient": k_clustering,
        "k_core_decomposition": k_kcore,
    }



def build_graphtool(n, edges, variant):
    import graph_tool as gt

    if variant and "threads" in variant:
        gt.openmp_set_num_threads(resolve_thread_count(variant["threads"]))

    def _build():
        g = gt.Graph(directed=False)
        g.add_vertex(n)
        if edges:
            g.add_edge_list(edges)
        return g
    g, elapsed = timed(_build)
    return g, elapsed, None


def kernels_graphtool(variant, start_node=0):
    import graph_tool.topology as topology
    import graph_tool.centrality as centrality
    import graph_tool.clustering as clustering
    import graph_tool.search as search

    def k_bfs(g):
        return timed(lambda: len(list(search.bfs_iterator(g, source=g.vertex(start_node)))))

    def k_dfs(g):
        return timed(lambda: len(list(search.dfs_iterator(g, source=g.vertex(start_node)))))

    def k_dijkstra(g):
        return timed(lambda: list(topology.shortest_distance(g, source=g.vertex(start_node)).a))

    def k_connected_components(g):
        def _run():
            _, hist = topology.label_components(g)
            return len(hist)
        return timed(_run)

    def k_pagerank(g):
        return timed(lambda: list(centrality.pagerank(g).a))

    def k_triangle_count(g):
        def _run():
            _, tri, _triads = clustering.global_clustering(g, ret_counts=True)
            return tri
        return timed(_run)

    def k_betweenness(g):
        return timed(lambda: list(centrality.betweenness(g)[0].a))

    def k_closeness(g):
        return timed(lambda: list(centrality.closeness(g).a))

    def k_clustering(g):
        def _run():
            (c, _stderr), _tri, _triads = clustering.global_clustering(g, ret_counts=True)
            return c
        return timed(_run)

    def k_kcore(g):
        return timed(lambda: list(topology.kcore_decomposition(g).a))

    return {
        "bfs": k_bfs,
        "dfs": k_dfs,
        "dijkstra": k_dijkstra,
        "connected_components": k_connected_components,
        "pagerank": k_pagerank,
        "triangle_count": k_triangle_count,
        "betweenness_centrality": k_betweenness,
        "closeness_centrality": k_closeness,
        "clustering_coefficient": k_clustering,
        "k_core_decomposition": k_kcore,
    }



def build_networkit(n, edges, variant):
    import networkit as nk

    if variant and "threads" in variant:
        nk.setNumberOfThreads(resolve_thread_count(variant["threads"]))

    def _build():
        g = nk.Graph(n, weighted=False, directed=False)
        for u, v in edges:
            g.addEdge(u, v)
        g.indexEdges()
        return g
    g, elapsed = timed(_build)
    return g, elapsed, None


def kernels_networkit(variant, start_node=0):
    import networkit as nk

    def k_bfs(g):
        def _run():
            order = []
            nk.traversal.Traversal.BFSfrom(g, start_node, lambda node, *_: order.append(node))
            return order
        return timed(_run)

    def k_dfs(g):
        def _run():
            order = []
            nk.traversal.Traversal.DFSfrom(g, start_node, lambda node, *_: order.append(node))
            return order
        return timed(_run)

    def k_dijkstra(g):
        def _run():
            d = nk.distance.Dijkstra(g, start_node)
            d.run()
            return d.getDistances()
        return timed(_run)

    def k_connected_components(g):
        def _run():
            cc = nk.components.ConnectedComponents(g)
            cc.run()
            return cc.numberOfComponents()
        return timed(_run)

    def k_pagerank(g):
        def _run():
            pr = nk.centrality.PageRank(g)
            pr.run()
            return pr.scores()
        return timed(_run)

    def k_triangle_count(g):
        def _run():
            tc = nk.sparsification.ChibaNishizekiTriangleEdgeScore(g)
            tc.run()
            return sum(tc.scores()) // 3
        return timed(_run)

    def k_betweenness(g):
        def _run():
            bt = nk.centrality.Betweenness(g, normalized=True)
            bt.run()
            return bt.scores()
        return timed(_run)

    def k_closeness(g):
        def _run():
            cl = nk.centrality.Closeness(g, True, nk.centrality.ClosenessVariant.GENERALIZED)
            cl.run()
            return cl.scores()
        return timed(_run)

    def k_clustering(g):
        return timed(lambda: nk.globals.ClusteringCoefficient.exactGlobal(g))

    def k_kcore(g):
        def _run():
            core = nk.centrality.CoreDecomposition(g)
            core.run()
            return core.scores()
        return timed(_run)

    return {
        "bfs": k_bfs,
        "dfs": k_dfs,
        "dijkstra": k_dijkstra,
        "connected_components": k_connected_components,
        "pagerank": k_pagerank,
        "triangle_count": k_triangle_count,
        "betweenness_centrality": k_betweenness,
        "closeness_centrality": k_closeness,
        "clustering_coefficient": k_clustering,
        "k_core_decomposition": k_kcore,
    }


BUILDERS = {
    "netfog": build_netfog,
    "igraph": build_igraph,
    "networkx": build_networkx,
    "graph_tool": build_graphtool,
    "networkit": build_networkit,
}

KERNELS = {
    "netfog": kernels_netfog,
    "igraph": kernels_igraph,
    "networkx": kernels_networkx,
    "graph_tool": kernels_graphtool,
    "networkit": kernels_networkit,
}

ALL_KERNEL_NAMES = [
    "bfs",
    "dijkstra",
    "connected_components",
    "pagerank",
    "triangle_count",
    "betweenness_centrality",
]
