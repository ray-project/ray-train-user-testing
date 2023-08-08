import time

from dataset import Example
import random
import numpy as np
from typing import List
import multiprocessing as mp
from tqdm import tqdm
import json
import multiprocessing.shared_memory


class BaseDemoSelection:
    def __init__(self, n_processes: int = 1, n_shots: int = 5):
        self.n_processes = n_processes
        self.n_shots = n_shots

    def get_demo(self, target: Example) -> List[Example]:
        raise NotImplementedError()

    def batch_get_demo(self, targets: List[Example]) -> List[List[Example]]:
        if self.n_processes <= 1:
            return [self.get_demo(target) for target in targets]

        with mp.Pool(self.n_processes) as pool:
            return list(tqdm(
                pool.imap(self.get_demo, targets),
                disable=False, total=len(targets)
            ))


class FixedDemoSelection(BaseDemoSelection):

    def __init__(self, examples: List[Example], demo_ids: List[str], n_processes: int = 1):
        super().__init__(n_processes=n_processes)

        assert isinstance(examples, list)

        eid2example = {ex.example_id: ex for ex in examples}
        self.demonstrations = [eid2example[demo_id] for demo_id in demo_ids if demo_id in eid2example]

    def get_demo(self, target: Example) -> List[Example]:
        return self.demonstrations


class RandomDemoSelection(BaseDemoSelection):

    def __init__(self, examples: List[Example], n_shots: int = 5, n_processes: int = 1):
        super().__init__(n_processes=n_processes, n_shots=n_shots)

        assert isinstance(examples, list)

        eid2example = {ex.example_id: ex for ex in examples}
        self.demonstrations = random.sample(list(eid2example.values()), n_shots)

    def get_demo(self, target: Example) -> List[Example]:
        return self.demonstrations


class CosineTopKDemoSelection(BaseDemoSelection):
    def __init__(self, examples: List[Example], n_shots: int = 5, n_processes: int = 1,
                 whiten: bool = False):
        super().__init__(n_processes=n_processes)

        assert isinstance(examples, list)
        assert len(examples) >= n_shots

        self.examples = examples
        self.n_shots = n_shots
        self.X_emb = np.array([ex.source_emb for ex in examples], dtype=np.float64)

        if whiten:
            u, s, vt = np.linalg.svd(self.X_emb, full_matrices=False)
            self.W_whiten = vt.T.dot(np.diag(1 / s)).dot(vt)

    def get_demo(self, target: Example) -> List[Example]:
        assert isinstance(target.source_emb, np.ndarray), f"{type(target.source_emb)}"
        X = self.X_emb
        y = target.source_emb
        # print("x_shape", X.shape)
        # print("y_shape", y.shape)

        if hasattr(self, "W_whiten"):
            X = X.dot(self.W_whiten)
            y = y.dot(self.W_whiten)
        # dist[i] = -cosine(X[i], y)
        #         = - X[i]^T y / |X[i]| |y|
        #         ~ - X[i]^T y / |X[i]|
        dist = -X.dot(y) / np.sqrt((X ** 2).sum(1))
        demo_ids = np.argsort(dist)[:self.n_shots]
        return [self.examples[i] for i in demo_ids[::-1]]


class MMRDemoSelection(BaseDemoSelection):
    def __init__(self, examples: List[Example], n_shots: int = 5, n_processes: int = 1,
                 whiten: bool = False):
        super().__init__(n_processes=n_processes)

        assert isinstance(examples, list)
        assert len(examples) >= n_shots

        self.examples = examples
        self.n_shots = n_shots
        self.X_emb = np.array([ex.source_emb for ex in examples], dtype=np.float64)

        if whiten:
            u, s, vt = np.linalg.svd(self.X_emb, full_matrices=False)
            self.W_whiten = vt.T.dot(np.diag(1 / s)).dot(vt)
            self.X_emb = self.X_emb.dot(self.W_whiten)

        self.sim_matrix = self._calculate_similarity_matrix(self.X_emb)
        self.sim_matrix_shared = mp.shared_memory.SharedMemory(create=True, size=self.sim_matrix.nbytes)
        np.ndarray(self.sim_matrix.shape, dtype=self.sim_matrix.dtype, buffer=self.sim_matrix_shared.buf)[:] = self.sim_matrix

    def _calculate_similarity_matrix(self, X):
        sim_matrix = X.dot(X.T)
        norms = np.array([np.sqrt(np.diagonal(sim_matrix))])
        sim_matrix = sim_matrix / norms / norms.T
        # print("calculate pair wise similarity matrix...")
        return sim_matrix

    def precompute_similarities(self, targets: List[Example]):
        if hasattr(self, "W_whiten"):
            targets_emb = np.array([target.source_emb.dot(self.W_whiten) for target in targets])
        else:
            targets_emb = np.array([target.source_emb for target in targets])
        targets_norms = np.sqrt((targets_emb ** 2).sum(axis=1)).reshape(-1, 1)
        self.target_sim_matrix = (targets_emb.dot(self.X_emb.T) / targets_norms)
        self.target_sim_matrix_shared = mp.shared_memory.SharedMemory(create=True, size=self.target_sim_matrix.nbytes)
        np.ndarray(self.target_sim_matrix.shape, dtype=self.target_sim_matrix.dtype, buffer=self.target_sim_matrix_shared.buf)[:] = self.target_sim_matrix
        # print("precompute target similarities...")
        # print("target_sim:", self.target_sim_matrix.shape)

    def batch_get_demo(self, target_indices: List[int]) -> List[List[Example]]:
        if self.n_processes <= 1:
            return [self.get_demo(target_index) for target_index in target_indices]

        with mp.Manager() as manager:
            shared_dict = manager.dict()
            shared_dict['sim_matrix'] = self.sim_matrix
            shared_dict['target_sim_matrix'] = self.target_sim_matrix

            with mp.Pool(self.n_processes) as pool:
                results = pool.starmap(self.get_demo, [(index, shared_dict) for index in target_indices])
            return results

    def get_demo(self, target_index: int, shared_dict=None) -> List[Example]:
        if shared_dict is not None:
            sim_matrix = shared_dict['sim_matrix']
            target_sim_matrix = shared_dict['target_sim_matrix']
        else:
            sim_matrix = self.sim_matrix
            target_sim_matrix = self.target_sim_matrix

        sim_target = target_sim_matrix[target_index]

        selected = []
        while len(selected) < self.n_shots:
            remaining = [i for i in range(len(sim_matrix)) if i not in selected]
            mmr_scores = []

            for i in remaining:
                if selected:
                    max_sim_selected = max(sim_matrix[i][j] for j in selected)
                else:
                    max_sim_selected = 0
                mmr_score = sim_target[i] - max_sim_selected
                mmr_scores.append(mmr_score)

            selected.append(remaining[np.argmax(mmr_scores)])
        return [self.examples[i] for i in selected]

    def __del__(self):
        self.sim_matrix_shared.close()
        self.sim_matrix_shared.unlink()
        self.target_sim_matrix_shared.close()
        self.target_sim_matrix_shared.unlink()

