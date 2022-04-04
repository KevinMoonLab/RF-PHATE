from proximity import RFProximity, RFProximityReg

# For PHATE part
from phate import PHATE
import numpy as np
import warnings
from scipy import sparse
import tasklogger
_logger = tasklogger.get_tasklogger("graphtools")

class RFPHATE(RFProximity, PHATE):

    def __init__(
        self,
        n_components=2,
        prox_method='oob',
        matrix_type='sparse',
        knn=5,
        decay=40,
        n_landmark=2000,
        t="auto",
        gamma=1,
        n_pca=100,
        mds_solver="sgd",
        knn_max=None,
        mds_dist="euclidean",
        mds="metric",
        n_jobs=1,
        random_state=None,
        verbose=0,
        identity_scale = 5,
        diag_nonzero = True,
        **kwargs
        ):

        super(RFPHATE, self).__init__(**kwargs)
        
        self.n_components = n_components
        self.decay = decay
        self.knn = knn
        self.t = t
        self.gamma = gamma
        self.n_landmark = n_landmark
        self.mds = mds
        self.n_pca = n_pca
        self.knn_max = knn_max
        self.knn_dist = 'precomputed_affinity'
        self.mds_dist = mds_dist
        self.mds_solver = mds_solver
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.graph = None
        self._diff_potential = None
        self.embedding = None
        self.X = None
        self.optimal_t = None
        self.prox_method = prox_method
        self.matrix_type = matrix_type
        self.verbose = verbose
        self.identity_scale = identity_scale
        self.diag_nonzero = diag_nonzero

    def _fit_transform(self, X, y):

        n, d = X.shape

        if self.verbose == 1:
            verbose = True
        else:
            verbose = False

        self.fit(X, y)

        proximity = self.get_proximities(X)


        # TODO: Not sure if this is the reason RF-PHATE takes so long with fashion

        if self.diag_nonzero:
            if self.prox_method == 'rfgap':

                if self.matrix_type == 'dense':
                    diag_indices = np.diag_indices_from(proximity)
                    # proximity[diag_indices] = np.minimum(self.identity_scale * np.max(proximity, axis = 1).flatten(), 1)
                    proximity[diag_indices] = 1
                else:
                    i = np.array([range(n)]).squeeze()
    
                    # replace_vals = self.identity_scale * proximity.max(axis = 1).data
                    # replace_vals[replace_vals > 1] = 1
                    # replace_vals = np.array(replace_vals)

                    # diag = sparse.csr_matrix((replace_vals, (i, i)))

                    diag = sparse.csr_matrix((np.ones(n), (i, i)), shape = (n, n))
                    proximity = proximity + diag

            self.proximity = proximity
                        

        phate_op = PHATE(n_components = self.n_components,
            decay = self.decay,
            knn = self.knn,
            t = self.t,
            n_landmark = self.n_landmark,
            mds = self.mds,
            n_pca = self.n_pca,
            knn_dist = self.knn_dist,
            knn_max = self.knn_max,
            mds_dist = self.mds_dist,
            mds_solver = self.mds_solver,
            random_state = self.random_state,
            verbose = self.verbose)

        self.embedding_ = phate_op.fit_transform(proximity)

    def fit_transform(self, X, y):
        self._fit_transform(X, y)
        return self.embedding_


class RFPHATEReg(RFProximityReg, PHATE):

    def __init__(
        self,
        n_components=2,
        prox_method='oob',
        matrix_type='sparse',
        knn=5,
        decay=40,
        n_landmark=2000,
        t="auto",
        gamma=1,
        n_pca=100,
        mds_solver="sgd",
        knn_max=None,
        mds_dist="euclidean",
        mds="metric",
        n_jobs=1,
        random_state=None,
        verbose=0,
        identity_scale = 5,
        diag_nonzero = True,
        **kwargs
        ):

        super(RFPHATE, self).__init__(**kwargs)
        
        self.n_components = n_components
        self.decay = decay
        self.knn = knn
        self.t = t
        self.gamma = gamma
        self.n_landmark = n_landmark
        self.mds = mds
        self.n_pca = n_pca
        self.knn_max = knn_max
        self.knn_dist = 'precomputed_affinity'
        self.mds_dist = mds_dist
        self.mds_solver = mds_solver
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.graph = None
        self._diff_potential = None
        self.embedding = None
        self.X = None
        self.optimal_t = None
        self.prox_method = prox_method
        self.matrix_type = matrix_type
        self.verbose = verbose
        self.identity_scale = identity_scale
        self.diag_nonzero = diag_nonzero

    def _fit_transform(self, X, y):

        n, d = X.shape

        if self.verbose == 1:
            verbose = True
        else:
            verbose = False

        self.fit(X, y)

        proximity = self.get_proximities(X)


        # TODO: Not sure if this is the reason RF-PHATE takes so long with fashion

        if self.diag_nonzero:
            if self.prox_method == 'rfgap':

                if self.matrix_type == 'dense':
                    diag_indices = np.diag_indices_from(proximity)
                    # proximity[diag_indices] = np.minimum(self.identity_scale * np.max(proximity, axis = 1).flatten(), 1)
                    proximity[diag_indices] = 1
                else:
                    i = np.array([range(n)]).squeeze()
    
                    # replace_vals = self.identity_scale * proximity.max(axis = 1).data
                    # replace_vals[replace_vals > 1] = 1
                    # replace_vals = np.array(replace_vals)

                    # diag = sparse.csr_matrix((replace_vals, (i, i)))

                    diag = sparse.csr_matrix((np.ones(n), (i, i)), shape = (n, n))
                    proximity = proximity + diag

            self.proximity = proximity
                        

        phate_op = PHATE(n_components = self.n_components,
            decay = self.decay,
            knn = self.knn,
            t = self.t,
            n_landmark = self.n_landmark,
            mds = self.mds,
            n_pca = self.n_pca,
            knn_dist = self.knn_dist,
            knn_max = self.knn_max,
            mds_dist = self.mds_dist,
            mds_solver = self.mds_solver,
            random_state = self.random_state,
            verbose = self.verbose)

        self.embedding_ = phate_op.fit_transform(proximity)

    def fit_transform(self, X, y):
        self._fit_transform(X, y)
        return self.embedding_
