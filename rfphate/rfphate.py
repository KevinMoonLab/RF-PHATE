from proximity import RFProximity, RFProximityReg

# For PHATE part
from phate import PHATE
import numpy as np
from scipy import sparse


def MakeRFPHATE(label_type = 'categorical',       
        n_components = 2,
        prox_method = 'oob',
        matrix_type = 'sparse',
        n_landmark = 2000,
        t = "auto",
        n_pca = 100,
        mds_solver = "sgd",
        mds_dist = "euclidean",
        mds = "metric",
        n_jobs = 1,
        random_state = None,
        verbose = 0,
        diag_nonzero = True,
        **kwargs):

    """A method to generate an instance of the class RFProximity, 
       determining whether the data labels are categorical or numeric

    Parameters
    ----------
    label_type : str
        The type of forest to be created, supported types are 'categorical' for a 
        classification forest, or 'numeric' for a regression forest

    n_components : int
        The number of dimensions for the RF-PHATE embedding

    prox_method : str
        The type of proximity to be constructed.  Options are 'original', 'oob', and
        'rfgap' (default is 'oob')

    matrix_type : str
        Whether the proximity type should be 'sparse' or 'dense' (default is sparse)
    
    n_landmark : int, optional
        number of landmarks to use in fast PHATE (default is 2000)

    t : int, optional
        power to which the diffusion operator is powered.
        This sets the level of diffusion. If 'auto', t is selected
        according to the knee point in the Von Neumann Entropy of
        the diffusion operator (default is 'auto')

    n_pca : int, optional
        Number of principal components to use for calculating
        neighborhoods. For extremely large datasets, using
        n_pca < 20 allows neighborhoods to be calculated in
        roughly log(n_samples) time (default is 100)

    mds : string, optional
        choose from ['classic', 'metric', 'nonmetric'].
        Selects which MDS algorithm is used for dimensionality reduction
        (default is 'metric')

    mds_solver : {'sgd', 'smacof'}
        which solver to use for metric MDS. SGD is substantially faster,
        but produces slightly less optimal results (default is 'sgd')

    mds_dist : string, optional
        Distance metric for MDS. Recommended values: 'euclidean' and 'cosine'
        Any metric from `scipy.spatial.distance` can be used. Custom distance
        functions of form `f(x, y) = d` are also accepted (default is 'euclidean')

    n_jobs : integer, optional
        The number of jobs to use for the computation.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging.
        For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for
        n_jobs = -2, all CPUs but one are used (default is 1)

    random_state : integer
        random seed state set for RF and MDS


    verbose : int or bool
        If `True` or `> 0`, print status messages (default is 0)

    diag_nonzero: bool
        Only used if prox_method == 'rfgap.  Replaces the zero-diagonal entries
        of the rfgap proximities with ones (default is True)

    **kwargs
        Keyward arguements specific to the RandomForestClassifer and RFPHATE algorithms

    """

    if label_type == 'categorical':
        return RFPHATE(n_components = n_components,
        prox_method = prox_method,
        matrix_type = matrix_type,
        n_landmark = n_landmark,
        t = t,
        n_pca = n_pca,
        mds_solver = mds_solver,
        mds_dist = mds_dist,
        mds = mds,
        n_jobs = n_jobs,
        random_state = random_state,
        verbose = verbose,
        diag_nonzero = diag_nonzero,
        **kwargs)


    elif label_type == 'numeric':
        return RFPHATEReg(n_components = n_components,
        prox_method = prox_method,
        matrix_type = matrix_type,
        n_landmark = n_landmark,
        t = t,
        n_pca = n_pca,
        mds_solver = mds_solver,
        mds_dist = mds_dist,
        mds = mds,
        n_jobs = n_jobs,
        random_state = random_state,
        verbose = verbose,
        diag_nonzero = diag_nonzero,
        **kwargs)

    else: 
        print('Only "categorical" or "numeric" types are supported.')


class RFPHATE(RFProximity, PHATE):

    """An RF-PHATE class which is used to fit a random forest, generate RF-proximities,
       and create RF-PHATE embeddings.

    Parameters
    ----------
    n_components : int
        The number of dimensions for the RF-PHATE embedding

    prox_method : str
        The type of proximity to be constructed.  Options are 'original', 'oob', and
        'rfgap' (default is 'oob')

    matrix_type : str
        Whether the proximity type should be 'sparse' or 'dense' (default is sparse)
    
    n_landmark : int, optional
        number of landmarks to use in fast PHATE (default is 2000)

    t : int, optional
        power to which the diffusion operator is powered.
        This sets the level of diffusion. If 'auto', t is selected
        according to the knee point in the Von Neumann Entropy of
        the diffusion operator (default is 'auto')

    n_pca : int, optional
        Number of principal components to use for calculating
        neighborhoods. For extremely large datasets, using
        n_pca < 20 allows neighborhoods to be calculated in
        roughly log(n_samples) time (default is 100)

    mds : string, optional
        choose from ['classic', 'metric', 'nonmetric'].
        Selects which MDS algorithm is used for dimensionality reduction
        (default is 'metric')

    mds_solver : {'sgd', 'smacof'}
        which solver to use for metric MDS. SGD is substantially faster,
        but produces slightly less optimal results (default is 'sgd')

    mds_dist : string, optional
        Distance metric for MDS. Recommended values: 'euclidean' and 'cosine'
        Any metric from `scipy.spatial.distance` can be used. Custom distance
        functions of form `f(x, y) = d` are also accepted (default is 'euclidean')

    n_jobs : integer, optional
        The number of jobs to use for the computation.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging.
        For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for
        n_jobs = -2, all CPUs but one are used (default is 1)

    random_state : integer
        random seed state set for RF and MDS


    verbose : int or bool
        If `True` or `> 0`, print status messages (default is 0)

    diag_nonzero: bool
        Only used if prox_method == 'rfgap.  Replaces the zero-diagonal entries
        of the rfgap proximities with ones (default is True)
    """

    def __init__(
        self,
        n_components = 2,
        prox_method = 'oob',
        matrix_type = 'sparse',
        n_landmark = 2000,
        t = "auto",
        n_pca = 100,
        mds_solver = "sgd",
        mds_dist = "euclidean",
        mds = "metric",
        n_jobs = 1,
        random_state = None,
        verbose = 0,
        diag_nonzero = True,
        **kwargs
        ):

        super(RFPHATE, self).__init__(**kwargs)
        
        self.n_components = n_components
        self.t = t
        self.n_landmark = n_landmark
        self.mds = mds
        self.n_pca = n_pca
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
        self.diag_nonzero = diag_nonzero

    def _fit_transform(self, X, y):

        """Internal method for fitting and transforming the data
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted to dtype=np.float32.
            If a sparse matrix is provided, it will be converted into a sparse csc_matrix.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in regression).
        """

        n,  _= X.shape
        self.fit(X, y)
        proximity = self.get_proximities(X)

        if self.diag_nonzero:
            if self.prox_method == 'rfgap':

                if self.matrix_type == 'dense':
                    diag_indices = np.diag_indices_from(proximity)
                    proximity[diag_indices] = 1
                else:
                    i = np.array([range(n)]).squeeze()
                    diag = sparse.csr_matrix((np.ones(n), (i, i)), shape = (n, n))
                    proximity = proximity + diag

            self.proximity = proximity
                        
        phate_op = PHATE(n_components = self.n_components,
            t = self.t,
            n_landmark = self.n_landmark,
            mds = self.mds,
            n_pca = self.n_pca,
            knn_dist = self.knn_dist,
            mds_dist = self.mds_dist,
            mds_solver = self.mds_solver,
            random_state = self.random_state,
            verbose = self.verbose)

        self.embedding_ = phate_op.fit_transform(proximity)

    def fit_transform(self, X, y):

        """Applies _fit_tranform to the data, X, y, and returns the RF-PHATE embedding

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted to dtype=np.float32.
            If a sparse matrix is provided, it will be converted into a sparse csc_matrix.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in regression).


        Returns
        -------
        array-like (n_features, n_components)
            A lower-dimensional representation of the data following the RF-PHATE algorithm
        """
        self._fit_transform(X, y)
        return self.embedding_

class RFPHATEReg(RFProximityReg, PHATE):

    """An RF-PHATE class which is used to fit a random forest, generate RF-proximities,
       and create RF-PHATE embeddings.

    Parameters
    ----------
    n_components : int
        The number of dimensions for the RF-PHATE embedding

    prox_method : str
        The type of proximity to be constructed.  Options are 'original', 'oob', and
        'rfgap' (default is 'oob')

    matrix_type : str
        Whether the proximity type should be 'sparse' or 'dense' (default is sparse)
    
    n_landmark : int, optional
        number of landmarks to use in fast PHATE (default is 2000)

    t : int, optional
        power to which the diffusion operator is powered.
        This sets the level of diffusion. If 'auto', t is selected
        according to the knee point in the Von Neumann Entropy of
        the diffusion operator (default is 'auto')

    n_pca : int, optional
        Number of principal components to use for calculating
        neighborhoods. For extremely large datasets, using
        n_pca < 20 allows neighborhoods to be calculated in
        roughly log(n_samples) time (default is 100)

    mds : string, optional
        choose from ['classic', 'metric', 'nonmetric'].
        Selects which MDS algorithm is used for dimensionality reduction
        (default is 'metric')

    mds_solver : {'sgd', 'smacof'}
        which solver to use for metric MDS. SGD is substantially faster,
        but produces slightly less optimal results (default is 'sgd')

    mds_dist : string, optional
        Distance metric for MDS. Recommended values: 'euclidean' and 'cosine'
        Any metric from `scipy.spatial.distance` can be used. Custom distance
        functions of form `f(x, y) = d` are also accepted (default is 'euclidean')

    n_jobs : integer, optional
        The number of jobs to use for the computation.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging.
        For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for
        n_jobs = -2, all CPUs but one are used (default is 1)

    random_state : integer
        random seed state set for RF and MDS


    verbose : int or bool
        If `True` or `> 0`, print status messages (default is 0)

    diag_nonzero: bool
        Only used if prox_method == 'rfgap.  Replaces the zero-diagonal entries
        of the rfgap proximities with ones (default is True)
    """

    def __init__(
        self,
        n_components = 2,
        prox_method = 'oob',
        matrix_type = 'sparse',
        n_landmark = 2000,
        t = "auto",
        n_pca = 100,
        mds_solver = "sgd",
        mds_dist = "euclidean",
        mds = "metric",
        n_jobs = 1,
        random_state = None,
        verbose = 0,
        diag_nonzero = True,
        **kwargs
        ):

        super(RFPHATEReg, self).__init__(**kwargs)
        
        self.n_components = n_components
        self.t = t
        self.n_landmark = n_landmark
        self.mds = mds
        self.n_pca = n_pca
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
        self.diag_nonzero = diag_nonzero

    def _fit_transform(self, X, y):

        """Internal method for fitting and transforming the data
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted to dtype=np.float32.
            If a sparse matrix is provided, it will be converted into a sparse csc_matrix.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in regression).
        """

        n,  _= X.shape
        self.fit(X, y)
        proximity = self.get_proximities(X)

        if self.diag_nonzero:
            if self.prox_method == 'rfgap':

                if self.matrix_type == 'dense':
                    diag_indices = np.diag_indices_from(proximity)
                    proximity[diag_indices] = 1
                else:
                    i = np.array([range(n)]).squeeze()
                    diag = sparse.csr_matrix((np.ones(n), (i, i)), shape = (n, n))
                    proximity = proximity + diag

            self.proximity = proximity
                        
        phate_op = PHATE(n_components = self.n_components,
            t = self.t,
            n_landmark = self.n_landmark,
            mds = self.mds,
            n_pca = self.n_pca,
            knn_dist = self.knn_dist,
            mds_dist = self.mds_dist,
            mds_solver = self.mds_solver,
            random_state = self.random_state,
            verbose = self.verbose)

        self.embedding_ = phate_op.fit_transform(proximity)

    def fit_transform(self, X, y):

        """Applies _fit_tranform to the data, X, y, and returns the RF-PHATE embedding

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted to dtype=np.float32.
            If a sparse matrix is provided, it will be converted into a sparse csc_matrix.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in regression).


        Returns
        -------
        array-like (n_features, n_components)
            A lower-dimensional representation of the data following the RF-PHATE algorithm
        """
        self._fit_transform(X, y)
        return self.embedding_