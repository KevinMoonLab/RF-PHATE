# Imports
import numpy as np
from scipy import sparse

#sklearn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import sklearn

from distutils.version import LooseVersion
if LooseVersion(sklearn.__version__) >= LooseVersion("0.24"):
    # In sklearn version 0.24, forest module changed to be private.
    from sklearn.ensemble._forest import _generate_unsampled_indices
    from sklearn.ensemble import _forest as forest
else:
    # Before sklearn version 0.24, forest was public, supporting this.
    from sklearn.ensemble.forest import _generate_unsampled_indices # Remove underscore from _forest
    from sklearn.ensemble import forest

from sklearn.utils.validation import check_is_fitted

class RFProximity(RandomForestClassifier):

    """
    This class takes on a random forest predictors (sklearn) and addes methods to 
    construct proximities from the random forest object. 

    Note: most methods here will not work until your forest is fit.  That is, use 
    RFProximity.fit(X, y) prior to generating proximities.
    """

    def __init__(self, prox_method = 'oob', matrix_type = 'sparse', **kwargs):
        super(RFProximity, self).__init__(**kwargs)

        self.prox_method = prox_method
        self.matrix_type = matrix_type
        #TODO: get args and such from below for 'paperwork'


    #TODO: Make sure this is still working after defining this method
    def fit(self, X, y, sample_weight = None):
        super().fit(X, y, sample_weight)
        self.leaf_matrix = self.apply(X)
        self.oob_indices = self.get_oob_indices(X)

    
    def _get_oob_samples(self, data):
        
      """
      This is a helper function for get_oob_indices. 
      
      Parameters
      ----------
      data : (n, d) array_like (numeric)
      
      """
      n = len(data)
      oob_samples = []
      for tree in self.estimators_:
    
        # Here at each iteration we obtain out of bag samples for every tree.
        oob_indices = _generate_unsampled_indices(tree.random_state, n, n)
        oob_samples.append(oob_indices)
      return oob_samples


    def get_oob_indices(self, data): #The data here is your X_train matrix
        
      """
      This generates a matrix of out-of-bag samples for each decision tree in the forest
      
      
      Parameters
      ----------
      data : (n, d) array_like (numeric)
      
      
      Returns
      -------
      oob_matrix : (n, n_estimators) array_like
      
      
      """
      n = len(data)
      num_trees = self.n_estimators
      oob_matrix = np.zeros((n, num_trees))
      oob_samples = self._get_oob_samples(data)
      for i in range(n):
        for j in range(num_trees):
          if i in oob_samples[j]:
            oob_matrix[i, j] = 1
      return oob_matrix
    
    def get_proximity_vector(self, ind, leaf_matrix, oob_indices):
        """
        This method produces a vector of proximity values for a given observation index. This is typically
        used in conjunction with get_proximities.
        
        Parameters
        ----------
        leaf_matrix : (n, n_estimators) array_like
        oob_indices : (n, n_estimators) array_like
        method      : string: methods may be 'original' or 'oob' (default)
        
        Returns
        ------
        prox_vec : (n, 1) array)_like: a vector of proximity values
        """
        n, num_trees = leaf_matrix.shape
        prox_vec = np.zeros((1, n))
        
        
        if self.prox_method == 'oob':
            treeCounts = np.zeros((1, n)) 

            for t in range(num_trees): 
                if oob_indices[ind, t] == 0:
                    continue
                else:
                    index = leaf_matrix[ind, t]
                    oob_matches = leaf_matrix[:, t] * oob_indices[:, t] == index 
                    oob_obs = oob_indices[:, t] == 1
                    treeCounts[0, oob_obs] += 1
                    prox_vec[0, oob_matches] += 1

            treeCounts[treeCounts == 0] = 1
            prox_vec /= treeCounts 

            cols = np.nonzero(prox_vec)[1]
            rows = np.ones(len(cols), dtype = int) * ind
            data = prox_vec[0, cols]
            
        elif self.prox_method == 'original':

            treeCounts = np.zeros((1, n)) 
            for t in range(num_trees): 

                index = leaf_matrix[ind, t]
                matches = leaf_matrix[:, t] == index
                prox_vec[0, matches] += 1
            prox_vec /= num_trees 

            cols = np.nonzero(prox_vec)[1]
            rows = np.ones(len(cols), dtype = int) * ind
            data = prox_vec[0, cols]

        elif self.prox_method == 'rfgap':

            in_bag_indices = 1 - oob_indices
            in_bag_leaves = leaf_matrix * in_bag_indices

            for t in range(num_trees): 
                if in_bag_indices[ind, t] == 1:
                    continue
                else:
                    index = leaf_matrix[ind, t] 
                    matches = in_bag_leaves[:, t] == index 
                    k = np.count_nonzero(matches)
                    if k > 0:
                        prox_vec[0, matches] += 1 / k
                    
            S = np.count_nonzero(in_bag_indices[ind, :] == 0)         
            prox_vec /= S
            prox_vec[0, ind] = 0 
            cols = np.nonzero(prox_vec)[1]
            rows = np.ones(len(cols), dtype = int) * ind
            data = prox_vec[0, cols]
 
        return data.tolist(), rows.tolist(), cols.tolist()
    
    
    def get_proximities(self, data):
        
        """
        This method produces a proximity matrix for the random forest object.
        
        
        Parameters
        ----------
        data : (n, d) array_like (numeric)
        method : string: methods may be 'original' or 'oob' (default)
        matrix_type: string: 'dense' (default) to return a dense matrix, 'sparse' to return a sparse crs matrix
        verbose : bool: should progress be printed to console (default: False)
        
        
        Returns
        -------
        prox (if matrix_type = dense) : a matrix of random forest proximities
        prox_sparse (if matrix_type = sparse) : a sparse crs_matrix of proximities
        
        """
        check_is_fitted(self)

        oob_indices  = self.get_oob_indices(data)
        leaf_matrix  = self.apply(data)
        n, _ = leaf_matrix.shape

        for i in range(n):
            if i == 0:
                    prox_vals, rows, cols = self.get_proximity_vector(i, leaf_matrix, oob_indices)
            else:
                if self.verbose:
                    if i % 100 == 0:
                        print('Finished with {} rows'.format(i))
                prox_val_temp, rows_temp, cols_temp = self.get_proximity_vector(i, leaf_matrix, oob_indices)
                prox_vals.extend(prox_val_temp)
                rows.extend(rows_temp)
                cols.extend(cols_temp)

        prox_sparse = sparse.csr_matrix((np.array(prox_vals), (np.array(rows), np.array(cols))), shape = (n, n)) 
        
        if self.matrix_type == 'dense':
            return np.array(prox_sparse.todense())
        
        else:
            return prox_sparse

    def prox_extend(self, extended_data):
        """
        Method to compute proximities between the original training observations and a set of new observations
        """
        check_is_fitted(self)
        n, num_trees = self.leaf_matrix.shape

        extended_leaf_matrix = self.apply(extended_data)
        n_ext, _ = extended_leaf_matrix.shape


        prox_vals = []
        rows = []
        cols = []
        
        if self.prox_method == 'oob':
            for ind in range(n):

                treeCounts = np.zeros((1, n_ext)) 
                prox_vec = np.zeros((1, n_ext))


                for t in range(num_trees): 
                    if self.oob_indices[ind, t] == 0:
                        continue
                    else:
                        index = self.leaf_matrix[ind, t]
                        oob_matches = extended_leaf_matrix[:, t] == index

                        oob_obs = self.oob_indices[ind, t] == 1
                        treeCounts[0, oob_obs] += 1
                        prox_vec[0, oob_matches] += 1

                prox_vec /= treeCounts

                col_inds = np.nonzero(prox_vec)[1]
                row_inds = np.ones(len(col_inds), dtype = int) * ind
                vals = prox_vec[0, col_inds]


                prox_vals.extend(vals)
                rows.extend(row_inds)
                cols.extend(col_inds)

  
        elif self.prox_method == 'original':

            for ind in range(n):

                prox_vec = np.zeros((1, n_ext))

                for t in range(num_trees): 

                    index = self.leaf_matrix[ind, t]
                    matches = extended_leaf_matrix[:, t] == index
                    prox_vec[0, matches] += 1

                prox_vec /= num_trees 

                col_inds = np.nonzero(prox_vec)[1]
                row_inds = np.ones(len(col_inds), dtype = int) * ind
                vals = prox_vec[0, col_inds]


                prox_vals.extend(vals)
                rows.extend(row_inds)
                cols.extend(col_inds)

        elif self.prox_method == 'rfgap':

            in_bag_indices = 1 - self.oob_indices
            in_bag_leaves = self.leaf_matrix * in_bag_indices

            
            for ind in range(n_ext):

                prox_vec = np.zeros((1, n))

                for t in range(num_trees): 

                    index = extended_leaf_matrix[ind, t] 
                    matches = in_bag_leaves[:, t] == index 
                    k = np.count_nonzero(matches)
                    if k > 0:
                        prox_vec[0, matches] += 1 / k
                    
                prox_vec /= num_trees
                prox_vec[0, ind] = 0 

                row_inds = np.nonzero(prox_vec)[1]
                col_inds = np.ones(len(row_inds), dtype = int) * ind
                vals = prox_vec[0, row_inds]

                prox_vals.extend(vals)
                rows.extend(row_inds)
                cols.extend(col_inds)
 

        prox_sparse = sparse.csr_matrix((np.array(prox_vals), (np.array(cols), np.array(rows))), shape = (n_ext, n))

        if self.matrix_type == 'dense':
            return prox_sparse.todense() 
        else:
            return prox_sparse



class RFProximityReg(RandomForestRegressor):

    """
    This class takes on a random forest predictors (sklearn) and addes methods to 
    construct proximities from the random forest object. 

    Note: most methods here will not work until your forest is fit.  That is, use 
    RFProximity.fit(X, y) prior to generating proximities.
    """

    def __init__(self, prox_method = 'oob', matrix_type = 'sparse', **kwargs):
        super(RFProximity, self).__init__(**kwargs)

        self.prox_method = prox_method
        self.matrix_type = matrix_type
        #TODO: get args and such from below for 'paperwork'


    #TODO: Make sure this is still working after defining this method
    def fit(self, X, y, sample_weight = None):
        super().fit(X, y, sample_weight)
        self.leaf_matrix = self.apply(X)
        self.oob_indices = self.get_oob_indices(X)

    
    def _get_oob_samples(self, data):
        
      """
      This is a helper function for get_oob_indices. 
      
      Parameters
      ----------
      data : (n, d) array_like (numeric)
      
      """
      n = len(data)
      oob_samples = []
      for tree in self.estimators_:
    
        # Here at each iteration we obtain out of bag samples for every tree.
        oob_indices = _generate_unsampled_indices(tree.random_state, n, n)
        oob_samples.append(oob_indices)
      return oob_samples


    def get_oob_indices(self, data): #The data here is your X_train matrix
        
      """
      This generates a matrix of out-of-bag samples for each decision tree in the forest
      
      
      Parameters
      ----------
      data : (n, d) array_like (numeric)
      
      
      Returns
      -------
      oob_matrix : (n, n_estimators) array_like
      
      
      """
      n = len(data)
      num_trees = self.n_estimators
      oob_matrix = np.zeros((n, num_trees))
      oob_samples = self._get_oob_samples(data)
      for i in range(n):
        for j in range(num_trees):
          if i in oob_samples[j]:
            oob_matrix[i, j] = 1
      return oob_matrix
    
    def get_proximity_vector(self, ind, leaf_matrix, oob_indices):
        """
        This method produces a vector of proximity values for a given observation index. This is typically
        used in conjunction with get_proximities.
        
        Parameters
        ----------
        leaf_matrix : (n, n_estimators) array_like
        oob_indices : (n, n_estimators) array_like
        method      : string: methods may be 'original' or 'oob' (default)
        
        Returns
        ------
        prox_vec : (n, 1) array)_like: a vector of proximity values
        """
        n, num_trees = leaf_matrix.shape
        prox_vec = np.zeros((1, n))
        
        
        if self.prox_method == 'oob':
            treeCounts = np.zeros((1, n)) 

            for t in range(num_trees): 
                if oob_indices[ind, t] == 0:
                    continue
                else:
                    index = leaf_matrix[ind, t]
                    oob_matches = leaf_matrix[:, t] * oob_indices[:, t] == index 
                    oob_obs = oob_indices[:, t] == 1
                    treeCounts[0, oob_obs] += 1
                    prox_vec[0, oob_matches] += 1

            treeCounts[treeCounts == 0] = 1
            prox_vec /= treeCounts 

            cols = np.nonzero(prox_vec)[1]
            rows = np.ones(len(cols), dtype = int) * ind
            data = prox_vec[0, cols]
            
        elif self.prox_method == 'original':

            treeCounts = np.zeros((1, n)) 
            for t in range(num_trees): 

                index = leaf_matrix[ind, t]
                matches = leaf_matrix[:, t] == index
                prox_vec[0, matches] += 1
            prox_vec /= num_trees 

            cols = np.nonzero(prox_vec)[1]
            rows = np.ones(len(cols), dtype = int) * ind
            data = prox_vec[0, cols]

        elif self.prox_method == 'rfgap':

            in_bag_indices = 1 - oob_indices
            in_bag_leaves = leaf_matrix * in_bag_indices

            for t in range(num_trees): 
                if in_bag_indices[ind, t] == 1:
                    continue
                else:
                    index = leaf_matrix[ind, t] 
                    matches = in_bag_leaves[:, t] == index 
                    k = np.count_nonzero(matches)
                    if k > 0:
                        prox_vec[0, matches] += 1 / k
                    
            S = np.count_nonzero(in_bag_indices[ind, :] == 0)         
            prox_vec /= S
            prox_vec[0, ind] = 0 
            cols = np.nonzero(prox_vec)[1]
            rows = np.ones(len(cols), dtype = int) * ind
            data = prox_vec[0, cols]
 
        return data.tolist(), rows.tolist(), cols.tolist()
    
    
    def get_proximities(self, data):
        
        """
        This method produces a proximity matrix for the random forest object.
        
        
        Parameters
        ----------
        data : (n, d) array_like (numeric)
        method : string: methods may be 'original' or 'oob' (default)
        matrix_type: string: 'dense' (default) to return a dense matrix, 'sparse' to return a sparse crs matrix
        verbose : bool: should progress be printed to console (default: False)
        
        
        Returns
        -------
        prox (if matrix_type = dense) : a matrix of random forest proximities
        prox_sparse (if matrix_type = sparse) : a sparse crs_matrix of proximities
        
        """
        check_is_fitted(self)

        oob_indices  = self.get_oob_indices(data)
        leaf_matrix  = self.apply(data)
        n, _ = leaf_matrix.shape

        for i in range(n):
            if i == 0:
                    prox_vals, rows, cols = self.get_proximity_vector(i, leaf_matrix, oob_indices)
            else:
                if self.verbose:
                    if i % 100 == 0:
                        print('Finished with {} rows'.format(i))
                prox_val_temp, rows_temp, cols_temp = self.get_proximity_vector(i, leaf_matrix, oob_indices)
                prox_vals.extend(prox_val_temp)
                rows.extend(rows_temp)
                cols.extend(cols_temp)

        prox_sparse = sparse.csr_matrix((np.array(prox_vals), (np.array(rows), np.array(cols))), shape = (n, n)) 
        
        if self.matrix_type == 'dense':
            return np.array(prox_sparse.todense())
        
        else:
            return prox_sparse

    def prox_extend(self, extended_data):
        """
        Method to compute proximities between the original training observations and a set of new observations
        """
        check_is_fitted(self)
        n, num_trees = self.leaf_matrix.shape

        extended_leaf_matrix = self.apply(extended_data)
        n_ext, _ = extended_leaf_matrix.shape


        prox_vals = []
        rows = []
        cols = []
        
        if self.prox_method == 'oob':
            for ind in range(n):

                treeCounts = np.zeros((1, n_ext)) 
                prox_vec = np.zeros((1, n_ext))


                for t in range(num_trees): 
                    if self.oob_indices[ind, t] == 0:
                        continue
                    else:
                        index = self.leaf_matrix[ind, t]
                        oob_matches = extended_leaf_matrix[:, t] == index

                        oob_obs = self.oob_indices[ind, t] == 1
                        treeCounts[0, oob_obs] += 1
                        prox_vec[0, oob_matches] += 1

                prox_vec /= treeCounts

                col_inds = np.nonzero(prox_vec)[1]
                row_inds = np.ones(len(col_inds), dtype = int) * ind
                vals = prox_vec[0, col_inds]


                prox_vals.extend(vals)
                rows.extend(row_inds)
                cols.extend(col_inds)

  
        elif self.prox_method == 'original':

            for ind in range(n):

                prox_vec = np.zeros((1, n_ext))

                for t in range(num_trees): 

                    index = self.leaf_matrix[ind, t]
                    matches = extended_leaf_matrix[:, t] == index
                    prox_vec[0, matches] += 1

                prox_vec /= num_trees 

                col_inds = np.nonzero(prox_vec)[1]
                row_inds = np.ones(len(col_inds), dtype = int) * ind
                vals = prox_vec[0, col_inds]


                prox_vals.extend(vals)
                rows.extend(row_inds)
                cols.extend(col_inds)

        elif self.prox_method == 'rfgap':

            in_bag_indices = 1 - self.oob_indices
            in_bag_leaves = self.leaf_matrix * in_bag_indices

            
            for ind in range(n_ext):

                prox_vec = np.zeros((1, n))

                for t in range(num_trees): 

                    index = extended_leaf_matrix[ind, t] 
                    matches = in_bag_leaves[:, t] == index 
                    k = np.count_nonzero(matches)
                    if k > 0:
                        prox_vec[0, matches] += 1 / k
                    
                prox_vec /= num_trees
                prox_vec[0, ind] = 0 

                row_inds = np.nonzero(prox_vec)[1]
                col_inds = np.ones(len(row_inds), dtype = int) * ind
                vals = prox_vec[0, row_inds]

                prox_vals.extend(vals)
                rows.extend(row_inds)
                cols.extend(col_inds)
 

        prox_sparse = sparse.csr_matrix((np.array(prox_vals), (np.array(cols), np.array(rows))), shape = (n_ext, n))

        if self.matrix_type == 'dense':
            return prox_sparse.todense() 
        else:
            return prox_sparse