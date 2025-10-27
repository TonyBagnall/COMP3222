import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Encoder for string columns.

    strategy='attribute': one column per string feature
        enc_j(cat) = P(X_j = cat)

    strategy='class': K columns per string feature (K = #classes in y)
        enc_j(cat, c) = P(y = c | X_j = cat)   [target/impact encoding]

    Parameters
    ----------
    strategy : {'attribute', 'class'}, default='attribute'
    """

    def __init__(self, strategy='attribute'):
        self.strategy = strategy

    # ---------- helpers ----------
    @staticmethod
    def _is_string_col(col: np.ndarray) -> bool:
        if col.dtype.kind in ('U', 'S'):  # unicode/bytes
            return True
        return False

    # ---------- sklearn API ----------
    def fit(self, X, y=None):
        X = check_array(X, dtype=None, force_all_finite=True, accept_sparse=False)
        n, d = X.shape
        self.n_features_in_ = d

        if self.strategy == "class":
            self.classes_ = np.unique(y)
            self.class_to_idx_ = {c: k for k, c in enumerate(self.classes_)}
            counts = np.array([np.sum(y == c) for c in self.classes_], dtype=float)
            self.class_priors_ = counts / counts.sum()  # (K,)

        # column metadata
        self.string_cols_ = []
        self.numeric_cols_ = []

        # per string column state
        self.col_cats_ = {}        # j -> categories (sorted) (n_cat,)
        self.col_cat_to_idx_ = {}  # j -> dict(cat -> idx)
        self.attr_freqs_ = {}      # j -> (n_cat,)
        self.class_freqs_ = {}     # j -> (n_cat, K)

        for j in range(d):
            col = X[:, j]
            if self._is_string_col(col):
                self.string_cols_.append(j)
                cats = np.unique(col)  # sorted
                self.col_cats_[j] = cats
                # fast mapping via search sorted because cats is sorted
                codes = np.searchsorted(cats, col)
                self.col_cat_to_idx_[j] = {cats[k]: k for k in range(cats.size)}
                counts = np.bincount(codes, minlength=cats.size).astype(float)

                if self.strategy == "attribute":
                    self.attr_freqs_[j] = counts / n
                else:
                    K = len(self.classes_)
                    cat_class = np.zeros((cats.size, K), dtype=float)
                    y_idx = np.vectorize(self.class_to_idx_.get, otypes=[int])(y)
                    for k in range(K):
                        mask = (y_idx == k)
                        if mask.any():
                            cat_class[:, k] = np.bincount(codes[mask], minlength=cats.size)
                    denom = counts[:, None] + self.alpha
                    numer = cat_class + self.alpha * self.class_priors_
                    with np.errstate(divide='ignore', invalid='ignore'):
                        self.class_freqs_[j] = np.divide(
                            numer, denom, out=np.zeros_like(numer), where=denom > 0
                        )
            else:
                self.numeric_cols_.append(j)

        # compute output width
        self.output_dimensions_ = 0
        for j in range(d):
            if j in self.numeric_cols_:
                self.output_dimensions_ += 1
            elif self.strategy == "attribute":
                self.output_dimensions_ += 1
            else:
                self.output_dimensions_ += len(self.classes_)

        return self

    def transform(self, X):
        X = check_array(X, dtype=None, force_all_finite=True, accept_sparse=False)
        n = X.shape[0]
        out = np.empty((n, self.output_dimensions_), dtype=float)
        out_col = 0

        for j in range(self.n_features_in_):
            col = X[:, j]
            if j in self.numeric_cols_:
                out[:, out_col] = col.astype(float, copy=False)
                out_col += 1

            # string column
            cats = self.col_cats_[j]
            # map via searchsorted; detect unknowns
            codes = np.searchsorted(cats, col)
            known = (codes >= 0) & (codes < cats.size) & (cats[codes] == col)
            unknown_idx = np.where(~known)[0]
            if unknown_idx.size:
                if self.handle_unknown == "error":
                    first = unknown_idx[0]
                    raise ValueError(
                        f"Unknown category '{col[first]}' in column index {j} at row {first}."
                    )

            if self.strategy == "attribute":
                vec = np.zeros(n, dtype=float)
                vec[known] = self.attr_freqs_[j][codes[known]]
                # 'prior' → 0.0 default for unknowns (already zero)
                out[:, out_col] = vec
                out_col += 1
            else:
                K = len(self.classes_)
                block = np.zeros((n, K), dtype=float)
                block[known, :] = self.class_freqs_[j][codes[known], :]
                if unknown_idx.size and self.handle_unknown == "prior":
                    block[unknown_idx, :] = self.class_priors_
                out[:, out_col : out_col + K] = block
                out_col += K

        return out
