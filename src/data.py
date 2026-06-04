"""BUSI dataset loader: image/mask pairing, label extraction, validation.

ISP Characteristics (W6):
  C1 - root path: {exists, missing}
  C2 - folder structure: {complete, missing subfolder}
  C3 - image/mask pair: {matched, size mismatch}
"""


def load_busi(root):
    """Load BUSI dataset from root directory.

    Returns list of (image, mask, label) tuples where
    label in {'benign', 'malignant', 'normal'}.

    Raises:
        FileNotFoundError: if root or a required subfolder is missing.
        ValueError: if any image/mask pair has mismatched spatial dimensions.

    Graph Coverage edges (W7):
    [START] -> check_root -> [root_missing? -> FileNotFoundError] -> iterate_labels
    iterate_labels -> [label_dir_missing? -> FileNotFoundError] -> load_pairs
    load_pairs -> [size_mismatch? -> ValueError] -> [END]
    """
    raise NotImplementedError
