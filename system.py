from utils import *
import numpy as np
import scipy.linalg

# Training images are in the shape (3000, 784)

def image_to_reduced_feature(images, split='test', max_components=48):
    """
    Applies PCA for dimensionality reduction. 
    Keeps consistency between training and testing phases.
    """
    global pca_mean, pca_std, pca_components

    if split == 'train':
        # Step 1: Compute statistics for the training data
        mean_vector = np.mean(images, axis=0)
        std_vector = np.std(images, axis=0)

        # Handle any zero or near-zero standard deviation to avoid division issues
        std_vector[std_vector < 1e-10] = 1e-10  # Minimal threshold for stability
        pca_mean, pca_std = mean_vector, std_vector

        # Step 2: Center and scale the training data
        normalized_images = (images - mean_vector) / std_vector

        # Step 3: Compute covariance matrix and eigen decomposition
        covariance_matrix = np.cov(normalized_images, rowvar=False)
        eigenvalues, eigenvectors = scipy.linalg.eigh(covariance_matrix)

        # Debugging note: Uncomment to inspect eigenvalues
        # print(f"Eigenvalues (max): {eigenvalues.max()}, Eigenvalues (min): {eigenvalues.min()}")

        # Step 4: Select top principal components based on max_components
        max_components = min(max_components, normalized_images.shape[1])
        sorted_indices = np.argsort(eigenvalues)[::-1]
        top_components_indices = sorted_indices[:max_components]
        pca_components = eigenvectors[:, top_components_indices]

        # Project training data onto the top principal components
        reduced_features = np.dot(normalized_images, pca_components)
        return reduced_features

    elif split == 'test':
        
        # Load PCA parameters saved during training
        model = load_model()
        pca_mean, pca_std, pca_components = model.get_pca_params()

        # Validation: Ensure PCA parameters are properly loaded
        if pca_mean is None or pca_std is None or pca_components is None:
            raise ValueError("PCA parameters missing. Check if the model was trained and saved properly.")

        # Handle zero standard deviations
        pca_std[pca_std < 1e-10] = 1e-10

        # Debugging note: Uncomment to inspect PCA parameters
        # print(f"PCA Mean: {pca_mean.shape}, PCA Std Min/Max: {pca_std.min()} - {pca_std.max()}")

        # Validate test data range and adjust outliers
        lower_bound = pca_mean - 3 * pca_std
        upper_bound = pca_mean + 3 * pca_std
        if not np.all((images >= lower_bound) & (images <= upper_bound)):
            print("Notice: Some test data points fall outside the training range. Adjusting them.")
            images = np.clip(images, lower_bound, upper_bound)

        # Center and scale test data using training PCA parameters
        normalized_images = (images - pca_mean) / pca_std

        # Project test data onto the saved PCA components
        reduced_features = np.dot(normalized_images, pca_components)
        return reduced_features

    else:
        raise ValueError("Invalid split specified. Use 'train' or 'test'.")


class NearestNeighborClassifier:
    def __init__(self):
        self.train_data = None
        self.train_labels = None

    def fit(self, features, labels):
        """
        Stores training features and corresponding labels.
        """
        self.train_data = features
        self.train_labels = labels

    def predict(self, test_features, neighbors=10):
        """
        Predicts labels for test samples using k-Nearest Neighbors.
        Resolves ties by selecting the class with the smallest total distance.
        """
        if self.train_data is None or self.train_labels is None:
            raise ValueError("Training data is missing. Call fit() before predict().")

        # Step 1: Calculate distances between test and training samples
        distances = np.linalg.norm(test_features[:, np.newaxis] - self.train_data, axis=2)

        # Step 2: Identify the nearest neighbors
        nearest_indices = np.argsort(distances, axis=1)[:, :neighbors]
        nearest_labels = self.train_labels[nearest_indices]
        nearest_distances = np.take_along_axis(distances, nearest_indices, axis=1)

        # Step 3: Resolve ties using the smallest total distance heuristic
        def resolve_tie(candidates, candidate_distances):
            counts = np.bincount(candidates)
            max_count = counts.max()
            tied_classes = np.where(counts == max_count)[0]
            # Debugging note: Uncomment to inspect tie-breaking behavior
            # print(f"Tie among classes: {tied_classes}")
            return min(tied_classes, key=lambda cls: candidate_distances[candidates == cls].sum())

        predictions = np.array([
            resolve_tie(nearest_labels[i], nearest_distances[i])
            for i in range(len(test_features))
        ])
        return predictions


def training_model(features, labels):
    """
    Trains a k-Nearest Neighbor classifier and stores PCA parameters.
    """
    global pca_mean, pca_std, pca_components

    # Initialize and train the classifier
    knn = NearestNeighborClassifier()
    knn.fit(features, labels)

    # Wrap the classifier and PCA parameters
    model = ModelWrapper(
        classifier=knn,
        pca_mean=pca_mean,
        pca_std=pca_std,
        pca_components=pca_components
    )

    return model


class ModelWrapper:
    """
    Combines PCA parameters with the classifier to allow consistent predictions.
    """
    def __init__(self, classifier, pca_mean, pca_std, pca_components):
        self.classifier = classifier
        self.pca_mean = pca_mean
        self.pca_std = pca_std
        self.pca_components = pca_components

    def predict(self, features):
        return self.classifier.predict(features)

    def get_pca_params(self):
        return self.pca_mean, self.pca_std, self.pca_components
