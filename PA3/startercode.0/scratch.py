import numpy as np
X = np.array([[1, -2], [-3, 4]])
print(np.array(X > 0.0).astype(float))
print(np.maximum(X, 0.0))