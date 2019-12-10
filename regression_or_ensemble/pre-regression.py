data = None

with open("../pickles/preprocessed_data.pkl","rb") as f:
    data = pickle.load(f)

labels = data['log_price']
data = data.drop('log_price', 1)

# Normalize data
data = (data - data.min())/(data.max() - data.min())

print(data.shape)
cov_matrix = data.cov()
eigen_values,eigen_vectors = np.linalg.eig(cov_matrix)
plt.plot(range(len(eigen_values)),eigen_values,alpha=0.7, color = "blue")
plt.xlabel("Principle Component")
plt.ylabel("Eigen value")
plt.title("Eigen Value Plot")
plt.show()
