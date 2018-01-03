import scipy.io

mat_path = '/Users/lmao/Downloads/wiki/wiki.mat'

mat = scipy.io.loadmat(mat_path)

print(mat.keys())
# print(mat.get("wiki"))
data = mat.get("wiki")
print(data)