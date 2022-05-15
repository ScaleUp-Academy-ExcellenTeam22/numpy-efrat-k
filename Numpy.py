import numpy as np
import matplotlib.pyplot as plt

# 1
arr_0_to_20 = np.array(range(21))
arr_0_to_20_after_scale = np.concatenate((arr_0_to_20[:8], (arr_0_to_20[9:16] * -1), arr_0_to_20[16:]))
print("tar1: \n", arr_0_to_20_after_scale)

# 2
arr_div_to_10_part = np.linspace(5, 50, 10)
print("tar2: \n", arr_div_to_10_part)

# 3
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("tar3: \n", matrix.shape)

# 4
matrix_10x10 = np.zeros((10, 10))
matrix_10x10[0] = np.ones(10)
matrix_10x10[9] = np.ones(10)
matrix_10x10[:, 0] = np.ones(10)
matrix_10x10[:, 9] = np.ones(10)
print("tar4: \n", matrix_10x10)

# 5
matrix_to_add_vector = np.zeros((3, 3))
vec_for_added = np.ones((5))
matrix_after_added = np.zeros((matrix_to_add_vector.shape[0], matrix_to_add_vector.shape[1] + vec_for_added.shape[0]),int)
matrix_after_added[:, :matrix_to_add_vector.shape[1]] = matrix_to_add_vector
matrix_after_added[:, matrix_to_add_vector.shape[1]:] = vec_for_added
print("tar5: \n", matrix_after_added)

# 6

x = np.arange(0,4*np.pi,0.1)
y = np.sin(x)
plt.plot(x,y)
print("tar6: \n")
print(plt)
plt.show()


# 7
matrix_randon = array = np.random.randint(20, size=(4, 4))
print("tar7: \nbefor\n ", matrix_randon)
matrix_randon[[0, 3]] = matrix_randon[[3, 0]]
matrix_randon[[1, 2]] = matrix_randon[[2, 1]]
print("after \n", matrix_randon)

# 8
matrix = np.array([[5, 3, 9],
              [6, 8, 6],
              [1, 2, 9]])
print("tar8: \nbefor\n ", matrix)
replace_equal = np.where(matrix == 6, 10, matrix)
print("after replace equal \n", replace_equal)
replace_small = np.where(matrix < 6, 10, matrix)
print("after replace small \n", replace_small)
replace_big = np.where(matrix > 6, 10, matrix)
print("after replace big \n", replace_big)

# 9
arr_a = np.array([1, 2, 3, 4, 5])
arr_b = np.array([2, 3, 4, 5, 6])
arr_result = arr_a * arr_b
print("tar9: \n", arr_result)

# 10

matrix_to_sort = np.array([[4, 6], [2, 1]])
print("tar10: \n", matrix_to_sort)
print("Sort along the first axis: \n", np.sort(matrix_to_sort, axis=0))
print("Sort along the last axis: \n", np.sort(matrix_to_sort))

# 11
unit_matrix = np.eye(3)
print("tar11: \n", unit_matrix)

# 12
arr_delete_1 = np.array([1, 2, 3, 4, 5, 1, 6, 7, 8, 9, 1])
num_to_delete = np.array([1])
after_delete = np.setdiff1d(arr_delete_1, num_to_delete)
print("tar12: \n", after_delete)

# 13
arr_a = np.array((10, 20, 30))
arr_b = np.array((40, 50, 60))
after_integration = np.dstack((arr_a, arr_b))
print("tar13: \n", after_integration)

# 14

arr_1d = np.array([0, 1, 2, 3])
arr_2d = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
print("tar14: \n")
for a, b in np.nditer([arr_1d, arr_2d]):
    print("%d:%d" % (a, b), )
