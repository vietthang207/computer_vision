import sys
import numpy as np
import cv2
original_image_name = sys.argv[1]
intensity_range = 256
extension = 'jpg'
image = cv2.imread(original_image_name, cv2.IMREAD_GRAYSCALE)
num_of_rows = len(image)
num_of_cols = len(image[0])
image_size = num_of_rows * num_of_cols

sobel_horizontal = np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1])
sobel_vertical = np.array([-1, -2, -1, 0, 0, 0, 1, 2, 1])
sobel_image = np.zeros((num_of_rows,num_of_cols));

def get_adjacent_pixels(row_number, col_number):
	mat = np.zeros((3,3))
	for i in range(3):
		row = max(0, min(row_number - 1 + i, num_of_rows - 1))
		for j in range(3):
			col = max(0, min(col_number - 1 + j, num_of_cols - 1))
			mat[i, j] = image[row, col]
	mat.resize((1, 9))
	return mat
sobel_max = 0
sobel_min = 1000
for row_number in range(num_of_rows):
	for col_number in range(num_of_cols):
		adjacent_pixels = get_adjacent_pixels(row_number, col_number)

		sobel_vertical_strength = adjacent_pixels.dot(sobel_horizontal)
		sobel_horizontal_strength = adjacent_pixels.dot(sobel_vertical)
		sobel_strength = np.sqrt(sobel_vertical_strength * sobel_vertical_strength + sobel_horizontal_strength * sobel_horizontal_strength)
		if sobel_strength > sobel_max:
			sobel_max = sobel_strength
		if sobel_strength < sobel_min:
			sobel_min = sobel_strength

		sobel_image[row_number, col_number] = sobel_strength

for row_number in range(num_of_rows):
	for col_number in range(num_of_cols):
		sobel_image[row_number, col_number] = int(float(sobel_max - sobel_image[row_number, col_number])/(sobel_max - sobel_min) * (intensity_range - 1))

sobel_image_name = original_image_name + '_sobel_result.' + extension
cv2.imwrite(sobel_image_name, sobel_image)
