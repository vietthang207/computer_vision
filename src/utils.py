import numpy as np

def conv2d(image, kernel, padding='same'):
	# return image
	image_height = image.shape[0]
	image_width = image.shape[1]
	kernel_height = kernel.shape[0]
	kernel_width = kernel.shape[1]
	offset_height = kernel_height // 2
	offset_width = kernel_width // 2
	padded_image = np.zeros(((image_height + offset_height * 2), (image_width + offset_width * 2)))
	output = np.zeros(((image_height + offset_height * 2), (image_width + offset_width * 2)))
	
	for i in range(image_height):
		for j in range(image_width):
			padded_image[i + offset_height][j + offset_width] = image[i][j]

	for x in range(image_height):
		for y in range(image_width):
			i = x + offset_height
			j = y + offset_width
			pi = padded_image[x:x+kernel_height, y:y+kernel_width]
			output[i][j] = np.sum(np.sum(pi * kernel))

	return output[offset_height : offset_height + image_height, offset_width : offset_width + image_width]

def gaussWin(N, alpha=2.5):
    gw = np.zeros(N);
    for i in range(N):
        n = i - (N-1.0)/2
        arg = -1.0/2 * ((alpha * n/((N-1.0)/2)) ** 2)
        gw[i] = np.exp(arg)
    gw.shape = (N, 1) 
    return gw

if __name__ == "__main__":
	image = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
	kernel = np.array([[1.0/9, 1.0/9, 1.0/9], [1.0/9, 1.0/9, 1.0/9], [1.0/9, 1.0/9, 1.0/9]])
	res = conv2d(image, kernel)
	print(res)
