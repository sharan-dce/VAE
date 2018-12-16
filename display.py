import numpy as np
import matplotlib.pyplot as plt

def display(img):
	w = 28
	h = 28
	fig = plt.figure(figsize=(11, 11))
	columns = 15
	rows = 15
	for i in range(1, columns * rows + 1):
	    image = img[i - 1].reshape([28, 28])
	    fig.add_subplot(rows, columns, i)
	    plt.imshow(image, cmap = 'gray')
	plt.show()
	