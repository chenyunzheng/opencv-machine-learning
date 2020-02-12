import numpy as np
import cv2
import matplotlib.pyplot as plt

plt.style.use('ggplot')

x = range(0,100)
plt.plot(x, np.sin(x))
plt.show()

# --> it's convenient to keep results in jupyter, and there are advantages using vs-code to run .py, 
# that is, to debug code and check the source code, but these functions are not necessary 
# for initial pythoner in scientific computing