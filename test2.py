import numpy as np
import math

a = np.asarray([2+1j, 3+2j])
b = np.asarray([4+2j, 3+4j])

a = np.dot(np.conjugate(a-b), a-b)

print(a)