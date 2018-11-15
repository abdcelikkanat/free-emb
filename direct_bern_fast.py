import pyximport; pyximport.install()
from direct_bern_cython import *

mynums = [0, 1, 2, 3]

num = say(0, mynums)

print(num)