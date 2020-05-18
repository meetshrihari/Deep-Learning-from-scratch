import numpy as np
import matplotlib.pyplot as plt
#import sys
#sys.path.append("/home/shrihari/Downloads/Deep Learning/Exercise1/pycharm/")
import pattern as pat

#C = pat.Checkers(100, 25)
#C.draw()
#C.show()

#RGB = pat.Spectrum(255)
#RGB.draw()
#RGB.show()

#Circ = pat.Circle(1024, 200, (512, 256))
#Circ.draw()
#Circ.show()

from generator import ImageGenerator

label_path = '/home/shrihari/Downloads/Deep Learning/Exercise1/src_to_implement/data/Labels.json'
file_path = '/home/shrihari/Downloads/Deep Learning/Exercise1/src_to_implement/data/exercise_data/'

gen = ImageGenerator(file_path, label_path, 12, [32, 32, 3], rotation=True, mirroring=False, shuffle=True)

#b,l = gen.next()

#print(b)
#print(l)


gen.show()
