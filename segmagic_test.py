import os
import glob
import sys

# Done: figure out the current exe directory
if hasattr(sys, '_MEIPASS'):
    base = sys._MEIPASS
else:
    base = os.getcwd()

print(glob.glob(base+"/*"))
print("\n\n")
print(glob.glob(base+"/*/*"))