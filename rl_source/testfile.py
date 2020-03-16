import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
	sys.path.append(module_path)

print(os.listdir('.'))

import testfile2 as tf2



print ("{} {}, you age is {}".format(tf2.greet,tf2.name,tf2.age))
os.mkdir(tf2.mkdirpath + 'ppo')