#encoding="utf-8"
import numpy as np

#使用一个*，可以使用变长参数列表，参数作为一个元组传递进来
def testOneStar(*pa):
	print("pa=", pa)
	print("pa type=", type(pa))

#使用两个*，传递进去的变长参数会作为字典保存
def testTwoStar(**pa):
	print ("pa=", pa)
	print("pa type=", type(pa))

testOneStar(1, 2, 3, "a", "b", "c")
#('pa=', (1, 2, 3, 'a', 'b', 'c'))
#('pa type=', <type 'tuple'>)

testTwoStar(a=1, b=2, c=3)
#('pa=', {'a': 1, 'c': 3, 'b': 2})
#('pa type=', <type 'dict'>)

x = np.arange(1, 9).reshape(2,4)
print x