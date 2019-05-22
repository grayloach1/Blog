# --*-- coding:utf-8 --*--
'''
三个警察和三个囚徒过河问题

三个警察和三个囚徒【需要】过一条河，河边只有一条船。
1）船每次最多只能载两个人。
2）无论在河的哪边，当囚徒人数多于警察的人数时，警察会被囚徒杀死。

用程序求解让所有人安全过河的方案。

数据结构：
用一个元组来表示两边的人员分布，例如：
('3，2，1')表示 左岸：3警察2罪犯1船，因为右岸状态与左岸状态完全一一对应，可以用一个简单的函数推算，因此程序计算过程中不直接保存右岸状态。

STEP 1：
生成所有可能的组合，过滤这些组合，把所有人都安全的可能状态及船的状态，保存在一个列表里

STEP 2：
对列表里这些状态进行排序，使每一种状态可以过渡为下一个状态（满足船是来回划，每次最多乘两人的规定），如果通过一系列中间状态，可以把初始状态（3，3，1）与目标状态（0，0，0）连接起来的，那么就找到了问题的解。
'''

from collections import deque

def calcRight(left):
    '''根据左岸状态，返回右岸的状态'''
    police,criminal,boat = left
    right = POLICES - police,CRIMINALS - criminal, 1 - boat
    return right

def isSafe(status):
    '''该状态是安全的吗？若囚犯数量多于警察数量，就不安全'''
    if 0 < status[0] < status[1]:
        return False
    else:
        return True

def filterSafe(allStatus):
    '''过滤掉那些不安全的状态'''
    filted = []
    for left in allStatus:
        right = calcRight(left)
        if isSafe(right) and isSafe(left):
            filted.append(left)
    return filted

def isNext(preItem=(3, 3, 1),item=(3, 1, 0)):
    '''可以从上一个状态转变为下一个状态吗？'''
    if preItem[2] == 1:
        flag = 1
    else:
        flag = -1

    police = flag*(preItem[0] - item[0])
    craminal = flag*(preItem[1] - item[1])

    if (preItem[2]+item[2] == 1) and (police,craminal) in [(1,0),(0,1),(2,0),(0,2),(1,1)]:    #船只有一只，且最多只能运两个人
        return True
    else:
        return False
def statusFormat(status):
    right = calcRight(status)
    return '左岸：{}  {}  {},  右岸：{}  {}  {}'.format(*status, *right)

def find(p):
    global s
    # 在副本中循环而不是在S中循环，因为后面的操作会破环s中元素的相互位置关系
    for item in s[p:]:
        if isNext(s[p-1], item):
            s[s.index(item)], s[p] = s[p], s[s.index(item)]
            #找到满足条件的下一个元素，则把它保存到当前位置
            if s[p] == (0,0,0): solutions.append(s[:p+1])    #递归结束条件：找到目标状态元素(0,0,0)
            find(p+1)                                        #递归查找下一个满足条件的元素
    else:
        return
if __name__ == '__main__':

    '''预设警察和囚犯的数量'''
    POLICES = 3
    CRIMINALS = 3

    '''生成(警察，囚徒，船)的状态所有组合'''
    allStatus = [(p, c, b) for p in range(POLICES+1) for c in range(CRIMINALS+1) for b in (0, 1)]
    '''过滤掉不安全的状态，把结果存入一个数组'''
    statusCollection = filterSafe(allStatus)

    '''状态始化，左边岸(警察，囚徒，船)的数量为(3, 3, 1)'''
    s = statusCollection
    init = s.index((POLICES, CRIMINALS, 1))
    s[0], s[init] = s[init], s[0]
    solutions = []  # 用来保存最终的解决方案

    find(p=1)       #第一个元素为初始状态，因此从第二元素开始查找。

    print(f'共找到{len(solutions)}种解决方案。第一种解决方案为：')
    print('     警 囚 船        警 囚 船')
    for status in solutions[0]:
        print(statusFormat(status))
#增加了一些没有必要的说明
#增加1