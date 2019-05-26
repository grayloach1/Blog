# -*- coding:utf-8 -*-
'''
Bulls and Cows
一个人出数字，一方猜。出数字的人要先想好一个没有重复数字的4位数，例:8123，不能让猜的人知道。猜的人就可以开始猜。每猜一个数，出数者就要根据这个数字给出几A几B，例猜1562，则为 0A2B ，其中A前面的数字表示位置正确的数的个数，而B前的数字表示数字正确而位置不对的数的个数。

接着猜的人再根据出题者的几A几B继续猜，直到猜中为止。
'''
from itertools import permutations

def isLegal(i,historyAnswer):
    '''historyAnswer构成一把筛子，筛选出满足条件的元素'''
    lst = (matchAnswer(i,item,answer) for item, answer in historyAnswer)
    return all(lst)


def matchAnswer(i, j, refer):
    '''用两个元素i,j的关系是否满足参考答案refer'''
    A = [i for i,j in zip(i, j) if i == j]
    B = set(i).union(j)  #利用集合的特性去掉重复元素
    xA = len(A)
    yB = 8 - len(B) - xA
    result = f'{xA}A{yB}B'
    return result == refer

def main():
    historyAnswer = []
    for i in permutations(range(9), 4):
        if not isLegal(i,historyAnswer):
            continue
        #print(historyAnswer)
        print(i,end=' ')
        answer = input('xAyB:')
        if answer in ('4A0B','4a0b','Quit', 'quit','Q','q','e','Exit','exit','E'):
            print(f'Computer have try {len(historyAnswer)} times. Bye!')
            break
        else:
            historyAnswer.append((i,answer.upper()))
    print('Game OVER')
if __name__ == '__main__':
    main()
