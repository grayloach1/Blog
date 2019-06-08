# -*- coding:utf-8 -*-
'''
Bulls and Cows

一个人出数字，一方猜。出数字的人要先想好一个没有重复数字的4位数，
例:8123，不能让猜的人知道。猜的人就可以开始猜。每猜一个数，出数
者就要根据这个数字给出几A几B，例猜1562，则为 0A2B ，其中A前面的
数字表示位置正确的数的个数，而B前的数字表示数字正确而位置不对的
数的个数。

接着猜的人再根据出题者的几A几B继续猜，直到猜中为止。
'''
from itertools import permutations

def is_legal(i, history_answer):
    '''history_answer构成一把筛子，筛选出满足条件的元素'''
    lst = (match_answer(i, item, answer) for item, answer in history_answer)
    return all(lst)


def match_answer(i, j, refer):
    '''用两个元素i,j的关系是否满足参考答案refer'''
    list_a = [i for i, j in zip(i, j) if i == j]
    set_b = set(i).union(j)  #利用集合的特性去掉重复元素
    x_a = len(list_a)
    y_b = 8 - len(set_b) - x_a
    result = f'{x_a}A{y_b}B'
    return result == refer

def main():
    history_answer = []
    for i in permutations(range(9), 4):
        if not is_legal(i, history_answer):
            continue
        #print(history_answer)
        print(i, end=' ')
        answer = input('xAyB:')
        if answer in ('4A0B', '4a0b', 'Quit', 'quit', 'Q', 'q', 'e', 'Exit', 'exit', 'E'):
            print(f'Computer have try {len(history_answer)} times. Bye!')
            break
        else:
            history_answer.append((i, answer.upper()))
    print('Game OVER')
if __name__ == '__main__':
    main()
