"""
寻找一种抽象基类的替代方法

定义了一个基类，抽象基类主要用来执行检查以确保子类实现了某些特定的方法。
为此抽象基类必须引入两种高级的工具：函数装饰器和元类声明。既然抽象基类
的目的是在直接实例化基类或者实例化未实现期望方法的子类时引发错误，从而
实现检查的目的（示例见Super1类），那么是不是可以简单的在专门初始化实例
的__init__()函数中，使用getattr(self, 'method')来检查实例是否定义了
method属性呢？示例见如下的Super2类。
"""
from abc import ABCMeta, abstractmethod


class Super1(metaclass=ABCMeta):
    @abstractmethod
    def method(self): pass


class Super2:
    def __init__(self):
        getattr(self, 'method')   #在实例中搜索'method'属性


class Sub(Super2):
    def method(self):
        print('spam')
    pass


x = Sub()
# s = Super()                     #直接实例化基类会引发错误