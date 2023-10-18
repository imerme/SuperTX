from typing import Dict, List
from itertools import chain
from utils.constant import *

_NUMBER = 'A2' + NUMBER


def _gen(start, end, n):
    cards = ""
    for i in range(start, end):
        cards += _NUMBER[i] * n
    return cards


def singles():
    for n in [1] + list(range(7, 14)):
        for start in range(0, 14 - n + 1):
            yield _gen(start, start + n, 1)
            if n == 13:
                break


def pairs():
    for n in list(range(1, 14)):
        for start in range(0, 14 - n + 1):
            yield _gen(start, start + n, 2)
            if n == 13:
                break


def triads():
    for n in list(range(1, 14)):
        for start in range(0, 14 - n + 1):
            yield _gen(start, start + n, 3)
            if n == 13:
                break


def bomb():
    for n in range(4, 8):
        for card in NUMBER[:-2]:
            yield card * n


def triads_plus():
    return triads()


def ftk():
    for i in range(5):
        return '5XK'


def con():
    for card in NUMBER[:-2]:
        for i in range(4):
            yield card * 3

    for card in NUMBER[-2:]:
        yield card * 3


if __name__ == "__main__":
    for hand in con():
        print(hand)

