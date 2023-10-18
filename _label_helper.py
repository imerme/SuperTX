
# 动作和标签之间的转换
# 统一为torch操作
# checked
from typing import Dict
from collections import Counter
import numpy as np

from .utils import str2arr
from .constant import *


# __all__ = [
#     "LABEL2ARRAY", "main2label", "sub2label"
# ]


def _get_value(label: int):
    act_arr = np.zeros((15,), dtype=int)
    if not label:
        pass
    elif 1 <= label <= 43:
        n = (label - 1) // 15 + 1
        return if_1_2_3_32(1, label, n)
    elif 44 <= label <= 58:  # todo: 3+2
        return if_1_2_3_32(44, label, 3)
    # 顺子
    elif 59 <= label <= 92:
        return if_combo_1(59, label)
    # 对顺子
    elif 93 <= label <= 181:
        return if_combo_2_3_32(93, label, 2)

    # 三顺子
    elif 182 <= label <= 270:
        return if_combo_2_3_32(182, label, 3)

    # 3顺子 + 2
    elif 271 <= label <= 340:
        return if_combo_2_3_32(271, label, 3)

    # bomb
    elif 341 <= label <= 405:
        return if_combo_bomb(341, label)

    # 510k
    elif 406 <= label <= 410:
        return if_combo_510k()

    # concentric
    elif 411 <= label <= 464:
        return if_combo_concentric(label)
    return act_arr


def if_1_2_3_32(start, label, n):
    # 3-Joker
    act_arr = np.zeros((15,), dtype=int)
    act_arr[(label - start) % 15] = n
    return act_arr


def if_combo_1(start, label):
    act_arr = np.zeros((15,), dtype=int)
    _increment = list(range(8, 2, -1))
    _n_card = list(range(7, 14, 1))
    _increment.append(1)
    _increment.insert(0, start)

    _increment = np.array(_increment, dtype=int)
    _flag = np.cumsum(_increment, 0)

    starts: np.ndarray = _flag[:-1]
    ends: np.ndarray = _flag[1:] - 1

    index = list(range(2, 13))
    index.extend([0, 1, 13, 14])

    # index = list(range())
    vindex = np.array(sorted(index, key=lambda x: index[x]), dtype=int)
    key = list(range(0, 13)) + [0]
    for ind in range(len(starts)):
        if starts[ind] <= label <= ends[ind]:
            act_arr[
                vindex[
                    key[(label - starts[ind]).item(): (label - starts[ind] + _n_card[ind]).item()]
                ]
            ] += 1
    return act_arr


def if_combo_2_3_32(start, label, n):
    act_arr = np.zeros((15,), dtype=int)
    _increment = list(range(13, 2, -1))
    _n_card = list(range(2, 14, 1))
    _increment.append(1)
    _increment.insert(0, start)
    _flag = np.cumsum(np.array(_increment), 0)

    starts: np.ndarray = _flag[:-1]
    ends: np.ndarray = _flag[1:] - 1

    index = list(range(2, 13))
    index.extend([0, 1, 13, 14])
    vindex = np.array(sorted(index, key=lambda x: index[x]), dtype=int)

    key = list(range(0, 13)) + [0]
    for ind in range(len(starts)):
        if starts[ind] <= label <= ends[ind]:
            act_arr[vindex[key[label - starts[ind]: label - starts[ind] + _n_card[ind]]]] += n   # 猪脑过载了
            break

    return act_arr


def if_combo_bomb(start, label):
    act_arr = np.zeros((15,), dtype=int)
    _n_card = list(range(4, 9, 1))

    end = start + 13 * 5
    starts = np.arange(start, end, 13)
    ends = starts + 12
    n = (label - start) // 13 + 4  # 361 start label of bomb

    for ind in range(len(starts)):
        if starts[ind] <= label <= ends[ind]:
            act_arr[label - starts[ind]] += n
    return act_arr


def if_combo_510k():
    act_arr = np.zeros((15,), dtype=int)
    act_arr[[I5, I10, IK]] += 1
    return act_arr


def if_combo_concentric(label, start=411, special=464):
    act_arr = np.zeros((15,), dtype=int)
    n = (label - start) // 4
    if label == special:
        n += 1
    act_arr[n] = 3
    return act_arr


def label2array() -> Dict[int, np.ndarray]:
    """
    Get a dict that map label to 1d card array.
    :return: Dict[int, nd.ndarray]
    """
    _label2array = dict(
        zip(
            range(465),
            [_get_value(_label) for _label in range(465)]
        ))
    return _label2array


LABEL2ARRAY = label2array()


def get_array2label_without_pair():
    #
    with_pairs = set(range(44, 59))
    with_pairs.update(set(range(271, 341)))

    ftk = set(range(406, 411))
    concentric = set(range(411, 465))

    _set = with_pairs | ftk | concentric

    array2label_1, array2label_2 = {}, {}
    for label, array in LABEL2ARRAY.items():
        if label not in _set:
            array2label_1[tuple(array)] = label

    for label in with_pairs:
        array2label_2[tuple(LABEL2ARRAY[label])] = label

    return array2label_1, array2label_2


ARRAY2LABEL_WITHOUT_PAIR, ARRAY2LABEL_WITH_PAIR = get_array2label_without_pair()


def is_ftk(action_str):
    action = action_str[1::2]
    flower = action_str[::2]
    if len(action) != 3:
        return False, None

    if len(action) == 3:
        assert len(flower) == 3
    if_ftk = '5' in action and 'X' in action and 'K' in action
    if not if_ftk:
        return False, None
    pure_ftk = flower[0] == flower[1] == flower[2]

    if if_ftk and not pure_ftk:
        return True, 406
    else:
        label = 407 + SSTX_CONST.color_idx[flower[0]]
        return True, label


def is_concentric(action_str):
    action = action_str[1::2]
    flower = action_str[::2]
    if not len(action) == 3:
        return False, None

    if_concentric = (action[0] == action[1] == action[2] and flower[0] == flower[1] == flower[2])
    if not if_concentric:
        return False, None
    else:
        if SSTX_CONST.number_idx[action[0]] == 13:
            return True, 463
        elif SSTX_CONST.number_idx[action[0]] == 14:
            return True, 464
        else:
            return True, 411 + 4 * SSTX_CONST.number_idx[action[0]] + SSTX_CONST.color_idx[flower[0]]


def main2label(main_action, sub_action):
    if not main_action:
        return 0
    # main_action, sub_action = parse_action(action)
    if not sub_action:
        label = -1
        if_ftk, label = is_ftk(main_action)
        if not if_ftk:
            if_concentric, label = is_concentric(main_action)
            if not if_concentric:
                key = tuple(str2arr(main_action).sum(1))
                label = ARRAY2LABEL_WITHOUT_PAIR.get(key, -1)

        assert label != -1, "Lable Error!"
    else:
        key = tuple(str2arr(main_action).sum(1))
        label = ARRAY2LABEL_WITH_PAIR[key]

    return label


def sub2label(sub_action):
    label = []
    for i in range(0, len(sub_action), 4):
        _card = sub_action[i: i+4]
        action = _card[1::2]

        assert (len(action) == 2) and action[0] == action[1]
        label.append(SSTX_CONST.number_idx[action[0]])
    return label


def parse_action(action: str):
    flowers = action[0::2]
    numbers = action[1::2]

    counter = Counter(numbers)
    if len(counter) == 1:
        return action, None

    twos = []
    for number in counter:
        while True:
            if counter[number] in [2, 4, 5, 6, 7, 8, 9, 11]:
                counter[number] -= 2
                twos.append(number)
            else:
                break
    threes = sum(counter.values()) // 3

    filters = set()
    if threes != len(twos):
        return action, None
    else:
        for two in twos:
            target = 2
            for i, number in enumerate(numbers):
                if number == two and i not in filters:
                    filters.add(i)
                    target -= 1
                if target == 0:
                    break
        action = ""
        sub_action = ""
        for i, (f, n) in enumerate(zip(flowers, numbers)):
            if i not in filters:
                action += f + n
            else:
                sub_action += f + n
        return action, sub_action

