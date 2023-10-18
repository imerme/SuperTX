

COLOR = 'DCHS'

COLOR_IDX = dict([(f, i) for i, f in enumerate(COLOR)])

NUMBER = '3456789XJQKA2LB'

NUMBER_IDX = dict([(n, i) for i, n in enumerate(NUMBER)])


F = NUMBER_IDX['5']

X = NUMBER_IDX['X']

K = NUMBER_IDX['K']


teammate = {
    0: 2, 1: 3, 2: 0, 3: 1
}


opponent = {
    0: [1, 3], 1: [0, 2],
    2: [1, 3], 3: [0, 2]
}