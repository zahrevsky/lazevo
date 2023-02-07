# Encoded coordinates: 0 is for x, 1 is for y, 2 is for z
# Helpful for iterating over list of triplets (x, y, z)
#TODO: move to dataclasses
XYZ = [0, 1, 2]


def dist_squared(begin, end) -> float:
    return sum([
        (end[q] - begin[q]) ** 2
        for q in XYZ
    ])