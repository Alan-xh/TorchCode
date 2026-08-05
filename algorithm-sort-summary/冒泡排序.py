''' 时间复杂度 n^2  '''

def bubble_sort_v1(blist):
    length = len(blist)
    for i in range(length):
        for j in range(1, length - 1):
            if blist[j - 1] > blist[j]:
                blist[j - 1], blist[j] = blist[j], blist[j - 1]
    return blist