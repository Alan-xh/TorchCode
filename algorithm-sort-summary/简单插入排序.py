def insert_sort(ilist):
    for i in range(len(ilist)):
        for j in range(i):
            if ilist[i] < ilist[j]:
                ilist.insert(j, ilist.pop(i)) # 将 i 位置的数字插进 j 位置
                break
    return ilist

if __name__ == '__main__':
    ilist = insert_sort([4, 5, 6, 7, 3, 2, 6, 9, 8])
    print(ilist)