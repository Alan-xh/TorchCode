''' 右侧作为基准位置，从左向右找第一个小于基准的值与基准进行交换， 时间复杂度平均 O(n log n)，最坏 O(n²) '''

def quicksort_inplace(arr, low, high):
    """ 原地快速排序，直接修改原列表 """
    if low < high:
        # 分区操作，返回基准元素的最终位置
        pi = partition(arr, low, high)
        # 递归排序左半部分和右半部分
        quicksort_inplace(arr, low, pi - 1)
        quicksort_inplace(arr, pi + 1, high)


def partition(arr, low, high):
    """ 使用 Lomuto 分区方案，选择最后一个元素作为基准 """
    pivot = arr[high]  # 基准元素
    i = low - 1        # 较小元素的索引
    
    for j in range(low, high):
        # 如果当前元素小于或等于基准
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]  # 交换
    
    # 将基准元素放到正确位置
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


# 使用示例
arr = [10, 7, 8, 9, 1, 5]
quicksort_inplace(arr, 0, len(arr) - 1)
print(arr)  # 输出: [1, 5, 7, 8, 9, 10]