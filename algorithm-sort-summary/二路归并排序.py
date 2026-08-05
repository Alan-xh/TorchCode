''' 递归分成两半，然后对两个子数组的进行排序, O(n log n) '''

def merge_sort(arr):
    """
    归并排序（二分归并排序）
    时间复杂度：O(n log n)
    空间复杂度：O(n)
    """
    # 递归终止条件：数组长度小于等于1时已经有序
    if len(arr) <= 1:
        return arr
    
    # 1. 分：将数组从中间分成两个子数组
    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]
    
    # 2. 递归排序左右子数组
    left = merge_sort(left)
    right = merge_sort(right)
    
    # 3. 并：合并两个已排序的子数组
    return merge(left, right)


def merge(left, right):
    """
    合并两个已排序的数组
    """
    result = []
    i = j = 0
    
    # 比较两个数组的元素，将较小的放入结果数组
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # 将剩余元素添加到结果数组
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result


# 测试代码
if __name__ == "__main__":
    arr = [38, 27, 43, 3, 9, 82, 10]
    print("原始数组:", arr)
    sorted_arr = merge_sort(arr)
    print("排序后:", sorted_arr)