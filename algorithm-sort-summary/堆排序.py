
def heapify(arr, n, i):
    """
    将以i为根的子树调整为最大堆
    n: 堆的大小
    i: 当前节点索引
    """
    largest = i          # 假设当前节点最大
    left = 2 * i + 1     # 左子节点
    right = 2 * i + 2    # 右子节点
    
    # 如果左子节点存在且大于根节点
    if left < n and arr[left] > arr[largest]:
        largest = left
    
    # 如果右子节点存在且大于当前最大值
    if right < n and arr[right] > arr[largest]:
        largest = right
    
    # 如果最大值不是根节点，交换并继续下沉
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)  # 递归调整受影响的子树


def heap_sort(arr):
    """
    堆排序主函数（升序排列）
    时间复杂度: O(n log n)
    空间复杂度: O(1) 原地排序
    """
    n = len(arr)
    
    # 第一步：构建最大堆（从最后一个非叶子节点开始）
    # 最后一个非叶子节点索引 = n//2 - 1
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    
    # 第二步：一个个取出堆顶元素（最大值）
    for i in range(n - 1, 0, -1):
        # 将堆顶（最大值）交换到末尾
        arr[i], arr[0] = arr[0], arr[i]
        # 对剩余的堆（大小减1）重新调整
        heapify(arr, i, 0)
    
    return arr


# 测试
if __name__ == "__main__":
    test_arr = [12, 11, 13, 5, 6, 7]
    print("原数组:", test_arr)
    heap_sort(test_arr)
    print("排序后:", test_arr)
    # 输出: [5, 6, 7, 11, 12, 13]