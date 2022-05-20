def partition(A: list, start: int, end: int):
    pivot_index = start
    pivot = A[start]

    while start < end:
        while start < len(A) and A[start] <= pivot:
            start += 1
        while A[end] > pivot:
            end -= 1
        if (start < end):
            A[start], A[end] = A[end], A[start]
    
    A[end], A[pivot_index] = A[pivot_index], A[end]
    return end

def quick_sort(A: list, start: int, end: int):
    if (start<end):
        p = partition(A, start, end)
        
        quick_sort(A, start, p - 1)
        quick_sort(A, p + 1, end)
    return A