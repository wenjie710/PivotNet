import numpy as np

def seq_matching_dist_parallel(cost, gt_lens, coe_endpts=0):
    # Time complexity: O(m*n)
    bs, m, n = cost.shape
    assert m <= n
    min_cost = np.ones((bs, m, n)) * np.inf  
    mem_sort_value = np.ones((bs, m, n)) * np.inf  # v[i][j] = np.min(min_cost[i][:j+1])

    # initialization
    for j in range(0, n):
        if j == 0:
            min_cost[:, 0, j] = cost[:, 0, j] 
        mem_sort_value[:, 0, j] = min_cost[:, 0, 0]
        
    for i in range(1, m):
        for j in range(i, n):
            min_cost[:, i, j] = mem_sort_value[:, i-1, j-1] + cost[:, i, j]
            indexes = (min_cost[:, i, j] < mem_sort_value[:, i, j-1])
            indexes_inv = np.array(1-indexes, dtype=np.bool)
            mem_sort_value[indexes, i, j] = min_cost[indexes, i, j]
            mem_sort_value[indexes_inv, i, j] = mem_sort_value[indexes_inv, i, j-1]

    indexes = []
    for i, ll in enumerate(gt_lens):
        indexes.append([i, ll-1, n-1])
    indexes = np.array(indexes)
    xs, ys, zs = indexes[:, 0], indexes[:, 1], indexes[:, 2]
    res_cost = min_cost[xs, ys, zs] + (cost[xs, 0, 0] + cost[xs, ys, zs]) * coe_endpts
    return  res_cost / (indexes[:, 1]+1+coe_endpts*2)

def pivot_dynamic_matching(cost: np.array):
    # Time complexity: O(m*n)
    m, n = cost.shape
    assert m <= n

    min_cost = np.ones((m, n)) * np.inf  
    mem_sort_value = np.ones((m, n)) * np.inf  
    match_res1 = [[] for _ in range(n)]   
    match_res2 = [[] for _ in range(n)]   

    # initialization
    for j in range(0, n-m+1):
        match_res1[j] = [0]
        mem_sort_value[0][j] = cost[0][0]
        if j == 0:
            min_cost[0][j] = cost[0][0]
            
    for i in range(1, m):
        for j in range(i, n-m + i+1):
            min_cost[i][j] = mem_sort_value[i-1][j-1] + cost[i][j]
            if min_cost[i][j] < mem_sort_value[i][j-1]: 
                mem_sort_value[i][j] = min_cost[i][j]
                if i < m-1: 
                    match_res2[j] = match_res1[j-1] + [j]  
            else:
                mem_sort_value[i][j] = mem_sort_value[i][j-1]
                if i < m -1:
                    match_res2[j] = match_res2[j-1]
        if i < m-1:
            match_res1, match_res2 = match_res2.copy(), [[] for _ in range(n)] 

    total_cost =  min_cost[-1][-1]
    final_match_res = match_res1[-2] + [n-1]
    return total_cost, final_match_res