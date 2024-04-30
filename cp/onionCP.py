# library
import numpy as np
import cv2
import copy
import math
from scipy.ndimage import label
from collections import Counter
from skimage.filters import gaussian
import time
from tqdm import tqdm


# local
from utils import misc


def onion_cp(img_source, img_target, mask_source, mask_target, dataset):
    '''
    = input =
    img : (h, w, 3) shape of np.ndarry
    mask : (h, w) shape of np.ndarry. It contains [0 or 1].

    = output =
    Synthesized sample.
    img : img : (h, w, 3) shape of np.ndarry
    mask : img : (h, w) shape of np.ndarry. It contains [0 or 1].
    '''
    assert len(mask_source.shape) == 2
    assert len(mask_target.shape) == 2
    assert isinstance(img_source, np.ndarray)
    assert isinstance(img_target, np.ndarray)
    assert isinstance(mask_source, np.ndarray)
    assert isinstance(mask_target, np.ndarray)

    # 0. init
    imsize = img_target.shape[1]

    # 1. peeling onion
    edges_source, onions_source = onion_peeling(mask_source.astype(np.uint8), prune_all=False)
    edges_target, onions_target = onion_peeling(mask_target.astype(np.uint8), prune_all=False)
    return_mask = onions_target[0]
    if dataset in ['kumar', 'cpm17']: # kumar and cpm17 are too chbby
        edges_source, onions_source = edges_source[1:], onions_source[1:]
        edges_target, onions_target = edges_target[1:], onions_target[1:]

    # 2. resize source to fit target's objet size
    img_source_resized, mask_source_resized, edges_source_resized, onions_source_resized, imsize_source = resize_and_onion_peeling(img_source,
                                                                                                                                   edges_source,
                                                                                                                                   edges_target,
                                                                                                                                   onions_source[0],
                                                                                                                                   imsize)
    
    
    # 3. Moore's algorithm
    ## source
    coordinate = get_coordinate(imsize_source,imsize_source)
    coordinate_source_resized = coordinate * mask_source_resized[...,None] # (imsize_source,imsize_source,2)
    centroid_source_resized = np.sum(np.sum(coordinate_source_resized, 0),0) / np.sum(mask_source_resized)
    centroid_x_source_resized, centroid_y_source_resized = int(centroid_source_resized[0]), int(centroid_source_resized[1])
    flatten_source_resized = moore_algorithm(edges_source_resized[0])

    centroid_moved_flatten_source = copy.deepcopy(flatten_source_resized)
    centroid_moved_flatten_source[...,0] -= centroid_x_source_resized
    centroid_moved_flatten_source[...,1] -= centroid_y_source_resized

    ## target
    coordinate = get_coordinate(imsize,imsize)
    pruned_target_edge = prune_edge(edges_target[0])
    flatten_target = moore_algorithm(pruned_target_edge)
    coordinate_target = coordinate * onions_target[0][...,None] # (imsize,imsize,2)
    centroid_target = np.sum(np.sum(coordinate_target, 0),0) / np.sum (onions_target[0])
    centroid_x_target, centroid_y_target = int(centroid_target[0]), int(centroid_target[1])
    
    centroid_moved_flatten_target = copy.deepcopy(flatten_target)
    centroid_moved_flatten_target[...,0] -= centroid_x_target
    centroid_moved_flatten_target[...,1] -= centroid_y_target


    # 4. DTW matching and first edge
    warping_idx, _, _ = dtw(centroid_moved_flatten_source[0:-1], centroid_moved_flatten_target[0:-1], True, pre_matching=2)

    T_1 = {}
    for x,y in flatten_target:
        T_1[f'{x},{y}'] = {'S_coor':[], 'S_represent':None}
    for idx_source, idx_target in warping_idx:
        x_target, y_target = flatten_target[idx_target]
        x_source, y_source = flatten_source_resized[idx_source]
        T_1[f'{x_target},{y_target}']['S_coor'].append([x_source, y_source])
    for key in T_1.keys():
        allocated_S_coors = len(T_1[key]['S_coor'])
        assert allocated_S_coors>=1
        T_1[key]['S_represent'] = T_1[key]['S_coor']
    
    new = np.zeros_like(img_target)
    for key in T_1.keys():
        length = len(T_1[key]['S_coor'])
        x_target, y_target = map(int, key.split(','))
        if length > 1:
            rgb = np.array([[[0,0,0]]])
            for pixels in T_1[key]['S_coor']:
                rgb += img_source_resized[pixels[0], pixels[1]]
            rgb = rgb.astype(np.float64)
            rgb /= length
            new[x_target, y_target, 0:] = rgb
        else:
            x_source, y_source = T_1[key]['S_represent'][0]
            # x_source, y_source = T_1[key]['S_represent']
            new[x_target, y_target, 0:] = img_source_resized[x_source, y_source, 0:]
    new = new.astype(np.uint8)

    
    # 5. From second to last edge
    T_i_minus_1 = copy.deepcopy(T_1)
    edge_source_i_minus_1 = edges_source_resized[0]
    edge_target_i_minus_1 = edges_target[0]

    h_max_t, w_max_t = imsize, imsize
    h_max_s, w_max_s = imsize_source, imsize_source

    for edge_idx in tqdm(range(1,len(edges_source_resized)), desc='from second edge...'):
        edge_source = edges_source_resized[edge_idx]
        edge_target = edges_target[edge_idx]

        edge_idx_target_h, edge_idx_target_w = np.where(edge_target==1)

        T_i = {}
        for idx in range(len(edge_idx_target_h)):
            T_i[f'{edge_idx_target_h[idx]},{edge_idx_target_w[idx]}'] = {'S_coor':[], 'S_represent':[]}
        
        no_represent_key_list = []
        for idx, key in enumerate(T_i.keys()):
            h_target, w_target = map(int, key.split(','))
            mask_target = np.zeros_like(edge_target)
            mask_target[max(0,h_target-1):min(h_target+1+1, h_max_t),
                        max(0,w_target-1):min(w_target+1+1, w_max_t)] = 1
            neighbor_t_i = edge_target_i_minus_1 * mask_target
            neighbor_t_i_idx = np.stack(np.where(neighbor_t_i==1),-1) # T_i 상의 어떤 점 t_j에 대해 T_i-1에 있었던 이웃의 좌표 # T_{i-1}로 올라가고
            
            s_i_minus_1_represents = []
            for h_t, w_t in neighbor_t_i_idx: # T_i-1에 있던 좌표에 대해 # S_{i-1}에 대해
                for represent in T_i_minus_1[f'{h_t},{w_t}']['S_represent']: # S_i로 내려온다
                    s_i_minus_1_represents.append(represent)
                    h_source, w_source = represent
                    mask_source = np.zeros_like(edge_source)
                    mask_source[max(0,h_source-1):min(h_source+1+1, h_max_s),
                                max(0,w_source-1):min(w_source+1+1, w_max_s)] = 1
                    neighbor_s_i = edge_source * mask_source
                    neighbor_s_i_idx = np.stack(np.where(neighbor_s_i==1),-1) # S_i 상의 이웃하는 coordinates # (length, 2)
                    T_i[key]['S_coor'] += neighbor_s_i_idx.tolist()

            # 실제 픽셀 값 주기
            if len(T_i[key]['S_coor']) != 0:
                counter = Counter(map(tuple, T_i[key]['S_coor']))
                count_elements = counter.most_common() # [((51, 147), 5), ((52, 147), 5), ... ]
                biggest_num = count_elements[0][1]
                T_i[key]['S_represent'] = [list(item[0]) for item in count_elements if item[1]==biggest_num]
                new[h_target, w_target] = np.sum(np.stack([img_source_resized[item[0], item[1]] for item in T_i[key]['S_represent']], axis=-1), axis=-1)/len(T_i[key]['S_represent'])
            else: # 아래껍질에 짝꿍없는애들
                s_i_minus_1_negihbors = []
                no_represent_key_list.append([h_target, w_target])
                for represent in s_i_minus_1_represents:
                    h_source, w_source = represent
                    mask_source = np.zeros_like(edge_source)
                    mask_source[max(0,h_source-1):min(h_source+1+1, h_max_s),
                                max(0,w_source-1):min(w_source+1+1, w_max_s)] = 1
                    neighbor_s_i_minus_1 = edge_source_i_minus_1 * mask_source
                    neighbor_s_i_minus_1_idx = np.stack(np.where(neighbor_s_i_minus_1==1),-1).tolist()
                    s_i_minus_1_negihbors += neighbor_s_i_minus_1_idx
                    new[h_target, w_target] = np.sum(np.stack([img_source_resized[item[0], item[1]] for item in s_i_minus_1_negihbors], axis=-1), axis=-1)/len(s_i_minus_1_negihbors)
        
        # S_represent없는 key 들에게 할당해주기
        start = time.time()
        while len(no_represent_key_list) != 0:
            if edge_idx+1 == len(edges_source_resized):
                break
            for item in no_represent_key_list:
                h_target, w_target = item
                key = f'{h_target},{w_target}'
                s_represent = []
                negihbors = [f'{int(h_target-1)},{int(w_target-1)}',
                            f'{int(h_target-1)},{int(w_target)}',
                            f'{int(h_target-1)},{int(w_target+1)}',
                            f'{int(h_target)},{int(w_target+1)}',
                            f'{int(h_target)},{int(w_target-1)}',
                            f'{int(h_target+1)},{int(w_target-1)}',
                            f'{int(h_target+1)},{int(w_target)}',
                            f'{int(h_target+1)},{int(w_target+1)}',]
                for neigbor in negihbors:
                    try:
                        s_represent += T_i[neigbor]['S_represent']
                    except:
                        pass

                if len(s_represent) >= 1:
                    T_i[key]['S_represent'] = s_represent
                    no_represent_key_list.remove(item)

            if time.time() - start > 60:
                raise TimeoutError('Timeout by key allocation')

        # variables for next step
        T_i_minus_1 = T_i
        edge_source_i_minus_1 = edges_source_resized[edge_idx]
        edge_target_i_minus_1 = edges_target[edge_idx]

    return new, gaussian(onions_target[0], 0.2), return_mask #gaussian(onions_target[0], 0.2)


def onion_peeling(selected_object, prune_all=False, stop_iter=-1):
    '''
    = input =
    selected_object : mask which contains source object only. np.array. (h,w) shape
    prune_all : pruning for all edges. When it is False; the algorithm prunes first edge only.
    stop_iter : you can set specific nums of onions. If stop_iter = 2, len(onions)==2.

    = output =
    edges : Onion edges from selected_object. (h,w) shape of np.array is in the list.
    onions : Onion that removed edge_i -> repeat.... it is list that containts (h,w) shape of np.ndarray onions.
    '''
    # 초기 선언
    i = 0
    onions = [selected_object]
    edges = []
    pad_size = 1
    laplacianFilter = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])

    # edge detecting
    while True:
        i += 1
        
        # detect edge and pruning. The first edge must be pruned.
        if i == 1:
            edge = np.where(cv2.filter2D(np.pad(onions[-1],(pad_size,pad_size),mode='constant'),-1,laplacianFilter)[pad_size:-pad_size,pad_size:-pad_size]>0,1.0,0)
            edge = prune_edge(edge)
            edges.append(edge)
        elif prune_all:
            edge = np.where(cv2.filter2D(np.pad(onions[-1],(pad_size,pad_size),mode='constant'),-1,laplacianFilter)[pad_size:-pad_size,pad_size:-pad_size]>0,1.0,0)
            edge = prune_edge(edge)
            edges.append(edge)
        else:
            edge = np.where(cv2.filter2D(np.pad(onions[-1],(pad_size,pad_size),mode='constant'),-1,laplacianFilter)[pad_size:-pad_size,pad_size:-pad_size]>0,1.0,0)
            edges.append(edge)
        
        # make onion      
        onions.append(onions[-1] - edge)

        # break
        if np.sum(onions[-1]) == 0:
            break
        elif np.sum(onions[-1]) < 0:
            raise ValueError('Onion peeling error')
        elif stop_iter == i:
            break
    
    return edges, onions



def resize_and_onion_peeling(image_source, edges_source, edges_target, selected_source, imsize):
    '''
    = input =

    = output =

    '''
    # set image ratio
    ratio = len(edges_target)/len(edges_source)
    imsize_source = int(imsize * ratio)
    imsize_source_list = [copy.deepcopy(imsize_source)]

    # image resize
    image_source_resized = cv2.resize(image_source, (imsize_source, imsize_source))
    selected_source_resized = cv2.resize(selected_source.astype(np.float32), (imsize_source, imsize_source))
    selected_source_resized = np.where(selected_source_resized>0,1.0,0)

    # onion peeling
    while True:
        edges_source_resized, onions_source_resized = onion_peeling(selected_source_resized)
        if len(edges_source_resized) == len(edges_target):
            break
        elif len(edges_source_resized) > len(edges_target):
            imsize_source -= 5
            image_source_resized = cv2.resize(image_source, (imsize_source, imsize_source))
            selected_source_resized = cv2.resize(selected_source.astype(np.float32), (imsize_source, imsize_source))
            selected_source_resized = np.where(selected_source_resized>0,1.0,0)
        else:
            imsize_source += 5
            image_source_resized = cv2.resize(image_source, (imsize_source, imsize_source))
            selected_source_resized = cv2.resize(selected_source.astype(np.float32), (imsize_source, imsize_source))
            selected_source_resized = np.where(selected_source_resized>0,1.0,0)

        if imsize_source not in imsize_source_list:
            imsize_source_list.append(imsize_source)
        else:
            raise ValueError('Resize source error')

    return image_source_resized, selected_source_resized, edges_source_resized, onions_source_resized, imsize_source




def prune_edge(edge_mask):
    '''
    It prunes isolated edge.

    = input =
    edge_mask : unpruned np.array (h,w) shape. Mask that have its edge only.
    
    = output =
    pruned_edge : pruned np.array (h,w) shape
    '''
    neighbor_filter = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    pruned_edge = copy.deepcopy(edge_mask)
    start = time.time()
    while True:
        sum_edge = np.sum(pruned_edge)
        pruned_edge -= np.where(cv2.filter2D(pruned_edge,-1,neighbor_filter)==-7.0, 1.0, 0)
        if np.sum(pruned_edge) == sum_edge:
            break
        elif time.time() - start > 30:
            raise TimeoutError('Pruning error')
    return pruned_edge



##############################
#     Moore's Algorithm      #
##############################

def get_coordinate(h,w):
    '''
    (h,w,2) shape의 np.array를 출력
    [h,w]의 성분인 2개의 채널은 
    0번 채널에 h의 coordinate
    1번 채널에 w의 coordinate를 포함
    '''
    coordinate = []
    h_coordinate = []
    w_coordinate = []
    for i in range(h):
        h_coordinate.append([i for _ in range(w)])
    for i in range(w):
        w_coordinate.append([i for _ in range(h)])
    h_coordinate = np.array(h_coordinate)
    w_coordinate = np.array(w_coordinate)
    w_coordinate = w_coordinate.T
    coordinate = np.stack([h_coordinate, w_coordinate], -1)
    return coordinate


def find_nth_smallest(arr, n):
    return int(np.sort(np.unique(arr.flatten()))[n])


def is_in(arr,target):    
    return np.any(np.all(arr == target, axis=-1))


def moore_algorithm(mask):
    '''
    mask : edged mask
    '''
    assert len(mask.shape)==2
    labeled_array, num_features = label(mask, structure=np.ones((3, 3)))
    edge_list = []

    for i in range(1,num_features+1):
        mask = np.where(labeled_array==i, 1, 0)
        edge_list += _moore_algorithm(mask)

    edge_list = np.array(edge_list) - 1
    return edge_list


def _moore_algorithm(mask):
    '''
    mask : edged mask
    '''
    assert len(mask.shape)==2
    h, w = mask.shape
    coordinate = get_coordinate(h,w)
    coordinate += 1
    coordinate_edge = coordinate * mask[...,None]
    h_start = find_nth_smallest(coordinate_edge[...,0],-1) # 가장 아래
    w_start = find_nth_smallest(coordinate_edge[coordinate_edge[...,0]==h_start][...,1],0) # 가장 왼쪽
    mask = np.pad(mask,(1,1))
    
    edge_list = [[h_start, w_start]]
    h_now = copy.deepcopy(h_start)
    w_now = copy.deepcopy(w_start)

    iteration = 0
    stop = True
    while stop:
        iteration += 1
        # next_steps = [[h_now+1, w_now],
        #               [h_now+1, w_now-1],
        #               [h_now, w_now-1],
        #               [h_now-1, w_now-1],
        #               [h_now-1, w_now],
        #               [h_now-1, w_now+1],
        #               [h_now, w_now+1],
        #               [h_now+1, w_now+1]]
        next_steps = [[h_now, w_now-1],
                      [h_now-1, w_now-1],
                      [h_now-1, w_now],
                      [h_now-1, w_now+1],
                      [h_now, w_now+1],
                      [h_now+1, w_now+1],
                      [h_now+1, w_now],
                      [h_now+1, w_now-1]]
        for h_temp, w_temp in next_steps:
            if mask[h_temp,w_temp] == 1.0:
                if [h_temp,w_temp] == [h_start, w_start]:
                    if len(edge_list) > np.sum(mask)-2:
                        stop = False
                        edge_list.append([h_start, w_start])
                        break
                elif [h_temp,w_temp] in edge_list:
                    pass
                else:
                    edge_list.append([h_temp,w_temp])
                    h_now = copy.deepcopy(h_temp)
                    w_now = copy.deepcopy(w_temp)
                    break
        else:
            '''
            뭐던간에 2개 겹친거는 알고리즘을 바보로 만든다.
            '''
            edge_list[-1], edge_list[-2] = edge_list[-2], edge_list[-1]
            h_now, w_now = edge_list[-1]
            # raise NotImplementedError(f'The edge [{h_now-1}, {w_now-1}] seems to be an isolated.')

        if len(edge_list) == np.sum(mask):
            stop = False
            edge_list.append([h_start, w_start])
        
        if iteration > np.sum(mask) + 10:
            raise ValueError("Moore Algorithm error")
        
    return edge_list



##############################
#            DTW             #
##############################


def dtw(coor_a, coor_b, constraints=True, pre_matching=None):
    '''
    = input = 
    coor_a & coor_b : flatten된 edge coordinates. np.array. (n, 2) shape.
                      coor_a는 source, coor_b는 target을 넣는 것을 추천.
    constraints : 두 시퀀스 중 더 긴 쪽은 중복해서 매칭이 불가능하다(True)
    pre_matching : 몇개의 점을 미리 매칭해둘건지

    = output =
    warping_path : 최소 cost 경로 idx. np.array. (n, 2) shape이다.
                   [1, 4]라는 성분이 있다면 coor_a의 1번 idx는 coor_b의 4번 idx와 매칭되어야한다는 의미다.
    cost_matrix : 두 시퀀스의 경로 누적 cost를 저장한 matrix. np.array. (n,m) shape
    self_cost_matrix : 두 시퀀스의 cost matrix. np.array. (n,m) shape
    '''
    if pre_matching != None:
        assert pre_matching > 1
        assert type(pre_matching) == int
        w_a = int((np.min(coor_a[:,1])+np.max(coor_a[:,1]))/2)
        w_b = int((np.min(coor_b[:,1])+np.max(coor_b[:,1]))/2)
        indices_a = np.where(coor_a[:, 1] == w_a)[0][0]
        indices_b = np.where(coor_b[:, 1] == w_b)[0][0]
        _coor_a = np.concatenate((coor_a[indices_a:], coor_a[:indices_a]))
        _coor_b = np.concatenate((coor_b[indices_b:], coor_b[:indices_b]))
        length_a = int(len(coor_a)/pre_matching)
        length_b = int(len(coor_b)/pre_matching)

        warping_idx = []
        for i in range(pre_matching):
            if i+1 == pre_matching:
                temp_a = _coor_a[i*length_a:]
                temp_b = _coor_b[i*length_b:]
            else:
                temp_a = _coor_a[i*length_a:(i+1)*length_a+1]
                temp_b = _coor_b[i*length_b:(i+1)*length_b+1]
            idx, _, _ = dtw(temp_a, temp_b, constraints=constraints, pre_matching=None)

            if i+1 != pre_matching:
                idx = idx[0:-1]
            
            for j in range(len(idx)):
                idx[j,0] += i*length_a + indices_a if idx[j,0] + i*length_a + indices_a < len(coor_a) else (i*length_a + indices_a) - len(coor_a)
                idx[j,1] += i*length_b + indices_b if idx[j,1] + i*length_b + indices_b < len(coor_b) else (i*length_b + indices_b) - len(coor_b)
            
            idx = idx.tolist()
            warping_idx += idx

        return np.array(warping_idx), _, _
    
    else:
        n = max(len(coor_a), len(coor_b))
        m = min(len(coor_a), len(coor_b))
        constraints = False if n==m else constraints
        cost_matrix = np.zeros((n,m))
        self_cost_matrix = np.zeros((n,m))
        warping_idx = []

        # self cost matrix
        if len(coor_a) >= len(coor_b):
            for i in range(n):
                for j in range(m):
                    self_cost_matrix[i,j] = round(math.sqrt((coor_a[i][0] - coor_b[j][0])**2 +
                                                            (coor_a[i][1] - coor_b[j][1])**2),
                                                3)
        else:
            for i in range(n):
                for j in range(m):
                    self_cost_matrix[i,j] = round(math.sqrt((coor_b[i][0] - coor_a[j][0])**2 +
                                                            (coor_b[i][1] - coor_a[j][1])**2),
                                                3)
        
        # relative cost matrix
        for i in range(n):
            for j in range(m):
                if i==0 and j==0:
                    minimum = 0
                elif i == 0:
                    minimum = cost_matrix[i,j-1]
                elif j == 0:
                    minimum = cost_matrix[i-1,j]
                else:
                    minimum = min(cost_matrix[i,j-1], cost_matrix[i-1,j-1], cost_matrix[i-1,j])
                cost_matrix[i,j] = self_cost_matrix[i,j] + minimum

        # dtw
        ## init
        i, j = 0, 0
        warping_idx.append([0,0])
        
        ## to prevent out of idx Error
        temp_cost_matrix = np.full((n+1,m+1), math.inf)
        temp_cost_matrix[0:n, 0:m] = cost_matrix

        ## if constrains, define possible movements
        num_down_dir = n - m

        while True:
            if constraints:
                down = temp_cost_matrix[i+1,j]
                diag = temp_cost_matrix[i+1,j+1]
                if down < diag and num_down_dir>0:
                    warping_idx.append([i+1,j])
                    i += 1
                    num_down_dir -= 1
                else:
                    warping_idx.append([i+1,j+1])
                    i += 1
                    j += 1
            else: 
                down = temp_cost_matrix[i+1,j]
                diag = temp_cost_matrix[i+1,j+1]
                right = temp_cost_matrix[i,j+1]
                minimum = min(down, diag, right)
                if minimum == down:
                    warping_idx.append([i+1, j])
                    i += 1
                elif minimum == diag:
                    warping_idx.append([i+1,j+1])
                    i += 1
                    j += 1
                elif minimum == right:
                    warping_idx.append([i, j+1])
                    j += 1

            
            if i==n-1 and j==m-1:
                break
        
        warping_idx = np.array(warping_idx)
        if len(coor_a) < len(coor_b):
            warping_idx = np.concatenate([warping_idx[...,1][...,None], warping_idx[...,0][...,None]], axis=-1)

        return warping_idx, cost_matrix, self_cost_matrix