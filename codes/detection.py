import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time


# in  : train
# out : increased data ( original + edge)
def get_edge(train_x, train_y):
    """훈련 데이터를 넣으면 훈련데이터 + 에지를 반환하는 함수"""
    ref  = [cv2.cvtColor(train_x[i], cv2.COLOR_GRAY2BGR) for i, v in enumerate(train_x)]
    edge = [cv2.Canny(ref[i], 50, 100) for i, v in enumerate(ref)]

    new_train_x = np.concatenate([train_x, edge], 0)
    new_train_y = np.vstack([train_y, train_y]).flatten()
    return new_train_x, new_train_y


# in : RGB src image
# out: edge image
def get_canny_edge(img):    
    """get_contours의 helper function (get_edge 함수와 헷갈리지 말 것)"""
    ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    edge = cv2.Canny(ref, 50, 100)

    return edge

def get_contours(edge, imgContour):
    """contour를 얻는 함수"""
    contours, hierarchy =cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#    contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                        #2번째 매개변수설명         contour 추출 모드 (EXTERNAL:바깥쪽만, TREE:전부,
                        #     LIST : 계층 구조 상관x 추출,
                        #3번째 매개변수설명         contour 근사 방법 (SIMPLE: 꼭짓점 4개만, NONE: 모든 점)
    min_area = 150 # 200
    max_area = edge.shape[0] * edge.shape[1] / 8

    ptsCandidate = []
    approxCandidate = []

    for cnt in contours:
        if len(cnt) == 0: continue
        area = cv2.contourArea(cnt) #   윤곽선의 면적
        # print(area)
        if  area < min_area: continue  #최소 크기 제약조건
        cv2.drawContours(imgContour, cnt, -1, (255,0,0), 3)
        peri = cv2.arcLength(cnt, True)     #윤곽선의 길이
        epsilon = 0.02 * peri
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)
        if w*h > max_area: continue     #최대 크기 제약조건

        ptsCandidate.append([x,y,x+w,y+h])
        approxCandidate.append(approx)
    return ptsCandidate, approxCandidate


# 이미지가 많이 그려진 그림을 넣으면 분리한 뒤 28x28로 작아진 그림을 출력하는 함수
# in  : 파일 경로, 그린 그림 개수
# out : localize된 28 x 28로 잘린 이미지들
def cluster_localize(path, nb_imgs):
    """cluster 방식을 사용하여 이미지를 localize한다."""
    src = cv2.imread(path)

    # Get Contour
    # edges = get_morphological_edge(src)       #   엣지를 얻는다
    edges = get_canny_edge(src)       #   엣지를 얻는다
    imgContour = src.copy()                 #   contour를 얻고
    # #   퀴즈 후보 & 퀴즈
    ptsCandidate, _ = get_contours(edges, imgContour)

    for i in range(len(ptsCandidate)):
        x1, y1, x2, y2 = ptsCandidate[i]

    columns = ["x1", "y1", "x2", "y2"]
    points_df = pd.DataFrame(ptsCandidate, None, columns)

    points_df['cx'] = (points_df['x1'] + points_df['x2'])//2
    points_df['cy'] = (points_df['y1'] + points_df['y2'])//2

    points_df['group'] = 0
    
    # Gaussian Mixture
    x = points_df['cx'].values
    y = points_df['cy'].values

    xy = np.array(list(zip(x, y)))

    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(n_components = nb_imgs, n_init=10, random_state=42)
    gm.fit(xy)

    points_df['group'] = gm.predict(xy)

    
    # 가장 큰 직사각형은 무엇이야?
    columns = ['min_x', 'min_y', 'max_x', 'max_y']

    group_points_df = pd.DataFrame(None, None, columns) # 행이 group이다.

    group_points_df['min_x'] = [min(points_df[points_df['group'] == group_i]['x1']) for group_i in range(nb_imgs)]
    group_points_df['min_y'] = [min(points_df[points_df['group'] == group_i]['y1']) for group_i in range(nb_imgs)]
    group_points_df['max_x'] = [max(points_df[points_df['group'] == group_i]['x2']) for group_i in range(nb_imgs)]
    group_points_df['max_y'] = [max(points_df[points_df['group'] == group_i]['y2']) for group_i in range(nb_imgs)]


    # margin을 직사각형 길이의 몇 %로 줄 것인지
    # 만약 scale이 0.1이면 총 10%늘어남 (왼쪽으로 5%, 오른쪽으로 5% 늘어남)
    scale = 0.1


    w = group_points_df['max_x'] - group_points_df['min_x']
    h = group_points_df['max_y'] - group_points_df['min_y']


    group_points_df['min_x'] -= w * (scale/2)
    group_points_df['max_x'] += w * (scale/2)

    group_points_df['min_y'] -= h * (scale/2)
    group_points_df['max_y'] += h * (scale/2)

    group_points_df = group_points_df.astype(int)

    # weight와 hegiht의 크기를 큰 쪽으로 같게 맞춰준다.
    ref = np.zeros((src.shape[0]*2, src.shape[1]*2), dtype=np.uint8)
    ref[:,:] = 255
    ref = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)

    W = src.shape[1]
    H = src.shape[0]

    ref[H//2:H//2+src.shape[0], W//2:W//2+src.shape[1]] = src

    # 입력 이미지는 정사각형을 넣어야한다.
    w = group_points_df['max_x'] - group_points_df['min_x']
    h = group_points_df['max_y'] - group_points_df['min_y']

    greater = np.max([w, h], 0)

    cx = group_points_df['min_x'] + w // 2
    cy = group_points_df['min_y'] + h // 2

    group_points_df['min_x'] = cx - greater//2 + W//2
    group_points_df['max_x'] = cx + greater//2 + W//2
    group_points_df['min_y'] = cy - greater//2 + H//2
    group_points_df['max_y'] = cy + greater//2 + H//2

    # 입력 이미지는 28 x 28 이다.
    imgs = [ref[group_points_df['min_y'][i]:group_points_df['max_y'][i], 
            group_points_df['min_x'][i]:group_points_df['max_x'][i]] for i in range(nb_imgs)]

    output_imgs = []
    for i in range(nb_imgs):
        output_img = cv2.resize(imgs[i], (28, 28), interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
        gray[:,:] = 255 - gray
        output_imgs.append(gray)
    
    return np.array(output_imgs)


def union(a,b):
    """두 시각형을 합쳐서 반환"""
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return (x, y, w, h)

def intersection(a,b):
    """두 사각형이 안겹치면 0반환 겹치면 합쳐서 반환"""
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0: return 0
    return (x, y, w, h)
   
def absorption_localize(file, num):
    """absorption방식으로 겹치는 이미지를 하나의 rect으로 만든다."""
    import cv2
    import numpy as np
    import pandas as pd
    
    # 이미지 읽어오기
    img = cv2.imread(file)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_th = cv2.threshold(imgray, 127, 250, cv2.THRESH_BINARY_INV)[1]
    
    # contours 찾기
    contours, hierachy = cv2.findContours(img_th, cv2.RETR_LIST, 
                                     cv2.CHAIN_APPROX_NONE)
    rect = []
    # 면적이 1보다는 크면 rect만든 후 리스트에 담는다.
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1:
            x,y,w,h = cv2.boundingRect(cnt)
            rect.append([x,y,w,h])
    
    # 면적 순으로 정렬
    df = pd.DataFrame(rect, columns=["x", "y", "w", "z"])
    df["area"] = [w*z for (x,y,w,z) in rect]
    df.sort_values(by='area', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    del rect, hierachy

    rect_list = []

    # 각 그림마다 5번 반복하면서 합쳐준다.
    try:
        for h in range(num):
            for i in range(5):
                if i == 0:
                    a = df.iloc[0, :-1].values
                for j in range(df.shape[0]-1):
                    b = df.loc[j+1, "x":"z"].values
                    if intersection(a, b) != 0:
                        a = union(a, b)
                        df.drop(j+1, inplace=True)
                df.reset_index(drop=True, inplace=True)
                if i == 4:
                    rect_list.append(a)
                    df.drop(0, inplace=True)
                    df.reset_index(drop=True, inplace=True)
    except IndexError:
        # 너무 가깝게 그리면 예외처리
        print("띄어서 그려 주세요.")

    # 찾은 구역마다 마진 10만큼 추가
    margin_pixel = 10
    seg_img = []
    for x,y,w,z in rect_list:
        cropped = img.copy()[y - margin_pixel:y + z + margin_pixel, x - margin_pixel:x + w + margin_pixel] 
        seg_img.append(cropped)

    # 이미지를 모두 gray로 만든 후 weight와 hegiht의 크기를 큰 쪽으로 같게 맞춰준다.
    re_seg_img = []
    for i in range(len(seg_img)):
        gray = cv2.cvtColor(seg_img[i], cv2.COLOR_BGR2GRAY)
        gray = 255 - gray

        W = gray.shape[1]
        H = gray.shape[0]
        gray = gray[: gray.shape[0] //2 *2 , : gray.shape[1] //2 *2 ]
        ref = np.zeros((np.max([gray.shape[0], gray.shape[1]]),
                        np.max([gray.shape[0], gray.shape[1]])), dtype=np.uint8)
        ref[ref.shape[0]//2 - H//2:ref.shape[0]//2 + H//2, ref.shape[1]//2 - W//2:ref.shape[1]//2 + W//2] = gray
        ref = cv2.resize(ref, (28, 28), interpolation = cv2.INTER_AREA)
        re_seg_img.append(ref)
    
    return np.array(re_seg_img)

def Img_localize(path, num, mothod="cluster"):
    """Img localize한다. 입력인자로 경로, 수, 방법(default cluster)을 받아 
        loacalize된 이미지를 반환한다.."""
    if mothod=="cluster": return cluster_localize(path, num)
    else: return absorption_localize(path, num)