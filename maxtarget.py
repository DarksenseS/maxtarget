import numpy as np
from scipy.spatial import distance_matrix
from gurobipy import *
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point
from numpy import random

def generate_s(points,M=100):
    '''
     Вход:
         points:в виде массива Numpy вида (N,2)
         M: количество кандидатов для создания
     Выход:
         sites: в виде массива Numpy вида (M,2)
    '''
    hull = ConvexHull(points)
    polygon_points = points[hull.vertices]
    poly = Polygon(polygon_points)
    min_x, min_y, max_x, max_y = poly.bounds
    sites = []
    while len(sites) < M:
        random_point = Point([random.uniform(min_x, max_x),
                             random.uniform(min_y, max_y)])
        if (random_point.within(poly)):
            sites.append(random_point)
    return np.array([(p.x,p.y) for p in sites])

def model_c(points,K,radius,M):
    '''
      Вход:
         points: входные точки, массив Numpy в форме [N,2]
         K: количество мест для выбора
         radius: радиус круга
         M: количество мест-кандидатов, которые будут сгенерированы случайным образом внутри
         ConvexHull, обернутый полигоном
     Возврат:
         opt_sites: расположение K оптимальных мест, массив Numpy в форме [K,2]
         f: оптимальное значение целевой функции
    '''
    import time
    start = time.time()
    sites = generate_s(points,M)
    A = sites.shape[0]
    I = points.shape[0]
    D = distance_matrix(points,sites)
    mask1 = D<=radius
    D[mask1]=1
    D[~mask1]=0

    # строим модель

    m = Model()

    # Добавляем переменные

    x = {}
    y = {}
    for i in range(I):
      y[i] = m.addVar(vtype=GRB.BINARY, name="y%d" % i)
    for a in range(A):
      x[a] = m.addVar(vtype=GRB.BINARY, name="x%d" % j)

    m.update()

    # Добавляем константы

    m.addConstr(quicksum(x[a] for a in range(A)) == K)

    for i in range(I):
        m.addConstr(quicksum(x[a] for a in np.where(D[i]==1)[0]) >= y[i])

    m.setObjective(quicksum(y[i]for i in range(I)),GRB.MAXIMIZE)
    m.setParam('OutputFlag', 0)
    m.optimize()

    
    solution = []
    if m.status == GRB.Status.OPTIMAL:
        for v in m.getVars():
            # print v.varName,v.x
            if v.x==1 and v.varName[0]=="x":
               solution.append(int(v.varName[1:]))
    opt_sites = sites[solution]
    return opt_sites,m.objVal


def plot_r(points,opt_sites,radius):
    '''
      Вход:
         points: входные точки, массив Numpy в форме [N,2]
         opt_sites: расположение K оптимальных мест, массив Numpy в форме [K,2]
         radius: радиус круга
    '''
    from matplotlib import pyplot as plt
    fig = plt.figure("Result", figsize=(10,10))
    plt.scatter(points[:,0],points[:,1],c='C9')
    ax = plt.gca()
    plt.scatter(opt_sites[:,0],opt_sites[:,1],c='C3',marker='+')
    for site in opt_sites:
        circle = plt.Circle(site, radius, color='C3',fill=False,lw=2)
        ax.add_artist(circle)
    ax.axis('equal')
    ax.tick_params(axis='both',left=False, top=False, right=False,
                       bottom=False, labelleft=False, labeltop=False,
                       labelright=False, labelbottom=False)
