# %%
import sys
import numpy as np
import matplotlib.pyplot as plt
# import pygame as pg
from scipy.spatial import Delaunay
import time
# from HalfEdge import *
from geompreds import orient2d, incircle
import gmsh
import sys
from bisect import bisect

# %%
class TriangularMesh:
    def __init__(self, vertices=None, faces=None, halfedges=None):
        self.vertices = [] if vertices == None else vertices
        self.faces = [] if faces == None else faces
        self.halfedges = [] if halfedges == None else halfedges
        self.version = int(sys.version.split('.')[1]) >= 10

    def sort_vertices(self):
        self.vertices.sort(key=lambda x: x.Hilbert)

    def add_triangle(self, vertices):
        halfedges = [Halfedge() for i in range(3)]
        face = Face(halfedges[0])
        for i in range(3):
            halfedges[i].next = halfedges[i-2]
            halfedges[i].prev = halfedges[i-1]
            halfedges[i].vertex = vertices[i]
            halfedges[i].facet = face
            if not vertices[i].halfedge: vertices[i].halfedge = halfedges[i]
            print(i, vertices[i].halfedge.vertex.coords)
        self.halfedges += halfedges
        return halfedges
    
    def add_Hvertex(self, coords, halfedge, Hilbert, index=None):
        if self.version:
            ind = bisect(self.vertices, Vertex(Hilbert=Hilbert), key=lambda x: x.Hilbert)
            self.vertices.insert(ind, Vertex(coords, halfedge, Hilbert, index))
        else:
            self.vertices.append(Vertex(coords, halfedge, Hilbert, index))
            self.vertices.sort(key=lambda x: x.Hilbert)
            ind = self.vertices.index(Vertex(coords, halfedge, Hilbert, index))
        return ind
    
    def add_Hvertices(self, vertices):
        self.vertices += vertices
        self.vertices.sort(key=lambda x: x.Hilbert)
        return list(filter(lambda i: self.vertices[i] in vertices, range(len(self.vertices))))
    
    def add_mesh(self, mesh):
        self.vertices += mesh.vertices
        self.vertices.sort(key=lambda x: x.Hilbert)
        self.faces += mesh.faces
        self.halfedges += mesh.halfedges

    def rem_mesh(self, mesh):
        self.vertices = list(filter(lambda v: v not in mesh.vertices, self.vertices))
        self.faces = list(filter(lambda f: f not in mesh.faces, self.faces))
        self.halfedges = list(filter(lambda h: h not in mesh.halfedges, self.halfedges))

class Vertex:
    def __init__(self, coords=None, halfedge=None, Hilbert=None, next=None, prev=None, index=None):
        self.coords = coords
        self.halfedge = halfedge
        self.Hilbert = Hilbert
        self.next = next
        self.prev = prev
        self.index = index

class Face:
    def __init__(self, halfedge=None, visited=False, index=None):
        self.halfedge = halfedge # halfedge going ccw around this facet
        self.visited = visited
        self.index = index

class Halfedge:
    def __init__(self, next=None, prev=None, twin=None, vertex=None, facet=None, index=None):
        self.next = next
        self.prev = prev
        self.twin = twin
        self.vertex = vertex
        self.facet = facet
        self.index = index

# %%
input_file = 'pts.dat' if '-i' not in sys.argv else sys.argv[sys.argv.index('-i') + 1]
output_file = 'triangles.dat' if '-o' not in sys.argv else sys.argv[sys.argv.index('-o') + 1]

# %%
points = np.random.rand(2, 10).T if input_file == '' else np.loadtxt(input_file, skiprows=1)
N = len(points)
border = [min(points[:, 0]), max(points[:, 0]), min(points[:, 1]), max(points[:, 1])]

# %%
# expected result
start = time.time()
tri = Delaunay(points)
print('Time:', time.time() - start)
plt.triplot(points[:,0], points[:,1], tri.simplices, 'r-', linewidth=0.5)
plt.plot(points[:,0], points[:,1], 'ko', markersize=2)
plt.show()

# %%
def Hilbert(point, depth, O, R, B):
    point, O, R, B = map(np.array, [point, O, R, B])
    index = [0] * depth
    for i in range(depth):
        dotR = np.dot(point-O, R) > 0.
        dotB = np.dot(point-O, B) > 0.
        R /= 2.
        B /= 2.
        if dotR:
            if dotB:
                index[i] = 2
                O += R + B
            else:
                index[i] = 3
                O += R - B
                R, B = B, R
        else:
            if dotB:
                index[i] = 1
                O -= R - B
            else:
                index[i] = 0
                O -= R + B
                R, B = B, R
    return index

# %%
a = (0.79,0.15)
res = Hilbert(a, 8, (0.5, 0.5), (0.5, 0), (0, 0.5))
print(res)

# %%
def Ball(interior, boundary, p):
    # connect p to all vertices of the cavity
    prev_edge1 = None
    for edge in boundary.halfedges:
        print(edge.vertex.coords)
        edge1, edge2, face = Halfedge(), Halfedge(), Face()
        edge1.vertex = p
        edge1.next = edge
        edge1.twin = edge2
        edge1.facet = face
        edge2.vertex = edge.vertex
        edge2.prev = edge.prev
        edge2.twin = edge1
        edge2.facet = face
        edge2.next = prev_edge1
        prev_edge1 = edge1
        face.halfedge = edge
        boundary.halfedges += [edge1, edge2]
        boundary.faces.append(face)
        boundary.halfedges[1].next = boundary.halfedges[-2]
    return boundary

# %%
# a, b, c = (2., 3.), (np.inf, -np.inf), (np.inf, np.inf)
# sum(map(lambda x: (np.array(x)>0) * np.isinf(x), [a, b, c]))
# tuple(map(lambda x: (np.array(x)>0) * np.isinf(x), [a, b, c]))
# tuple(sum(x) for x in zip(a, b, c))
# %timeit next(filter(lambda x: not np.isinf(x[0]), (a,b,c)))

# %%
from collections import deque
def Cavity(mesh, p, ind):
    visited = []
    # find a triangle containing p
    def check_triangle(halfedge):
        # print(halfedge.vertex.coords)
        # print(halfedge.next.vertex.coords)
        # print(halfedge.prev.vertex.coords)
        a, b, c = halfedge.vertex.coords, halfedge.next.vertex.coords, halfedge.prev.vertex.coords
        res = incircle(a, b, c, p.coords)
        if np.isnan(res):
            # number of None in [a, b, c]
            k = [halfedge.twin, halfedge.next.twin, halfedge.prev.twin].count(None)
            if k == 0:
                # do orient2d test instead
                return orient2d(a, b, p.coords) >= 0 or orient2d(b, c, p.coords) >= 0 or orient2d(c, a, p.coords) >= 0
            elif k == 1:
                # find index of non infinite point in [a, b, c]
                d = a if not np.isinf(a[0]) else b if not np.isinf(b[0]) else c
                quadrant = tuple(sum(x) for x in zip(a, b, c))
                if np.isnan(quadrant[1]):
                    return p.coords[0] >= d[0] if quadrant[0] > 0 else p.coords[0] <= d[0]
                else:
                    return p.coords[1] >= d[1] if quadrant[1] > 0 else p.coords[1] <= d[1]
            else:
                print('Error: 3 infinite points in triangle')
                print(a, b, c)

        return res >= 0

    def check_vertex(vertex):
        halfedge = vertex.halfedge
        if not halfedge: return None
        while True:
            if check_triangle(halfedge):
                return halfedge
            halfedge = halfedge.twin.next # clockwise
            if halfedge == vertex.halfedge or halfedge == None:
                break
        return None
    
    def expand(init_hedge):
        interior = TriangularMesh(faces=[init_hedge.facet])
        boundary = TriangularMesh()

        stack = deque([init_hedge, init_hedge.next, init_hedge.prev])
        while stack:
            hedge = stack.popleft()
            # if hedge.facet.visited:
            #     continue
            if hedge.twin == None or not check_triangle(hedge.twin):
                boundary.halfedges.append(hedge)
                # boundary.vertices.append(hedge.vertex)
                continue
            hedge.twin.facet.visited = True
            visited.append(hedge.twin.facet)
            interior.halfedges += [hedge.twin, hedge]
            interior.faces.append(hedge.twin.facet)

            for e in [hedge.twin.next, hedge.twin.prev]:
                stack.append(e)
        return interior, boundary
    
    i = 2
    while True:
        u = mesh.vertices[ind + (-1)**i * i // 2]
        init_hedge = check_vertex(u)
        if init_hedge:
            break
        i += 1
    
    print(init_hedge.vertex.coords, init_hedge.next.vertex.coords, init_hedge.prev.vertex.coords)

    return expand(init_hedge)


# %%
depth = 8
O, R, B = (0.5, 0.5), (0.5, 0), (0, 0.5)
mesh = TriangularMesh()
for p in points:
    mesh.vertices.append(Vertex(tuple(p.tolist()), Hilbert=Hilbert(p, depth, O, R, B)))
for p in mesh.vertices:
    print(p.coords, p.Hilbert)

# %%
# add points at +-infinity
infite_pts = [(np.inf, np.inf), (-np.inf, np.inf), (-np.inf, -np.inf), (np.inf, -np.inf)] # keep order!
mesh.vertices += list(map(lambda x: Vertex(x, Hilbert=[-1]*depth), infite_pts))
mesh.sort_vertices()
for p in mesh.vertices:
    print(p.coords, p.Hilbert)

# %%
# add the 4 initial triangles with infinite points and point p[0]
p = mesh.vertices[4]
for i in range(4):
    mesh.add_triangle([mesh.vertices[i], mesh.vertices[(i+1)%4], mesh.vertices[4]])
# stick edges together
for i in range(4):
    mesh.halfedges[3*i + 1].twin = mesh.halfedges[(3*i + 5) % 12]
    mesh.halfedges[(3*i + 5) % 12].twin = mesh.halfedges[3*i + 1]

# %%
# for he in mesh.halfedges:
    # print(he.vertex.coords, he.next.vertex.coords, he.twin.vertex.coords if he.twin else None)

# %%
def draw_mesh(mesh):
    for p in mesh.vertices:
        plt.plot(*p.coords, 'ko')
    for he in mesh.halfedges:
        # if he.visited:
        #     continue
        a, b = he.vertex.coords, he.next.vertex.coords
        plt.plot(*zip(a, b), 'r-')
        # plt.arrow(a[0], a[1], b[0]-a[0], b[1]-a[1], head_width=0.5, head_length=1, fc='r', ec='k')
    plt.show()

draw_mesh(mesh)


# %%
p = mesh.vertices[5]
interior, boundary = Cavity(mesh, p, 5)
pass