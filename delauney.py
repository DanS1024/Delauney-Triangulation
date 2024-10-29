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
from collections import deque

class TriangularMesh:
    def __init__(self, vertices=None, faces=None, halfedges=None):
        self.vertices = [] if vertices == None else vertices
        self.faces = [] if faces == None else faces
        self.halfedges = [] if halfedges == None else halfedges
        self.version = int(sys.version.split('.')[1]) >= 10
    
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


input_file = 'pts.dat' if '-i' not in sys.argv else sys.argv[sys.argv.index('-i') + 1]
output_file = 'triangles.dat' if '-o' not in sys.argv else sys.argv[sys.argv.index('-o') + 1]

points = np.random.rand(2, 10).T if input_file == '' else np.loadtxt(input_file, skiprows=1)
N = len(points)
border = [min(points[:, 0]), max(points[:, 0]), min(points[:, 1]), max(points[:, 1])]

# expected result
start = time.time()
tri = Delaunay(points)
print('Time:', time.time() - start)
# plt.triplot(points[:,0], points[:,1], tri.simplices, 'r-', linewidth=0.5)
# plt.plot(points[:,0], points[:,1], 'ko', markersize=2)
# plt.show()

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

def Ball(cavity, p, index):
    # connect p to all vertices of the cavity
    new_mesh = TriangularMesh()
    new_mesh.vertices = cavity.vertices + [p]
    prev_edge1 = None
    for edge in cavity.halfedges:
        print(edge.vertex.coords)
        edge1, edge2, face = Halfedge(), Halfedge(), Face()
        edge1.vertex = p
        edge1.next = edge
        edge1.twin = edge2
        edge1.facet = face
        edge1.index = index
        edge2.vertex = edge.vertex
        edge2.prev = edge.prev
        edge2.twin = edge1
        edge2.facet = face
        edge2.index = index + 1
        edge2.next = prev_edge1
        prev_edge1 = edge1
        face.halfedge = edge
        face.index = index + 2
        index += 3
        new_mesh.halfedges += [edge1, edge2]
        new_mesh.faces.append(face)
        new_mesh.halfedges[1].next = new_mesh.halfedges[-2]
    return new_mesh

def Cavity(mesh, p, ind):
    visited = []
    # find a triangle containing p
    def check_triangle(halfedge):
        if halfedge.facet.visited:
            return False
        halfedge.facet.visited = True
        visited.append(halfedge.facet)
        a, b, c = halfedge.vertex.coords, halfedge.next.vertex.coords, halfedge.prev.vertex.coords
        res = incircle(a, b, c, p.coords)
        if np.isnan(res):
            # number of None in [a, b, c]
            k = [halfedge.twin, halfedge.next.twin, halfedge.prev.twin].count(None)
            if k == 1 or k == 3:
                # do orient2d test instead
                return orient2d(a, b, p.coords) >= 0 or orient2d(b, c, p.coords) >= 0 or orient2d(c, a, p.coords) >= 0
            elif k == 2:
                # find index of non infinite point in [a, b, c]
                d = a if not np.isinf(a[0]) else b if not np.isinf(b[0]) else c
                hello = tuple(sum(x) for x in zip(a, b, c))
                if np.isnan(hello[1]):
                    return p.coords[0] >= d.coords[0] if hello[0] > 0 else p.coords[0] <= d.coords[0]
                else:
                    return p.coords[1] >= d.coords[1] if hello[1] > 0 else p.coords[1] <= d.coords[1]

        return res >= 0

    def check_vertex(vertex):
        halfedge = vertex.halfedge
        while True:
            if check_triangle(halfedge):
                return halfedge
            halfedge = halfedge.twin.next
            if halfedge == vertex.halfedge or halfedge == None:
                break
        return None
    
    i = 2
    while True:
        u = mesh.vertices[ind + (-1)**i * i // 2]
        ok_tr = check_vertex(u)
        if ok_tr:
            break
        i += 1
    
    def expand(halfedge):
        res = TriangularMesh(vertices=[halfedge.vertex, halfedge.next.vertex, halfedge.prev.vertex], faces=[halfedge.facet], halfedges=[halfedge, halfedge.next, halfedge.prev])
        stack = deque([halfedge.twin, halfedge.next.twin, halfedge.prev.twin])
        while stack:
            edge = stack.popleft()
            if not check_triangle(edge):
                continue
            edge.facet.visited = True
            visited.append(edge.facet)
            res.vertices.append(edge.prev.vertex)
            res.halfedges += [edge, edge.next, edge.prev]
            res.faces.append(edge.facet)

            for e in [edge.next.twin, edge.prev.twin]:
                if e and not e.facet.visited:
                    stack.append(e)
        return res

    return expand(ok_tr)

test_p = [(0, 0), (2, 0), (3.1, 1.2), (1.5, 2), (0.5, 1.5)]
test_p = [Vertex(p, index=i) for i, p in enumerate(test_p)]
test_e = [Halfedge(vertex=p) for p in test_p]
test_e = [Halfedge(vertex=p, next=test_e[(i+1)%len(test_e)], prev=test_e[(i-1)%len(test_e)]) for i, p in enumerate(test_p)]
test_mesh = TriangularMesh()
test_mesh.vertices = test_p
test_mesh.halfedges = test_e
# plot points
plt.plot(*zip(*map(lambda x: x.coords, test_mesh.vertices)), 'ko')
# plot edges
for edge in test_mesh.halfedges:
    plt.plot(*zip(edge.vertex.coords, edge.next.vertex.coords), 'r-')
plt.show()

ball = Ball(test_mesh, Vertex((1, 1)), 0)
# plot points
plt.plot(*zip(*map(lambda x: x.coords, ball.vertices)), 'bo')
# plot edges
for abc in ball.halfedges:
    print(abc.vertex.coords, abc.next.vertex.coords)
    plt.plot(*zip(abc.vertex.coords, abc.next.vertex.coords), 'g-')
plt.show()