import sys
import math
import gmsh
import numpy as np

class TriangularMesh:

    def __init__(self, vertices, triangles):
        # vertices, faces, halfedges


        self.vertices = []
        j = 0
        for v in vertices :
            self.vertices.append(Vertex(v[0],v[1],v[2],j,None))            
            j = j+1
            
        self.faces = []
        self.halfedges = []
        edges = {}
        j = 0
        index = 0
        for t in triangles :
            indices = [index, index+1, index + 2]
            index = index + 3
            for i in range (3) :
                self.vertices[t[i]].halfedge = indices[i]
                self.halfedges.append(Halfedge(indices[(i+1)%3], None, indices[(i+2)%3], t[i], j, indices[i]))
                edges[(t[i], t[(i+1)%3])] = indices [i]
            self.faces.append(Face(j,indices[0]))
            j = j+1
        for e,ind1 in edges.items() :
            if (e[1],e[0]) in edges :
                ind2 = edges[(e[1],e[0])]
                self.halfedges[ind1].opposite = ind2
                self.halfedges[ind2].opposite = ind1


            
class Vertex:

    def __init__(self, x=0, y=0, z=0, index=None, halfedge=None):
        self.x = x
        self.y = y
        self.z = z
        self.index = index
        self.halfedge = halfedge

class Face:

    def __init__(self, index=None, halfedge=None):
        self.index = index
        # halfedge going ccw around this facet.
        self.halfedge = halfedge

class Halfedge:

    def __init__(self, next=None, opposite=None, prev=None, vertex=None,
                 facet=None, index=None):
        self.opposite = opposite
        self.next = next
        self.prev = prev
        self.vertex = vertex
        self.facet = facet
        self.index = index


if __name__ == '__main__':
    print(len(sys.argv))
    if len(sys.argv) == 2 :
        gmsh.initialize()
        gmsh.open (sys.argv[1])
        gmsh.model.mesh.renumber_nodes()
        tags, x, _ = gmsh.model.mesh.get_nodes()
        order = np.argsort(tags)
        verts = x.reshape([-1,3])[order]
        
        _,el = gmsh.model.mesh.get_elements_by_type(2)
        tris = el.reshape((-1,3))-1
    else  :   
        verts = [[0.,0.,0.], [1.,0.,0.], [1.,1.,0.], [0.,1.,0.]]
        tris  = [[0,1,2], [2,3,0]]
    T  = TriangularMesh(verts, tris)
    for he in T.halfedges:
        print (he.index,he.prev,he.next,he.opposite)
        
