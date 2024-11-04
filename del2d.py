'''
2D Delaunay triangulation using the Bowyer-Watson incremental algorithm, with Hilbert sorting for vertices.

Usage:
    python del2d.py [-i input_file.*] [-o output_file.*] [-c] [-a] [-s output_video.mp4]

Dependencies:
    - numpy
    - geompreds

Settings:
    -i input_file: str, path to the input file
    -o output_file: str, path to the output file
    -c complexity: bool, plot time complexity comparison (requires matplotlib, scipy)
    -a animation: bool, show pygame animation of triangulated mesh (requires pygame)
    -s save_anim: str, save animation; affects real-time performance, but not video (requires vidmaker)

Notes:
    - Tested in Python 3.10, Windows 10.
    - The Bowyer-Watson algorithm is initialised with 4 points at infinity (+-inf, +-inf) as to directly have the correct triangulation and convex hull.
    - Vertex coordinates are stored as tuples of floats, to be efficient with geompreds methods.
    - The time complexity is tested as approximately O(n log n), however, scipy's implementation is still almost two orders of magnitude faster.
    - The pygame animation could be improved, as it currently computes the entire triangulation from scratch at each frame. I however did not have time to finish a good update method. The same applies for edges being rendered twice.
    - Mouse interactions can be very easily implemented in the pygame animation (e.g. vertices attracted/repelled by the mouse). It did not look great, so it is omitted here.

Daniel Smolders
4 November 2024
'''


# settings
from sys import argv
complexity = True if '-c' in argv else False
animation = True if '-a' in argv else False
save_anim = '' if '-s' not in argv else argv[argv.index('-s') + 1]
input_file = '' if '-i' not in argv else argv[argv.index('-i') + 1]
output_file = '' if '-o' not in argv else argv[argv.index('-o') + 1]

if not (complexity or animation or input_file or output_file):
    animation = True


# imports
import numpy as np
from geompreds import orient2d, incircle
from collections import deque
if complexity:
    import timeit
    import matplotlib.pyplot as plt
    from scipy.spatial import Delaunay
if animation:
    import pygame
    import pygame.gfxdraw
    import gc
    if save_anim:
        import vidmaker


# half-edge mesh data structure
class TriangularMesh:
    # initialise mesh with list of vertices, sets of faces and halfedges
    # None required as default arguments to avoid single shared instance
    def __init__(self, vertices=None, faces=None, halfedges=None):
        self.vertices = [] if vertices == None else vertices
        self.faces = set() if faces == None else faces
        self.halfedges = set() if halfedges == None else halfedges

    # sort all vertices by Hilbert curve index
    def sort_vertices(self):
        self.vertices.sort(key=lambda x: x.Hilbert)

    # add a triangle to the mesh, given 3 vertices, and optionally 3 halfedges
    def add_triangle(self, vertices, halfedges=None, out=False):
        halfedges = [Halfedge() for _ in range(3)] if halfedges is None else halfedges
        face = Face(halfedge=halfedges[0], index=min([v.index for v in vertices]))
        for i in range(3): # connect halfedges anti-clockwise
            halfedges[i].next = halfedges[i-2]
            halfedges[i].prev = halfedges[i-1]
            halfedges[i].vertex = vertices[i]
            halfedges[i].facet = face
            vertices[i].halfedge = halfedges[i] # update vertex halfedge to always be up-to-date
        self.halfedges.update(halfedges)
        self.faces.add(face)
        if out:
            return halfedges
    
    # add faces and halfedges of a mesh to the current mesh
    def add_mesh(self, mesh):
        self.faces.update(mesh.faces)
        self.halfedges.update(mesh.halfedges)

    # remove faces and halfedges of a mesh from the current mesh
    def rem_mesh(self, mesh):
        self.faces.difference_update(mesh.faces)
        self.halfedges.difference_update(mesh.halfedges)

    # representation for debugging
    def __repr__(self):
        return f"Vertices: {len(self.vertices)}, Faces: {len(self.faces)}, Halfedges: {len(self.halfedges)}"

class Vertex:
    def __init__(self, coords=None, halfedge=None, Hilbert=None, index=None):
        self.coords = coords
        self.halfedge = halfedge
        self.Hilbert = Hilbert
        self.index = index
        self.vel = (np.random.uniform(-1, 1), np.random.uniform(-1, 1)) # for pygame animation
        self.acc = (0., 0.) # for pygame animation

    def __repr__(self):
        return f"({self.coords[0]:.4g}, {self.coords[1]:.4g})"

class Halfedge:
    def __init__(self, next=None, prev=None, twin=None, vertex=None, facet=None, index=None):
        self.next = next
        self.prev = prev
        self.twin = twin
        self.vertex = vertex
        self.facet = facet
        self.index = index

    def __repr__(self):
        return f"{self.vertex} -> {self.next.vertex}"

class Face:
    def __init__(self, halfedge=None, index=None):
        self.halfedge = halfedge
        self.index = index
    
    def __repr__(self):
        return f"[{self.halfedge.vertex} {self.halfedge.next.vertex} {self.halfedge.prev.vertex}]"


# calculate Hilbert index for a point
def Hilbert(point, depth, O, R, B): # O, R, B: origin, red, and blue vectors
    point, O, R, B = map(np.array, [point, O, R, B]) # convert to numpy arrays for vector operations
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

# check if a point is inside the circumcircle of the triangle containing the halfedge
def check_triangle(halfedge, p):
    # coordinates of non-infinite vertices
    coords = [v.coords for v in [halfedge.vertex, halfedge.next.vertex, halfedge.prev.vertex] if v.index >= 0]

    if len(coords) == 3:
        return incircle(*coords, p.coords) >= 0
    elif len(coords) == 2:
        # one vertex is at infinity, incircle test becomes an orientation test
        if halfedge.next.vertex.index < 0:
            coords = [coords[1], coords[0]] # reverse order of non-infinite vertices
        return orient2d(*coords, p.coords) >= 0
    else:
        # two vertices are at infinity, incircle test becomes a coordinate test
        quadrant = -sum(idx for idx in [halfedge.vertex.index, halfedge.next.vertex.index, halfedge.prev.vertex.index] if idx < 0)
        if quadrant == 3:
            return p.coords[1] >= coords[0][1]
        elif quadrant == 6:
            return p.coords[0] <= coords[0][0]
        elif quadrant == 12:
            return p.coords[1] <= coords[0][1]
        elif quadrant == 9:
            return p.coords[0] >= coords[0][0]
    
    print('Error: 2 or more incompatible vertices at infinity') # should not happen
    return False

# check_triangle for triangles around a vertex, returns halfedge of the triangle containing the point p if found
def check_vertex(vertex, p):
    hedge = vertex.halfedge
    while True:
        if check_triangle(hedge, p):
            return hedge
        if hedge.twin is None: # boundary edge
            break
        hedge = hedge.twin.next # moves clockwise around vertex
        if hedge == vertex.halfedge: # checked all triangles around vertex
            break
    return None # no triangle containing p found

# find the Delaunay cavity around a point p, returns the interior and boundary meshes
def Cavity(mesh, p, idx):
    for i in range(idx-1, -1, -1): # loop over vertices before p
        init_hedge = check_vertex(mesh.vertices[i], p) # find a triangle halfedge containing p
        if init_hedge:
            break

    # expand found cavity around p
    interior = TriangularMesh(faces={init_hedge.facet})
    boundary = []

    # BFS around cavity
    stack = deque([init_hedge, init_hedge.next, init_hedge.prev])
    while stack:
        hedge = stack.popleft()
        if hedge.twin is None or not check_triangle(hedge.twin, p): # twin halfedge is not in cavity, so hedge is a boundary edge
            boundary.append(hedge)
            continue
        # else hedge is an interior edge
        interior.halfedges.update({hedge, hedge.twin})
        interior.faces.add(hedge.twin.facet)

        # add remaining halfedges of the twin triangle to the stack
        stack.append(hedge.twin.next)
        stack.append(hedge.twin.prev)
    
    return interior, boundary

# construct Delaunay ball around a point p, connects p to all vertices of the cavity boundary
def Ball(boundary, p):
    ball = TriangularMesh()
    boundary.sort(key=lambda hedge: hedge.next.vertex.coords)
    boundary_next = sorted(boundary, key=lambda hedge: hedge.vertex.coords)

    for hedge in boundary: # add triangles connecting p to the vertices of the boundary
        ball.add_triangle([p, hedge.vertex, hedge.next.vertex], [Halfedge(), hedge, Halfedge()]) # reuse hedge to not have to remove it from the mesh
    
    for i, hedge in enumerate(boundary): # connect twin halfedges together
        hedge.next.twin = boundary_next[i].next.next
        boundary_next[i].next.next.twin = hedge.next

    return ball

# generate random points or load from file
def get_points(input_file='', N=100):
    # N random points if no input file is specified
    return np.random.rand(N, 2) if input_file == '' else np.loadtxt(input_file, skiprows=1)

# initialise mesh from points
def init_mesh(points, depth=8): # depth: Hilbert sorting depth
    border = [points[:, 0].min(), points[:, 0].max(), points[:, 1].min(), points[:, 1].max()]
    O, R, B = ((border[0] + border[1])/2, (border[2] + border[3])/2), ((border[1] - border[0])/2, 0), (0, (border[3] - border[2])/2)

    mesh = TriangularMesh()
    for i, p in enumerate(points): # add vertices with Hilbert coords to mesh
        mesh.vertices.append(Vertex(tuple(p.tolist()), Hilbert=Hilbert(p, depth, O, R, B), index=i))

    # add the 4 points at infinity
    inf_pts = [(np.inf, np.inf), (-np.inf, np.inf), (-np.inf, -np.inf), (np.inf, -np.inf)] # order is important!
    for i, p in enumerate(inf_pts):
        mesh.vertices.append(Vertex(p, Hilbert = [-1]*depth, index=-(1 << i))) # all Hilbert coords of -1, negative powers of 2 indices for determining quadrant in check_triangle
    
    # sort vertices by Hilbert index
    mesh.sort_vertices()

    return mesh

# Delaunay triangulation
def triangulate(mesh):
    # manually add the first point to the mesh
    p = mesh.vertices[4]
    init_hedges = []
    for i in range(4): # 4 triangles around p
        init_hedges += mesh.add_triangle([mesh.vertices[i], mesh.vertices[(i+1)%4], mesh.vertices[4]], out=True)
    
    # connect twin halfedges of the 4 triangles
    for i in range(4):
        init_hedges[3*i + 1].twin = init_hedges[(3*i + 5) % 12]
        init_hedges[(3*i + 5) % 12].twin = init_hedges[3*i + 1]

    # incrementally add the remaining points to the mesh with the Bowyer-Watson algorithm
    for i in range(5, len(mesh.vertices)):
        p = mesh.vertices[i]
        interior, boundary = Cavity(mesh, p, i)
        ball = Ball(boundary, p)
        mesh.rem_mesh(interior)
        mesh.add_mesh(ball)

    return mesh

# time complexity comparison
def time_complexity(avg_loops=5):
    nums = np.linspace(1, 4, 50)
    nums = np.power(10, nums) # powers of 10 from 10 to 10^4, evenly spaced in log space
    del2d_times = []
    scipy_times = []
    points = get_points(N=int(nums[-1]))

    for num in nums:
        test_points = points[:int(num)]

        # test del2d
        start = timeit.default_timer()
        for _ in range(avg_loops):
            tri = triangulate(init_mesh(test_points))
        end = timeit.default_timer()
        del2d_times.append((end - start) / avg_loops)

        # test scipy
        start = timeit.default_timer()
        for _ in range(avg_loops):
            tri = Delaunay(test_points)
        end = timeit.default_timer()
        scipy_times.append((end - start) / avg_loops)

    # plot results, compare to O(n^2) and O(n log n)
    plt.plot(nums, nums*nums/(nums[0]*nums[0])*del2d_times[0], label=r'O($n^2$)')
    plt.plot(nums, nums*np.log(nums)/(nums[0]*np.log(nums[0]))*del2d_times[0], label=r'O($n \log n$)')
    plt.plot(nums, del2d_times, label='del2d')
    plt.plot(nums, scipy_times, label='scipy')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of points')
    plt.ylabel('Time (s)')
    plt.title('Time complexity comparison')
    plt.legend()
    plt.show()

# pygame animation
def animate(FPS, vel, acc, save_anim): # vel: velocity factor, acc: acceleration/jitter factor
    pygame.init()
    w, h = 800, 600
    screen = pygame.display.set_mode((w, h))
    clock = pygame.time.Clock()
    gc.disable() # disable garbage collection for performance, might be necessary to enable it for long runs

    if save_anim:
        video = vidmaker.Video(save_anim, FPS, late_export=True)

    # initialise mesh
    points = get_points(N=100) * [w, h]
    mesh = triangulate(init_mesh(points))

    def clip(x, a, b):
        return max(a, min(b, x))

    # update velocities and positions
    def update(mesh, w, h, vel, acc):
        # re-triangulate mesh
        mesh.faces = set()
        mesh.halfedges = set()
        mesh = triangulate(mesh)

        # update velocities and positions
        for p in mesh.vertices:
            p.vel = (clip(p.vel[0] + acc*p.acc[0], -1, 1), clip(p.vel[1] + acc*p.acc[1], -1, 1))
            p.coords = (p.coords[0] + vel*p.vel[0], p.coords[1] + vel*p.vel[1])

            # reset acceleration
            p.acc = (np.random.uniform(-1, 1), np.random.uniform(-1, 1))

            # boundary conditions
            x, y = p.coords
            if x <= 0 or x >= w:
                p.vel = (-p.vel[0], p.vel[1])
                p.coords = (clip(x, 0, w), y)
            if y <= 0 or y >= h:
                p.vel = (p.vel[0], -p.vel[1])
                p.coords = (x, clip(y, 0, h))

    # render mesh to screen
    def render(screen, mesh, w, h):
        screen.fill((255, 255, 255))
        for face in mesh.faces:
            vertices = [face.halfedge.vertex, face.halfedge.next.vertex, face.halfedge.prev.vertex]
            if any(v.index < 0 for v in vertices):
                continue
            
            # colour based on triangle centroid
            avgX, avgY = np.mean([v.coords for v in vertices], axis=0)
            colour = (int(255*(1-avgX/w)), int(255*avgY/h), int(255*avgX/w))

            # render triangle colour and anti-aliased outline
            pygame.gfxdraw.filled_polygon(screen, list(v.coords for v in vertices), colour)
            pygame.gfxdraw.aapolygon(screen, list(v.coords for v in vertices), (0, 0, 0))

    # main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        update(mesh, w, h, vel, acc)
        render(screen, mesh, w, h)
        pygame.display.flip()
        clock.tick(FPS)

        if save_anim:
            video.update(pygame.surfarray.pixels3d(screen).swapaxes(0, 1))
    
    pygame.quit()

    if save_anim:
        video.export(verbose=True)


if __name__ == "__main__":
    # get points and initialise mesh
    points = get_points(input_file)
    
    # triangulate mesh
    mesh = triangulate(init_mesh(points))

    # save output to file
    if output_file:
        faces = [face for face in mesh.faces if face.index >= 0]
        with open(output_file, 'w') as f:
            f.write(f'{len(faces)}\n')
            for face in faces:
                f.write(f'{face.halfedge.vertex.index} {face.halfedge.next.vertex.index} {face.halfedge.prev.vertex.index}\n')

    # complexity comparison
    if complexity:
        time_complexity()

    # pygame animation
    if animation:
        animate(60, 1, 0.05, save_anim)
