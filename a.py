import plotly.graph_objects as go
import numpy as np
from scipy.spatial import Delaunay

# Example points
points = np.random.rand(10, 2)
tri = Delaunay(points)

# Create initial plot
fig = go.Figure()

# Plot points
fig.add_trace(go.Scatter(x=points[:, 0], y=points[:, 1], mode='markers'))

# Plot triangulation edges
for triangle in tri.simplices:
    for i in range(3):
        fig.add_trace(go.Scatter(x=[points[triangle[i], 0], points[triangle[(i+1) % 3], 0]],
                                 y=[points[triangle[i], 1], points[triangle[(i+1) % 3], 1]],
                                 mode='lines'))

fig.show()

# You can make it interactive using callbacks in Dash for point updates
