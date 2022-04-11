from logging import FileHandler
import open3d as o3d
import numpy as np
from particleFilter.geometry import pose2
from mpl_toolkits.mplot3d import art3d
import matplotlib.pyplot as plt

def createStructure():

    wallWidth = 0.2
    wallHeight = 2.0
    drop = -1.0
    wallColor = [0.5, 0.5, 0.3]
    firstwallColor = [0.2, 0.2, 0.3]

    FloorColor = [0.8, 0.8, 0.8]

    products = []

    ###---- axis frame
    worldFrame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=4.0, origin=[0.0, 0.0, 0.0])

    ###---- Room 1
    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=3.0, depth=wallHeight)
    wall.paint_uniform_color(firstwallColor)
    T = pose2(0,0,0).T3d()
    T[2,3] = drop
    wall.transform(T)
    products.append(wall)

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=3.0, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    T = pose2(0,0,np.pi/2).T3d()
    T[2,3] = drop
    wall.transform(T)
    products.append(wall)

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=3.0, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    T = pose2(0,3-wallWidth,np.pi/2).T3d()
    T[2,3] = drop
    wall.transform(T)
    products.append(wall)

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=1.5, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    T = pose2(-3,wallWidth,0).T3d()
    T[2,3] = drop
    wall.transform(T)
    products.append(wall)

    ###--- Room 2
    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=3.0, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    T = pose2(-3,3,0).T3d()
    T[2,3] = drop
    wall.transform(T)
    products.append(wall)

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=4.0, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    T = pose2(-3,0,np.pi/2).T3d()
    T[2,3] = drop
    wall.transform(T)
    products.append(wall)

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=4.0, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    T = pose2(-7-wallWidth,0,0).T3d()
    T[2,3] = drop
    wall.transform(T)
    products.append(wall)

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=3.0, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    T = pose2(-7,4,np.pi/2).T3d()
    T[2,3] = drop
    wall.transform(T)
    products.append(wall)

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=4.0, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    T = pose2(-10,4+wallWidth,0).T3d()
    T[2,3] = drop
    wall.transform(T)
    products.append(wall)

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=7.0, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    T = pose2(-3,8+wallWidth,np.pi/2).T3d()
    T[2,3] = drop
    wall.transform(T)
    products.append(wall)

    ###--- Room 3

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=3.0, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    T = pose2(0,8+wallWidth,np.pi/2).T3d()
    T[2,3] = drop
    wall.transform(T)
    products.append(wall)

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=5.0+2*wallWidth, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    T = pose2(0,3,0).T3d()
    T[2,3] = drop
    wall.transform(T)
    products.append(wall)

    #### Floor
    floor = o3d.geometry.TriangleMesh.create_box(width=20.0,height=20.0, depth=0.1)
    floor.paint_uniform_color(FloorColor)
    T = pose2(-14,-5,0).T3d()
    T[2,3] = -0.1 + drop
    floor.transform(T)
    products.append(floor)

    return products, worldFrame

def showPyplot(meshes):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    for mesh in meshes:
        v = mesh["vertices"]
        f = mesh["faces"]
        pc = art3d.Poly3DCollection(v[f], facecolors=mesh["color"][:3], edgecolor="none")
        ax.add_collection(pc)

    allV = np.concatenate([mesh["vertices"] for mesh in meshes])
    maxV = allV.max(axis=0)
    minV = allV.min(axis=0)
    ax.set(xlim=(minV[0], maxV[0]), ylim=(minV[1], maxV[1]), zlim=(minV[2], maxV[2]))
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.view_init(azim=-90, elev=90)
    ax.set_proj_type('ortho')
 
    return ax

def o3d_to_mymesh(o3d_mesh):
    v = np.array(o3d_mesh.vertices)
    f = np.asarray(o3d_mesh.triangles)
    color = np.asarray(np.average(o3d_mesh.vertex_colors,0))
    mymesh = {"vertices": v, "faces": f, "color": color}
    return mymesh

products, worldFrame = createStructure()

#open3d visualization
o3d.visualization.draw_geometries(products+[worldFrame],mesh_show_wireframe=True)

#matplotlib visualization
mymeshes = [o3d_to_mymesh(p) for p in products]
ax = showPyplot(mymeshes)
plt.show()

