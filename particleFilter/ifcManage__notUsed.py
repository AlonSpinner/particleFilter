#made from https://github.com/AlonSpinner/IFC-Demos - 05_IFC2trimesh.ipynb

import ifcopenshell, ifcopenshell.geom
import numpy as np

def ifc2trimesh(filename):

    # Open the IFC file using IfcOpenShell
    ifc = ifcopenshell.open(filename)

    # Get a list of all walls in the file
    products = ifc.by_type("IfcProduct")
    settingsGeom = ifcopenshell.geom.settings()
    settingsGeom.set(settingsGeom.USE_PYTHON_OPENCASCADE, False) #ifcopenshell.geom.create_shape behaives diffrently
    settingsGeom.set(settingsGeom.USE_WORLD_COORDS,True)
    settingsStyles = ifcopenshell.geom.settings()
    settingsStyles.set(settingsStyles.USE_PYTHON_OPENCASCADE, True)
    settingsStyles.set(settingsStyles.USE_WORLD_COORDS,True)

    meshes = []
    for product in products:
        if product.is_a("IfcOpeningElement"): continue
        if product.Representation:
            try:
                #collect color
                shapeStyle = ifcopenshell.geom.create_shape(settingsStyles, inst=product)
                color = np.array(shapeStyle.styles[0]) # the shape color

                #collect geometry
                shape = ifcopenshell.geom.create_shape(settingsGeom, inst=product)
                element = ifc.by_guid(shape.guid)
                verts = shape.geometry.verts # X Y Z of vertices in flattened list e.g. [v1x, v1y, v1z, v2x, v2y, v2z, ...]
                verts = np.array(verts).reshape((-1,3))
                faces = shape.geometry.faces  #Indices of vertices per triangle face e.g. [f1v1, f1v2, f1v3, f2v1, f2v2, f2v3, ...]
                faces = np.array(faces).reshape((-1,3))
                dict = {
                    "guid" : shape.guid,
                    "element": element,
                    "vertices": verts,
                    "faces": faces,
                    "color": color,
                    }
                meshes.append(dict)
                print(product)
            except:
                print(f'failed to include {product}')
    
    return meshes