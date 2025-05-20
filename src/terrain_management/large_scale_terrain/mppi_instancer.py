import omni
from colorsys import hsv_to_rgb
import numpy as np

from pxr import UsdGeom, Usd, Vt, Gf, UsdShade, Sdf

class VisualizeMPPI:
    def __init__(self, stage, instancer_path):
        self.stage = stage
        self.instancer_path = instancer_path

    
    def createColor(self, material_path, color):
        """
        Creates a color material."""

        material_path = omni.usd.get_stage_next_free_path(self.stage, material_path, False)
        material = UsdShade.Material.Define(self.stage, material_path)
        shader = UsdShade.Shader.Define(self.stage, material_path + "/shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(color))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material
    
    def applyMaterial(self, prim: Usd.Prim, material: UsdShade.Material) -> None:
        """
        Applies a material to a prim."""

        binder = UsdShade.MaterialBindingAPI.Apply(prim)
        binder.Bind(material)

    def build_visualizer(self):
        self.instancer_path = omni.usd.get_stage_next_free_path(self.stage, self.instancer_path, False)
        self.instancer = UsdGeom.PointInstancer.Define(self.stage, self.instancer_path)
        self.instancer_prim = self.instancer.GetPrim()

        # Make a color_map from HSV, sample Hue from 0 to 240, 8bit encoding
        self.color_path = self.instancer_path + "/looks"

        prim_paths = []
        for i in range(256):
            hue = (i / 255.0) * (240.0 / 360.0)
            rgb = hsv_to_rgb(hue, 1.0, 1.0)
            color_name = "rgb_"+str(i)
            color_path = self.color_path + "/" + color_name
            material = self.createColor(color_path, rgb)        

            # Add a sphere as a prototype to the point instancer
            sphere_path = self.instancer_path + "/sphere_" + str(i)
            sphere = UsdGeom.Sphere.Define(self.stage, sphere_path)
            cs = UsdGeom.PrimvarsAPI(sphere.GetPrim()).CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool)
            cs.Set(True)
            prim_paths.append(sphere.GetPath())
            sphere.GetRadiusAttr().Set(0.125)
            self.applyMaterial(sphere.GetPrim(), material)

            #sphere.GetRefinementLevelAttr().Set(2) # Make the sphere smoother

        # Add the sphere as a prototype
        self.instancer.GetPrototypesRel().SetTargets(prim_paths)

        # Set the instancer to use the sphere prototype
        self.update_visualizer(np.array([[0, 0, 0]]), np.array([1]))
    
    def update_visualizer(self, position, weight: np.ndarray = None):
        """
        Update the position of the point instancer.

        Args:
            position (list | np.array): A list of x, y, z coordinates for the instancer. [N, 3]
        """

        self.instancer.GetPositionsAttr().Set(position)

        if weight is None:
            self.instancer.GetProtoIndicesAttr().Set([0] * len(position))
        else:
            # If weight == 0 then assign id 0 (red), if weight == 1 then assign id 255 (blue)
            # shape weight == shape position
            # weight = weight / np.max(weight)

            id = ((1-weight) * 255).astype(np.int32)

            self.instancer.GetProtoIndicesAttr().Set(id)
            

        try:
            self.update_extent()
        except Exception as e:
            print("Error updating extent", e)


    def update_extent(self) -> None:
        """
        Updates the extent of an instancer.
        """

        # Compute the extent of the objetcs.
        extent = self.instancer.ComputeExtentAtTime(Usd.TimeCode(0), Usd.TimeCode(0))
        # Applies the extent to the instancer.
        self.instancer.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(extent[0]), Gf.Vec3f(extent[1])]))

# Example usage
if __name__ == "__main__":
    stage = omni.usd.get_context().get_stage()
    instancer_path = "/World/MPPI_Visualizer"
    
    visualizer = VisualizeMPPI(stage, instancer_path)
    visualizer.build_visualizer()
    
    # Example position update
    positions = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
    visualizer.update_visualizer(positions)