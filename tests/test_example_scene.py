from pathlib import Path

from pxr import Usd, UsdGeom, UsdPhysics


def test_asset_free_example_scene_has_static_collider():
    path = Path(__file__).resolve().parent.parent / "examples/example_scene.usd"
    assert path.stat().st_size < 10_000
    stage = Usd.Stage.Open(str(path))
    assert stage
    assert [layer.identifier for layer in stage.GetUsedLayers() if not layer.anonymous] == [str(path)]
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert str(stage.GetDefaultPrim().GetPath()) == "/World"
    collider = stage.GetPrimAtPath("/World/Collision")
    assert collider.HasAPI(UsdPhysics.CollisionAPI)
    assert len(UsdGeom.Mesh(collider).GetFaceVertexCountsAttr().Get()) == 24
