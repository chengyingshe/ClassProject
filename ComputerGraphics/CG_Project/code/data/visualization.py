import open3d as o3d

pcd = o3d.io.read_point_cloud('/home/scy/ComputerGraphics/CG_Project/temp/data/scannet/scans/scene0000_00/scene0000_00_vh_clean_2.ply')
pcd_label = o3d.io.read_point_cloud('/home/scy/ComputerGraphics/CG_Project/temp/data/scannet/scans/scene0000_00/scene0000_00_vh_clean_2.labels.ply')
# o3d.visualization.draw_geometries([pcd])
o3d.visualization.draw_geometries([pcd_label])