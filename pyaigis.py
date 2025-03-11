import pyvista
import vtk
import math
import numpy as np

def latlon_grid(mesh, lat_interval, lon_interval):
    lat_grid = {}
    lon_grid = {}
    
    length=math.ceil(mesh.length)

    plane = pyvista.Plane(direction=(0,0,1)).triangulate()
    cutter = vtk.vtkCutter()
    implicitFunction = vtk.vtkImplicitPolyDataDistance()
    implicitFunction.SetInput(plane)
    cutter.SetInputData(mesh)
    cutter.SetCutFunction(implicitFunction)
    cutter.Update()
    lat_grid[0] = pyvista.wrap(cutter.GetOutput())

    for lat in range(0, 90, lat_interval):
        if lat != 0:
            cone_N = pyvista.Cone(center=(0, 0, length/2), height=length, direction=(0, 0, -1), angle=lat, resolution=30).triangulate()
            cone_S = pyvista.Cone(center=(0, 0, (-1)*length/2), height=length, direction=(0, 0, 1), angle=lat, resolution=30).triangulate()
            
            cutter_N = vtk.vtkCutter()
            implicitFunction_N = vtk.vtkImplicitPolyDataDistance()
            implicitFunction_N.SetInput(cone_N)
            cutter_N.SetInputData(mesh)
            cutter_N.SetCutFunction(implicitFunction_N)
            cutter_N.Update()

            cutter_S = vtk.vtkCutter()
            implicitFunction_S = vtk.vtkImplicitPolyDataDistance()
            implicitFunction_S.SetInput(cone_S)
            cutter_S.SetInputData(mesh)
            cutter_S.SetCutFunction(implicitFunction_S)
            cutter_S.Update()


            lat_grid[lat] = pyvista.wrap(cutter_N.GetOutput())
            lat_grid[lat*(-1)] = pyvista.wrap(cutter_S.GetOutput())


    for lon in range(0, 180, lon_interval):
        plane = pyvista.Plane(direction=(math.tan(math.radians(lon)), 1, 0), i_size=length, j_size=length).triangulate()

        cutter = vtk.vtkCutter()
        implicitFunction = vtk.vtkImplicitPolyDataDistance()
        implicitFunction.SetInput(plane)
        cutter.SetInputData(mesh)
        cutter.SetCutFunction(implicitFunction)
        cutter.Update()
        
        lon_grid[lon] = pyvista.wrap(cutter.GetOutput())


    lat_grid = dict(sorted(lat_grid.items()))

    return lat_grid, lon_grid
    
# モデル上に点をマーキングする関数
def model_marking(mesh, mapdata=[], mode=2):
    global marking_list
    marking_list = []
    marking = None

    class Picker:
        def __init__(self, plotter, mesh):
            self.plotter = plotter
            self.mesh = mesh
            self._points = []
            self.count = 0

        @property
        def points(self):
            return self._points

        def __call__(self, *args):
            picked_pt = np.array(self.plotter.pick_mouse_position())
            direction = picked_pt - self.plotter.camera_position[0]
            direction = direction / np.linalg.norm(direction)
            start = picked_pt - 1000 * direction
            end = picked_pt + 10000 * direction
            global point
            point, ix = self.mesh.ray_trace(start, end, first_point=True)

            if len(point) > 0:
                global sphere
                global marking

                self._points.append(point)

                if self.count == 0:
                    sphere = 0
                elif mode == 1:
                    _ = self.plotter.remove_actor(sphere)

                sphere = p.add_mesh(pyvista.Sphere(radius=0.005, center=point), color='red')

                marking_x = point[0]
                marking_y = point[1]
                marking_z = point[2]

                if mode == 1:
                    marking = [marking_x, marking_y, marking_z]
                elif mode == 2:
                    marking_list.append([marking_x, marking_y, marking_z])

                self.count += 1

            return

    # PyVistaプロッターのセットアップ
    # p = pyvista.Plotter()
    p = pyvista.Plotter(notebook=False)
    p.background_color = 'white'

    if mapdata != []:
        p.add_mesh(mesh, scalars=mapdata)
    else:
        p.add_mesh(mesh, scalars=None)

    # ピッカーのセットアップ
    picker = Picker(p, mesh)
    p.track_click_position(picker, side='right')
    p.show()

    # modeによって返却するデータを選択
    if mode == 1:
        return marking
    elif mode == 2:
        return marking_list
    else:
        print('mode選択が正しくありません')

# モデル上に点をマーキングする関数
def model_marking_inline(mesh, mapdata=[], mode=2):
    global marking_list
    marking_list = []
    marking = None

    class Picker:
        def __init__(self, plotter, mesh):
            self.plotter = plotter
            self.mesh = mesh
            self._points = []
            self.count = 0

        @property
        def points(self):
            return self._points

        def __call__(self, *args):
            picked_pt = np.array(self.plotter.pick_mouse_position())
            direction = picked_pt - self.plotter.camera_position[0]
            direction = direction / np.linalg.norm(direction)
            start = picked_pt - 1000 * direction
            end = picked_pt + 10000 * direction
            global point
            point, ix = self.mesh.ray_trace(start, end, first_point=True)

            if len(point) > 0:
                global sphere
                global marking

                self._points.append(point)

                if self.count == 0:
                    sphere = 0
                elif mode == 1:
                    _ = self.plotter.remove_actor(sphere)

                sphere = p.add_mesh(pyvista.Sphere(radius=0.005, center=point), color='red')

                marking_x = point[0]
                marking_y = point[1]
                marking_z = point[2]

                if mode == 1:
                    marking = [marking_x, marking_y, marking_z]
                elif mode == 2:
                    marking_list.append([marking_x, marking_y, marking_z])

                self.count += 1

            return

    # PyVistaプロッターのセットアップ
    # p = pyvista.Plotter()
    p = pyvista.Plotter(notebook=True)
    p.background_color = 'white'

    if mapdata != []:
        p.add_mesh(mesh, scalars=mapdata)
    else:
        p.add_mesh(mesh, scalars=None)

    # ピッカーのセットアップ
    picker = Picker(p, mesh)
    p.track_click_position(picker, side='right')
    p.show(jupyter_backend='trame') # Trame

    # modeによって返却するデータを選択
    if mode == 1:
        return marking
    elif mode == 2:
        return marking_list
    else:
        print('mode選択が正しくありません')


def get_circle(points, mesh):
    if len(points) != 2:
        print("Exactly two points are required.")
        return

    #中心と半径の計算
    center = np.array(points[0])
    radius_point = np.array(points[1])
    radius = np.linalg.norm(radius_point - center)

    #球体の生成
    sphere = pyvista.Sphere(center=center, radius=radius, theta_resolution=180, phi_resolution=180).triangulate()

    #メッシュと球体の交差部分を計算
    cutter = vtk.vtkCutter()
    implicit_function = vtk.vtkImplicitPolyDataDistance()
    implicit_function.SetInput(sphere)

    cutter.SetInputData(mesh)
    cutter.SetCutFunction(implicit_function)
    cutter.Update()

    #結果の取得とPyVista用のラップ
    cut_result = cutter.GetOutput()
    cut_polydata = pyvista.wrap(cut_result)

    return cut_polydata

def get_line(points, mesh, mode=1):
    if len(points) != 2:
        print("Exactly two points are required.")
        return

    start_point = np.array(points[0])
    end_point = np.array(points[1])
    # midpoint = (start_point + end_point) / 2
    # direction = end_point - start_point
    # normal = np.cross(direction, [0, 0, 1])
    # if np.linalg.norm(normal) == 0:
    #     normal = [1, 0, 0]

    from vtk import vtkMath
    p1  = start_point
    p2 = end_point
    p_origin = [0, 0, 0]
    v1 = [0, 0, 0] # make it a list because a list is assignable
    v2 = [0, 0, 0]
    normal = [0, 0, 0]
    vtkMath.Subtract(p2, p1, v1);
    vtkMath.Normalize(v1);
    vtkMath.Subtract(p_origin, p1, v2);
    vtkMath.Normalize(v2);
    vtkMath.Cross(v1, v2, normal);

    plane = vtk.vtkPlane()
    plane.SetOrigin(p1)
    # plane.SetOrigin(midpoint)
    plane.SetNormal(normal)

    cutter = vtk.vtkCutter()
    cutter.SetInputData(mesh)
    cutter.SetCutFunction(plane)
    cutter.Update()

    cut_polydata = pyvista.wrap(cutter.GetOutput())
    cut_points = cut_polydata.points

    cut_points = np.array(cut_polydata.points)
    cut_points = np.unique(cut_points, axis=0)

    if cut_polydata.n_points > 0:
        remaining_points = list(cut_points)
        sorted_points = []

        current_point = start_point
        while len(remaining_points) > 0:
            distances = np.linalg.norm(remaining_points - current_point, axis=1)
            idx = np.argmin(distances)
            chosen_point = remaining_points[idx]

            sorted_points.append(chosen_point)
            remaining_points.pop(idx)
            current_point = chosen_point

        sorted_points = np.array(sorted_points)
        end_point_index = next((i for i, point in enumerate(sorted_points) if np.allclose(point, end_point, atol=1e-6)), None)
        line_points = []
        

        if mode == 1:
            if end_point_index <= (len(sorted_points)/2):
                # Explore sorted_points to find points from index 0 to end_point
                for point in sorted_points:
                    line_points.append(point)
                    if np.allclose(point, end_point, atol=1e-6):
                        break
        
                line_points = np.array(line_points)
        
                line_polydata = pyvista.PolyData()
                line_polydata.points = line_points
                line_polydata.lines = np.hstack(([line_points.shape[0]], np.arange(line_points.shape[0])))

                return line_polydata
        
                # plotter.add_mesh(line_polydata, color="blue", line_width=2)
                # plotter.add_mesh(mesh)
                # plotter.add_mesh(np.array(points[0]),color='red')
                # plotter.add_mesh(np.array(points[1]),color='orange')
                # print("1-a")
            else:
                # Explore sorted_points to find points from index end_point to last_point
                for point in sorted_points[end_point_index:]:
                    line_points.append(point)
    
                line_points = np.array(line_points)
        
                line_polydata = pyvista.PolyData()
                line_polydata.points = line_points
                line_polydata.lines = np.hstack(([line_points.shape[0]], np.arange(line_points.shape[0])))
        
                return line_polydata

                # plotter.add_mesh(line_polydata.points[0])
                # plotter.add_mesh(line_polydata, color="blue", line_width=2)
                # plotter.add_mesh(mesh)
                # plotter.add_mesh(np.array(points[0]),color='red')
                # plotter.add_mesh(np.array(points[1]),color='orange')
                # print("1-b")
                
        if mode == 2:
            if end_point_index <= (len(sorted_points)/2):
                # Explore sorted_points to find points from index end_point to last_point
                for point in sorted_points[end_point_index:]:
                    line_points.append(point)
        
                line_points = np.array(line_points)
        
                line_polydata = pyvista.PolyData()
                line_polydata.points = line_points
                line_polydata.lines = np.hstack(([line_points.shape[0]], np.arange(line_points.shape[0])))

                return line_polydata

                # plotter.add_mesh(line_polydata.points[0])
                # plotter.add_mesh(line_polydata, color="blue", line_width=2)
                # plotter.add_mesh(mesh)
                # plotter.add_mesh(np.array(points[0]),color='red')
                # plotter.add_mesh(np.array(points[1]),color='orange')
                # print("2-b")
            else:
                # Explore sorted_points to find points from index 0 to end_point
                for point in sorted_points:
                    line_points.append(point)
                    if np.allclose(point, end_point, atol=1e-6):
                        break
        
                line_points = np.array(line_points)
                
                line_polydata = pyvista.PolyData()
                line_polydata.points = line_points
                line_polydata.lines = np.hstack(([line_points.shape[0]], np.arange(line_points.shape[0])))
        
                return line_polydata
                # plotter.add_mesh(line_polydata, color="blue", line_width=2)
                # plotter.add_mesh(mesh)
                # plotter.add_mesh(np.array(points[0]),color='red')
                # plotter.add_mesh(np.array(points[1]),color='orange')
                # print("2-a")

def extract_profile(line, mesh, mapdata):
    line_points = np.array(line.points)

    closest_points_indices = []
    for point in line_points:
        closest_point_index = mesh.find_closest_point(point)
        np.insert
        closest_points_indices.append(closest_point_index)
        
    # 元のモデルの最も近い点を取得する
    closest_points = []
    
    for index in closest_points_indices:
        closest_point = mesh.points[index-1]
        closest_points.append(closest_point)
    
    reordered_closest_points = reorder_points_by_distance(np.array(closest_points)) #近い順に並び替えたポイント
    
    reordered_closest_points_corrected = []
    for s_point in line_points:
        distances = [np.linalg.norm(s_point - r_point) for r_point in reordered_closest_points]
        closest_point_index = np.argmin(distances)
        reordered_closest_points_corrected.append(reordered_closest_points[closest_point_index])
    
    reordered_closest_points_corrected = reorder_points_by_distance(np.array(reordered_closest_points_corrected))

    distances = [0]
    sum_distance = 0
    
    for i in range(1, len(reordered_closest_points_corrected)):
        distance = np.linalg.norm(reordered_closest_points_corrected[i] - reordered_closest_points_corrected[i-1])
        sum_distance += distance
        distances.append(sum_distance)

    cell = []
    for point in reordered_closest_points_corrected:
        cell.append(mesh.find_closest_cell(point))
        
    filtered_mapdata = [mapdata[i] for i in cell]

    return np.stack([np.array(distances), np.array(filtered_mapdata)]).T

def reorder_points_by_distance(points):
    ordered_points = [points[0]]  # 最初の点をスタート地点にする
    points_to_order = list(points[1:])  # 残りの点をリストとして扱う

    while len(points_to_order) > 0:
        last_point = ordered_points[-1]
        # 最後に追加された点に最も近い点を探す
        next_point = min(points_to_order, key=lambda p: np.linalg.norm(p - last_point))
        ordered_points.append(next_point)
        
        # インデックスを使ってリストから削除
        index_to_remove = np.where(np.all(points_to_order == next_point, axis=1))[0][0]
        points_to_order.pop(index_to_remove)

    return ordered_points

def meshforglobalmap(mesh):
    points = mesh.points
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    longitude = np.arctan2(y, x)
    longitude = (longitude + np.pi) % (2 * np.pi) - np.pi
    
    longitude = longitude + np.pi
    longitude = (longitude + np.pi) % (2 * np.pi) - np.pi
    
    r_xy = np.sqrt(x**2 + y**2)
    latitude = np.arctan2(z, r_xy)
    
    U = (longitude + np.pi) / (2 * np.pi)
    V = (latitude + (np.pi / 2)) / np.pi
    
    uv = np.column_stack([U, V])

    # --- シーム検出と頂点複製 ---
    # 新しい頂点座標、UV、及びセルの接続情報のリストを初期化
    new_points = mesh.points.tolist()     # 複製する際、元の座標リストに追加していきます
    new_uv     = uv.tolist()
    new_faces  = []  # 接続配列は1次元リストで管理
    
    # オリジナルの faces 配列は各セルごとに [頂点数, idx0, idx1, ..., idx_{n-1}] となっている
    orig_faces = mesh.faces
    i = 0
    while i < len(orig_faces):
        n_vertices = int(orig_faces[i])
        cell = orig_faces[i+1 : i+1+n_vertices].tolist()
        
        # 各セル内の U 座標を取得し、その範囲を計算
        cell_uv = np.array([uv[idx] for idx in cell])
        u_values = cell_uv[:, 0]
        u_min = u_values.min()
        u_max = u_values.max()
        
        # セル内の U 差が 0.5 を超えていればシームと判断
        new_cell = []
        if u_max - u_min > 0.5:
            for idx in cell:
                if uv[idx, 0] < 0.5:
                    # 複製：元の頂点から新たな頂点を作成し、Uに1.0加算
                    new_pt = mesh.points[idx]
                    new_uv_val = uv[idx].copy()
                    new_uv_val[0] += 1.0
                    new_idx = len(new_points)
                    new_points.append(new_pt.tolist())
                    new_uv.append(new_uv_val.tolist())
                    new_cell.append(new_idx)
                else:
                    new_cell.append(idx)
        else:
            # シームでなければ元の頂点インデックスをそのまま使用
            new_cell = cell.copy()
        
        # 新しいセルの情報: [頂点数, idx0, idx1, ..., idx_{n-1}] を new_faces に追加
        new_faces.append(n_vertices)
        new_faces.extend(new_cell)
        
        # 次のセルへ
        i += n_vertices + 1

    #新しいメッシュの生成
    new_faces = np.array(new_faces, dtype=np.int64)
    new_points = np.array(new_points)
    new_uv     = np.array(new_uv)
    
    new_mesh = pyvista.PolyData(new_points, faces=new_faces)
    new_mesh.point_data["Texture Coordinates"] = new_uv
    new_mesh.point_data.active_texture_coordinates = new_uv

    return new_mesh