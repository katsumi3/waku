# 2.8用

import bpy, mathutils, bmesh,bpy.ops
from math import pi, radians, degrees, sqrt, cos, acos, tan, sin, atan
from mathutils import Vector, Matrix
from bpy.props import FloatProperty, EnumProperty,BoolProperty

bl_info = {
    "name": "多角形の平面の内側に枠を付けるアドオン",
    "author": "勝己（kastumi）",
    "version": (1, 0),
    "blender": (2, 80, 0),
    "location": "3Dビューポート > 追加 > メッシュ",
    "description": "多角形の平面の内側に枠を付けるアドオン",
    "warning": "",
    "support": "TESTING",
    "wiki_url": "",
    "tracker_url": "",
    "category": "Object"
}

############################################################
#def center_mat-->辺の中心のmatrix_worldの値を返す


def center_mat(xv, yv, p_mat, p_norm):
    xv = xv.vert.co
    yv = yv.vert.co
    loc = (p_mat @ xv + p_mat @ yv) / 2
    muki = (p_mat @ yv - p_mat @ xv)
    print("p_matp_matp_mat",p_mat,p_norm)
    mx_inv = p_mat.inverted()
    mx_norm = mx_inv.transposed().to_3x3()
    world_no = mx_norm @ p_norm
    world_no.normalize()
    cro = muki.cross(world_no)
    cro.normalize()
    muki.normalize()
    m = Matrix([[muki.x, -cro.x, world_no.x, loc.x],
                [muki.y, -cro.y, world_no.y, loc.y],
                [muki.z, -cro.z, world_no.z, loc.z], [0, 0, 0, 1]])
    return m



###########################
def shear_rad(bm, rad, id, n):
    bm.faces.ensure_lookup_table()
    axis = 'Z'
    if axis == 'Z':
        selected = [v.co.y for v in bm.verts if v.select]
    elif axis == 'Y':
        selected = [v.co.z for v in bm.verts if v.select]
    t = max(selected) - min(selected)
    shear_va = tan(rad)
    va = [t * shear_va / 2, 0, 0]
    bpy.ops.transform.shear(value=shear_va,
                            orient_axis=axis,
                            orient_axis_ortho='X',
                            orient_type='LOCAL',
                            orient_matrix_type='GLOBAL')
    bpy.ops.transform.translate(value=va,
                                orient_type='LOCAL',
                                orient_matrix_type='GLOBAL')


################################################
#def obj_length()->辺の長さにobjの大きさを合わせる
def obj_length(mat, edge):
    l = (mat @ edge.verts[0].co - mat @ edge.verts[1].co).length
    #scale = mat.to_scale() * l
    scale = [l]
    scale_mat = Matrix.Scale(scale[0], 4, (1, 0, 0))
    return scale_mat


#############################################
def end_face(ids, width, length_adj, xv, bme, t, rad, n):
    bpy.ops.mesh.select_all(action='DESELECT')
    id = ids
    bme.faces[id].select_set(True)
    bpy.ops.transform.translate(value=(0, 0, length_adj), orient_type='NORMAL')
    if round(degrees(rad)) != 0:
        shear_rad(bme, rad, id, n)


def edg_length3(mat, edge):
    l = (mat @ edge.verts[0].co - mat @ edge.verts[1].co).length
    return l


class hasigo:
    def __init__(self):
        self.waku_type = 0
        self.sw = 0
        self.cont = 0
        self.p_len = 0
        self.angle = 0.0
        self.another_angle = 0.0
        self.other_angle = 0.0
        self.another_width = 0.0
        self.width = 0.0
        self.is_conv = 0
        self.other_is_conv = 0
        self.p_mat = 0.0
        self.another_edge = 0.0
        self.edge = 0.0
        self.rad = 0.0
        self.length_adj = 0.0
        self.n = 0.0
      

    def tijimi(self):
        if round(degrees(self.angle)) in {180, 0}:
            self.length_adj = 0
            self.rad = 0

        elif self.waku_type == "3":
            self.length_adj = 0
            self.rad = (self.angle / 2 - pi / 2)
            if self.is_conv == 0:
                self.rad = -(self.angle / 2 - pi / 2)

        elif edg_length3(
                self.p_mat, self.another_edge) < self.another_width / sin(
                    self.angle) and self.is_conv and 90 < degrees(self.angle):
            self.rad = (self.angle / 2 - pi / 2)
            self.length_adj = 0
            print("el1")

        elif edg_length3(self.p_mat,
                         self.another_edge) < self.another_width / sin(
                             self.angle) and not self.is_conv:
            self.rad = -(self.angle / 2 - pi / 2)
            self.length_adj = 0
            print("el2")

        elif edg_length3(self.p_mat, self.edge) < self.width / sin(
                self.angle) and not self.is_conv:
            self.rad = -(self.angle / 2 - pi / 2)
            self.length_adj = 0
            print("el3")

        elif edg_length3(self.p_mat, self.edge) < self.another_width / sin(
                self.angle) and self.is_conv:
            self.rad = (self.angle / 2 - pi / 2)
            self.length_adj = 0
            print("el4")



        elif edg_length3(self.p_mat, self.another_edge) * abs(tan(
                self.angle)) < self.another_width and self.is_conv and round(
                    degrees(self.angle)) != 0:
            self.rad = (self.angle - pi / 2)
            self.another_width = edg_length3(
                self.p_mat, self.another_edge) * abs(tan(self.angle))
            self.length_adj = self.sw * self.is_conv * (-self.another_width /
                                                        sin(self.angle))
            print("el7")
        

        elif self.cont == 0 and self.p_len % 2 == 1 and self.n%2 == 0 :
            #０個目で面の頂点が奇数の場合
            self.length_adj = 0
            self.rad = [-1, 1][self.is_conv] * self.angle / 2 - pi / 2
            print(3)
           
        elif self.cont == self.p_len - 1 and self.p_len % 2 == 1 and self.n%2 == 1:
            #最後の1個目で面の頂点が奇数の場合
          
            self.rad = [-1, 1][self.is_conv] * self.angle / 2 - pi / 2
            self.length_adj = 0
            print("el8")
    

        elif 0 <= self.cont < self.p_len:
            if self.sw == 0 and self.is_conv == 0:
                print("futu-if")
                self.rad =self.sw* [-1, 1][self.is_conv] * (self.angle - pi / 2)
                self.length_adj = (self.another_width / sin(self.angle))
            else:
                #普通の場合
                print("futuu-else")
                self.rad = [-1, 1][self.is_conv] * (self.angle - pi / 2)
                self.length_adj = self.sw * self.is_conv * (-self.another_width /
                                                            sin(self.angle))
            print("angle", degrees(self.angle))
        print("end")

    
class WAKU_OT_CreateObject(bpy.types.Operator):

    bl_idname = "object.waku_create_object"
    bl_label = "枠"
    bl_description = "多角形の平面の内側に枠を付けます"
    bl_options = {'REGISTER', 'UNDO'}


    waku_type: EnumProperty(
        name="枠の形状",
        description="移動軸を設定します",
        default='1',
        items=[
            ('1', "梯子", "梯子風にします"),
            ('2', "風車", "風車風にします"),
            ('3', "額縁", "額縁風にします"),
        ]
    )
    v_width: FloatProperty(
        name="縦の枠の幅",
        description="幅を設定します",
        default=8.0,
     )
    h_width: FloatProperty(
            name="横の枠の幅",
            description="幅を設定します",
            default=1.0,
     )
    t: FloatProperty(
            name="厚み",
            description="厚みを設定します",
            default=1.0,
        )
    inv: BoolProperty(
         name = "反転",
         description="反転するかを設定します",
         default = False,
     )

    # メニューを実行したときに呼ばれる関数
    def execute(self, context):
        
#        bpy.ops.object.mode_set(mode='OBJECT')
#        bpy.ops.object.select_all(action='DESELECT')
#        bpy.data.objects['円'].select_set(True)
#        bpy.ops.object.select_all(action='INVERT')
#        bpy.ops.object.delete(use_global=False, confirm=False)
#        bpy.context.view_layer.objects.active = bpy.data.objects['円']

        #初期設定####################################
        #waku_type = 1  #1:梯子状　2:風車風　#3平留め継ぎ(額縁風)
        waku_type = self.waku_type
        #v_width = 8  # 梯子の縦になってる木の幅
        v_width = self.v_width
        #h_width = 1  # 梯子の横の方の木の幅
        h_width = self.h_width
        #width = 0  # 枠の木の幅（仮）
        #inv_vh = 0
        #t = 1.5  # 厚み
        t = self.t
        if self.inv == False :
            inv_vh = 0
        else:
            inv_vh = 1  #
        ############################################

        plane = bpy.context.object
        #bmshをオブジェクトモードのまま使う
        bm = bmesh.new()
        bm.from_mesh(plane.data)
        bm.faces.ensure_lookup_table()
        loops = bm.faces[0].loops
        p_mat = plane.matrix_world
        p_norm = bm.faces[0].normal
        mw_loc, mw_rot, mw_scale = p_mat.decompose()
        p_norm = mw_rot.to_matrix() @ p_norm
        p_len = len(loops)
        #cont = 2 ; xv=loops[cont]
        #if 1==1:

        for cont, xv in enumerate(loops):
            #bpy.ops.mesh.primitive_cube_add(size=1,
            #                                enter_editmode=False,
            #                                location=(0, 0, 0))
            
            currentPath=bpy.utils.script_paths() [2] + "/addons/"
            filename="hikide.blend"
            path=currentPath+filename+"/"
            obj=bpy.ops.wm.append(
                directory=path+"Object/",
                link=False,
                filename="ki")
            bpy.context.view_layer.objects.active=bpy.context.selected_objects[0]
            obj = bpy.context.object
            #縦か横か？
            if (xv.index + inv_vh) % 2 == 0:
                width = h_width
                another_width = v_width
            else:
                width = v_width
                another_width = h_width
            bpy.ops.transform.resize(value=(1, width, t), orient_type='LOCAL')
            bpy.ops.object.transform_apply(scale=True)
            yv = xv.link_loop_next
            obj.matrix_world = center_mat(xv, yv, p_mat, p_norm)
            #オブジェクトを辺の長さに合わせる
            scale_mat = obj_length(p_mat, xv.edge)
            obj.data.transform(scale_mat)
            obj.data.update()
            #位置調節
            offset = Vector((0, width / 2, t / 2))
            bpy.ops.transform.translate(value=offset,
                                        orient_type='LOCAL',
                                        orient_matrix_type='LOCAL')
            bpy.ops.object.mode_set(mode='EDIT')
            obm = bmesh.from_edit_mesh(obj.data)
            obm.faces.ensure_lookup_table()
            obm.edges.ensure_lookup_table()
            obm.verts.ensure_lookup_table()
            #
            #面[0]をどれだけ回転させてどれだけ移動するか
            #index=(id)
            index = (0, 2)
            n = xv.index + inv_vh
            length_adj = [0, 0]
            rad = [0, 0]
            off_l = [0, 0]
            zero = hasigo()
            if waku_type == "1":
                zero.sw = (n % 2 == 0)
            elif waku_type == "2":
                zero.sw = 1
            zero.waku_type = waku_type
            zero.cont = cont
            zero.p_len = p_len
            zero.angle = xv.calc_angle()
            zero.another_angle = xv.link_loop_prev.calc_angle()
            zero.other_angle = xv.link_loop_next.calc_angle()
            zero.another_width = another_width
            zero.width = width
            zero.is_conv = xv.is_convex
            zero.other_is_conv = xv.link_loop_next.is_convex
            zero.p_mat = p_mat
            zero.another_edge = xv.link_loop_prev.edge
            zero.edge = xv.edge
            zero.n = 0
            zero.tijimi()
            

            one = hasigo()
            if waku_type == "1":
                one.sw = (n % 2 == 0)
            elif waku_type == "2":
                one.sw = 0
            one.waku_type = waku_type
            one.cont = cont
            one.p_len = p_len
            one.angle = xv.link_loop_next.calc_angle()
            one.another_angle = xv.link_loop_next.link_loop_next.calc_angle()
            one.other_angle = xv.calc_angle()
            one.another_width = another_width
            one.width = width
            one.is_conv = xv.link_loop_next.is_convex
            one.other_is_conv = xv.link_loop_prev.is_convex
            one.p_mat = p_mat
            one.another_edge = xv.link_loop_next.edge
            one.edge = xv.edge
            one.n = 1
            one.tijimi()
            
         

            rad[0], length_adj[0] = zero.rad, zero.length_adj
            rad[1], length_adj[1] = one.rad, one.length_adj

            rad = [-rad[0], rad[1]]

            for i, ids in enumerate(index):
                end_face(ids, width, length_adj[i], xv, obm, t, rad[i], n)

            bpy.ops.object.mode_set(mode='OBJECT')
            print(cont)

        return {'FINISHED'}


def menu_fn(self, context):
    self.layout.separator()
    self.layout.operator(WAKU_OT_CreateObject.bl_idname)


# Blenderに登録するクラス
classes = [
    WAKU_OT_CreateObject,
]


# アドオン有効化時の処理
def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.VIEW3D_MT_mesh_add.append(menu_fn)
    print("waku: アドオン『waku』が有効化されました。")


# アドオン無効化時の処理
def unregister():
    bpy.types.VIEW3D_MT_mesh_add.remove(menu_fn)
    for c in classes:
        bpy.utils.unregister_class(c)
    print("waku: アドオン『waku』が無効化されました。")


# メイン処理
if __name__ == "__main__":
    register()