"""OpenGL 3.3 renderer (moderngl): Blinn-Phong lighting, shadow mapping, MSAA."""
import os

import moderngl
import numpy as np
from PIL import Image

from . import math3d as m3

MESH_VS = """
#version 330
uniform mat4 u_model;
uniform mat4 u_view_proj;
uniform mat4 u_light_vp;
uniform mat3 u_normal_mat;
in vec3 in_pos;
in vec3 in_normal;
in vec2 in_uv;
in vec3 in_color;
out vec3 v_world;
out vec3 v_normal;
out vec2 v_uv;
out vec3 v_color;
out vec4 v_light_pos;
void main() {
    vec4 world = u_model * vec4(in_pos, 1.0);
    v_world = world.xyz;
    v_normal = u_normal_mat * in_normal;
    v_uv = in_uv;
    v_color = in_color;
    v_light_pos = u_light_vp * world;
    gl_Position = u_view_proj * world;
}
"""

MESH_FS = """
#version 330
uniform vec3 u_color;
uniform bool u_use_tex;
uniform sampler2D u_tex;
uniform sampler2DShadow u_shadow;
uniform float u_shadow_texel;
uniform vec3 u_cam_pos;
uniform vec3 u_sun_dir;      // direction the light travels
uniform vec3 u_sun_color;
uniform vec3 u_ambient;
uniform vec3 u_sky;
uniform float u_shininess;
uniform float u_specular;
in vec3 v_world;
in vec3 v_normal;
in vec2 v_uv;
in vec3 v_color;
in vec4 v_light_pos;
out vec4 f_color;

float shadow_factor(vec3 n, vec3 l) {
    vec3 p = v_light_pos.xyz / v_light_pos.w * 0.5 + 0.5;
    if (p.z > 1.0 || any(lessThan(p.xy, vec2(0.0))) || any(greaterThan(p.xy, vec2(1.0))))
        return 1.0;
    float bias = max(0.0025 * (1.0 - dot(n, l)), 0.0006);
    float s = 0.0;
    for (int x = -1; x <= 1; ++x)        // 3x3 PCF on a hardware-compare sampler
        for (int y = -1; y <= 1; ++y)
            s += texture(u_shadow, vec3(p.xy + vec2(x, y) * u_shadow_texel, p.z - bias));
    return s / 9.0;
}

void main() {
    vec3 n = normalize(v_normal);
    if (!gl_FrontFacing) n = -n;         // double-sided materials
    vec3 albedo = pow(u_color * v_color, vec3(2.2));  // material and vertex colours are sRGB
    if (u_use_tex) albedo *= pow(texture(u_tex, v_uv).rgb, vec3(2.2));  // sRGB -> linear

    vec3 l = normalize(-u_sun_dir);
    vec3 v = normalize(u_cam_pos - v_world);
    vec3 h = normalize(l + v);
    float ndl = max(dot(n, l), 0.0);
    float spec = ndl > 0.0 ? pow(max(dot(n, h), 0.0), u_shininess) * u_specular : 0.0;
    float sh = shadow_factor(n, l);

    // hemisphere ambient: sky colour from above, darker ground bounce from below
    vec3 amb = u_ambient * mix(vec3(0.45), u_sky, n.y * 0.5 + 0.5);
    vec3 col = albedo * amb + (albedo * ndl + spec) * u_sun_color * sh;
    f_color = vec4(pow(col, vec3(1.0 / 2.2)), 1.0);  // linear -> sRGB
}
"""

DEPTH_VS = """
#version 330
uniform mat4 u_model;
uniform mat4 u_light_vp;
in vec3 in_pos;
void main() { gl_Position = u_light_vp * u_model * vec4(in_pos, 1.0); }
"""

DEPTH_FS = """
#version 330
void main() {}
"""


class GpuMesh:
    def __init__(self, ctx, geometry, mesh_prog, depth_prog):
        self.vbo = ctx.buffer(geometry.interleaved().tobytes())
        self.ibo = ctx.buffer(geometry.indices.tobytes())
        self.vao = ctx.vertex_array(
            mesh_prog, [(self.vbo, "3f 3f 2f 3f", "in_pos", "in_normal", "in_uv", "in_color")], self.ibo)
        self.depth_vao = ctx.vertex_array(
            depth_prog, [(self.vbo, "3f 12x 8x 12x", "in_pos")], self.ibo)


class Renderer:
    def __init__(self, ctx, samples=4, shadow_size=2048):
        self.ctx = ctx
        self.samples = min(samples, ctx.max_samples)
        self.mesh_prog = ctx.program(vertex_shader=MESH_VS, fragment_shader=MESH_FS)
        self.depth_prog = ctx.program(vertex_shader=DEPTH_VS, fragment_shader=DEPTH_FS)
        self._meshes = {}
        self._textures = {}

        self.shadow_tex = ctx.depth_texture((shadow_size, shadow_size))
        self.shadow_tex.compare_func = "<="
        self.shadow_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.shadow_tex.repeat_x = self.shadow_tex.repeat_y = False
        self.shadow_fbo = ctx.framebuffer(depth_attachment=self.shadow_tex)
        self.shadow_size = shadow_size

        self._msaa_fbo = None
        self._size = (0, 0)
        self.stats = {}

    # -- resources -----------------------------------------------------------
    def _gpu_mesh(self, geometry):
        key = id(geometry)
        if key not in self._meshes:
            self._meshes[key] = GpuMesh(self.ctx, geometry, self.mesh_prog, self.depth_prog)
        return self._meshes[key]

    def _texture(self, path):
        if path not in self._textures:
            img = Image.open(path).convert("RGB").transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            img.thumbnail((2048, 2048))
            tex = self.ctx.texture(img.size, 3, img.tobytes())
            tex.build_mipmaps()
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            tex.anisotropy = 8.0
            self._textures[path] = tex
        return self._textures[path]

    def resize(self, width, height):
        """(Re)create the MSAA framebuffer for a new framebuffer size in pixels."""
        width, height = max(1, int(width)), max(1, int(height))
        if (width, height) == self._size:
            return
        if self._msaa_fbo is not None:
            for a in (*self._msaa_fbo.color_attachments, self._msaa_fbo.depth_attachment):
                a.release()
            self._msaa_fbo.release()
        color = self.ctx.renderbuffer((width, height), 4, samples=self.samples)
        depth = self.ctx.depth_renderbuffer((width, height), samples=self.samples)
        self._msaa_fbo = self.ctx.framebuffer(color, depth)
        self._size = (width, height)

    # -- frame ---------------------------------------------------------------
    def _light_view_proj(self, scene):
        lo, hi = scene.bounds()
        center = (lo + hi) / 2
        radius = float(np.linalg.norm(hi - lo) / 2) or 1.0
        d = scene.sun.direction
        eye = center - d * radius * 2.0
        up = (0, 0, 1) if abs(d[1]) > 0.99 else (0, 1, 0)
        view = m3.look_at(eye, center, up)
        proj = m3.orthographic(-radius, radius, -radius, radius, radius * 0.5, radius * 3.5)
        return proj @ view

    def render(self, scene, camera, target_fbo):
        ctx = self.ctx
        meshes = scene.meshes()
        light_vp = self._light_view_proj(scene)
        light_vp_gl = light_vp.T.copy().tobytes()

        # 1) shadow map pass
        self.shadow_fbo.use()
        self.shadow_fbo.clear(depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE)
        self.depth_prog["u_light_vp"].write(light_vp_gl)
        for n in meshes:
            if n.cast_shadow:
                self.depth_prog["u_model"].write(n.world_matrix.T.copy().tobytes())
                self._gpu_mesh(n.geometry).depth_vao.render()

        # 2) main pass into the multisampled framebuffer
        self.resize(*target_fbo.size)
        fbo = self._msaa_fbo
        fbo.use()
        fbo.clear(*scene.background, 1.0, depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.depth_func = "<"

        p = self.mesh_prog
        p["u_view_proj"].write((camera.projection_matrix() @ camera.view_matrix()).T.copy().tobytes())
        p["u_light_vp"].write(light_vp_gl)
        p["u_cam_pos"].value = tuple(camera.position)
        p["u_sun_dir"].value = tuple(scene.sun.direction)
        p["u_sun_color"].value = tuple(scene.sun.color * scene.sun.intensity)
        p["u_ambient"].value = tuple(scene.ambient.color * scene.ambient.intensity)
        p["u_sky"].value = (1.0, 1.0, 1.0)
        p["u_shadow_texel"].value = 1.0 / self.shadow_size
        p["u_shadow"].value = 0
        p["u_tex"].value = 1
        self.shadow_tex.use(location=0)

        tris = 0
        for n in meshes:
            mat = n.material
            if mat.double_sided:
                ctx.disable(moderngl.CULL_FACE)
            else:
                ctx.enable(moderngl.CULL_FACE)
            ctx.wireframe = mat.wireframe
            p["u_model"].write(n.world_matrix.T.copy().tobytes())
            p["u_normal_mat"].write(m3.normal_matrix(n.world_matrix).T.copy().tobytes())
            p["u_color"].value = mat.color
            p["u_shininess"].value = mat.shininess
            p["u_specular"].value = mat.specular
            use_tex = bool(mat.texture) and os.path.exists(mat.texture)
            p["u_use_tex"].value = use_tex
            if use_tex:
                self._texture(mat.texture).use(location=1)
            self._gpu_mesh(n.geometry).vao.render()
            tris += n.geometry.triangle_count

        ctx.wireframe = False

        # 3) resolve MSAA into the target (window or offscreen framebuffer)
        ctx.copy_framebuffer(target_fbo, fbo)
        self.stats = {"meshes": len(meshes), "triangles": tris}
