//! A software z-buffer rasterizer: a [`Scene`] seen through a [`View`], fitted into a canvas.
//!
//! The camera is orthographic and looks down −Z after the view's turn. Triangles are drawn
//! double-sided (STL and OBJ windings are unreliable) with interpolated normals where the
//! primitive has them and flat ones otherwise, lit by a key light from the upper left, a fill
//! light from the lower right, an ambient term and a small highlight. Base colours are linear
//! (glTF's `baseColorFactor`); the output is straight-alpha sRGB on a transparent background,
//! supersampled 2 × 2 when the canvas is at most a megapixel.

use crate::Scene;
use crate::math::{Mat4, normalize};

/// Largest canvas side [`render`] draws.
pub const MAX_SIDE: u32 = 4096;
/// Space left around the model, as a fraction of the canvas on each side.
const MARGIN: f32 = 0.02;

/// How the model is turned in front of the camera.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    /// Degrees about the vertical axis, turning the model's right side towards the viewer.
    pub yaw: f32,
    /// Degrees about the horizontal axis, tipping the model's top towards the viewer.
    pub pitch: f32,
}

impl Default for View {
    /// A three-quarter view from slightly above, so a box shows three faces.
    fn default() -> Self {
        View { yaw: -30.0, pitch: 20.0 }
    }
}

impl View {
    fn matrix(&self) -> Mat4 {
        let finite = |v: f32| if v.is_finite() { v as f64 } else { 0.0 };
        Mat4::axis_rotation(0, finite(self.pitch)).mul(&Mat4::axis_rotation(1, finite(self.yaw)))
    }
}

/// Straight-alpha sRGB pixels, row by row from the top.
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// The view-space box (min x, min y, max x, max y) around every referenced vertex.
fn screen_box(turned: &[Vec<[f32; 3]>], scene: &Scene) -> Option<[f32; 4]> {
    let mut b: Option<[f32; 4]> = None;
    for (p, pos) in scene.primitives.iter().zip(turned) {
        for &i in &p.indices {
            let Some(v) = pos.get(i as usize) else { continue };
            b = Some(match b {
                None => [v[0], v[1], v[0], v[1]],
                Some(b) => [b[0].min(v[0]), b[1].min(v[1]), b[2].max(v[0]), b[3].max(v[1])],
            });
        }
    }
    b.filter(|b| b.iter().all(|c| c.is_finite()))
}

fn turn(scene: &Scene, view: &View) -> (Vec<Vec<[f32; 3]>>, Vec<Vec<[f32; 3]>>) {
    let m = view.matrix();
    let nm = m.normal_matrix();
    let pos = scene.primitives.iter().map(|p| p.positions.iter().map(|v| m.point(*v)).collect()).collect();
    let nrm = scene
        .primitives
        .iter()
        .map(|p| match &nm {
            Some(nm) => p.normals.iter().map(|n| crate::math::apply3(nm, *n)).collect(),
            None => Vec::new(),
        })
        .collect();
    (pos, nrm)
}

/// The model's width and height on screen under `view`, in model units (`None` with no
/// triangles). Its ratio is the frame a model is best inserted in.
pub fn projected_size(scene: &Scene, view: &View) -> Option<(f32, f32)> {
    let (pos, _) = turn(scene, view);
    screen_box(&pos, scene).map(|b| (b[2] - b[0], b[3] - b[1]))
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Linear light to an sRGB byte (the 2.2 power curve).
fn encode(v: f32) -> u8 {
    (v.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0 + 0.5) as u8
}

struct Lights {
    key: [f32; 3],
    fill: [f32; 3],
    half: [f32; 3],
}

impl Lights {
    fn new() -> Self {
        let key = normalize([-0.5, 0.7, 0.6]);
        Lights { key, fill: normalize([0.6, -0.2, 0.4]), half: normalize([key[0], key[1], key[2] + 1.0]) }
    }

    /// Linear RGB of a surface of `color` facing `n` (towards the camera).
    fn shade(&self, color: [f32; 4], n: [f32; 3]) -> [f32; 3] {
        let diffuse = 0.28 + 0.72 * dot(n, self.key).max(0.0) + 0.22 * dot(n, self.fill).max(0.0);
        let spec = 0.12 * dot(n, self.half).max(0.0).powi(24);
        [color[0] * diffuse + spec, color[1] * diffuse + spec, color[2] * diffuse + spec]
    }
}

/// Draws `scene` turned by `view`, scaled to fit a `width` × `height` canvas and centred.
/// `None` when the canvas is empty or larger than [`MAX_SIDE`], or the model has no extent.
pub fn render(scene: &Scene, view: &View, width: u32, height: u32) -> Option<Image> {
    if width == 0 || height == 0 || width > MAX_SIDE || height > MAX_SIDE {
        return None;
    }
    let ss: u32 = if (width as u64) * (height as u64) <= 1 << 20 { 2 } else { 1 };
    let (w, h) = (width * ss, height * ss);
    let (pos, nrm) = turn(scene, view);
    let b = screen_box(&pos, scene)?;
    let (dx, dy) = (b[2] - b[0], b[3] - b[1]);
    let room = |side: u32| side as f32 * (1.0 - 2.0 * MARGIN);
    let scale = match (dx > 1e-12, dy > 1e-12) {
        (true, true) => (room(w) / dx).min(room(h) / dy),
        (true, false) => room(w) / dx,
        (false, true) => room(h) / dy,
        (false, false) => return None,
    };
    if !(scale.is_finite() && scale > 0.0) {
        return None;
    }
    let (cx, cy) = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0);
    let (hw, hh) = (w as f32 / 2.0, h as f32 / 2.0);
    let n_px = (w as usize) * (h as usize);
    let mut depth = vec![f32::NEG_INFINITY; n_px];
    // Linear premultiplied colour per sample.
    let mut px = vec![[0f32; 4]; n_px];
    let lights = Lights::new();

    for ((p, pos), nrm) in scene.primitives.iter().zip(&pos).zip(&nrm) {
        let screen: Vec<[f32; 3]> = pos.iter().map(|v| [(v[0] - cx) * scale + hw, hh - (v[1] - cy) * scale, v[2]]).collect();
        let smooth = nrm.len() == pos.len();
        let alpha = p.color[3].clamp(0.0, 1.0);
        for t in p.indices.as_chunks::<3>().0 {
            let (Some(&a), Some(&bb), Some(&c)) = (screen.get(t[0] as usize), screen.get(t[1] as usize), screen.get(t[2] as usize)) else { continue };
            let edge = |p: [f32; 3], q: [f32; 3], x: f32, y: f32| (x - p[0]) * (q[1] - p[1]) - (y - p[1]) * (q[0] - p[0]);
            let area = edge(a, bb, c[0], c[1]);
            if !(area.is_finite() && area.abs() > 1e-12) {
                continue;
            }
            // The face normal in view space, turned towards the camera (double-sided).
            let (Some(&va), Some(&vb), Some(&vc)) = (pos.get(t[0] as usize), pos.get(t[1] as usize), pos.get(t[2] as usize)) else { continue };
            let mut face = normalize(cross(sub(vb, va), sub(vc, va)));
            let flip = face[2] < 0.0;
            if flip {
                face = [-face[0], -face[1], -face[2]];
            }
            let corner_normals = smooth.then(|| {
                let g = |i: u32| nrm.get(i as usize).copied().unwrap_or(face);
                [g(t[0]), g(t[1]), g(t[2])]
            });
            let flat = lights.shade(p.color, face);
            let x0 = a[0].min(bb[0]).min(c[0]).floor().max(0.0) as u32;
            let y0 = a[1].min(bb[1]).min(c[1]).floor().max(0.0) as u32;
            let x1 = (a[0].max(bb[0]).max(c[0]).ceil().min(w as f32 - 1.0)).max(0.0) as u32;
            let y1 = (a[1].max(bb[1]).max(c[1]).ceil().min(h as f32 - 1.0)).max(0.0) as u32;
            if x0 > x1 || y0 > y1 {
                continue;
            }
            for y in y0..=y1 {
                let fy = y as f32 + 0.5;
                for x in x0..=x1 {
                    let fx = x as f32 + 0.5;
                    let w0 = edge(bb, c, fx, fy) / area;
                    let w1 = edge(c, a, fx, fy) / area;
                    let w2 = 1.0 - w0 - w1;
                    if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                        continue;
                    }
                    let z = w0 * a[2] + w1 * bb[2] + w2 * c[2];
                    let i = (y as usize) * (w as usize) + x as usize;
                    let Some(d) = depth.get_mut(i) else { continue };
                    if z <= *d {
                        continue;
                    }
                    *d = z;
                    let rgb = match corner_normals {
                        Some([na, nb, nc]) => {
                            let mut n = normalize([
                                w0 * na[0] + w1 * nb[0] + w2 * nc[0],
                                w0 * na[1] + w1 * nb[1] + w2 * nc[1],
                                w0 * na[2] + w1 * nb[2] + w2 * nc[2],
                            ]);
                            if flip {
                                n = [-n[0], -n[1], -n[2]];
                            }
                            lights.shade(p.color, n)
                        }
                        None => flat,
                    };
                    if let Some(out) = px.get_mut(i) {
                        *out = [rgb[0] * alpha, rgb[1] * alpha, rgb[2] * alpha, alpha];
                    }
                }
            }
        }
    }

    // Down to the canvas: average the samples (premultiplied), then unpremultiply and encode.
    let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);
    let k = (ss * ss) as f32;
    for y in 0..height {
        for x in 0..width {
            let mut acc = [0f32; 4];
            for sy in 0..ss {
                for sx in 0..ss {
                    let i = ((y * ss + sy) as usize) * (w as usize) + (x * ss + sx) as usize;
                    if let Some(s) = px.get(i) {
                        for (a, v) in acc.iter_mut().zip(s) {
                            *a += v;
                        }
                    }
                }
            }
            let a = acc[3] / k;
            if a <= 0.0 {
                rgba.extend([0, 0, 0, 0]);
            } else {
                rgba.extend([encode(acc[0] / k / a), encode(acc[1] / k / a), encode(acc[2] / k / a), (a.clamp(0.0, 1.0) * 255.0 + 0.5) as u8]);
            }
        }
    }
    Some(Image { width, height, rgba })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DEFAULT_COLOR, Primitive};

    /// A unit cube, its faces wound outward, one colour per face pair.
    fn cube() -> Scene {
        let v: Vec<[f32; 3]> = (0..8).map(|i| [(i & 1) as f32, ((i >> 1) & 1) as f32, ((i >> 2) & 1) as f32]).collect();
        let quads = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]];
        let indices = quads.iter().flat_map(|q| [q[0], q[1], q[2], q[0], q[2], q[3]]).collect();
        Scene { primitives: vec![Primitive { name: "cube".into(), positions: v, indices, color: [0.8, 0.2, 0.1, 1.0], ..Default::default() }] }
    }

    fn at(img: &Image, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * img.width + x) * 4) as usize;
        [img.rgba[i], img.rgba[i + 1], img.rgba[i + 2], img.rgba[i + 3]]
    }

    #[test]
    fn a_cube_fills_the_middle_and_leaves_the_corners_clear() {
        let img = render(&cube(), &View::default(), 120, 120).expect("render");
        assert_eq!(img.rgba.len(), 120 * 120 * 4);
        assert_eq!(at(&img, 60, 60)[3], 255, "the middle is the cube");
        assert_eq!(at(&img, 1, 1)[3], 0, "a three-quarter view leaves the corners empty");
        // Three faces with three different shades: the top (lit) is brighter than the right side.
        let top = at(&img, 60, 22);
        let right = at(&img, 95, 75);
        assert!(top[3] == 255 && right[3] == 255, "{top:?} {right:?}");
        assert_ne!(top, right);
    }

    #[test]
    fn a_front_view_is_a_flat_square() {
        let img = render(&cube(), &View { yaw: 0.0, pitch: 0.0 }, 100, 50).expect("render");
        // Fitted to the height, centred: a 48 px square in the middle.
        assert_eq!(at(&img, 50, 25)[3], 255);
        assert_eq!(at(&img, 10, 25)[3], 0);
        let c = at(&img, 50, 25);
        assert!((0..100).filter(|&x| at(&img, x, 25) == c).count() >= 44, "one flat shade across the face");
        let (w, h) = projected_size(&cube(), &View { yaw: 0.0, pitch: 0.0 }).expect("size");
        assert!((w - 1.0).abs() < 1e-5 && (h - 1.0).abs() < 1e-5);
    }

    #[test]
    fn nearer_faces_hide_farther_ones() {
        // Two squares facing the camera, the red one in front: red wins whatever the order.
        let sq = |z: f32, color: [f32; 4]| Primitive {
            name: String::new(),
            positions: vec![[0.0, 0.0, z], [1.0, 0.0, z], [1.0, 1.0, z], [0.0, 1.0, z]],
            normals: vec![],
            indices: vec![0, 1, 2, 0, 2, 3],
            color,
        };
        let front = View { yaw: 0.0, pitch: 0.0 };
        for order in [[0, 1], [1, 0]] {
            let prims = [sq(1.0, [1.0, 0.0, 0.0, 1.0]), sq(0.0, [0.0, 0.0, 1.0, 1.0])];
            let scene = Scene { primitives: order.iter().map(|&i| prims[i].clone()).collect() };
            let p = at(&render(&scene, &front, 20, 20).expect("render"), 10, 10);
            assert!(p[0] > 100 && p[2] < 30, "{order:?}: {p:?}");
        }
    }

    #[test]
    fn degenerate_input_draws_nothing_or_refuses() {
        let empty = Scene::default();
        assert!(render(&empty, &View::default(), 10, 10).is_none());
        let point = Scene {
            primitives: vec![Primitive { positions: vec![[1.0; 3]; 3], indices: vec![0, 1, 2], color: DEFAULT_COLOR, ..Default::default() }],
        };
        assert!(render(&point, &View::default(), 10, 10).is_none());
        assert!(render(&cube(), &View::default(), 0, 10).is_none());
        assert!(render(&cube(), &View::default(), MAX_SIDE + 1, 10).is_none());
        assert!(render(&cube(), &View { yaw: f32::NAN, pitch: f32::INFINITY }, 10, 10).is_some());
    }
}
