//! Insert ▸ 3D Models: STL, OBJ (with its `.mtl` colours), FBX and glTF binary files, embedded as
//! `.glb` the way PowerPoint keeps them, so a saved `.pptx` carries a model PowerPoint can open.

use std::collections::HashMap;
use std::path::{Component, Path};

use deckcraft_geom::Xfrm;
use deckcraft_mesh::Format;
use deckcraft_model::{MediaId, Presentation, Shape, ShapeKind};
use serde_json::{Value, json};

use super::insert::{add_shape, media_bytes};
use super::*;
use crate::{Result, Session};

pub fn specs() -> Vec<CommandSpec> {
    vec![cmd!(
        "insert.model3d",
        "3D Model from File…",
        ["Insert", "Illustrations", "3D Models"],
        None,
        "{path? | data: base64, name?, rect?: [x,y,w,h]} (.glb, .fbx, .obj, .stl; an OBJ's .mtl colours are read from beside `path`) → {id, format, triangles}",
        has_slide,
        insert_model
    )]
}

/// Whether `name` / `bytes` look like a 3-D model file [`insert_model`] reads.
pub fn is_model(name: &str, bytes: &[u8]) -> bool {
    Format::detect(name, bytes).is_some()
}

/// Reads files a model names next to itself (an OBJ's `mtllib`), only by plain relative names
/// inside the model's folder, and through the same access checks as every other read.
fn sidecar_reader(path: Option<&str>) -> impl FnMut(&str) -> Option<Vec<u8>> {
    let dir = path.and_then(|p| Path::new(p).parent()).map(Path::to_path_buf);
    move |name: &str| {
        let rel = Path::new(name);
        if name.is_empty() || !rel.components().all(|c| matches!(c, Component::Normal(_))) {
            log::warn!("insert.model3d: not reading `{name}` (only files beside the model are read)");
            return None;
        }
        let full = dir.as_ref()?.join(rel);
        crate::io::read_file(full.to_str()?).map_err(|e| log::info!("insert.model3d: {e}")).ok()
    }
}

/// The frame for a model whose default view is `view_size` (width, height): half the slide's
/// height, kept inside the slide and centred.
fn default_rect(slide_w: f64, slide_h: f64, view_size: Option<(f32, f32)>) -> Xfrm {
    let aspect = view_size.map(|(w, h)| w as f64 / h as f64).filter(|a| a.is_finite() && *a > 0.0).unwrap_or(1.0).clamp(0.25, 4.0);
    let (mut w, mut h) = (slide_h * 0.5 * aspect, slide_h * 0.5);
    if w > slide_w * 0.9 {
        w = slide_w * 0.9;
        h = w / aspect;
    }
    Xfrm::new((slide_w - w) / 2.0, (slide_h - h) / 2.0, w, h)
}

/// Fallback pictures are drawn at 2 px per point, their longer side within these bounds.
const POSTER_MIN: f64 = 256.0;
const POSTER_MAX: f64 = 1024.0;

/// Pixel size of the fallback picture for a frame `w` × `h` points.
fn poster_size(w: f64, h: f64) -> (u32, u32) {
    let (w, h) = if w.is_finite() && h.is_finite() && w > 0.0 && h > 0.0 { (w, h) } else { (1.0, 1.0) };
    let k = (POSTER_MAX / w.max(h)).min((2.0f64).max(POSTER_MIN / w.max(h)));
    (((w * k).round() as u32).clamp(1, POSTER_MAX as u32), ((h * k).round() as u32).clamp(1, POSTER_MAX as u32))
}

/// The model in `glb` drawn in its default view on a transparent `w` × `h` PNG, as older readers
/// show it in place of the model.
pub fn poster_png(glb: &[u8], w: u32, h: u32) -> Option<Vec<u8>> {
    let scene = deckcraft_mesh::glb::read(glb).ok()?;
    let img = deckcraft_mesh::raster::render(&scene, &deckcraft_mesh::raster::View::default(), w, h)?;
    let img = image::RgbaImage::from_raw(img.width, img.height, img.rgba)?;
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png).ok()?;
    Some(out.into_inner())
}

/// Calls `f` on every 3-D model in `shapes` (inside groups too) that has no fallback picture.
fn bare_models<'a>(shapes: &'a [Shape], f: &mut dyn FnMut(&'a Shape)) {
    for s in shapes {
        match &s.kind {
            ShapeKind::Group { children, .. } => bare_models(children, f),
            ShapeKind::Model3d { poster: None, .. } => f(s),
            _ => {}
        }
    }
}

fn bare_models_mut(shapes: &mut [Shape], f: &mut dyn FnMut(&mut Shape)) {
    for s in shapes {
        match &mut s.kind {
            ShapeKind::Group { children, .. } => bare_models_mut(children, f),
            ShapeKind::Model3d { poster: None, .. } => f(s),
            _ => {}
        }
    }
}

fn poster_key(s: &Shape) -> Option<(MediaId, (u32, u32))> {
    let ShapeKind::Model3d { media, .. } = s.kind else { return None };
    Some((media, s.xfrm.map_or((512, 512), |x| poster_size(x.w, x.h))))
}

/// `doc` with a fallback picture drawn for every 3-D model on its slides that has none, for a
/// `.pptx` (PowerPoint keeps one beside each model for readers that can't show models).
/// `None` when every model already has one, or there are none.
pub fn with_posters(doc: &Presentation) -> Option<Presentation> {
    let mut wanted = Vec::new();
    for slide in &doc.slides {
        bare_models(&slide.shapes, &mut |s| {
            if let Some(k) = poster_key(s)
                && !wanted.contains(&k)
            {
                wanted.push(k);
            }
        });
    }
    if wanted.is_empty() {
        return None;
    }
    let mut out = doc.clone();
    let mut made: HashMap<(MediaId, (u32, u32)), MediaId> = HashMap::new();
    for (media, (w, h)) in wanted {
        let Some(item) = out.media(media) else { continue };
        let stem = Path::new(&item.name).file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "model".into());
        let Some(png) = poster_png(&item.data, w, h) else {
            log::warn!("export: no fallback picture for 3-D model {}", item.name);
            continue;
        };
        if let Some(id) = out.add_media(&format!("{stem}-poster.png"), "image/png", png) {
            made.insert((media, (w, h)), id);
        }
    }
    if made.is_empty() {
        return None;
    }
    for slide in &mut out.slides {
        let mut has = false;
        bare_models(&slide.shapes, &mut |s| has |= poster_key(s).is_some_and(|k| made.contains_key(&k)));
        if !has {
            continue;
        }
        bare_models_mut(&mut std::sync::Arc::make_mut(slide).shapes, &mut |s| {
            let id = poster_key(s).and_then(|k| made.get(&k).copied());
            if let (ShapeKind::Model3d { poster, .. }, Some(id)) = (&mut s.kind, id) {
                *poster = Some(id);
            }
        });
    }
    Some(out)
}

fn insert_model(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "insert.model3d";
    let (name, bytes) = media_bytes(p, CMD)?;
    let imported = deckcraft_mesh::import(&name, &bytes, &mut sidecar_reader(str_param(p, "path"))).map_err(|e| bad(CMD, format!("{name}: {e}")))?;
    let size = s.doc()?.doc.slide_size;
    let rect = rect_param(p, "rect").unwrap_or_else(|| default_rect(size.width, size.height, imported.view_size));
    let stem = Path::new(&name).file_stem().map(|n| n.to_string_lossy().into_owned()).filter(|n| !n.is_empty()).unwrap_or_else(|| "model".into());
    let part_name = format!("{stem}.glb");
    let media = s.edit(|doc, _| doc.add_media(&part_name, "model/gltf-binary", imported.glb).ok_or_else(|| bad(CMD, NO_ID_SPACE)))?;
    let shape = Shape { xfrm: Some(rect), kind: ShapeKind::Model3d { media, poster: None }, descr: name, ..Default::default() };
    let id = add_shape(s, CMD, shape, true)?;
    Ok(json!({"id": id, "format": imported.format.name(), "triangles": imported.triangles}))
}

#[cfg(test)]
mod tests {
    use deckcraft_model::ShapeKind;

    use super::*;

    const OBJ: &str = "mtllib parts.mtl\nv 0 0 0\nv 2 0 0\nv 2 1 0\nv 0 1 0\nusemtl Teal\nf 1 2 3 4\n";
    const STL: &str = "solid t\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 0 3\nendloop\nendfacet\nendsolid t\n";

    fn session() -> Session {
        let mut s = Session::with_new();
        s.execute("slide.new", &json!({"layout": "blank"})).expect("slide");
        s
    }

    fn models(s: &Session) -> Vec<Shape> {
        s.doc()
            .ok()
            .and_then(|d| d.current_slide().map(|sl| sl.shapes.iter().filter(|x| matches!(x.kind, ShapeKind::Model3d { .. })).cloned().collect()))
            .unwrap_or_default()
    }

    fn glb_of(s: &Session, shape: &Shape) -> Vec<u8> {
        let ShapeKind::Model3d { media, .. } = shape.kind else { return Vec::new() };
        s.doc().ok().and_then(|d| d.doc.media(media)).map(|m| m.data.to_vec()).unwrap_or_default()
    }

    #[test]
    fn inserts_a_model_as_glb_shaped_like_its_front_view() {
        let mut s = session();
        let r = s.execute("insert.model3d", &json!({"name": "bracket.obj", "data": base64_encode(OBJ.as_bytes())})).expect("insert");
        assert_eq!((r["format"].as_str(), r["triangles"].as_u64()), (Some("OBJ"), Some(2)));
        let m = models(&s);
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].name, "3D Model 1");
        assert_eq!(m[0].descr, "bracket.obj");
        let x = m[0].xfrm.expect("xfrm");
        let scene = deckcraft_mesh::glb::read(&glb_of(&s, &m[0])).expect("glb");
        let (vw, vh) = deckcraft_mesh::raster::projected_size(&scene, &deckcraft_mesh::raster::View::default()).expect("size");
        assert!((x.w / x.h - vw as f64 / vh as f64).abs() < 1e-6, "shaped like the default view: {x:?}");
        assert!(x.w > x.h, "a 2 × 1 plate stays wider than tall");
        let size = s.doc().expect("doc").doc.slide_size;
        assert!((x.x + x.w / 2.0 - size.width / 2.0).abs() < 1e-9 && (x.h - size.height * 0.5).abs() < 1e-9, "centred, half the slide high");
        let item = s.doc().ok().and_then(|d| match m[0].kind {
            ShapeKind::Model3d { media, .. } => d.doc.media(media).map(|i| (i.name.clone(), i.content_type.clone())),
            _ => None,
        });
        assert_eq!(item, Some(("bracket.glb".into(), "model/gltf-binary".into())));
        assert_eq!(deckcraft_mesh::glb::read(&glb_of(&s, &m[0])).map(|g| g.triangle_count()), Ok(2));
        assert!(s.doc().is_ok_and(|d| d.selection.shapes.contains(&m[0].id)), "the new model is selected");

        s.execute("edit.undo", &json!({})).expect("undo");
        assert!(models(&s).is_empty(), "one undo takes it out");
    }

    #[test]
    fn each_format_and_an_explicit_rect() {
        let mut s = session();
        let r =
            s.execute("insert.model3d", &json!({"name": "tall.stl", "data": base64_encode(STL.as_bytes()), "rect": [10, 20, 30, 40]})).expect("stl");
        assert_eq!(r["format"], "STL");
        let x = models(&s)[0].xfrm.expect("xfrm");
        assert_eq!((x.x, x.y, x.w, x.h), (10.0, 20.0, 30.0, 40.0));
        // A .glb is embedded as it is.
        let glb = glb_of(&s, &models(&s)[0]);
        let r = s.execute("insert.model3d", &json!({"name": "again.glb", "data": base64_encode(&glb)})).expect("glb");
        assert_eq!(r["format"], "glTF binary");
        assert_eq!(glb_of(&s, &models(&s)[1]), glb);
    }

    #[test]
    fn obj_colours_come_from_the_mtl_beside_it() {
        let dir = std::env::temp_dir().join(format!("deckcraft-model3d-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        std::fs::write(dir.join("bracket.obj"), OBJ).expect("obj");
        std::fs::write(dir.join("parts.mtl"), "newmtl Teal\nKd 0 0.5 0.5\n").expect("mtl");
        let mut s = session();
        let path = dir.join("bracket.obj");
        s.execute("insert.model3d", &json!({"path": path.to_str()})).expect("insert");
        let scene = deckcraft_mesh::glb::read(&glb_of(&s, &models(&s)[0])).expect("glb");
        assert_eq!(scene.primitives[0].color, [0.0, 0.5, 0.5, 1.0]);

        // Library names that leave the folder are not followed.
        std::fs::write(dir.join("escape.obj"), OBJ.replace("parts.mtl", "../parts.mtl")).expect("obj");
        let path = dir.join("escape.obj");
        s.execute("insert.model3d", &json!({"path": path.to_str()})).expect("insert");
        let scene = deckcraft_mesh::glb::read(&glb_of(&s, &models(&s)[1])).expect("glb");
        assert_eq!(scene.primitives[0].color, deckcraft_mesh::DEFAULT_COLOR);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn unreadable_files_are_refused_and_change_nothing() {
        let mut s = session();
        for (name, data) in [
            ("notes.txt", &b"hello"[..]),
            ("broken.obj", b"v 0 0 0\nf 1 2 3\n"),
            ("empty.stl", b"solid x\nendsolid x\n"),
            ("x.glb", b"glTF\x02\0\0\0"),
        ] {
            let e = s.execute("insert.model3d", &json!({"name": name, "data": base64_encode(data)})).expect_err(name);
            assert!(e.to_string().contains(name), "{name}: {e}");
        }
        assert!(s.execute("insert.model3d", &json!({})).is_err());
        assert!(models(&s).is_empty());
    }

    #[test]
    fn a_model_survives_a_pptx_round_trip() {
        let mut s = session();
        s.execute("insert.model3d", &json!({"name": "bracket.obj", "data": base64_encode(OBJ.as_bytes())})).expect("insert");
        let glb = glb_of(&s, &models(&s)[0]);
        let bytes = deckcraft_pptx::export(&s.doc().expect("doc").doc).expect("export");
        let back = deckcraft_pptx::import(&bytes).expect("import");
        let shapes: Vec<&Shape> = back.slides.iter().flat_map(|sl| &sl.shapes).filter(|x| matches!(x.kind, ShapeKind::Model3d { .. })).collect();
        assert_eq!(shapes.len(), 1);
        let ShapeKind::Model3d { media, .. } = shapes[0].kind else { return };
        assert_eq!(back.media(media).map(|m| m.data.to_vec()), Some(glb));
    }

    #[test]
    fn default_rect_keeps_odd_shapes_on_the_slide() {
        let b = |x: f32, y: f32| Some((x, y));
        for bounds in [b(100.0, 1.0), b(1.0, 100.0), b(1.0, 0.0), b(0.0, 0.0), b(f32::NAN, 1.0), None] {
            let r = default_rect(960.0, 540.0, bounds);
            assert!(r.x >= 0.0 && r.y >= 0.0 && r.x + r.w <= 960.0 && r.y + r.h <= 540.0 && r.w > 0.0 && r.h > 0.0, "{bounds:?}: {r:?}");
        }
    }

    #[test]
    fn poster_sizes_follow_the_frame() {
        assert_eq!(poster_size(260.0, 300.0), (520, 600));
        assert_eq!(poster_size(10.0, 5.0), (256, 128));
        assert_eq!(poster_size(2000.0, 1000.0), (1024, 512));
        assert_eq!(poster_size(f64::NAN, 0.0), (256, 256));
        assert_eq!(poster_size(1e6, 1e-6), (1024, 1));
    }

    fn pptx_parts(bytes: &[u8]) -> Vec<(String, Vec<u8>)> {
        use std::io::Read;
        let mut z = zip::ZipArchive::new(std::io::Cursor::new(bytes)).expect("zip");
        (0..z.len())
            .map(|i| {
                let mut f = z.by_index(i).expect("entry");
                let mut v = Vec::new();
                f.read_to_end(&mut v).expect("read");
                (f.name().to_string(), v)
            })
            .collect()
    }

    #[test]
    fn a_saved_pptx_carries_a_fallback_picture_of_the_model() {
        let mut s = session();
        s.execute("insert.model3d", &json!({"name": "bracket.obj", "data": base64_encode(OBJ.as_bytes()), "rect": [100, 100, 200, 100]}))
            .expect("insert");
        let doc = s.doc().expect("doc").doc.clone();
        let bytes = super::super::file::save_bytes(&doc, "pptx").expect("save");
        assert!(matches!(models(&s)[0].kind, ShapeKind::Model3d { poster: None, .. }), "the open presentation is left as it was");

        let parts = pptx_parts(&bytes);
        let slide =
            parts.iter().find(|(n, x)| n.starts_with("ppt/slides/slide") && String::from_utf8_lossy(x).contains("p:contentPart")).expect("slide");
        let xml = String::from_utf8_lossy(&slide.1);
        assert!(xml.contains("mc:AlternateContent") && xml.contains("mc:Fallback") && xml.contains("p:pic"), "{xml}");
        let png = parts.iter().find(|(n, _)| n.starts_with("ppt/media/") && n.ends_with(".png")).expect("poster part");
        let img = image::load_from_memory(&png.1).expect("png").to_rgba8();
        assert_eq!(img.dimensions(), (400, 200), "2 px per point, the frame's shape");
        assert_eq!(img.get_pixel(200, 100)[3], 255, "the model is in the middle");
        assert_eq!(img.get_pixel(2, 2)[3], 0, "around it the picture is see-through");

        // Opened again, the model and its picture come back; saving again adds no second picture.
        let back = deckcraft_pptx::import(&bytes).expect("import");
        let model = back.slides.iter().flat_map(|sl| &sl.shapes).find(|x| matches!(x.kind, ShapeKind::Model3d { .. })).expect("model");
        assert!(matches!(model.kind, ShapeKind::Model3d { poster: Some(_), .. }), "{:?}", model.kind);
        assert!(with_posters(&back).is_none());
    }

    #[test]
    fn models_in_groups_get_pictures_and_unreadable_ones_are_saved_without() {
        let mut s = session();
        s.execute("insert.model3d", &json!({"name": "a.stl", "data": base64_encode(STL.as_bytes())})).expect("a");
        s.execute("insert.textBox", &json!({"text": "label"})).expect("text");
        s.execute("edit.selectAll", &json!({})).expect("select");
        s.execute("arrange.group", &json!({})).expect("group");
        // A .glb whose mesh can't be read (compressed) is kept, drawn as a box, and saved bare.
        let json = br#"{"asset":{"version":"2.0"},"meshes":[{"primitives":[{"attributes":{},"extensions":{"KHR_draco_mesh_compression":{}}}]}]}  "#;
        let mut glb = b"glTF".to_vec();
        glb.extend(2u32.to_le_bytes());
        glb.extend(((12 + 8 + json.len()) as u32).to_le_bytes());
        glb.extend((json.len() as u32).to_le_bytes());
        glb.extend(0x4E4F_534Au32.to_le_bytes());
        glb.extend(json);
        s.execute("insert.model3d", &json!({"name": "draco.glb", "data": base64_encode(&glb)})).expect("draco");
        let out = with_posters(&s.doc().expect("doc").doc).expect("one picture made");
        let mut posters = Vec::new();
        for slide in &out.slides {
            fn walk(shapes: &[Shape], out: &mut Vec<bool>) {
                for x in shapes {
                    match &x.kind {
                        ShapeKind::Group { children, .. } => walk(children, out),
                        ShapeKind::Model3d { poster, .. } => out.push(poster.is_some()),
                        _ => {}
                    }
                }
            }
            walk(&slide.shapes, &mut posters);
        }
        posters.sort();
        assert_eq!(posters, vec![false, true], "the grouped STL has one, the unreadable model none");
        assert!(super::super::file::save_bytes(&s.doc().expect("doc").doc, "pptx").is_ok());
    }
}
