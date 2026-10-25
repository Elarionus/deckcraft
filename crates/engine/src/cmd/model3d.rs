//! Insert ▸ 3D Models: STL, OBJ (with its `.mtl` colours), FBX and glTF binary files, embedded as
//! `.glb` the way PowerPoint keeps them, so a saved `.pptx` carries a model PowerPoint can open.

use std::path::{Component, Path};

use deckcraft_geom::Xfrm;
use deckcraft_mesh::{Bounds, Format};
use deckcraft_model::{Shape, ShapeKind};
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

/// The model's front view (X across, Y up) at half the slide's height, kept inside the slide
/// and centred.
fn default_rect(slide_w: f64, slide_h: f64, bounds: Option<Bounds>) -> Xfrm {
    let aspect = bounds
        .map(|b| {
            let s = b.size();
            s[0] as f64 / s[1] as f64
        })
        .filter(|a| a.is_finite() && *a > 0.0)
        .unwrap_or(1.0)
        .clamp(0.25, 4.0);
    let (mut w, mut h) = (slide_h * 0.5 * aspect, slide_h * 0.5);
    if w > slide_w * 0.9 {
        w = slide_w * 0.9;
        h = w / aspect;
    }
    Xfrm::new((slide_w - w) / 2.0, (slide_h - h) / 2.0, w, h)
}

fn insert_model(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "insert.model3d";
    let (name, bytes) = media_bytes(p, CMD)?;
    let imported = deckcraft_mesh::import(&name, &bytes, &mut sidecar_reader(str_param(p, "path"))).map_err(|e| bad(CMD, format!("{name}: {e}")))?;
    let size = s.doc()?.doc.slide_size;
    let rect = rect_param(p, "rect").unwrap_or_else(|| default_rect(size.width, size.height, imported.bounds));
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
        assert!((x.w / x.h - 2.0).abs() < 1e-9, "2 × 1 front view: {x:?}");
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
        let b = |x: f32, y: f32| Some(Bounds { min: [0.0; 3], max: [x, y, 1.0] });
        for bounds in [b(100.0, 1.0), b(1.0, 100.0), b(1.0, 0.0), b(0.0, 0.0), None] {
            let r = default_rect(960.0, 540.0, bounds);
            assert!(r.x >= 0.0 && r.y >= 0.0 && r.x + r.w <= 960.0 && r.y + r.h <= 540.0 && r.w > 0.0 && r.h > 0.0, "{bounds:?}: {r:?}");
        }
    }
}
