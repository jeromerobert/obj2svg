use clap::Parser;
use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};
use svg::{
    Document,
    node::element::{Path, path::Data},
};

type Position2D = (f32, f32);
type Edge = (usize, usize);
type Positions = Vec<Position2D>;
type Edges = Vec<Edge>;

#[derive(Parser)]
#[command(
    name = "obj2svg",
    version,
    about = "Converts OBJ line elements (l v1 v2) into connected SVG polylines (ignoring Z)."
)]
struct Cli {
    /// Path to input .obj file
    input: PathBuf,

    /// Path to output .svg file
    output: PathBuf,

    /// Stroke color for SVG paths
    #[arg(short, long, default_value = "black")]
    color: String,

    /// Stroke width (auto-calculated if omitted)
    #[arg(short, long)]
    stroke_width: Option<f32>,
}

fn load_obj_geometry(input_path: &PathBuf) -> Result<(Positions, Edges), Box<dyn std::error::Error>> {
    let load_options = tobj::LoadOptions {
        triangulate: false,
        ignore_lines: false,
        ..Default::default()
    };

    let (models, _) = tobj::load_obj(input_path, &load_options)?;

    if models.is_empty() {
        eprintln!("Warning: No models found in the OBJ file.");
        return Ok((Vec::new(), Vec::new()));
    }

    let mut raw_positions: Positions = Vec::new();
    let mut raw_edges: Edges = Vec::new();
    let mut vertex_offset = 0;

    for model in &models {
        let mesh = &model.mesh;
        let num_verts = mesh.positions.len() / 3;

        // Extract 2D points (ignoring Z; invert Y for SVG top-down coordinates)
        for i in 0..num_verts {
            let x = mesh.positions[3 * i];
            let y = -mesh.positions[3 * i + 1];
            raw_positions.push((x, y));
        }

        // Read edges where face_arities == 2 (line segments)
        let mut next_face = 0;
        for &arity in &mesh.face_arities {
            let end = next_face + arity as usize;
            if arity == 2 {
                let u = mesh.indices[next_face] as usize;
                let v = mesh.indices[next_face + 1] as usize;
                raw_edges.push((vertex_offset + u, vertex_offset + v));
            } else if arity > 2 {
                // If a line element specifies a polyline (l v1 v2 v3 ...)
                let slice = &mesh.indices[next_face..end];
                for w in slice.windows(2) {
                    raw_edges.push((vertex_offset + w[0] as usize, vertex_offset + w[1] as usize));
                }
            }
            next_face = end;
        }

        vertex_offset += num_verts;
    }

    Ok((raw_positions, raw_edges))
}

fn canonicalize_and_deduplicate(
    raw_positions: &Positions,
    raw_edges: Edges,
) -> (Positions, Edges) {
    // Canonicalize 2D coordinates to automatically weld identical endpoints (5 decimal precision)
    let mut coord_map = HashMap::new();
    let mut canonical_id = Vec::with_capacity(raw_positions.len());
    let mut canonical_positions = Vec::new();

    for &(x, y) in raw_positions {
        let key = (
            (x * 100_000.0).round() as i64,
            (y * 100_000.0).round() as i64,
        );
        let id = *coord_map.entry(key).or_insert_with(|| {
            let new_id = canonical_positions.len();
            canonical_positions.push((x, y));
            new_id
        });
        canonical_id.push(id);
    }

    // Deduplicate undirected edges
    let mut edge_set = HashSet::new();
    let mut edges = Vec::new();

    for (u, v) in raw_edges {
        let u_c = canonical_id[u];
        let v_c = canonical_id[v];
        if u_c != v_c {
            let edge = if u_c < v_c { (u_c, v_c) } else { (v_c, u_c) };
            if edge_set.insert(edge) {
                edges.push((u_c, v_c));
            }
        }
    }

    (canonical_positions, edges)
}

fn build_adjacency_graph(edges: &[Edge]) -> HashMap<usize, Vec<Edge>> {
    let mut adj: HashMap<usize, Vec<Edge>> = HashMap::new();
    for (edge_idx, &(u, v)) in edges.iter().enumerate() {
        adj.entry(u).or_default().push((v, edge_idx));
        adj.entry(v).or_default().push((u, edge_idx));
    }
    adj
}

fn unvisited_degree(
    node: usize,
    adj: &HashMap<usize, Vec<Edge>>,
    visited: &[bool],
) -> usize {
    adj.get(&node).map_or(0, |neighbors| {
        neighbors
            .iter()
            .filter(|(_, e_idx)| !visited[*e_idx])
            .count()
    })
}

fn extract_polylines(
    adj: &HashMap<usize, Vec<Edge>>,
    edges: &[Edge],
) -> Vec<Vec<usize>> {
    let mut edge_visited = vec![false; edges.len()];
    let mut polylines: Vec<Vec<usize>> = Vec::new();
    let nodes: Vec<usize> = adj.keys().copied().collect();

    // Pass 1: Walk paths starting from odd-degree vertices (endpoints of open lines)
    for &start_node in &nodes {
        while unvisited_degree(start_node, adj, &edge_visited) % 2 == 1 {
            let path = walk_path(start_node, adj, &mut edge_visited);
            if path.len() > 1 {
                polylines.push(path);
            }
        }
    }

    // Pass 2: Walk remaining edges (closed cycles/loops where all degrees are even)
    for &start_node in &nodes {
        while unvisited_degree(start_node, adj, &edge_visited) > 0 {
            let path = walk_path(start_node, adj, &mut edge_visited);
            if path.len() > 1 {
                polylines.push(path);
            }
        }
    }

    polylines
}

fn walk_path(
    start_node: usize,
    adj: &HashMap<usize, Vec<Edge>>,
    edge_visited: &mut [bool],
) -> Vec<usize> {
    let mut path = vec![start_node];
    let mut curr = start_node;

    while let Some(&(next, e_idx)) = adj
        .get(&curr)
        .and_then(|neighbors| neighbors.iter().find(|(_, e_idx)| !edge_visited[*e_idx]))
    {
        edge_visited[e_idx] = true;
        path.push(next);
        curr = next;
    }

    path
}

fn compute_bounding_box(
    adj: &HashMap<usize, Vec<Edge>>,
    canonical_positions: &[Position2D],
) -> (f32, f32, f32, f32) {
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;

    for &node in adj.keys() {
        let (x, y) = canonical_positions[node];
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }

    (min_x, min_y, max_x, max_y)
}

fn create_svg_document(
    polylines: &[Vec<usize>],
    canonical_positions: &[Position2D],
    color: &str,
    stroke_width: f32,
    view_box: (f32, f32, f32, f32),
) -> Document {
    let mut document = Document::new()
        .set("viewBox", view_box)
        .set("xmlns", "http://www.w3.org/2000/svg");

    // Add each connected polyline as an SVG <path>
    for polyline in polylines {
        let first = polyline[0];
        let is_closed = polyline.len() > 2 && polyline.first() == polyline.last();

        let mut data = Data::new().move_to(canonical_positions[first]);

        if is_closed {
            for &node in &polyline[1..polyline.len() - 1] {
                data = data.line_to(canonical_positions[node]);
            }
            data = data.close();
        } else {
            for &node in &polyline[1..] {
                data = data.line_to(canonical_positions[node]);
            }
        }

        let path = Path::new()
            .set("fill", "none")
            .set("stroke", color)
            .set("stroke-width", stroke_width)
            .set("stroke-linecap", "round")
            .set("stroke-linejoin", "round")
            .set("d", data);

        document = document.add(path);
    }

    document
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let (raw_positions, raw_edges) = load_obj_geometry(&cli.input)?;

    if raw_edges.is_empty() {
        eprintln!("Warning: No line elements (face_arities == 2) found in the OBJ file.");
        return Ok(());
    }

    let (canonical_positions, edges) = canonicalize_and_deduplicate(&raw_positions, raw_edges);
    let adj = build_adjacency_graph(&edges);
    let polylines = extract_polylines(&adj, &edges);

    // Compute bounding box
    let (min_x, min_y, max_x, max_y) = compute_bounding_box(&adj, &canonical_positions);

    let width = (max_x - min_x).max(1.0);
    let height = (max_y - min_y).max(1.0);
    let pad_x = width * 0.05;
    let pad_y = height * 0.05;

    let view_box = (
        min_x - pad_x,
        min_y - pad_y,
        f32::mul_add(pad_x, 2.0, width),
        f32::mul_add(pad_y, 2.0, height),
    );

    let stroke_width = cli
        .stroke_width
        .unwrap_or_else(|| (width.max(height) * 0.002).max(0.0001));

    let document = create_svg_document(&polylines, &canonical_positions, &cli.color, stroke_width, view_box);

    svg::save(&cli.output, &document)?;
    println!(
        "Extracted {} connected polylines into {}",
        polylines.len(),
        cli.output.display()
    );

    Ok(())
}
