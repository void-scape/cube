use crate::{
    camera::Camera,
    platform::debug,
    render::{
        byte_slice,
        voxel::{VOXEL_FACES, VOXEL_FACES_INDICES, VoxelVertex},
    },
};
use glam::{FloatExt, Vec2};
use std::collections::{HashMap, HashSet};
use wgpu::util::DeviceExt;

pub const CHUNK_SIZE: usize = 32;

#[derive(Default)]
pub struct Chunks(pub HashMap<(i64, i64), Chunk>);

impl Chunks {
    pub fn update(&mut self, device: &wgpu::Device, camera: &Camera) {
        let view_distance = 4;

        let (x, z) = (
            camera.translation.x as i64 / CHUNK_SIZE as i64,
            camera.translation.z as i64 / CHUNK_SIZE as i64,
        );
        let zrange = z - view_distance..=z + view_distance;
        let xrange = x - view_distance..=x + view_distance;

        self.0
            .retain(|(x, z), _| xrange.contains(x) && zrange.contains(z));

        let (dur, out) = debug::debug_time_millis(|| {
            std::thread::scope(|s| {
                let count = 2;
                let mut handles = Vec::with_capacity(count);
                'outer: for z in zrange.clone() {
                    for x in xrange.clone() {
                        if !self.0.contains_key(&(x, z)) {
                            handles.push(s.spawn(move || ((x, z), generate_voxels(x, z))));
                            // limit to 3 chunks a frame
                            if handles.len() >= count {
                                break 'outer;
                            }
                        }
                    }
                }

                let generated = handles.len();
                for handle in handles.into_iter() {
                    let ((x, z), (vertices, indices)) = handle.join().unwrap();
                    self.0
                        .insert((x, z), Chunk::new(device, &vertices, &indices));
                }

                generated
            })
        });
        if out > 0 {
            println!("generated {out} chunks in {dur}ms");
        }
    }
}

pub struct Chunk {
    pub vertices: wgpu::Buffer,
    pub indices: wgpu::Buffer,
    pub indices_count: u32,
}

impl Chunk {
    fn new(device: &wgpu::Device, vertices: &[VoxelVertex], indices: &[u32]) -> Self {
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk vertex buffer"),
            contents: byte_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let indices_count = indices.len() as u32;
        let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk index buffer"),
            contents: byte_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            vertices,
            indices,
            indices_count,
        }
    }
}

fn generate_voxels(chunk_x: i64, chunk_z: i64) -> (Vec<VoxelVertex>, Vec<u32>) {
    let perlin_scale = 200;
    let noise_layers = [(1.5, 80.0), (3.0, 40.0), (8.0, 30.0)];

    let mut translations = Vec::new();

    let chunk_x = chunk_x as f32 * CHUNK_SIZE as f32;
    let chunk_z = chunk_z as f32 * CHUNK_SIZE as f32;

    for z in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            let uv = Vec2::new(
                (x as f32 + chunk_x) / perlin_scale as f32,
                (z as f32 + chunk_z) / perlin_scale as f32,
            );

            let mut surface = 0.0;
            for (uv_scale, weight) in noise_layers.iter() {
                surface += (perlin(uv * *uv_scale) * 0.5 + 0.5) * weight;
            }

            for y in 0..surface.round() as usize {
                translations.push((x as i32, y as i32, z as i32));
            }
        }
    }

    let hash: HashSet<(i32, i32, i32)> = translations.iter().copied().collect();
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let neighbors = [
        (1, 0, 0, 3),
        (-1, 0, 0, 2),
        (0, 1, 0, 5),
        (0, -1, 0, 4),
        (0, 0, 1, 1),
        (0, 0, -1, 0),
    ];

    for (x, y, z) in translations.iter() {
        for (dx, dy, dz, face_idx) in neighbors.iter() {
            let neighbor_pos = (x + dx, y + dy, z + dz);

            if !hash.contains(&neighbor_pos) {
                let vertex_offset = vertices.len() as u32;
                for mut vertex in VOXEL_FACES[*face_idx].iter().cloned() {
                    debug_assert!(*x >= 0);
                    debug_assert!(*y >= 0);
                    debug_assert!(*z >= 0);
                    vertex.offset(glam::UVec3::new(*x as u32, *y as u32, *z as u32));
                    if *y > 50 {
                        vertex.set_color(1);
                    }
                    vertices.push(vertex);
                }
                for index in VOXEL_FACES_INDICES[*face_idx].iter() {
                    indices.push(vertex_offset + index);
                }
            }
        }
    }

    (vertices, indices)
}

// https://thebookofshaders.com/edit.php#11/2d-gnoise.frag
fn perlin(st: Vec2) -> f32 {
    fn random2(st: Vec2) -> Vec2 {
        fn random(v: f32) -> f32 {
            -1.0 + 2.0 * (v.sin() * 43758.547).fract()
        }

        let st = Vec2::new(
            st.dot(Vec2::new(127.1, 311.7)),
            st.dot(Vec2::new(269.5, 183.3)),
        );
        Vec2::new(random(st.x), random(st.y))
    }

    let i = st.floor();
    let f = st - i;
    let u = f * f * (3.0 - 2.0 * f);

    let left = random2(i)
        .dot(f)
        .lerp(random2(i + Vec2::X).dot(f - Vec2::X), u.x);
    let right = random2(i + Vec2::Y)
        .dot(f - Vec2::Y)
        .lerp(random2(i + Vec2::ONE).dot(f - Vec2::ONE), u.x);
    left.lerp(right, u.y)
}
