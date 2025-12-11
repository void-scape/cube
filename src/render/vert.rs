use glam::IVec3;

// Pre-shrinkification process memory usage sits at ~39MB.
// valgrind sits at roughly 13,626,576b heap allocation.
//
// Pose-shrinkification process memory sits at ~35.5MB.
// valgrind sits at roughly 27,962,600b heap allocation.
//
// I don't think I am using valgrind correctly lol.
//
// Seems to almost double performance? The fps measurement is
// not very scientific, but it seems like it went from ~450 to ~750.
//
// TODO: Color needs to be changed to a more generic voxel id that stores all
// related voxel information in the wgsl shader. Maybe this information should
// be stored in a uniform so that the logic could be centralized?
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VoxelVertex {
    /// voxel_id | normal_index | z | y | x
    /// ---------+--------------+---+---+---
    /// 11       | 3            | 6 | 6 | 6
    packed: u32,
}

impl VoxelVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Uint32];

    const MAX_XYZ: u32 = 63;
    const Y_OFFSET: u32 = 6;
    const Z_OFFSET: u32 = 12;
    const NORMAL_OFFSET: u32 = 18;
    const MAX_ID: u32 = 3;
    const ID_OFFSET: u32 = 21;
    const ID_MASK: u32 = 0xff400000;

    pub const fn new(x: u32, y: u32, z: u32, normal: IVec3) -> Self {
        debug_assert!(x <= Self::MAX_XYZ);
        debug_assert!(y <= Self::MAX_XYZ);
        debug_assert!(z <= Self::MAX_XYZ);

        let mut packed = 0;
        packed |= x;
        packed |= y << Self::Y_OFFSET;
        packed |= z << Self::Z_OFFSET;

        let normal = match normal {
            IVec3::X => 0,
            IVec3::NEG_X => 1,
            IVec3::Y => 2,
            IVec3::NEG_Y => 3,
            IVec3::Z => 4,
            IVec3::NEG_Z => 5,
            _ => unreachable!(),
        };
        packed |= normal << Self::NORMAL_OFFSET;

        Self { packed }
    }

    pub fn id(&mut self, id: u32) {
        debug_assert!(id <= Self::MAX_ID);
        let packed = self.packed & !Self::ID_MASK;
        self.packed = packed | (id << Self::ID_OFFSET);
    }

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<VoxelVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

// TODO: Voxel vertices should store a size value for x, y, and z to represent how
// scaled they are, instead of specifying here.
pub const VOXEL_FACES: [[VoxelVertex; 4]; 6] = [
    // Back face
    [
        VoxelVertex::new(1, 0, 0, IVec3::NEG_Z),
        VoxelVertex::new(1, 1, 0, IVec3::NEG_Z),
        VoxelVertex::new(0, 0, 0, IVec3::NEG_Z),
        VoxelVertex::new(0, 1, 0, IVec3::NEG_Z),
    ],
    // Front face
    [
        VoxelVertex::new(1, 0, 1, IVec3::Z),
        VoxelVertex::new(1, 1, 1, IVec3::Z),
        VoxelVertex::new(0, 1, 1, IVec3::Z),
        VoxelVertex::new(0, 0, 1, IVec3::Z),
    ],
    // Left face
    [
        VoxelVertex::new(0, 1, 0, IVec3::NEG_X),
        VoxelVertex::new(0, 0, 0, IVec3::NEG_X),
        VoxelVertex::new(0, 0, 1, IVec3::NEG_X),
        VoxelVertex::new(0, 1, 1, IVec3::NEG_X),
    ],
    // Right face
    [
        VoxelVertex::new(1, 1, 0, IVec3::X),
        VoxelVertex::new(1, 0, 0, IVec3::X),
        VoxelVertex::new(1, 1, 1, IVec3::X),
        VoxelVertex::new(1, 0, 1, IVec3::X),
    ],
    // Bottom face
    [
        VoxelVertex::new(1, 0, 0, IVec3::NEG_Y),
        VoxelVertex::new(1, 0, 1, IVec3::NEG_Y),
        VoxelVertex::new(0, 0, 1, IVec3::NEG_Y),
        VoxelVertex::new(0, 0, 0, IVec3::NEG_Y),
    ],
    // Top face
    [
        VoxelVertex::new(1, 1, 0, IVec3::Y),
        VoxelVertex::new(1, 1, 1, IVec3::Y),
        VoxelVertex::new(0, 1, 0, IVec3::Y),
        VoxelVertex::new(0, 1, 1, IVec3::Y),
    ],
];

pub const VOXEL_FACES_INDICES: [[u32; 6]; 6] = [
    // Back face
    [1, 0, 2, 1, 2, 3],
    // Front face
    [0, 1, 2, 0, 2, 3],
    // Left face
    [0, 1, 2, 0, 2, 3],
    // Right face
    [0, 2, 1, 2, 3, 1],
    // Bottom face
    [0, 1, 2, 0, 2, 3],
    // Top face
    [0, 2, 1, 2, 3, 1],
];
