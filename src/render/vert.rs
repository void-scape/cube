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
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VoxelVertex {
    /// color_index | normal_index | z | y | x
    /// ---------------------------------------
    /// 3             3              6   8   6
    packed: u32,
}

impl VoxelVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Uint32];

    const MAX_XZ: u32 = 63;
    const MAX_Y: u32 = 255;
    const Y_OFFSET: u32 = 6;
    const Z_OFFSET: u32 = 14;

    const NORMAL_OFFSET: u32 = 20;

    const MAX_COLOR_INDEX: u32 = 3;
    const COLOR_OFFSET: u32 = 23;
    const COLOR_MASK: u32 = 0x1800000;

    pub const fn new(x: u32, y: u32, z: u32, normal: IVec3) -> Self {
        debug_assert!(x <= Self::MAX_XZ);
        debug_assert!(y <= Self::MAX_Y);
        debug_assert!(z <= Self::MAX_XZ);

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

    pub fn offset(&mut self, x: u32, y: u32, z: u32) {
        let packed = self.packed;
        let x = (packed & 0x3f) + x;
        let y = ((packed >> Self::Y_OFFSET) & 0xff) + y;
        let z = ((packed >> Self::Z_OFFSET) & 0x3f) + z;

        debug_assert!(x <= Self::MAX_XZ);
        debug_assert!(y <= Self::MAX_Y);
        debug_assert!(z <= Self::MAX_XZ);

        let remaining = packed & !0xFFFFF;
        self.packed = remaining
            | (x & 0x3f)
            | ((y & 0xff) << Self::Y_OFFSET)
            | ((z & 0x3f) << Self::Z_OFFSET);
    }

    pub fn set_color(&mut self, color: u32) {
        debug_assert!(color <= Self::MAX_COLOR_INDEX);
        let packed = self.packed & !Self::COLOR_MASK;
        self.packed = packed | (color << Self::COLOR_OFFSET);
    }

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<VoxelVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

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
