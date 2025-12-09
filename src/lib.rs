#![allow(clippy::too_many_arguments)]

use crate::render::voxel;
use std::collections::VecDeque;
use winit::{
    event::{DeviceEvent, WindowEvent},
    keyboard::KeyCode,
};

mod camera;
pub mod platform;
mod render;

#[derive(Default)]
pub struct Memory {
    world: Option<World>,
}

struct World {
    camera: camera::Camera,
    voxel_pipeline: voxel::VoxelPipeline,
    sliding_fps: VecDeque<f32>,
}

#[unsafe(no_mangle)]
pub fn handle_input(
    platform::PlatformInput { memory, input, .. }: platform::PlatformInput<Memory>,
) {
    let Some(world) = &mut memory.world else {
        return;
    };

    match input {
        platform::Input::Window(event) => {
            if let WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        physical_key: winit::keyboard::PhysicalKey::Code(key),
                        state,
                        ..
                    },
                ..
            } = event
            {
                if key == KeyCode::Escape {
                    std::process::exit(0);
                }
                world.camera.handle_key(key, state);
            }
        }
        platform::Input::Device(event) => {
            if let DeviceEvent::MouseMotion { delta } = event {
                world.camera.handle_mouse(delta.0 as f32, delta.1 as f32);
            }
        }
    }
}

pub fn update_and_render(
    platform::PlatformUpdate {
        memory,
        device,
        queue,
        surface_texture,
        surface_format,
        window,
        delta,
        width,
        height,
        ..
    }: platform::PlatformUpdate<Memory>,
) {
    let world = memory.world.get_or_insert_with(|| World {
        camera: camera::Camera {
            translation: glam::Vec3::new(-10.0, 130.0, -10.0),
            pitch: -0.80,
            yaw: -2.1,
            fov: 90f32.to_radians(),
            znear: 0.01,
            zfar: 1000.0,
            speed: 100.0,
            ..Default::default()
        },
        voxel_pipeline: voxel::VoxelPipeline::new(device, surface_format, width, height),
        sliding_fps: VecDeque::with_capacity(100),
    });

    if world.sliding_fps.len() >= 100 {
        world.sliding_fps.pop_front();
    }
    world.sliding_fps.push_back(1.0 / delta);
    window.set_title(&format!(
        "CUBE - {:.2}",
        world.sliding_fps.iter().sum::<f32>() / world.sliding_fps.len() as f32
    ));

    world.camera.update(delta);

    let view = surface_texture
        .texture
        .create_view(&wgpu::TextureViewDescriptor {
            format: Some(surface_format.add_srgb_suffix()),
            ..Default::default()
        });

    let mut encoder = device.create_command_encoder(&Default::default());
    world.voxel_pipeline.render(
        device,
        queue,
        &mut encoder,
        &view,
        &world.camera,
        width,
        height,
    );
    queue.submit([encoder.finish()]);
}
