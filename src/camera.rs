use glam::{Mat4, Vec3};
use std::f32::consts::FRAC_PI_2;
use winit::{event::ElementState, keyboard::KeyCode};

#[derive(Default)]
pub struct Camera {
    pub translation: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub fov: f32,
    pub znear: f32,
    pub zfar: f32,
    pub speed: f32,
    pub left: bool,
    pub right: bool,
    pub forward: bool,
    pub back: bool,
    pub up: bool,
    pub down: bool,
    pub disabled: bool,
    pub proj: bool,
}

impl Camera {
    pub fn handle_key(&mut self, key: KeyCode, state: ElementState) {
        match key {
            KeyCode::KeyE => {
                if state.is_pressed() {
                    self.disabled = !self.disabled;
                }
            }
            KeyCode::KeyO => {
                if state.is_pressed() {
                    self.proj = !self.proj;
                }
            }
            KeyCode::KeyA => {
                self.left = state.is_pressed();
            }
            KeyCode::KeyD => {
                self.right = state.is_pressed();
            }
            KeyCode::KeyW => {
                self.forward = state.is_pressed();
            }
            KeyCode::KeyS => {
                self.back = state.is_pressed();
            }
            KeyCode::Space | KeyCode::ControlLeft => {
                self.up = state.is_pressed();
            }
            KeyCode::ShiftLeft => {
                self.down = state.is_pressed();
            }
            _ => {}
        }
    }

    pub fn handle_mouse(&mut self, dx: f32, dy: f32) {
        if !self.disabled {
            let sensitivity = 0.005;

            self.yaw += dx * sensitivity;
            self.pitch += -dy * sensitivity;

            if self.pitch < -FRAC_PI_2 + 0.001 {
                self.pitch = -FRAC_PI_2 + 0.001;
            } else if self.pitch > FRAC_PI_2 + 0.001 {
                self.pitch = FRAC_PI_2 + 0.001;
            }
        }
    }

    pub fn update(&mut self, dt: f32) {
        if !self.disabled {
            let (yaw_sin, yaw_cos) = self.yaw.sin_cos();
            let forward = Vec3::new(yaw_cos, 0.0, yaw_sin).normalize();
            let right = Vec3::new(-yaw_sin, 0.0, yaw_cos).normalize();
            let mut dxz = Vec3::ZERO;
            dxz += forward * (self.forward as u32 as f32 - self.back as u32 as f32);
            dxz += right * (self.right as u32 as f32 - self.left as u32 as f32);
            self.translation += dxz.normalize_or_zero() * self.speed * dt;

            if self.down {
                self.translation.y -= self.speed * dt;
            }
            if self.up {
                self.translation.y += self.speed * dt;
            }
        }
    }

    pub fn view_matrix(&self) -> Mat4 {
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        Mat4::look_to_rh(
            self.translation,
            Vec3::new(cos_pitch * cos_yaw, sin_pitch, cos_pitch * sin_yaw).normalize_or(Vec3::Z),
            Vec3::Y,
        )
    }

    pub fn projection_matrix(&self, width: u32, height: u32) -> Mat4 {
        if !self.proj {
            let scale = 0.1;
            let w = width as f32 / 2.0 * scale;
            let h = height as f32 / 2.0 * scale;
            Mat4::orthographic_rh(-w, w, -h, h, self.znear, self.zfar)
        } else {
            Mat4::perspective_rh(
                self.fov,
                width as f32 / height as f32,
                self.znear,
                self.zfar,
            )
        }
    }
}
