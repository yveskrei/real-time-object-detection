//! Responsible for pre/post processing images before inference.
//! Performs operations on raw frames/inference results with SIMD optimizations

use std::sync::OnceLock;
use tokio::time::Instant;
use serde::Serialize;

// Custom modules
pub mod yolo;
pub mod dinov2;

/// Represents raw frame before performing inference on it
#[derive(Clone, Debug)]
pub struct RawFrame {
    pub data: Vec<u8>,
    pub height: usize,
    pub width: usize,
    pub added: Instant
}

/// Represents a single bbox instance from the model inference results
#[derive(Clone, Copy, Serialize)]
pub struct ResultBBOX {
    pub bbox: [f32; 4],
    pub class: usize, 
    pub score: f32
}

impl ResultBBOX {
    pub fn class_name(&self) -> &'static str {
        match self.class {
            0 => "person",
            1 => "bicycle",
            2 => "car",
            3 => "motorcycle",
            4 => "airplane",
            5 => "bus",
            _ => Box::leak(self.class.to_string().into_boxed_str())
        }
    }
}

/// Represents embedding output from the model inference results
#[derive(Clone, Serialize)]
pub struct ResultEmbedding {
    pub data: Vec<f32>
}

/// Lookup table for converting values from FP16 to FP32
pub static F16_TO_F32_LUT: OnceLock<Box<[f32; 65536]>> = OnceLock::new();
/// Lookup table for F32 to F16 conversion
pub static F32_TO_F16_LUT: OnceLock<Box<[u16; 32768]>> = OnceLock::new();
/// Lookup table for converting pixel values to FP16
pub static F16_LUT: OnceLock<Box<[u16; 256]>> = OnceLock::new();
/// Lookup table for converting pixel values to FP32
pub static F32_LUT: OnceLock<Box<[f32; 256]>> = OnceLock::new();

/// Create static lookup table for high speed conversion
fn create_f16_to_f32_lut() -> Box<[f32; 65536]> {
    let mut lut = Box::new([0.0f32; 65536]);
        
    for i in 0u16..=65535 {
        let sign = (i >> 15) & 0x1;
        let exp = (i >> 10) & 0x1f;
        let frac = i & 0x3ff;
        
        lut[i as usize] = if exp == 0 {
            if frac == 0 {
                if sign == 1 { -0.0 } else { 0.0 }
            } else {
                // Denormal
                let mut val = frac as f32 / 1024.0 / 16384.0;
                if sign == 1 { val = -val; }
                val
            }
        } else if exp == 31 {
            // Infinity or NaN
            if frac == 0 {
                if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY }
            } else {
                f32::NAN
            }
        } else {
            // Normal numbers
            let exp_f32 = (exp as i32 - 15 + 127) as u32;
            let frac_f32 = (frac as u32) << 13;
            let bits = (sign as u32) << 31 | exp_f32 << 23 | frac_f32;
            f32::from_bits(bits)
        };
    }
    
    lut
}

pub fn get_f16_to_f32_lut(val: u16) -> f32 {
    F16_TO_F32_LUT
        .get_or_init(create_f16_to_f32_lut)[val as usize]
}

/// Create static lookup table for F32 to F16 conversion
fn create_f32_to_f16_lut() -> Box<[u16; 32768]> {
    let mut lut = Box::new([0u16; 32768]);
    
    const MIN_VAL: f32 = -4.0;
    const MAX_VAL: f32 = 4.0;
    const RANGE: f32 = MAX_VAL - MIN_VAL;
    const STEP: f32 = RANGE / 32768.0;
    
    for i in 0..32768 {
        let val = MIN_VAL + (i as f32) * STEP;
        let bits = val.to_bits();
        let sign = (bits >> 16) & 0x8000;
        let exp = ((bits >> 23) & 0xff) as i32;
        let mantissa = bits & 0x7fffff;
        
        lut[i] = if exp == 0 {
            sign as u16
        } else {
            let exp_adj = exp - 127 + 15;
            if exp_adj >= 31 {
                (sign | 0x7c00) as u16
            } else if exp_adj <= 0 {
                sign as u16
            } else {
                let mantissa_adj = mantissa >> 13;
                (sign | ((exp_adj as u32) << 10) | mantissa_adj) as u16
            }
        };
    }
    
    lut
}

fn get_f32_to_f16_lut(val: f32) -> u16 {
    const MIN_VAL: f32 = -4.0;
    const MAX_VAL: f32 = 4.0;
    const RANGE: f32 = MAX_VAL - MIN_VAL;
    
    let clamped_val = val.clamp(MIN_VAL, MAX_VAL);
    let index = ((clamped_val - MIN_VAL) / RANGE * 32767.0) as usize;
    let index = index.min(32767);
    
    F32_TO_F16_LUT
        .get_or_init(create_f32_to_f16_lut)[index]
}

/// Create static lookup table for high speed conversion
fn create_f16_lut() -> Box<[u16; 256]> {
    let mut lut = Box::new([0u16; 256]);
    for i in 0..256 {
        let normalized = i as f32 / 255.0;
        let bits = normalized.to_bits();
        let sign = (bits >> 16) & 0x8000;
        let exp = ((bits >> 23) & 0xff) as i32;
        let mantissa = bits & 0x7fffff;
        lut[i] = if exp == 0 {
            sign as u16
        } else {
            let exp_adj = exp - 127 + 15;
            if exp_adj >= 31 {
                (sign | 0x7c00) as u16
            } else if exp_adj <= 0 {
                sign as u16
            } else {
                let mantissa_adj = mantissa >> 13;
                (sign | ((exp_adj as u32) << 10) | mantissa_adj) as u16
            }
        };
    }
    lut
}

pub fn get_f16_lut() -> &'static [u16; 256] {
    F16_LUT
        .get_or_init(create_f16_lut)
}

/// Create static lookup table for high speed conversion
fn create_f32_lut() -> Box<[f32; 256]> {
    let mut lut = Box::new([0.0f32; 256]);
    for i in 0..256 {
        lut[i] = i as f32 / 255.0;
    }
    lut
}

pub fn get_f32_lut() -> &'static [f32; 256] {
    F32_LUT
        .get_or_init(create_f32_lut)
}