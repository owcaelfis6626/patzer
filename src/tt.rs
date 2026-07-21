//! Transposition table. Fixed-size, power-of-two buckets, replace on (age differs | deeper | same key).
//! Moves are stored packed (from 6 bits | to 6 bits | promo 3 bits); they are only ever *compared*
//! against packed generated moves, never unpacked blind — so a corrupted move can cause a missed
//! ordering hint, never an illegal move.

use cozy_chess::{Move, Piece};

pub const BOUND_NONE: u8 = 0;
pub const BOUND_EXACT: u8 = 1;
pub const BOUND_LOWER: u8 = 2;
pub const BOUND_UPPER: u8 = 3;

pub fn pack(mv: Move) -> u16 {
    let promo = match mv.promotion {
        None => 0u16,
        Some(Piece::Knight) => 1,
        Some(Piece::Bishop) => 2,
        Some(Piece::Rook) => 3,
        Some(Piece::Queen) => 4,
        Some(_) => 0,
    };
    (mv.from as u16) | ((mv.to as u16) << 6) | (promo << 12)
}

#[derive(Clone, Copy, Default)]
pub struct Entry {
    pub key: u64,
    pub mv: u16,
    pub score: i16,
    pub depth: i8,
    pub bound: u8,
    pub age: u8,
}

pub struct TT {
    data: Vec<Entry>,
    mask: usize,
    pub age: u8,
}

impl TT {
    pub fn new(mb: usize) -> Self {
        let n = ((mb.max(1) << 20) / std::mem::size_of::<Entry>()).next_power_of_two() >> 1;
        TT {
            data: vec![Entry::default(); n.max(1024)],
            mask: n.max(1024) - 1,
            age: 0,
        }
    }

    pub fn clear(&mut self) {
        self.data.iter_mut().for_each(|e| *e = Entry::default());
        self.age = 0;
    }

    pub fn new_search(&mut self) {
        self.age = self.age.wrapping_add(1);
    }

    pub fn probe(&self, key: u64) -> Option<Entry> {
        let e = self.data[(key as usize) & self.mask];
        if e.key == key && e.bound != BOUND_NONE {
            Some(e)
        } else {
            None
        }
    }

    pub fn store(&mut self, key: u64, mv: u16, score: i32, depth: i32, bound: u8) {
        let idx = (key as usize) & self.mask;
        let e = &mut self.data[idx];
        if e.key != key || e.age != self.age || depth >= e.depth as i32 || bound == BOUND_EXACT {
            // keep an existing move hint if the new one is null and the key matches
            let mv = if mv == 0 && e.key == key { e.mv } else { mv };
            *e = Entry {
                key,
                mv,
                score: score.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                depth: depth.clamp(-128, 127) as i8,
                bound,
                age: self.age,
            };
        }
    }
}
