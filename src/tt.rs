//! Transposition table. Lock-free: two atomic words per slot (key^data, data), Relaxed
//! ordering — the standard technique for a table probed/stored by multiple search threads
//! without locks. A torn/racing read fails XOR-validation and is treated as a miss; it can
//! never produce a corrupted move, since TT moves are only ever *compared* against packed
//! generated moves, never unpacked blind (same invariant as before SMP).
//! Fixed-size, power-of-two buckets, replace on (age differs | deeper | same key).

use cozy_chess::{Move, Piece};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

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

// data word layout: mv:16 | score:16 | depth:8 | bound:8 | age:8 (56 of 64 bits used)
fn pack_data(mv: u16, score: i16, depth: i8, bound: u8, age: u8) -> u64 {
    (mv as u64)
        | ((score as u16 as u64) << 16)
        | ((depth as u8 as u64) << 32)
        | ((bound as u64) << 40)
        | ((age as u64) << 48)
}

fn unpack_data(data: u64) -> (u16, i16, i8, u8, u8) {
    let mv = data as u16;
    let score = (data >> 16) as u16 as i16;
    let depth = (data >> 32) as u8 as i8;
    let bound = (data >> 40) as u8;
    let age = (data >> 48) as u8;
    (mv, score, depth, bound, age)
}

struct Slot {
    key_xor_data: AtomicU64,
    data: AtomicU64,
}

pub struct TT {
    slots: Vec<Slot>,
    mask: usize,
    age: AtomicU8,
}

impl TT {
    pub fn new(mb: usize) -> Self {
        // same size_of (16 bytes) as the old plain Entry -- identical bucket count/mask at
        // a given Hash MB setting, which is what keeps single-thread bench bit-identical.
        let n = ((mb.max(1) << 20) / std::mem::size_of::<Slot>())
            .next_power_of_two()
            >> 1;
        let n = n.max(1024);
        let mut slots = Vec::with_capacity(n);
        slots.resize_with(n, || Slot {
            key_xor_data: AtomicU64::new(0),
            data: AtomicU64::new(0),
        });
        TT {
            slots,
            mask: n - 1,
            age: AtomicU8::new(0),
        }
    }

    pub fn clear(&self) {
        for s in &self.slots {
            s.key_xor_data.store(0, Ordering::Relaxed);
            s.data.store(0, Ordering::Relaxed);
        }
        self.age.store(0, Ordering::Relaxed);
    }

    pub fn new_search(&self) {
        self.age.fetch_add(1, Ordering::Relaxed);
    }

    pub fn probe(&self, key: u64) -> Option<Entry> {
        let slot = &self.slots[(key as usize) & self.mask];
        let kx = slot.key_xor_data.load(Ordering::Relaxed);
        let data = slot.data.load(Ordering::Relaxed);
        if kx ^ data != key {
            return None; // torn read or genuine miss -- either way, not a hit
        }
        let (mv, score, depth, bound, age) = unpack_data(data);
        if bound == BOUND_NONE {
            return None;
        }
        Some(Entry {
            key,
            mv,
            score,
            depth,
            bound,
            age,
        })
    }

    pub fn store(&self, key: u64, mv: u16, score: i32, depth: i32, bound: u8) {
        let idx = (key as usize) & self.mask;
        let slot = &self.slots[idx];
        let cur_age = self.age.load(Ordering::Relaxed);

        let kx = slot.key_xor_data.load(Ordering::Relaxed);
        let old_data = slot.data.load(Ordering::Relaxed);
        let existing_key = kx ^ old_data;
        let (old_mv, _old_score, old_depth, _old_bound, old_age) = unpack_data(old_data);

        let replace = existing_key != key
            || old_age != cur_age
            || depth >= old_depth as i32
            || bound == BOUND_EXACT;
        if !replace {
            return;
        }
        // keep an existing move hint if the new one is null and the key matches
        let mv = if mv == 0 && existing_key == key {
            old_mv
        } else {
            mv
        };
        let score = score.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        let depth = depth.clamp(-128, 127) as i8;
        let data = pack_data(mv, score, depth, bound, cur_age);
        slot.data.store(data, Ordering::Relaxed);
        slot.key_xor_data.store(key ^ data, Ordering::Relaxed);
    }
}
