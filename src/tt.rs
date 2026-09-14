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

/// What a probe hands back. `key` and `age` used to be carried here too; neither was ever read
/// by a caller (the caller already has the key, and age is a replacement-policy concern that
/// lives entirely inside `store`), so they were dead weight in the hot return path.
#[derive(Clone, Copy, Default)]
pub struct Entry {
    pub mv: u16,
    pub score: i16,
    pub depth: i8,
    pub bound: u8,
    /// Raw (uncorrected) static evaluation. `-8192` means unavailable.
    pub eval: i16,
    /// The score was searched as a PV node. This is sticky for an entry's lifetime.
    pub tt_pv: bool,
}

pub const EVAL_NONE: i16 = -8192;

#[derive(Clone, Copy)]
pub struct StoreData {
    pub mv: u16,
    pub score: i32,
    pub depth: i32,
    pub bound: u8,
    pub eval: i32,
    pub tt_pv: bool,
}

// data word layout: mv:16 | score:16 | depth:8 | bound:2 | age:5 | tt_pv:1 | eval:14.
// The two high bits remain zero. Keeping the value in the existing word preserves the
// four-entry cache-line layout and avoids a second atomic load on every probe.
fn pack_data(mv: u16, score: i16, depth: i8, bound: u8, age: u8, eval: i16, tt_pv: bool) -> u64 {
    let eval_bits = (eval.clamp(-8192, 8191) as i32 & 0x3fff) as u64;
    (mv as u64)
        | ((score as u16 as u64) << 16)
        | ((depth as u8 as u64) << 32)
        | (((bound & 0x3) as u64) << 40)
        | (((age & 0x1f) as u64) << 42)
        | ((tt_pv as u64) << 47)
        | (eval_bits << 48)
}

fn unpack_data(data: u64) -> (u16, i16, i8, u8, u8, i16, bool) {
    let mv = data as u16;
    let score = (data >> 16) as u16 as i16;
    let depth = (data >> 32) as u8 as i8;
    let bound = ((data >> 40) & 0x3) as u8;
    let age = ((data >> 42) & 0x1f) as u8;
    let tt_pv = ((data >> 47) & 1) != 0;
    let eval_bits = ((data >> 48) & 0x3fff) as i16;
    let eval = if eval_bits & 0x2000 != 0 {
        eval_bits | !0x3fff
    } else {
        eval_bits
    };
    (mv, score, depth, bound, age, eval, tt_pv)
}

struct Slot {
    key_xor_data: AtomicU64,
    data: AtomicU64,
}

/// Four slots occupy one cache line; Cluster also guarantees its starting alignment.
const CLUSTER: usize = 4;

#[repr(align(64))]
struct Cluster([Slot; CLUSTER]);

pub struct TT {
    clusters: Vec<Cluster>,
    n_slots: usize,
    mask: usize, // over CLUSTERS, not slots
    age: AtomicU8,
}

impl TT {
    pub fn new(mb: usize) -> Self {
        // Round the slot count DOWN to a power of two so the table fits in `mb` and no more.
        //
        // 2026-08-18 FIX: this was `.next_power_of_two() >> 1`, which halves unconditionally --
        // (mb << 20) / 16 is ALREADY a power of two for power-of-two mb, so the shift was pure
        // loss and every game this engine played at Hash=64 ran on 32 MB. ilog2 floors, so a
        // non-power-of-two request (Hash=100) rounds DOWN to 64 MB rather than overrunning the
        // budget the GUI set.
        let n = ((mb.max(1) << 20) / std::mem::size_of::<Slot>()).max(1);
        let n = (1usize << n.ilog2()).max(CLUSTER * 4);
        let mut clusters = Vec::with_capacity(n / CLUSTER);
        clusters.resize_with(n / CLUSTER, || Cluster(std::array::from_fn(|_| Slot {
            key_xor_data: AtomicU64::new(0),
            data: AtomicU64::new(0),
        })));
        TT {
            clusters,
            n_slots: n,
            mask: (n / CLUSTER) - 1,
            age: AtomicU8::new(0),
        }
    }

    /// Total slots. Memory is exactly `n_slots * size_of::<Slot>()`.
    ///
    /// Only the sizing tests call this in a release build, hence the allow: it is the accessor
    /// those assertions are written against, and inlining the field into them instead would put
    /// the sizing invariant out of reach of the gate that checks it.
    #[allow(dead_code)]
    pub fn n_slots(&self) -> usize {
        self.n_slots
    }

    #[inline]
    fn cluster(&self, key: u64) -> &[Slot] {
        &self.clusters[(key as usize) & self.mask].0
    }

    /// Pull the cluster's cache line in before the caller needs it. A TT probe is a guaranteed
    /// cache miss on any table that does not fit in L2, and the hash is known well before the
    /// probe result is used, so the latency can be overlapped with real work.
    #[inline]
    pub fn prefetch(&self, key: u64) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let base = (key as usize) & self.mask;
            std::arch::x86_64::_mm_prefetch(
                self.clusters.as_ptr().add(base) as *const i8,
                std::arch::x86_64::_MM_HINT_T0,
            );
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = key;
    }

    pub fn clear(&self) {
        for s in self.clusters.iter().flat_map(|c| &c.0) {
            s.key_xor_data.store(0, Ordering::Relaxed);
            s.data.store(0, Ordering::Relaxed);
        }
        self.age.store(0, Ordering::Relaxed);
    }

    pub fn new_search(&self) {
        self.age.fetch_add(1, Ordering::Relaxed);
    }

    pub fn probe(&self, key: u64) -> Option<Entry> {
        for slot in self.cluster(key) {
            let kx = slot.key_xor_data.load(Ordering::Relaxed);
            let data = slot.data.load(Ordering::Relaxed);
            if kx ^ data != key {
                continue; // torn read or genuine miss -- either way, not a hit
            }
            let (mv, score, depth, bound, _age, eval, tt_pv) = unpack_data(data);
            if bound == BOUND_NONE {
                continue;
            }
            return Some(Entry { mv, score, depth, bound, eval, tt_pv });
        }
        None
    }

    pub fn store(&self, key: u64, entry: StoreData) {
        let cur_age = self.age.load(Ordering::Relaxed) & 0x1f;
        let cluster = self.cluster(key);

        // Pick the destination. A slot already holding THIS position always wins -- two entries
        // for one key would waste the cluster and could disagree. Otherwise evict the least
        // valuable slot, where value = depth discounted by how many searches ago it was written.
        //
        // 2026-09-09: this replaces a single slot per index whose policy was
        // `existing_key != key || old_age != cur_age || depth >= old_depth || bound == EXACT`
        // -- i.e. essentially unconditional replacement. A deep entry was evicted by the next
        // shallow node that happened to collide with it, which at 40/15 means re-searching
        // subtrees that were already paid for. Four candidates per cluster on the same cache
        // line make that a choice instead of an accident.
        let mut victim = 0usize;
        let mut victim_value = i32::MAX;
        let mut found = None;
        for (i, slot) in cluster.iter().enumerate() {
            let kx = slot.key_xor_data.load(Ordering::Relaxed);
            let old_data = slot.data.load(Ordering::Relaxed);
            let (_, _, old_depth, old_bound, old_age, _, _) = unpack_data(old_data);
            if kx ^ old_data == key && old_bound != BOUND_NONE {
                found = Some((i, old_depth, old_data));
                break;
            }
            // an empty slot is worth nothing and is taken first
            let value = if old_bound == BOUND_NONE {
                i32::MIN
            } else {
                // each search of age costs 8 plies of nominal depth
                old_depth as i32 - 8 * age_distance(cur_age, old_age)
            };
            if value < victim_value {
                victim_value = value;
                victim = i;
            }
        }

        let (idx, keep_mv, keep_eval, keep_pv) = match found {
            Some((i, old_depth, old_data)) => {
                // Same position: only overwrite with something at least as deep, unless the
                // entry is stale (from an earlier search) or this is an exact score at equal
                // depth. A shallower bound must not erase a deeper one.
                let (old_mv, _, _, _, old_age, old_eval, old_pv) = unpack_data(old_data);
                if entry.depth < old_depth as i32 && old_age == cur_age && entry.bound != BOUND_EXACT {
                    return;
                }
                (i, if entry.mv == 0 { old_mv } else { entry.mv },
                    if entry.eval == EVAL_NONE as i32 { old_eval } else { entry.eval.clamp(-8192, 8191) as i16 }, old_pv || entry.tt_pv)
            }
            None => (victim, entry.mv, entry.eval.clamp(-8192, 8191) as i16, entry.tt_pv),
        };

        let score = entry.score.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        let depth = entry.depth.clamp(-128, 127) as i8;
        let data = pack_data(keep_mv, score, depth, entry.bound, cur_age, keep_eval, keep_pv);
        let slot = &cluster[idx];
        slot.data.store(data, Ordering::Relaxed);
        slot.key_xor_data.store(key ^ data, Ordering::Relaxed);
    }
}

/// How many searches ago `old` was written, wrapping. `age` is a u8 counter bumped once per
/// `go`, so plain subtraction would call a 1-vs-255 gap "254 searches ago" instead of one.
#[inline]
fn age_distance(cur: u8, old: u8) -> i32 {
    cur.wrapping_sub(old) as i32 & 0x1f
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gate for the 2026-08-18 sizing fix. Until then `.next_power_of_two() >> 1` halved the
    /// table unconditionally, so every game ever played at Hash=64 ran on 32 MB.
    #[test]
    fn table_uses_the_memory_it_was_given() {
        for mb in [16usize, 64, 256] {
            let tt = TT::new(mb);
            let bytes = tt.n_slots() * std::mem::size_of::<Slot>();
            assert_eq!(bytes, mb << 20, "Hash={mb} MB allocated {bytes} B");
        }
    }

    /// A non-power-of-two request must round DOWN. Rounding up would overrun the budget the
    /// GUI set -- the opposite failure, and the one a naive fix introduces.
    #[test]
    fn non_power_of_two_rounds_down_not_up() {
        let tt = TT::new(100);
        let bytes = tt.n_slots() * std::mem::size_of::<Slot>();
        assert!(bytes <= 100 << 20, "overran the budget: {bytes} B");
        assert_eq!(bytes, 64 << 20);
    }

    #[test]
    fn tiny_requests_still_give_a_usable_table() {
        let tt = TT::new(0);
        assert_eq!(tt.n_slots() * std::mem::size_of::<Slot>(), 1 << 20);
    }

    /// A cluster is exactly one cache line, which is the reason four candidates cost the same
    /// single miss as one did.
    #[test]
    fn a_cluster_is_one_cache_line() {
        assert_eq!(std::mem::size_of::<Cluster>(), 64);
        assert_eq!(std::mem::align_of::<Cluster>(), 64);
        let tt = TT::new(1);
        assert_eq!(tt.clusters.len() * std::mem::size_of::<Cluster>(), 1 << 20);
        for cluster in &tt.clusters {
            assert_eq!(cluster.0.as_ptr() as usize % 64, 0);
        }
    }

    /// THE POINT OF THE 2026-09-09 CLUSTER CHANGE. Under the old single-slot policy a deep
    /// entry was evicted by the very next shallow store that collided with it. Here three
    /// unrelated shallow positions land in the same cluster and the deep entry must survive
    /// all of them.
    #[test]
    fn a_deep_entry_survives_colliding_shallow_stores() {
        let tt = TT::new(1);
        let clusters = (tt.mask + 1) as u64;
        let deep_key = 0x1234_5678_9abc_def0u64;
        tt.store(deep_key, StoreData { mv: 42, score: 100, depth: 30, bound: BOUND_EXACT, eval: 321, tt_pv: true });
        // keys differing by a multiple of the cluster count share a cluster
        for i in 1..=3u64 {
            tt.store(deep_key.wrapping_add(i * clusters), StoreData { mv: 7, score: 5, depth: 1, bound: BOUND_UPPER, eval: -8192, tt_pv: false });
        }
        let e = tt.probe(deep_key).expect("the depth-30 entry was evicted by shallow stores");
        assert_eq!(e.depth, 30);
        assert_eq!(e.mv, 42);
    }

    /// Same key, shallower result, same search: must not overwrite the deeper one.
    #[test]
    fn a_shallow_result_does_not_erase_a_deeper_one_for_the_same_key() {
        let tt = TT::new(1);
        let k = 0xdead_beef_0000_0001u64;
        tt.store(k, StoreData { mv: 11, score: 50, depth: 20, bound: BOUND_LOWER, eval: 77, tt_pv: false });
        tt.store(k, StoreData { mv: 22, score: 60, depth: 3, bound: BOUND_LOWER, eval: -8192, tt_pv: false });
        let e = tt.probe(k).unwrap();
        assert_eq!(e.depth, 20, "a depth-3 store overwrote a depth-20 entry");
        assert_eq!(e.mv, 11);
    }

    #[test]
    fn static_eval_and_pv_metadata_survive_a_replacement() {
        let tt = TT::new(1);
        let k = 0xdead_beef_0000_0002u64;
        tt.store(k, StoreData { mv: 11, score: 50, depth: 20, bound: BOUND_LOWER, eval: 77, tt_pv: false });
        tt.store(k, StoreData { mv: 22, score: 60, depth: 20, bound: BOUND_LOWER, eval: -8192, tt_pv: true });
        let e = tt.probe(k).unwrap();
        assert_eq!(e.eval, 77, "a missing static eval erased the cached value");
        assert!(e.tt_pv, "PV metadata should be sticky");
    }

    /// A stale entry (written several searches ago) is worth less than a fresh one even if it
    /// is deeper, or the table would fill with entries no current search can use.
    #[test]
    fn age_discounts_depth() {
        assert!(age_distance(5, 5) == 0);
        assert_eq!(age_distance(0, 255), 1, "age must wrap, not read 255 searches ago");
        assert_eq!(age_distance(3, 1), 2);
    }
}
