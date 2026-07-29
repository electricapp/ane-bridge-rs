//! Field-presence bitmask used by [`crate::AneFuzzCase`].

/// `Name` key present.
pub const NAME: u32 = 1 << 0;
/// `Type` key present.
pub const TYPE: u32 = 1 << 1;
/// `Batches` key present.
pub const BATCHES: u32 = 1 << 2;
/// `Channels` key present.
pub const CHANNELS: u32 = 1 << 3;
/// `Depth` key present.
pub const DEPTH: u32 = 1 << 4;
/// `Height` key present.
pub const HEIGHT: u32 = 1 << 5;
/// `Width` key present.
pub const WIDTH: u32 = 1 << 6;
/// All dimension + name/type keys present.
pub const ALL: u32 = 0x7F;
