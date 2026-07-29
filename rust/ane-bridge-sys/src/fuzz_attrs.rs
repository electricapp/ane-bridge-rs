//! Mutation bitmask for [`crate::AneFuzzAttrsCase`]. Mirrors
//! `AneFuzzAttrsMutation`.

/// Replace `NetworkStatusList` with a non-array.
pub const NSL_NOT_ARRAY: u32 = 1 << 0;
/// Omit `NetworkStatusList` entirely.
pub const NSL_MISSING: u32 = 1 << 1;
/// `NetworkStatusList` is empty.
pub const NSL_EMPTY: u32 = 1 << 2;
/// `NetworkStatusList[0]` is not a dictionary.
pub const PROC_NOT_DICT: u32 = 1 << 3;
/// Omit `LiveInputList`.
pub const LIVEIN_MISSING: u32 = 1 << 4;
/// Omit `LiveOutputList`.
pub const LIVEOUT_MISSING: u32 = 1 << 5;
/// `LiveInputList` is not an array.
pub const LIVEIN_NOT_ARRAY: u32 = 1 << 6;
/// `LiveOutputList` is not an array.
pub const LIVEOUT_NOT_ARRAY: u32 = 1 << 7;
