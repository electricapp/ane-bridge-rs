//! Writer for the `CoreML` `MILBlob` weight-file format (`weights/weight.bin`).
//!
//! The layout below was reverse-engineered from Apple-shipped `weight.bin`
//! files and confirmed byte-for-byte against several of them:
//!
//! ```text
//! 0x00   u32  count        number of blob records in the file
//! 0x04   u32  version      always 2
//! 0x08   [u8; 56]          zero padding  (header is 64 bytes total)
//!
//! R      u32  0xDEAD_BEEF  record sentinel; R is always 64-byte aligned
//! R+4    u32  dtype        see `BlobDtype`
//! R+8    u64  size         payload length in bytes
//! R+16   u64  offset       absolute file offset of the payload, always R + 64
//! R+24   [u8; 40]          reserved, zero
//! R+64   payload           raw little-endian, C-order, contiguous
//! ```
//!
//! The next record begins at `align_up(R + 64 + size, 64)`.
//!
//! The critical subtlety: the `offset = uint64(N)` that a MIL `BLOBFILE`
//! reference carries is the offset of the 64-byte *record* `R`, **not** of the
//! payload. So the first blob in every file is at `offset = uint64(64)`.

use half::f16;

/// Element type of a stored blob, using the `MILBlob` dtype codes.
///
/// Only the variants this crate needs are listed. The codes are unrelated to
/// `ane_bridge::Dtype`, which describes the model's *I/O* schema.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlobDtype {
    /// IEEE binary16. The ANE's native compute type.
    Fp16,
    /// IEEE binary32.
    Fp32,
    /// Signed 32-bit integer.
    Int32,
}

impl BlobDtype {
    /// The code stored in a blob record's header.
    #[must_use]
    pub const fn code(self) -> u32 {
        match self {
            Self::Fp16 => 1,
            Self::Fp32 => 2,
            Self::Int32 => 14,
        }
    }
}

/// Byte offset of a blob record within the weights file.
///
/// This is what a MIL `BLOBFILE(...)` reference embeds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct BlobOffset(pub u64);

/// Bytes before the first record.
const HEADER_LEN: usize = 64;
/// Bytes of record metadata before a payload.
const RECORD_LEN: usize = 64;
/// Every record starts on this boundary.
const ALIGN: usize = 64;
/// Marks the start of a record.
const SENTINEL: u32 = 0xDEAD_BEEF;
/// The only format version Apple's files use.
const VERSION: u32 = 2;

/// A buffer length as the `u64` the file format stores offsets in.
///
/// A file this writer built is in memory first, so a length that does not fit
/// a `u64` cannot arise; saturating keeps that fact from needing a fallback
/// value that means something else.
fn offset(len: usize) -> u64 {
    u64::try_from(len).unwrap_or(u64::MAX)
}

/// Accumulates weight payloads and serializes them as a `MILBlob` file.
///
/// Blobs are appended in call order; each `write_*` returns the [`BlobOffset`]
/// to embed in the MIL text.
#[derive(Debug, Default)]
pub struct BlobWriter {
    /// Everything after the 64-byte header: records and their payloads.
    body: Vec<u8>,
    /// Number of records appended so far.
    count: u32,
}

#[expect(
    clippy::little_endian_bytes,
    reason = "`MILBlob` is a little-endian on-disk format; the byte order is the \
              file's, not a choice this code gets to make"
)]
impl BlobWriter {
    /// Create an empty writer.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            body: Vec::new(),
            count: 0,
        }
    }

    /// Number of blobs written so far.
    #[must_use]
    pub const fn len(&self) -> u32 {
        self.count
    }

    /// Whether no blobs have been written.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Append `payload` as one blob of type `dtype`, returning its record offset.
    fn write_raw(&mut self, dtype: BlobDtype, payload: &[u8]) -> BlobOffset {
        // Records are 64-byte aligned; pad the body out before starting one.
        self.body.resize(self.body.len().next_multiple_of(ALIGN), 0);

        let record_off = offset(HEADER_LEN.saturating_add(self.body.len()));
        let payload_off = record_off.saturating_add(offset(RECORD_LEN));

        let mut record = [0u8; RECORD_LEN];
        record[0..4].copy_from_slice(&SENTINEL.to_le_bytes());
        record[4..8].copy_from_slice(&dtype.code().to_le_bytes());
        record[8..16].copy_from_slice(&offset(payload.len()).to_le_bytes());
        record[16..24].copy_from_slice(&payload_off.to_le_bytes());
        // record[24..64] stays zero: reserved.

        self.body.extend_from_slice(&record);
        self.body.extend_from_slice(payload);
        self.count = self.count.saturating_add(1);

        BlobOffset(record_off)
    }

    /// Append an fp16 tensor, converting from `f32`.
    pub fn write_fp16(&mut self, values: &[f32]) -> BlobOffset {
        let mut payload = Vec::with_capacity(values.len().saturating_mul(2));
        for &v in values {
            payload.extend_from_slice(&f16::from_f32(v).to_bits().to_le_bytes());
        }
        self.write_raw(BlobDtype::Fp16, &payload)
    }

    /// Append an fp32 tensor.
    pub fn write_fp32(&mut self, values: &[f32]) -> BlobOffset {
        let mut payload = Vec::with_capacity(values.len().saturating_mul(4));
        for &v in values {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        self.write_raw(BlobDtype::Fp32, &payload)
    }

    /// Append an int32 tensor.
    pub fn write_int32(&mut self, values: &[i32]) -> BlobOffset {
        let mut payload = Vec::with_capacity(values.len().saturating_mul(4));
        for &v in values {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        self.write_raw(BlobDtype::Int32, &payload)
    }

    /// Serialize the complete weights file.
    #[must_use]
    pub fn finish(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_LEN.saturating_add(self.body.len()));
        out.extend_from_slice(&self.count.to_le_bytes());
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.resize(HEADER_LEN, 0);
        out.extend_from_slice(&self.body);
        out
    }
}

#[cfg(test)]
#[expect(
    clippy::inline_modules,
    reason = "these tests check the private record layout, which only a module \
              inside the file can reach"
)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::float_cmp,
        clippy::little_endian_bytes,
        reason = "in a test a panic is the failure report, a bit-exact format is \
                  checked with bit-exact comparisons, and reading the file back \
                  means the same byte order the writer emits"
    )]
    use super::*;

    /// `count` bytes at `at`, with the read bounded by an assertion rather than
    /// by an index panic, so an out-of-range read names the range it wanted.
    fn bytes_at(bytes: &[u8], at: usize, count: usize) -> &[u8] {
        let end = at.saturating_add(count);
        assert!(
            end <= bytes.len(),
            "read {at}..{end} past a {}-byte file",
            bytes.len()
        );
        bytes.get(at..end).unwrap_or_default()
    }

    /// The `u32` field at `at`.
    fn u32_at(bytes: &[u8], at: usize) -> u32 {
        u32::from_le_bytes(bytes_at(bytes, at, 4).try_into().unwrap())
    }

    /// The `u64` field at `at`.
    fn u64_at(bytes: &[u8], at: usize) -> u64 {
        u64::from_le_bytes(bytes_at(bytes, at, 8).try_into().unwrap())
    }

    /// A payload of `i32`.
    fn i32s(payload: &[u8]) -> Vec<i32> {
        payload
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    /// A record's fields, at the offsets the format fixes them at.
    fn record_at(bytes: &[u8], record: BlobOffset) -> (u32, u32, usize, usize) {
        let at = usize::try_from(record.0).unwrap();
        (
            u32_at(bytes, at),
            u32_at(bytes, at.saturating_add(4)),
            usize::try_from(u64_at(bytes, at.saturating_add(8))).unwrap(),
            usize::try_from(u64_at(bytes, at.saturating_add(16))).unwrap(),
        )
    }

    /// The header must be exactly 64 bytes and carry count + version.
    #[test]
    fn header_shape() {
        let mut writer = BlobWriter::new();
        let first = writer.write_fp16(&[1.0, 2.0]);
        assert_eq!(
            first,
            BlobOffset(64),
            "first record sits right after the header"
        );

        let bytes = writer.finish();
        assert_eq!(u32_at(&bytes, 0), 1);
        assert_eq!(u32_at(&bytes, 4), VERSION);
        assert!(
            bytes_at(&bytes, 8, 56).iter().all(|&b| b == 0),
            "header padding is zero"
        );
    }

    /// A record carries the sentinel, dtype, size and an absolute payload offset
    /// of exactly `record + 64`.
    #[test]
    fn record_fields() {
        let mut writer = BlobWriter::new();
        let record = writer.write_fp16(&[1.0, 2.0, 3.0, 4.0]);
        let bytes = writer.finish();

        let (sentinel, dtype, size, payload_off) = record_at(&bytes, record);
        assert_eq!(sentinel, SENTINEL);
        assert_eq!(dtype, BlobDtype::Fp16.code());
        assert_eq!(size, 8, "4 fp16 = 8 bytes");
        assert_eq!(payload_off, 128, "payload = record + 64");
        assert!(
            bytes_at(&bytes, 88, 40).iter().all(|&b| b == 0),
            "reserved area is zero"
        );
    }

    /// Every record after the first must land on a 64-byte boundary, and the
    /// returned offsets must point at those records.
    #[test]
    fn records_stay_aligned() {
        let mut writer = BlobWriter::new();
        // 3 fp16 = 6 bytes of payload: deliberately not a multiple of 64.
        let first = writer.write_fp16(&[1.0, 2.0, 3.0]);
        let second = writer.write_fp32(&[1.0]);
        let third = writer.write_int32(&[7, 8]);

        assert_eq!(first, BlobOffset(64));
        assert_eq!(
            second,
            BlobOffset(192),
            "64 header + 64 rec + 6 payload -> aligned to 192"
        );
        assert_eq!(third, BlobOffset(320));

        let bytes = writer.finish();
        assert_eq!(u32_at(&bytes, 0), 3);
        for record in [first, second, third] {
            assert_eq!(
                record.0.checked_rem(offset(ALIGN)),
                Some(0),
                "record {record:?} is 64-byte aligned"
            );
            let (sentinel, ..) = record_at(&bytes, record);
            assert_eq!(sentinel, SENTINEL, "record at {record:?} starts with it");
        }
    }

    /// Payload bytes must round-trip exactly at the advertised offset.
    #[test]
    fn payload_round_trips() {
        let mut writer = BlobWriter::new();
        let record = writer.write_int32(&[-1, 0, 1_000_000]);
        let bytes = writer.finish();

        let (_, _, size, payload_off) = record_at(&bytes, record);
        assert_eq!(
            i32s(bytes_at(&bytes, payload_off, size)),
            vec![-1, 0, 1_000_000]
        );
    }
}
