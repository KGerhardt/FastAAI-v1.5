//! Reader for `data.bin` — the interchange emitted by `export_bin.py` from a real
//! FastAAI v1 preprocessing run.

use std::io;

pub struct ScpRecord {
    pub acc: u16,
    pub seq: Vec<u8>,
    /// v1 decimal-ASCII codes, retained only to verify the dense encoding agrees.
    pub v1_codes: Vec<i32>,
}

pub struct Genome {
    pub name: String,
    pub scps: Vec<ScpRecord>,
}

pub struct Dataset {
    pub alphabet: Vec<u8>,
    pub n_acc: usize,
    pub genomes: Vec<Genome>,
}

struct Cursor<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cursor<'a> {
    fn u16(&mut self) -> u16 {
        let v = u16::from_le_bytes(self.b[self.p..self.p + 2].try_into().unwrap());
        self.p += 2;
        v
    }
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.b[self.p..self.p + 4].try_into().unwrap());
        self.p += 4;
        v
    }
    fn bytes(&mut self, n: usize) -> &'a [u8] {
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        s
    }
}

pub fn load(path: &str) -> io::Result<Dataset> {
    let raw = std::fs::read(path)?;
    let mut c = Cursor { b: &raw, p: 0 };

    let magic = c.bytes(8);
    if magic != b"FAAI0001" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
    }

    let n_genomes = c.u32() as usize;
    let n_acc = c.u32() as usize;
    let alen = c.u32() as usize;
    let alphabet = c.bytes(alen).to_vec();

    let mut genomes = Vec::with_capacity(n_genomes);
    for _ in 0..n_genomes {
        let nl = c.u16() as usize;
        let name = String::from_utf8_lossy(c.bytes(nl)).into_owned();
        let n_scp = c.u16() as usize;
        let mut scps = Vec::with_capacity(n_scp);
        for _ in 0..n_scp {
            let acc = c.u16();
            let sl = c.u32() as usize;
            let seq = c.bytes(sl).to_vec();
            let nk = c.u32() as usize;
            let mut v1_codes = Vec::with_capacity(nk);
            for _ in 0..nk {
                v1_codes.push(c.u32() as i32);
            }
            scps.push(ScpRecord { acc, seq, v1_codes });
        }
        genomes.push(Genome { name, scps });
    }

    Ok(Dataset { alphabet, n_acc, genomes })
}
