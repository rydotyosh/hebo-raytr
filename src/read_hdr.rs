use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::option::Option;
use std::path::Path;
type Vec3 = nalgebra::Vector3<f64>;

fn read_line(buf: &[u8], ofs: usize, width: usize, dest: &mut [u8]) -> usize {
    let mut cur = 4usize;
    for c in 0usize..4 {
        let mut j = 0;
        while j < width {
            let mut code = buf[ofs + cur];
            cur += 1;
            if code > 128 {
                // run length
                code &= 127;
                let val = buf[ofs + cur];
                cur = cur + 1;
                for _ in 0..code {
                    dest[j * 4 + c] = val;
                    j += 1;
                }
            } else {
                // direct value
                for _ in 0..code {
                    dest[j * 4 + c] = buf[ofs + cur];
                    j += 1;
                    cur += 1;
                }
            }
        }
    }
    return cur;
}

pub struct HDRData {
    pub w: usize,
    pub h: usize,
    pub data: Vec<Vec3>,
}

pub fn read_hdr(file_name: &str, width: usize, height: usize) -> Option<HDRData> {
    let path = Path::new(file_name);
    let file_ = File::open(path);
    if file_.is_err() {
        return None;
    }
    let file = file_.unwrap();
    let mut rd = BufReader::new(file);
    let mut buf = String::new();
    let _ = rd.read_line(&mut buf).unwrap();
    let _ = rd.read_line(&mut buf).unwrap();
    let _ = rd.read_line(&mut buf).unwrap();
    let _ = rd.read_line(&mut buf).unwrap();
    let _ = rd.read_line(&mut buf).unwrap();
    // println!("{}", buf);

    let mut buf: Vec<u8> = Vec::new();
    let _ = rd.read_to_end(&mut buf);

    // println!("{}", bf0.len());

    let mut buf_line: Vec<u8> = Vec::new();
    buf_line.resize(width * 4, 0);

    let mut data: Vec<Vec3> = Vec::new();
    data.resize(width * height, Vec3::new(0.0, 0.0, 0.0));

    let mut ofs = 0usize;
    for y in 0..height {
        let cur = read_line(&buf, ofs, width, &mut buf_line);
        ofs += cur;
        for x in 0..width {
            let r0 = buf_line[x * 4 + 0] as f64;
            let g0 = buf_line[x * 4 + 1] as f64;
            let b0 = buf_line[x * 4 + 2] as f64;
            let e0 = buf_line[x * 4 + 3] as f64;
            let r = r0 * (e0 - 128.0).exp2();
            let g = g0 * (e0 - 128.0).exp2();
            let b = b0 * (e0 - 128.0).exp2();
            data[y * width + x] = Vec3::new(r, g, b);
        }
    }
    return Some(HDRData {
        w: width,
        h: height,
        data: data,
    });
}
