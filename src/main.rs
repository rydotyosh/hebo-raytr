mod write_image;
use write_image::write_image;
mod read_hdr;
use rand::distributions::Uniform;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use read_hdr::read_hdr;
use read_hdr::HDRData;
use std::f64::consts::PI;
use std::option::Option;

type Vec3 = nalgebra::Vector3<f64>;

#[derive(Debug)]
struct Ray {
    org: Vec3,
    dir: Vec3, // outward is positive
    att: Vec3,
}

// fn test_image(file_name: &str) {
//     let width = 256usize;
//     let height = 256usize;
//     let mut data: Vec<u8> = Vec::new();
//     data.resize((width * height * 4) as usize, 0);
//     for j in 0usize..height {
//         for i in 0usize..width {
//             data[(j * width + i) * 4 + 0] = i as u8;
//             data[(j * width + i) * 4 + 1] = j as u8;
//             data[(j * width + i) * 4 + 2] = 0;
//             data[(j * width + i) * 4 + 3] = 255;
//         }
//     }
//     write_image(file_name, width, height, &data);
// }

#[derive(Debug, Clone)]
struct Mirror {
    att: Vec3,
}

#[derive(Debug, Clone)]
struct RefrMirror {
    att: Vec3,
    ior: f64,
}

#[derive(Debug, Clone)]
struct Diffuse {
    att: Vec3,
}

#[derive(Debug, Clone)]
struct FresnelDiffuse {
    f0: f64,
    attm: Vec3,
    attd: Vec3,
    alpha: f64,
}

#[derive(Debug, Clone)]
struct FresnelRefraction {
    attm: Vec3,
    attr: Vec3,
    ior: f64,
    alpha: f64,
}

#[derive(Debug, Clone)]
enum Material {
    Mir(Mirror),
    RefrMir(RefrMirror),
    Dif(Diffuse),
    FreDif(FresnelDiffuse),
    FreRefr(FresnelRefraction),
}
use Material::Dif;
use Material::FreDif;
use Material::FreRefr;
use Material::Mir;
use Material::RefrMir;

#[derive(Debug)]
struct Sphere {
    org: Vec3,
    rad: f64,
}

#[derive(Debug)]
struct Plane {
    org: Vec3,
    norm: Vec3,
}

#[derive(Debug)]
enum Shape {
    Sph(Sphere),
    Pla(Plane),
}
use Shape::Pla;
use Shape::Sph;

#[derive(Debug)]
struct Object {
    sha: Shape,
    mtr: Material,
}

#[derive(Debug)]
struct Proj {
    pos: Vec3,
    norm: Vec3,
    dist: f64,
}

fn project_sphere(ray: &Ray, sph: &Sphere) -> Option<Proj> {
    let v = &sph.org;
    let rad = sph.rad;
    let d = v - ray.org;
    let rad2 = rad * rad;
    if d.norm_squared() > rad2 {
        // exterior
        let l = -d.dot(&ray.dir);
        if l < 0. {
            return None;
        }
        let h = d + l * ray.dir;
        let r2 = h.norm_squared();
        if r2 > rad2 {
            return None;
        }
        let a = (rad2 - r2).sqrt();
        let w = -h + a * ray.dir;
        let pos = v + w;
        let norm = w.normalize();
        let dist = l - a;
        return Some(Proj { pos, norm, dist });
    } else {
        // interior
        let l = -d.dot(&ray.dir);
        let h = d + l * ray.dir;
        let r2 = h.norm_squared();
        if r2 > rad2 {
            return None;
        }
        let a = (rad2 - r2).sqrt();
        let w = -h - a * ray.dir;
        let pos = v + w;
        let norm = w.normalize();
        let dist = l + a;
        return Some(Proj { pos, norm, dist });
    }
}

fn project_plane(ray: &Ray, pla: &Plane) -> Option<Proj> {
    let m = pla.norm.dot(&ray.dir);
    if m.abs() < 1e-8 {
        return None;
    }
    let d = pla.org - ray.org;
    let l = -pla.norm.dot(&d);
    let h = l * pla.norm;
    if h.dot(&ray.dir) < 0. {
        return None;
    }
    let dist = l / m;
    let pos = ray.org - dist * ray.dir;
    let norm = pla.norm;
    return Some(Proj { pos, norm, dist });
}

fn project_shape<'a>(ray: &Ray, obj: &'a Object) -> Option<(Proj, &'a Object)> {
    match &obj.sha {
        Sph(s) => {
            if let Some(pj) = project_sphere(ray, &s) {
                return Some((pj, obj));
            } else {
                return None;
            }
        }
        Pla(p) => {
            if let Some(pj) = project_plane(ray, &p) {
                return Some((pj, obj));
            } else {
                return None;
            }
        }
    };
}

fn reflect_mirror(dir: &Vec3, norm: &Vec3) -> Vec3 {
    let bl = dir.dot(&norm);
    let bh = dir - bl * norm;
    let new_dir = bl * norm - bh;
    return new_dir;
}

fn refract_mirror(dir: &Vec3, norm: &Vec3, ior_: f64) -> Option<Vec3> {
    let bl = dir.dot(&norm);
    let ior_rev = if bl < 0.0 { ior_ } else { ior_.recip() };
    let bh = dir - bl * norm;
    let bn = bh.norm();
    if bn < 1e-8 {
        let new_dir = -bl * norm - bh * ior_rev; //
        return Some(new_dir);
    }
    let t = bn.asin() * ior_rev;
    if t > PI / 2.0 {
        return None;
    }
    let s = t.sin();
    let c = if bl < 0.0 { t.cos() } else { -t.cos() };
    let new_dir = c * norm - bh / bn * s;
    return Some(new_dir);
}

fn nearest<'a>(
    p0: Option<(Proj, &'a Object)>,
    p1: Option<(Proj, &'a Object)>,
) -> Option<(Proj, &'a Object)> {
    if let Some(q0) = &p0 {
        if let Some(q1) = &p1 {
            return if q0.0.dist < q1.0.dist { p0 } else { p1 };
        } else {
            return p0;
        }
    } else {
        if let Some(_) = p1 {
            return p1;
        } else {
            return p0;
        }
    }
}

// fn nearest_ref<'a>(p0: &'a Option<Proj>, p1: &'a Option<Proj>) -> &'a Option<Proj> {
//     if let Some(q0) = p0 {
//         if let Some(q1) = p1 {
//             return if q0.dist < q1.dist { p0 } else { p1 };
//         } else {
//             return p0;
//         }
//     } else {
//         if let Some(_) = p1 {
//             return p1;
//         } else {
//             return p0;
//         }
//     }
// }

struct UniformRng {
    rng: SmallRng,
}
impl UniformRng {
    fn seed_from_u64(seed: u64) -> UniformRng {
        let rng: SmallRng = SeedableRng::seed_from_u64(seed);
        return UniformRng { rng };
    }
    fn get(&mut self) -> f64 {
        let r: f64 = self.rng.sample(Uniform::new(0.0, 1.0));
        return r;
    }
}

fn sample_hemi_cos(rng: &mut UniformRng) -> Vec3 {
    let u0 = rng.get();
    let u1 = rng.get();
    let tc = (1.0 - u0).sqrt();
    let ts = u0.sqrt();
    let p = 2.0 * PI * u1;
    let pc = p.cos();
    let ps = p.sin();
    return Vec3::new(pc * tc, ps * tc, ts);
}

fn sample_hemi_cos_pow(rng: &mut UniformRng, pw: f64) -> Vec3 {
    let u0 = rng.get();
    let u1 = rng.get();
    let ts = u0.powf(1.0 / (pw + 2.0));
    let tc = (1.0 - ts * ts).sqrt();
    let p = 2.0 * PI * u1;
    let pc = p.cos();
    let ps = p.sin();
    return Vec3::new(pc * tc, ps * tc, ts);
}

fn tangent_vectors(norm: &Vec3) -> (Vec3, Vec3) {
    let i = norm.iamax();
    let j = (i as u32 + 1) % 3;
    let mut t0: Vec3 = nalgebra::zero();
    t0[j as usize] = 1.0;
    let t1 = norm.cross(&t0).normalize();
    t0 = t1.cross(&norm).normalize();
    return (t0, t1);
}

fn convert_normal(norm: &Vec3, dir: &Vec3) -> Vec3 {
    let (t0, t1) = tangent_vectors(norm);
    // println!("{:?}", t0);
    // println!("{:?}", t1);
    // println!("{:?}", norm);
    let n = dir.x * t0 + dir.y * t1 + dir.z * norm;
    return n;
}

fn reflect_diffuse(norm: &Vec3, rng: &mut UniformRng) -> Vec3 {
    let v = sample_hemi_cos(rng);
    let n = convert_normal(norm, &v);
    return n;
}

fn project(scene: &[Object], rng: &mut UniformRng, ray: &Ray) -> Option<Ray> {
    let projected = scene.iter().map(|x| project_shape(ray, x));
    let n = projected.fold(None, nearest);
    if let Some(pj) = n {
        let o = pj.0.pos;
        let dir: Vec3;
        let att0: Vec3;
        match &pj.1.mtr {
            Mir(m) => {
                dir = -reflect_mirror(&ray.dir, &pj.0.norm);
                att0 = m.att;
            }
            RefrMir(rm) => {
                let dir0 = refract_mirror(&ray.dir, &pj.0.norm, rm.ior);
                dir = match dir0 {
                    Some(d) => -d,
                    None => -reflect_mirror(&ray.dir, &pj.0.norm),
                };
                att0 = rm.att;
            }
            Dif(d) => {
                dir = -reflect_diffuse(&pj.0.norm, rng);
                att0 = d.att;
            }
            FreDif(fd) => {
                let n: Vec3;
                if fd.alpha > 0.0 {
                    let n0 = sample_hemi_cos_pow(rng, fd.alpha);
                    let n1 = convert_normal(&pj.0.norm, &n0);
                    if n1.dot(&ray.dir) < 0.0 {
                        n = reflect_mirror(&n1, &pj.0.norm);
                    } else {
                        n = n1;
                    }
                } else {
                    n = pj.0.norm;
                }
                let c = n.dot(&ray.dir);
                let f = fd.f0 + (1.0 - fd.f0) * (1.0 - c).powf(5.0);
                let r = rng.get();
                if r < f {
                    dir = -reflect_mirror(&ray.dir, &n);
                    att0 = fd.attm;
                } else {
                    dir = -reflect_diffuse(&pj.0.norm, rng);
                    att0 = fd.attd;
                }
            }
            FreRefr(fr) => {
                let c = pj.0.norm.dot(&ray.dir).abs();
                let f0 = ((fr.ior - 1.0) / (fr.ior + 1.0)) * ((fr.ior - 1.0) / (fr.ior + 1.0));
                let f = f0 + (1.0 - f0) * (1.0 - c).powf(5.0);
                let r = rng.get();
                if r < f {
                    dir = -reflect_mirror(&ray.dir, &pj.0.norm);
                    att0 = fr.attm;
                } else {
                    let dir0 = refract_mirror(&ray.dir, &pj.0.norm, fr.ior);
                    match dir0 {
                        Some(d) => {
                            dir = -d;
                            att0 = fr.attr;
                        }
                        None => {
                            dir = -reflect_mirror(&ray.dir, &pj.0.norm);
                            att0 = fr.attm;
                        }
                    }
                }
            }
        };
        let org = o - dir * 1e-8;
        let att = ray.att.component_mul(&att0);
        return Some(Ray { org, dir, att });
    } else {
        return None;
    }
}

fn bg_color(dir: &Vec3) -> Vec3 {
    let c = -dir * 0.5 + Vec3::new(0.5, 0.5, 0.5);
    let vv = Vec3::new(0.1, 1.0, 0.2).normalize().dot(&-dir);
    let v0 = vv.clamp(0.0, 1.0);
    let ww = v0.powf(200.) * 4.0;
    return c + ww * Vec3::new(1.0, 1.0, 1.0);
}

fn trace(
    scene: &[Object],
    rng: &mut UniformRng,
    ray: &Ray,
    depth: u32,
    bg: &dyn Fn(&Vec3) -> Vec3,
) -> Vec3 {
    // println!("{:?}",ray);
    if depth > 10 {
        // println!("depth over {:?}",ray);
        return Vec3::new(0., 0., 0.);
    }
    let p = project(scene, rng, ray);
    // println!("{:?}",p);
    return match p {
        Some(x) => trace(scene, rng, &x, depth + 1, bg),
        None => bg(&ray.dir).component_mul(&ray.att),
    };
}

fn color_to_bytes(c: &Vec3) -> [u8; 4] {
    let s = c * 255.0;
    let r = s.x.clamp(0.0, 255.0) as u8;
    let g = s.y.clamp(0.0, 255.0) as u8;
    let b = s.z.clamp(0.0, 255.0) as u8;
    let a = 255u8;
    return [r, g, b, a];
}

fn test_image2(file_name: &str, width: usize, height: usize, count: usize, exp: f64) {
    let sph = Sphere {
        org: Vec3::new(0., 0., 1.0),
        rad: 1.0,
    };
    let sph2 = Sphere {
        org: Vec3::new(2.618, 0., 0.),
        rad: 1.0,
    };
    let sph3 = Sphere {
        org: Vec3::new(-2.618, 0., 0.),
        rad: 1.0,
    };
    // let sph4 = Sphere {
    //     org: Vec3::new(1., 0., -3.),
    //     rad: 1.0,
    // };
    let pla = Plane {
        org: Vec3::new(0., -1., 0.),
        norm: Vec3::new(0., 1., 0.),
    };
    let copper_like = FresnelDiffuse {
        // copper like
        f0: 0.7,
        alpha: 0.0,
        attm: Vec3::new(0.95, 0.64, 0.54),
        attd: Vec3::new(0.99, 0.99, 0.99),
    };
    let silver_like = FresnelDiffuse {
        // silver like
        f0: 0.9,
        alpha: 0.0,
        attm: Vec3::new(0.95, 0.93, 0.88),
        attd: Vec3::new(0.99, 0.99, 0.99),
    };
    let plastic_like = FresnelDiffuse {
        // plastic like
        f0: 0.06,
        alpha: 0.0,
        attm: Vec3::new(0.99, 0.99, 0.99),
        attd: Vec3::new(0.99, 0.99, 0.99),
    };
    let dif3 = Diffuse {
        att: Vec3::new(0.9, 0.9, 0.9),
    };
    let rmir = RefrMirror {
        att: Vec3::new(0.9, 0.9, 0.9),
        ior: 1.5,
    };
    let glass_like = FresnelRefraction {
        // glass like
        attm: Vec3::new(0.99, 0.99, 0.99),
        attr: Vec3::new(0.9, 0.9, 0.9),
        ior: 1.5,
        alpha: 0.0,
    };
    let scene: Vec<_> = vec![
        Object {
            sha: Sph(sph),
            //mtr: FreDif(fre0),
            //mtr: RefrMir(rmir),
            mtr: FreRefr(glass_like),
        },
        Object {
            sha: Sph(sph2),
            mtr: FreDif(copper_like),
        },
        Object {
            sha: Sph(sph3),
            mtr: FreDif(plastic_like),
        },
        // Object {
        //     sha: Sph(sph4),
        //     mtr: FreDif(fre0),
        // },
        Object {
            sha: Pla(pla),
            mtr: Dif(dif3),
        },
    ];
    let mut rng = UniformRng::seed_from_u64(0);
    let mut data: Vec<u8> = Vec::new();
    data.resize((width * height * 4) as usize, 0);
    let aspect = width as f64 / height as f64;
    let org = Vec3::new(-15.0, 6.0, 25.0);
    let dir0 = -org.normalize();
    let mut up = Vec3::new(0.0, 1.0, 0.0);
    let right = dir0.cross(&up).normalize();
    up = right.cross(&dir0);
    let fovt = 0.06;
    for j in 0usize..height {
        for i in 0usize..width {
            let x = (i as f64 / (width as f64 - 1.0) - 0.5) * 2.0 * aspect;
            let y = (-(j as f64) / (height as f64 - 1.0) + 0.5) * 2.0;
            // orthogonal
            // let org = Vec3::new(x, y, 2.0);
            // let dir = Vec3::new(0.0, 0.0, -1.0);
            // perspective
            //let org = Vec3::new(0.0, 0.5, 10.0);
            //let dir = -Vec3::new(0.2 * x, 0.2 * y, -1.0).normalize();
            let dir = -(dir0 + right * (x * fovt) + up * (y * fovt)).normalize();
            let att = Vec3::new(1.0, 1.0, 1.0);
            let mut color: Vec3 = nalgebra::zero();
            for _ in 0..count {
                color += trace(&scene, &mut rng, &Ray { org, dir, att }, 0, &bg_color);
            }
            color /= count as f64;
            let [r, g, b, a] = color_to_bytes(&(color * exp));
            data[(j * width + i) * 4 + 0] = r;
            data[(j * width + i) * 4 + 1] = g;
            data[(j * width + i) * 4 + 2] = b;
            data[(j * width + i) * 4 + 3] = a;
        }
    }
    write_image(file_name, width, height, &data);
}

fn test_image3(width: usize, height: usize, count: usize, exp: f64, col: i32, data: &mut Vec<u8>) {
    let sph = Sphere {
        org: Vec3::new(0., 0., 1.0),
        rad: 1.0,
    };
    let sph2 = Sphere {
        org: Vec3::new(2.618, 0., 0.),
        rad: 1.0,
    };
    let sph3 = Sphere {
        org: Vec3::new(-2.618, 0., 0.),
        rad: 1.0,
    };
    // let sph4 = Sphere {
    //     org: Vec3::new(1., 0., -3.),
    //     rad: 1.0,
    // };
    let pla = Plane {
        org: Vec3::new(0., -1., 0.),
        norm: Vec3::new(0., 1., 0.),
    };
    let cop_ = vec![0.95, 0.64, 0.54][col as usize];
    let copper_like = FresnelDiffuse {
        // copper like
        f0: 0.7,
        alpha: 0.0,
        attm: Vec3::new(cop_, cop_, cop_),
        attd: Vec3::new(0.99, 0.99, 0.99),
    };
    // let silver_like = FresnelDiffuse {
    //     // silver like
    //     f0: 0.9,
    //     attm: Vec3::new(0.95, 0.93, 0.88),
    //     attd: Vec3::new(0.99, 0.99, 0.99),
    // };
    let plastic_like = FresnelDiffuse {
        // plastic like
        f0: 0.06,
        alpha: 0.0,
        attm: Vec3::new(0.99, 0.99, 0.99),
        attd: Vec3::new(0.99, 0.99, 0.99),
    };
    let dif3 = Diffuse {
        att: Vec3::new(0.9, 0.9, 0.9),
    };
    let rmir = RefrMirror {
        att: Vec3::new(0.9, 0.9, 0.9),
        ior: 1.5,
    };
    let gla_ = vec![1.518, 1.526, 1.532][col as usize];
    let glass_like = FresnelRefraction {
        // glass like
        attm: Vec3::new(0.99, 0.99, 0.99),
        attr: Vec3::new(0.9, 0.9, 0.9),
        ior: gla_,
        alpha: 0.0,
    };
    let scene: Vec<_> = vec![
        Object {
            sha: Sph(sph),
            //mtr: FreDif(fre0),
            //mtr: RefrMir(rmir),
            mtr: FreRefr(glass_like),
        },
        Object {
            sha: Sph(sph2),
            mtr: FreDif(copper_like),
        },
        Object {
            sha: Sph(sph3),
            mtr: FreDif(plastic_like),
        },
        // Object {
        //     sha: Sph(sph4),
        //     mtr: FreDif(fre0),
        // },
        Object {
            sha: Pla(pla),
            mtr: Dif(dif3),
        },
    ];
    let bg_colorx = |dir: &Vec3| {
        let c = bg_color(dir)[col as usize];
        return Vec3::new(c, c, c);
    };
    let mut rng = UniformRng::seed_from_u64(0);
    //let mut data: Vec<u8> = Vec::new();
    data.resize((width * height * 4) as usize, 0);
    let aspect = width as f64 / height as f64;
    let org = Vec3::new(-15.0, 6.0, 25.0);
    let dir0 = -org.normalize();
    let mut up = Vec3::new(0.0, 1.0, 0.0);
    let right = dir0.cross(&up).normalize();
    up = right.cross(&dir0);
    let fovt = 0.06;
    let dx = 1.0 / (width as f64 - 1.0) * 2.0 * aspect;
    let dy = 1.0 / (height as f64 - 1.0) * 2.0;
    for j in 0usize..height {
        for i in 0usize..width {
            let x = (i as f64 / (width as f64 - 1.0) - 0.5) * 2.0 * aspect;
            let y = (-(j as f64) / (height as f64 - 1.0) + 0.5) * 2.0;
            // orthogonal
            // let org = Vec3::new(x, y, 2.0);
            // let dir = Vec3::new(0.0, 0.0, -1.0);
            // perspective
            //let org = Vec3::new(0.0, 0.5, 10.0);
            //let dir = -Vec3::new(0.2 * x, 0.2 * y, -1.0).normalize();
            let att = Vec3::new(1.0, 1.0, 1.0);
            let mut color: Vec3 = nalgebra::zero();
            for _ in 0..count {
                let rx = dx * (rng.get() - 0.5);
                let ry = dy * (rng.get() - 0.5);
                let dir = -(dir0 + right * ((x + rx) * fovt) + up * ((y + ry) * fovt)).normalize();
                color += trace(&scene, &mut rng, &Ray { org, dir, att }, 0, &bg_colorx);
            }
            color /= count as f64;
            let [r, _, _, _] = color_to_bytes(&(color * exp));
            data[(j * width + i) * 4 + col as usize] = r;
            // data[(j * width + i) * 4 + 1] = g;
            // data[(j * width + i) * 4 + 2] = b;
            data[(j * width + i) * 4 + 3] = 255;
        }
    }
    // write_image(file_name, width, height, &data);
}

fn test_image4(
    file_name: &str,
    width: usize,
    height: usize,
    count: usize,
    hdr: &HDRData,
    exp: f64,
) {
    let sph = Sphere {
        org: Vec3::new(0., -1.0, 0.0),
        rad: 1.0,
    };
    let sph2 = Sphere {
        org: Vec3::new(2.618, 0., 0.),
        rad: 1.0,
    };
    let sph3 = Sphere {
        org: Vec3::new(-2.618, 0., 0.),
        rad: 1.0,
    };
    let sph4 = Sphere {
        // small
        org: Vec3::new(-0.618, 1.0, -1.0 + 0.381),
        rad: 0.381,
    };
    let sph5 = Sphere {
        // small
        org: Vec3::new(1.618, -1.618, -1.0 + 0.381),
        rad: 0.381,
    };
    let sph6 = Sphere {
        // small
        org: Vec3::new(0.618, 2.618, -1.0 + 0.381),
        rad: 0.381,
    };
    // let sph4 = Sphere {
    //     org: Vec3::new(1., 0., -3.),
    //     rad: 1.0,
    // };
    let pla = Plane {
        org: Vec3::new(0., 0.0, -1.0),
        norm: Vec3::new(0., 0.0, 1.0),
    };
    let copper_like = FresnelDiffuse {
        // copper like
        f0: 0.7,
        alpha: 5000.0,
        attm: Vec3::new(0.95, 0.64, 0.54),
        attd: Vec3::new(0.95, 0.64, 0.54),
    };
    let silver_like = FresnelDiffuse {
        // silver like
        f0: 0.9,
        alpha: 50000.0,
        attm: Vec3::new(0.95, 0.93, 0.88),
        attd: Vec3::new(0.95, 0.93, 0.88),
    };
    let plastic_like = FresnelDiffuse {
        // plastic like
        f0: 0.06,
        alpha: 1000.0,
        attm: Vec3::new(0.90, 0.90, 0.90),
        attd: Vec3::new(0.90, 0.90, 0.90),
    };
    let plastic_like_red = FresnelDiffuse {
        // plastic like
        f0: 0.06,
        alpha: 500.0,
        attm: Vec3::new(0.90, 0.90, 0.90),
        attd: Vec3::new(0.90, 0.10, 0.10),
    };
    let plastic_like_green = FresnelDiffuse {
        // plastic like
        f0: 0.06,
        alpha: 2000.0,
        attm: Vec3::new(0.90, 0.90, 0.90),
        attd: Vec3::new(0.30, 0.90, 0.20),
    };
    let plastic_like_blue = FresnelDiffuse {
        // plastic like
        f0: 0.06,
        alpha: 100.0,
        attm: Vec3::new(0.90, 0.90, 0.90),
        attd: Vec3::new(0.20, 0.30, 0.80),
    };
    let dif3 = Diffuse {
        att: Vec3::new(0.7, 0.7, 0.7),
    };
    let rmir = RefrMirror {
        att: Vec3::new(0.9, 0.9, 0.9),
        ior: 1.5,
    };
    let glass_like = FresnelRefraction {
        // glass like
        attm: Vec3::new(0.95, 0.95, 0.95),
        attr: Vec3::new(0.90, 0.90, 0.90),
        ior: 1.5,
        alpha: 10.0,
    };
    let sapphire_like = FresnelRefraction {
        // sapphire like
        attm: Vec3::new(0.95, 0.95, 0.95),
        attr: Vec3::new(0.20, 0.30, 0.90),
        ior: 1.76,
        alpha: 10.0,
    };
    let scene: Vec<_> = vec![
        Object {
            sha: Sph(sph),
            //mtr: FreDif(fre0),
            //mtr: RefrMir(rmir),
            mtr: FreRefr(glass_like),
        },
        Object {
            sha: Sph(sph2),
            mtr: FreDif(silver_like),
        },
        Object {
            sha: Sph(sph3),
            mtr: FreDif(plastic_like),
        },
        Object {
            sha: Sph(sph4),
            mtr: FreRefr(sapphire_like),
        },
        Object {
            sha: Sph(sph5),
            mtr: FreDif(plastic_like_green),
        },
        Object {
            sha: Sph(sph6),
            mtr: FreDif(copper_like),
        },
        // Object {
        //     sha: Sph(sph4),
        //     mtr: FreDif(fre0),
        // },
        Object {
            sha: Pla(pla),
            mtr: Dif(dif3),
        },
    ];
    let bg_colorx = |dir: &Vec3| {
        let dr = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt();
        let ay: f64;
        let ax: f64;
        if dr < 1e-8 {
            ax = 0.0;
            if dir[2] > 0.0 {
                ay = 0.0;
            } else {
                ay = 1.0;
            }
        } else {
            let p = -dir[1].atan2(dir[0]);
            let t = dir[2].atan2(dr);
            ax = (p / (2.0 * PI)) + 0.5;
            ay = (t / PI) + 0.5;
        }
        let x = ((ax * hdr.w as f64) as usize).clamp(0, hdr.w - 1);
        let y = ((ay * hdr.h as f64) as usize).clamp(0, hdr.h - 1);
        return hdr.data[y * hdr.w + x];
    };
    let mut rng = UniformRng::seed_from_u64(0);
    let mut data: Vec<u8> = Vec::new();
    data.resize((width * height * 4) as usize, 0);
    let aspect = width as f64 / height as f64;
    let org = Vec3::new(-15.0, -25.0, 8.0);
    let dir0 = -org.normalize();
    let mut up = Vec3::new(0.0, 0.0, 1.0);
    let right = dir0.cross(&up).normalize();
    up = right.cross(&dir0);
    let fovt = 0.06;
    let dx = 1.0 / (width as f64 - 1.0) * aspect;
    let dy = 1.0 / (height as f64 - 1.0);
    let mut internal_data: Vec<Vec3> = Vec::new();
    internal_data.resize(width * height, Vec3::new(0.0, 0.0, 0.0));
    for c in 0..count {
        println!("count {}", c);
        for j in 0usize..height {
            for i in 0usize..width {
                let x = (i as f64 / (width as f64 - 1.0) - 0.5) * 2.0 * aspect;
                let y = (-(j as f64) / (height as f64 - 1.0) + 0.5) * 2.0;
                // orthogonal
                // let org = Vec3::new(x, y, 2.0);
                // let dir = Vec3::new(0.0, 0.0, -1.0);
                // perspective
                //let org = Vec3::new(0.0, 0.5, 10.0);
                //let dir = -Vec3::new(0.2 * x, 0.2 * y, -1.0).normalize();
                let dir = -(dir0 + right * (x * fovt) + up * (y * fovt)).normalize();
                let att = Vec3::new(1.0, 1.0, 1.0);
                let mut color = internal_data[j * width + i];
                //for _ in 0..count {
                let rx = dx * (rng.get() - 0.5);
                let ry = dy * (rng.get() - 0.5);
                let dir = -(dir0 + right * ((x + rx) * fovt) + up * ((y + ry) * fovt)).normalize();
                color += trace(&scene, &mut rng, &Ray { org, dir, att }, 0, &bg_colorx);
                //}
                internal_data[j * width + i] = color;
                // color /= count as f64;
                // let gamma_rev = 1.0 / 2.2;
                // let rr = (color[0] * exp).powf(gamma_rev);
                // let gg = (color[1] * exp).powf(gamma_rev);
                // let bb = (color[2] * exp).powf(gamma_rev);
                // let [r, g, b, a] = color_to_bytes(&Vec3::new(rr, gg, bb));
                // data[(j * width + i) * 4 + 0] = r;
                // data[(j * width + i) * 4 + 1] = g;
                // data[(j * width + i) * 4 + 2] = b;
                // data[(j * width + i) * 4 + 3] = a;
            }
        }
        // dump
        for j in 0usize..height {
            for i in 0usize..width {
                let mut color = internal_data[j * width + i];
                color /= (c + 1) as f64;
                let gamma_rev = 1.0 / 2.2;
                let rr = (color[0] * exp).powf(gamma_rev);
                let gg = (color[1] * exp).powf(gamma_rev);
                let bb = (color[2] * exp).powf(gamma_rev);
                let [r, g, b, a] = color_to_bytes(&Vec3::new(rr, gg, bb));
                data[(j * width + i) * 4 + 0] = r;
                data[(j * width + i) * 4 + 1] = g;
                data[(j * width + i) * 4 + 2] = b;
                data[(j * width + i) * 4 + 3] = a;
            }
        }
        let dummy_name = file_name.to_string() + "dummy.png";
        write_image(&dummy_name, width, height, &data);
    }
    write_image(file_name, width, height, &data);
}

fn test_image5(
    file_name: &str,
    width: usize,
    height: usize,
    count: usize,
    hdr: &HDRData,
    exp: f64,
) {
    let sph = Sphere {
        org: Vec3::new(0., 0.0, 0.0),
        rad: 1.0,
    };
    // let sph2 = Sphere {
    //     org: Vec3::new(2.618, 0., 0.),
    //     rad: 1.0,
    // };
    // let sph3 = Sphere {
    //     org: Vec3::new(-2.618, 0., 0.),
    //     rad: 1.0,
    // };
    // let sph4 = Sphere {
    //     // small
    //     org: Vec3::new(-0.618, 1.0, -1.0 + 0.381),
    //     rad: 0.381,
    // };
    // let sph5 = Sphere {
    //     // small
    //     org: Vec3::new(1.618, -1.618, -1.0 + 0.381),
    //     rad: 0.381,
    // };
    // let sph6 = Sphere {
    //     // small
    //     org: Vec3::new(0.618, 2.618, -1.0 + 0.381),
    //     rad: 0.381,
    // };
    // let sph4 = Sphere {
    //     org: Vec3::new(1., 0., -3.),
    //     rad: 1.0,
    // };
    let pla = Plane {
        org: Vec3::new(0., 0.0, -1.0),
        norm: Vec3::new(0., 0.0, 1.0),
    };
    // let copper_like = FresnelDiffuse {
    //     // copper like
    //     f0: 0.7,
    //     alpha: 5000.0,
    //     attm: Vec3::new(0.95, 0.64, 0.54),
    //     attd: Vec3::new(0.95, 0.64, 0.54),
    // };
    // let silver_like = FresnelDiffuse {
    //     // silver like
    //     f0: 0.9,
    //     alpha: 50000.0,
    //     attm: Vec3::new(0.95, 0.93, 0.88),
    //     attd: Vec3::new(0.95, 0.93, 0.88),
    // };
    // let plastic_like = FresnelDiffuse {
    //     // plastic like
    //     f0: 0.06,
    //     alpha: 1000.0,
    //     attm: Vec3::new(0.90, 0.90, 0.90),
    //     attd: Vec3::new(0.90, 0.90, 0.90),
    // };
    // let plastic_like_red = FresnelDiffuse {
    //     // plastic like
    //     f0: 0.06,
    //     alpha: 500.0,
    //     attm: Vec3::new(0.90, 0.90, 0.90),
    //     attd: Vec3::new(0.90, 0.10, 0.10),
    // };
    // let plastic_like_green = FresnelDiffuse {
    //     // plastic like
    //     f0: 0.06,
    //     alpha: 2000.0,
    //     attm: Vec3::new(0.90, 0.90, 0.90),
    //     attd: Vec3::new(0.30, 0.90, 0.20),
    // };
    // let plastic_like_blue = FresnelDiffuse {
    //     // plastic like
    //     f0: 0.06,
    //     alpha: 100.0,
    //     attm: Vec3::new(0.90, 0.90, 0.90),
    //     attd: Vec3::new(0.20, 0.30, 0.80),
    // };
    let dif3 = Diffuse {
        att: Vec3::new(0.7, 0.7, 0.7),
    };
    let dif1 = Diffuse {
        att: Vec3::new(0.9, 0.9, 0.9),
    };
    let fredif1 = FresnelDiffuse {
        f0: 0.99,
        alpha: 100.0,
        attm: Vec3::new(0.9, 0.9, 0.9),
        attd: Vec3::new(0.9, 0.9, 0.9),
    };
    let mir1 = Mirror {
        att: Vec3::new(0.9, 0.9, 0.9),
    };
    let frefr1 = FresnelRefraction {
        attm: Vec3::new(0.9, 0.9, 0.9),
        attr: Vec3::new(0.9, 0.9, 0.9),
        ior: 1.5,
        alpha: 0.0,
    };
    // let rmir = RefrMirror {
    //     att: Vec3::new(0.9, 0.9, 0.9),
    //     ior: 1.5,
    // };
    // let glass_like = FresnelRefraction {
    //     // glass like
    //     attm: Vec3::new(0.95, 0.95, 0.95),
    //     attr: Vec3::new(0.90, 0.90, 0.90),
    //     ior: 1.5,
    //     alpha: 10.0,
    // };
    // let sapphire_like = FresnelRefraction {
    //     // sapphire like
    //     attm: Vec3::new(0.95, 0.95, 0.95),
    //     attr: Vec3::new(0.20, 0.30, 0.90),
    //     ior: 1.76,
    //     alpha: 10.0,
    // };
    let scene: Vec<_> = vec![
        Object {
            sha: Sph(sph),
            //mtr: FreDif(fre0),
            //mtr: RefrMir(rmir),
            //mtr: Dif(dif1),
            //mtr: FreDif(fredif1),
            //mtr: Mir(mir1),
            mtr: FreRefr(frefr1),
        },
        // Object {
        //     sha: Sph(sph2),
        //     mtr: FreDif(silver_like),
        // },
        // Object {
        //     sha: Sph(sph3),
        //     mtr: FreDif(plastic_like),
        // },
        // Object {
        //     sha: Sph(sph4),
        //     mtr: FreRefr(sapphire_like),
        // },
        // Object {
        //     sha: Sph(sph5),
        //     mtr: FreDif(plastic_like_green),
        // },
        // Object {
        //     sha: Sph(sph6),
        //     mtr: FreDif(copper_like),
        // },
        // Object {
        //     sha: Sph(sph4),
        //     mtr: FreDif(fre0),
        // },
        Object {
            sha: Pla(pla),
            mtr: Dif(dif3),
        },
    ];
    let bg_colorx = |dir: &Vec3| {
        let dr = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt();
        let ay: f64;
        let ax: f64;
        if dr < 1e-8 {
            ax = 0.0;
            if dir[2] > 0.0 {
                ay = 0.0;
            } else {
                ay = 1.0;
            }
        } else {
            let p = -dir[1].atan2(dir[0]);
            let t = dir[2].atan2(dr);
            ax = (p / (2.0 * PI)) + 0.5;
            ay = (t / PI) + 0.5;
        }
        let x = ((ax * hdr.w as f64) as usize).clamp(0, hdr.w - 1);
        let y = ((ay * hdr.h as f64) as usize).clamp(0, hdr.h - 1);
        return hdr.data[y * hdr.w + x];
    };
    let mut rng = UniformRng::seed_from_u64(0);
    let mut data: Vec<u8> = Vec::new();
    data.resize((width * height * 4) as usize, 0);
    let aspect = width as f64 / height as f64;
    let org = Vec3::new(-15.0, -25.0, 8.0);
    let dir0 = -org.normalize();
    let mut up = Vec3::new(0.0, 0.0, 1.0);
    let right = dir0.cross(&up).normalize();
    up = right.cross(&dir0);
    let fovt = 0.05;
    let dx = 1.0 / (width as f64 - 1.0) * aspect;
    let dy = 1.0 / (height as f64 - 1.0);
    let mut internal_data: Vec<Vec3> = Vec::new();
    internal_data.resize(width * height, Vec3::new(0.0, 0.0, 0.0));
    for c in 0..count {
        println!("count {}", c);
        for j in 0usize..height {
            for i in 0usize..width {
                let x = (i as f64 / (width as f64 - 1.0) - 0.5) * 2.0 * aspect;
                let y = (-(j as f64) / (height as f64 - 1.0) + 0.5) * 2.0;
                // orthogonal
                // let org = Vec3::new(x, y, 2.0);
                // let dir = Vec3::new(0.0, 0.0, -1.0);
                // perspective
                //let org = Vec3::new(0.0, 0.5, 10.0);
                //let dir = -Vec3::new(0.2 * x, 0.2 * y, -1.0).normalize();
                // let dir = -(dir0 + right * (x * fovt) + up * (y * fovt)).normalize();
                let att = Vec3::new(1.0, 1.0, 1.0);
                let mut color = internal_data[j * width + i];
                //for _ in 0..count {
                let rx = dx * (rng.get() - 0.5);
                let ry = dy * (rng.get() - 0.5);
                let dir = -(dir0 + right * ((x + rx) * fovt) + up * ((y + ry) * fovt)).normalize();
                color += trace(&scene, &mut rng, &Ray { org, dir, att }, 0, &bg_colorx);
                //}
                internal_data[j * width + i] = color;
                // color /= count as f64;
                // let gamma_rev = 1.0 / 2.2;
                // let rr = (color[0] * exp).powf(gamma_rev);
                // let gg = (color[1] * exp).powf(gamma_rev);
                // let bb = (color[2] * exp).powf(gamma_rev);
                // let [r, g, b, a] = color_to_bytes(&Vec3::new(rr, gg, bb));
                // data[(j * width + i) * 4 + 0] = r;
                // data[(j * width + i) * 4 + 1] = g;
                // data[(j * width + i) * 4 + 2] = b;
                // data[(j * width + i) * 4 + 3] = a;
            }
        }
        // dump
        for j in 0usize..height {
            for i in 0usize..width {
                let mut color = internal_data[j * width + i];
                color /= (c + 1) as f64;
                let gamma_rev = 1.0 / 2.2;
                let rr = (color[0] * exp).powf(gamma_rev);
                let gg = (color[1] * exp).powf(gamma_rev);
                let bb = (color[2] * exp).powf(gamma_rev);
                let [r, g, b, a] = color_to_bytes(&Vec3::new(rr, gg, bb));
                data[(j * width + i) * 4 + 0] = r;
                data[(j * width + i) * 4 + 1] = g;
                data[(j * width + i) * 4 + 2] = b;
                data[(j * width + i) * 4 + 3] = a;
            }
        }
        let dummy_name = file_name.to_string() + "dummy.png";
        write_image(&dummy_name, width, height, &data);
    }
    write_image(file_name, width, height, &data);
}

fn test_hdr() {
    let file_name = "MR_INT-001_NaturalStudio_NAD.hdr";
    // let path = Path::new(file_name);
    // let file = File::open(path).unwrap();
    // let mut rd = BufReader::new(file);
    // let mut buf = String::new();
    // let len = rd.read_line(&mut buf).unwrap();
    // let len = rd.read_line(&mut buf).unwrap();
    // let len = rd.read_line(&mut buf).unwrap();
    // let len = rd.read_line(&mut buf).unwrap();
    // buf.clear();
    // let len = rd.read_line(&mut buf).unwrap();
    // println!("{}", buf);

    let w = 6840usize;
    let h = 3420usize;

    // let mut bf0: Vec<u8> = Vec::new();
    // rd.read_to_end(&mut bf0);

    // println!("{}", bf0.len());

    // let mut bfl: Vec<u8> = Vec::new();
    // bfl.resize(w * 4, 0);

    // let mut ofs = 0usize;
    // for y in 0..h {
    //     let cur = read_line(&bf0, ofs, w, &mut bfl);
    //     ofs += cur;
    //     for x in 0..w {
    //         let r0 = bfl[x * 4 + 0] as f64;
    //         let g0 = bfl[x * 4 + 1] as f64;
    //         let b0 = bfl[x * 4 + 2] as f64;
    //         let e0 = bfl[x * 4 + 3] as f64;
    //         let r = r0 * (e0 - 128.0).exp2();
    //         let g = g0 * (e0 - 128.0).exp2();
    //         let b = b0 * (e0 - 128.0).exp2();
    //         let [rx, gx, bx, ax] = color_to_bytes(&(Vec3::new(r, g, b) * 0.02));
    //         data[(y * w + x) * 4 + 0] = rx;
    //         data[(y * w + x) * 4 + 1] = gx;
    //         data[(y * w + x) * 4 + 2] = bx;
    //         data[(y * w + x) * 4 + 3] = 255;
    //     }
    // }

    let hdr_data = read_hdr(file_name, w, h).unwrap();
    let dat = &hdr_data.data;

    println!("converted");

    let mut data: Vec<u8> = Vec::new();
    data.resize(w * h * 4, 0);

    for y in 0..h {
        for x in 0..w {
            let val = dat[y * w + x];
            let scale = 0.005;
            let gamma_rev = 1.0 / 2.2;
            let r = (val[0] * scale).powf(gamma_rev);
            let g = (val[1] * scale).powf(gamma_rev);
            let b = (val[2] * scale).powf(gamma_rev);
            let [rx, gx, bx, ax] = color_to_bytes(&Vec3::new(r, g, b));
            data[(y * w + x) * 4 + 0] = rx;
            data[(y * w + x) * 4 + 1] = gx;
            data[(y * w + x) * 4 + 2] = bx;
            data[(y * w + x) * 4 + 3] = ax;
        }
    }

    write_image("hoge.png", w, h, &data);
    //let lines = r.lines();

    // let f = File::open("hoge.txt").unwrap();
    // let mut reader = BufReader::new(f);
    // // let lines = reader.lines();
    // let mut c = String::new();
    // reader.read_to_string(&mut c);
}

fn main() {
    println!("Hello, world!");

    // test_image("test-image.png");

    // let mut count = 1000;
    // let mut w = 910;
    // let mut h = 512;
    // if cfg!(debug_assertions) {
    //     count = 5;
    //     w = 455;
    //     h = 256;
    // }
    // // final
    // count = 4000;
    // w = 1920;
    // h = 1080;

    // test_image2("test-image2.png", w, h, count, 1.0);

    // let mut data: Vec<u8> = Vec::new();
    // println!("r");
    // test_image3(w, h, count, 1.0, 0, &mut data);
    // write_image("test-image3-r.png", w, h, &data);
    // println!("g");
    // test_image3(w, h, count, 1.0, 1, &mut data);
    // write_image("test-image3-rg.png", w, h, &data);
    // println!("b");
    // test_image3(w, h, count, 1.0, 2, &mut data);
    // write_image("test-image3.png", w, h, &data);

    // test_hdr();

    let hdr = read_hdr("MR_INT-001_NaturalStudio_NAD.hdr", 6840, 3420).unwrap();
    // test_image4("test-image4.png", w, h, count, &hdr, 0.002);

    let mut count = 4000;
    let mut w = 512;
    let mut h = 512;
    if cfg!(debug_assertions) {
        count = 5;
        w = 256;
        h = 256;
    }
    // // final
    // count = 4000;
    // w = 1920;
    // h = 1080;

    test_image5("test-image5.png", w, h, count, &hdr, 0.002);

    // let mut rng = UniformRng::seed_from_u64(0);
    // for _ in 0..100 {
    //     // let r: f64 = rng.get();
    //     // let v=sample_hemi_cos(&mut rng);
    //     let v=sample_hemi_cos_pow(&mut rng, 100.0);
    //     println!("{} {} {}", v.x,v.y,v.z);
    // }

    // let mut rng = UniformRng::seed_from_u64(0);
    // let r = reflect_lambert(&Vec3::new(1.0, 0.0, 0.0), &mut rng);
    // println!("{:?}", r);

    // let org = Vec3::new(0.0, 0.0, 2.0);
    // let dir = Vec3::new(0.0, 1.0, 1.0).normalize();
    // let color = trace(&Ray { org, dir }, 0);
    // println!("{:?}",color);
    // let bytes = color_to_bytes(&color);
    // println!("{:?}",bytes);

    // let ref0 = refract_mirror(
    //     &Vec3::new(-1.0, 0.0, 1.0).normalize(),
    //     &Vec3::new(0.0, 0.0, 1.0),
    //     1.5,
    // );
    // println!("{:?}", ref0);
    // let ref1 = refract_mirror(
    //     &Vec3::new(-1.0, 0.0, 1.0).normalize(),
    //     &Vec3::new(0.0, 0.0, -1.0),
    //     1.5,
    // );
    // println!("{:?}", ref1);

    println!("test_image out.");
}
