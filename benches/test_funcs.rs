use rand::{rngs::SmallRng, Rng, SeedableRng};
use rust_pairwise_testing::Generator;
use std::{arch::asm, io};

#[derive(Clone)]
pub struct FixedStringGenerator {
    string: String,
}

impl Generator for FixedStringGenerator {
    type Output = String;

    fn next_payload(&mut self) -> Self::Output {
        self.string.clone()
    }
}

#[derive(Clone)]
pub struct RandomStringGenerator {
    string: String,
    char_indicies: Vec<usize>,
    rng: SmallRng,
    length: usize,
}

impl RandomStringGenerator {
    pub fn new() -> io::Result<Self> {
        let string = std::fs::read_to_string("./input.txt")?;
        let char_indicies = string
            .char_indices()
            .map(|(idx, _)| idx)
            .collect::<Vec<_>>();
        let rng = SmallRng::from_entropy();
        Ok(Self {
            string,
            char_indicies,
            rng,
            length: 50000,
        })
    }
}
impl Generator for RandomStringGenerator {
    type Output = String;

    fn next_payload(&mut self) -> Self::Output {
        let start = self
            .rng
            .gen_range(0..self.char_indicies.len() - self.length);

        let from = self.char_indicies[start];
        let to = self.char_indicies[start + self.length];
        self.string[from..to].to_string()
    }
}

const TIMES: usize = 1;

//#[repr(align(32))]
pub fn std(s: &String) -> usize {
    s.chars().count()
}

//#[repr(align(32))]
pub fn std_count(s: &String) -> usize {
    let mut l = 0;
    for _ in 0..TIMES {
        let mut chars = s.chars();
        while chars.next().is_some() {
            l += 1;
        }
    }
    l
}

//#[repr(align(32))]
pub fn std_count_rev(s: &String) -> usize {
    let mut l = 0;
    for _ in 0..TIMES {
        let mut chars = s.chars().rev();
        while chars.next().is_some() {
            l += 1;
        }
    }
    l
}

//#[repr(align(32))]
pub fn std_5000(s: &String) -> usize {
    let mut l = 0;
    for _ in 0..TIMES {
        l += s.chars().take(5000).count();
    }
    l
}

//#[repr(align(32))]
pub fn std_4925(s: &String) -> usize {
    let mut l = 0;
    for _ in 0..TIMES {
        l += s.chars().take(4925).count();
    }
    l / 2 + 100
}

//#[repr(align(32))]
pub fn std_5000_dupl(s: &String) -> usize {
    let mut l = 0;
    for _ in 0..TIMES {
        l += s.chars().take(5000).count();
    }
    l
}

#[inline(always)]
pub fn std_5000_n(s: &String, offset: usize) -> usize {
    let mut l = 0;
    for _ in 0..TIMES {
        l += s.chars().take(5000).count();
    }
    unsafe {
        asm!("nop");
        asm!("nop");
        asm!("nop");
        asm!("nop");
        asm!("nop");
        asm!("nop");
        asm!("nop");
        asm!("nop");
        asm!("nop");
        asm!("nop");
        asm!("nop");
        asm!("nop");
    }

    l + offset
}

//#[repr(align(32))]
pub fn std_5000_1(s: &String) -> usize {
    std_5000_n(s, 1)
}

//#[repr(align(32))]
pub fn std_5000_2(s: &String) -> usize {
    std_5000_n(s, 2)
}

//#[repr(align(32))]
pub fn std_5000_3(s: &String) -> usize {
    std_5000_n(s, 3)
}

//#[repr(align(32))]
pub fn std_5000_4(s: &String) -> usize {
    std_5000_n(s, 4)
}

//#[repr(align(32))]
pub fn std_5000_5(s: &String) -> usize {
    std_5000_n(s, 5)
}

//#[repr(align(32))]
pub fn std_5000_6(s: &String) -> usize {
    std_5000_n(s, 6)
}

//#[repr(align(32))]
pub fn std_5000_7(s: &String) -> usize {
    std_5000_n(s, 7)
}

//#[repr(align(32))]
pub fn std_5000_8(s: &String) -> usize {
    std_5000_n(s, 8)
}

//#[repr(align(32))]
pub fn std_5000_9(s: &String) -> usize {
    std_5000_n(s, 9)
}
