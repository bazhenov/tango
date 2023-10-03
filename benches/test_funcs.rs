use std::{arch::asm, hint::black_box};

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
