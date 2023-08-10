#[repr(align(32))]
pub fn std(s: &String) -> usize {
    s.chars().count()
}

#[repr(align(32))]
pub fn std_count_rev(s: &String) -> usize {
    let mut l = 0;
    let mut chars = s.chars().rev();
    while chars.next().is_some() {
        l += 1;
    }
    l
}

#[repr(align(32))]
pub fn std_count(s: &String) -> usize {
    let mut l = 0;
    let mut chars = s.chars();
    while chars.next().is_some() {
        l += 1;
    }
    l
}

#[repr(align(32))]
pub fn std_5000(s: &String) -> usize {
    s.chars().take(5000).count()
}

#[repr(align(32))]
pub fn std_5000_dupl(s: &String) -> usize {
    s.chars().take(5000).count()
}

#[repr(align(32))]
pub fn std_4925(s: &String) -> usize {
    s.chars().take(4925).count()
}
