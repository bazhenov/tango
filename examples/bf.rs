fn main() {
    let iterate_func = Func { func: iterate };
    let sort_func = SetupFunc {
        setup: prepare,
        inner: Func { func: sort },
    };
    let v = vec![0, 1, 2, 3];

    sort_func.measure(&v);
    iterate_func.measure(&v);
}

fn iterate(i: &Vec<u8>) -> u8 {
    for _ in i {}
    0
}

fn sort(mut i: Vec<u8>) -> u8 {
    i.sort();
    0
}

fn prepare(i: &Vec<u8>) -> Vec<u8> {
    i.clone()
}

trait Measure<P, O> {
    fn measure(&self, payload: P) -> (u64, O);
}

struct SetupFunc<S, M> {
    setup: S,
    inner: M,
}

impl<S, F, P, I, O> Measure<P, O> for SetupFunc<S, F>
where
    S: Fn(P) -> I,
    F: Measure<I, O>,
{
    fn measure(&self, payload: P) -> (u64, O) {
        let setup = (self.setup)(payload);
        self.inner.measure(setup)
    }
}

struct Func<F> {
    func: F,
}

impl<F, P, O> Measure<P, O> for Func<F>
where
    F: Fn(P) -> O,
{
    fn measure(&self, payload: P) -> (u64, O) {
        let time = 0;
        (0, (self.func)(payload))
    }
}
