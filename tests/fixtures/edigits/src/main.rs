//! e-digits: print e to N decimal places, truncated (not rounded).
//!
//! Supported range: 1 <= N <= 10000. Anything else (missing argument, non-digits,
//! signs, embedded whitespace, zero, or an unreasonable N) exits with status 2 and
//! a usage line on stderr. No external crates: the Peter worker sandbox builds
//! offline, so this uses only std with a small base-1e9 big-integer core.
//!
//! Method: floor(e * 10^(N+G)) = sum_k floor-accumulated 10^(N+G)/k! via repeated
//! integer division term/=k until the term vanishes. Each floor loses <1 unit, so
//! the result sits within ~K units of e*10^(N+G). G=20 guard digits make the
//! first N digits exact unless e's decimal has a 9-run straddling the cut; the
//! run is recomputed with G=30 and must agree before any digit is printed, so a
//! disagreement fails loudly instead of emitting plausible wrong digits.

use std::process::exit;

const MAX_N: usize = 10000;
const GUARD_A: usize = 20;
const GUARD_B: usize = 30;
const BASE: u64 = 1_000_000_000;

/// Little-endian base-1e9 limbs; most significant limb is never zero unless empty.
#[derive(Clone, PartialEq)]
struct BigUint(Vec<u32>);

impl BigUint {
    fn is_zero(&self) -> bool {
        self.0.is_empty()
    }

    fn trim(&mut self) {
        while self.0.last() == Some(&0) {
            self.0.pop();
        }
    }

    fn mul_small(&mut self, factor: u64) {
        let mut carry = 0u64;
        for limb in self.0.iter_mut() {
            let product = *limb as u64 * factor + carry;
            *limb = (product % BASE) as u32;
            carry = product / BASE;
        }
        while carry > 0 {
            self.0.push((carry % BASE) as u32);
            carry /= BASE;
        }
    }

    /// Integer division by a small divisor; exact quotient, discarded remainder.
    /// Safe while carry * BASE + limb fits u64; the remainder carry stays below
    /// the divisor (<= ~400 for the series, < BASE for power-of-ten shifts).
    fn div_small_assign(&mut self, divisor: u64) {
        let mut carry = 0u64;
        for limb in self.0.iter_mut().rev() {
            let current = carry * BASE + *limb as u64;
            *limb = (current / divisor) as u32;
            carry = current % divisor;
        }
        self.trim();
    }

    fn add(&mut self, other: &BigUint) {
        let mut carry = 0u64;
        for index in 0..other.0.len() {
            let slot = self.0.get_mut(index).expect("self covers other");
            let sum = *slot as u64 + other.0[index] as u64 + carry;
            *slot = (sum % BASE) as u32;
            carry = sum / BASE;
        }
        let mut index = other.0.len();
        while carry > 0 {
            match self.0.get_mut(index) {
                Some(slot) => {
                    let sum = *slot as u64 + carry;
                    *slot = (sum % BASE) as u32;
                    carry = sum / BASE;
                }
                None => {
                    self.0.push(carry as u32);
                    carry = 0;
                }
            }
            index += 1;
        }
    }

    fn pow10(exponent: usize) -> BigUint {
        let mut value = BigUint(vec![1]);
        for _ in 0..exponent {
            value.mul_small(10);
        }
        value
    }

    /// floor(self / 10^exponent) by repeated exact small division.
    fn div_pow10_assign(&mut self, exponent: usize) {
        let full = exponent / 9;
        let rest = exponent % 9;
        for _ in 0..full {
            self.div_small_assign(BASE);
        }
        if rest > 0 {
            self.div_small_assign(10u64.pow(rest as u32));
        }
    }

    fn to_string(&self) -> String {
        if self.0.is_empty() {
            return "0".to_string();
        }
        let mut text = self.0.last().unwrap().to_string();
        for limb in self.0.iter().rev().skip(1) {
            text.push_str(&format!("{limb:09}"));
        }
        text
    }
}

/// floor(e * 10^scale) via the factorial series. Term_k = 10^scale/k! is
/// reached from term_{k-1} by truncating division; k=0 and k=1 both contribute
/// the full scale (1/0! = 1/1! = 1), so the accumulator starts at 2*10^scale.
fn e_scaled(scale: usize) -> BigUint {
    let mut term = BigUint::pow10(scale);
    let mut total = term.clone();
    total.add(&term);
    let mut k = 2u64;
    loop {
        term.div_small_assign(k);
        if term.is_zero() {
            break;
        }
        total.add(&term);
        k += 1;
    }
    total
}

/// e truncated to `digits` decimals, verified across two guard depths.
fn e_to_digits(digits: usize) -> Result<String, String> {
    let mut a = e_scaled(digits + GUARD_A);
    let mut b = e_scaled(digits + GUARD_B);
    a.div_pow10_assign(GUARD_A);
    b.div_pow10_assign(GUARD_B);
    if a != b {
        return Err("guard bands disagree; digits at this scale are not certified".to_string());
    }
    let text = a.to_string();
    // Integer string is (1 leading integer digit) + `digits` fractional digits.
    Ok(format!("{}.{}", &text[..1], &text[1..]))
}

fn usage() -> ! {
    eprintln!("usage: edigits N   (prints e to N decimal places; 1 <= N <= {MAX_N}, truncated)");
    exit(2);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        usage();
    }
    let raw = &args[1];
    // Strict rejection before any parsing: ASCII digits only, no sign/space/plus.
    if raw.is_empty() || !raw.bytes().all(|byte| byte.is_ascii_digit()) {
        usage();
    }
    let Ok(parsed) = raw.parse::<usize>() else {
        usage();
    };
    if parsed < 1 || parsed > MAX_N {
        eprintln!("edigits: N={parsed} is outside the supported range 1..={MAX_N}");
        exit(2);
    }
    match e_to_digits(parsed) {
        Ok(line) => println!("{line}"),
        Err(reason) => {
            eprintln!("edigits: {reason}");
            exit(3);
        }
    }
}
