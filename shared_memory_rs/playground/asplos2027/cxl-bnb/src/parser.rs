use crate::Item;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Instance {
    pub name: String,
    pub n: usize,
    pub capacity: u64,
    pub optimal: u64,
    pub items: Vec<Item>,
}

impl Instance {
    /// Ground truth for correctness testing — profit sum of items flagged in_optimal.
    pub fn known_optimal_from_flags(&self) -> u64 {
        self.items
            .iter()
            .filter(|i| i.in_optimal)
            .map(|i| i.profit)
            .sum()
    }

    pub fn known_weight_from_flags(&self) -> u64 {
        self.items
            .iter()
            .filter(|i| i.in_optimal)
            .map(|i| i.weight)
            .sum()
    }
}

#[derive(Debug)]
pub enum ParseError {
    Malformed(String),
    Int(std::num::ParseIntError),
}

impl From<std::num::ParseIntError> for ParseError {
    fn from(e: std::num::ParseIntError) -> Self {
        ParseError::Int(e)
    }
}

/// Parses a Pisinger-format file, which may contain multiple instances
/// concatenated and separated by lines of dashes ("-----").
pub fn parse_file(path: impl AsRef<Path>) -> Result<Vec<Instance>, ParseError> {
    let content = fs::read_to_string(path).map_err(|e| ParseError::Malformed(e.to_string()))?;
    parse_str(&content)
}

pub fn parse_str(content: &str) -> Result<Vec<Instance>, ParseError> {
    let mut instances = Vec::new();
    let mut lines = content
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('-'))
        .peekable();

    while lines.peek().is_some() {
        let name = lines
            .next()
            .ok_or_else(|| ParseError::Malformed("expected instance name".into()))?
            .to_string();

        let n = parse_kv(lines.next(), "n")?;
        let capacity = parse_kv(lines.next(), "c")?;
        let optimal = parse_kv(lines.next(), "z")?;
        let _time = lines.next(); // ignored, present but unused

        let mut items = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let line = lines
                .next()
                .ok_or_else(|| ParseError::Malformed(format!("{name}: truncated item list")))?;
            let parts: Vec<&str> = line.split(',').map(str::trim).collect();
            if parts.len() != 4 {
                return Err(ParseError::Malformed(format!(
                    "{name}: bad item line '{line}'"
                )));
            }
            items.push(Item {
                id: parts[0].parse()?,
                profit: parts[1].parse()?,
                weight: parts[2].parse()?,
                in_optimal: parts[3].parse::<u32>()? != 0,
            });
        }

        instances.push(Instance {
            name,
            n: n as usize,
            capacity,
            optimal,
            items,
        });
    }

    Ok(instances)
}

fn parse_kv(line: Option<&str>, expected_key: &str) -> Result<u64, ParseError> {
    let line =
        line.ok_or_else(|| ParseError::Malformed(format!("expected '{expected_key} <value>'")))?;
    let mut parts = line.split_whitespace();
    let key = parts
        .next()
        .ok_or_else(|| ParseError::Malformed("empty line".into()))?;
    if key != expected_key {
        return Err(ParseError::Malformed(format!(
            "expected key '{expected_key}', got '{key}'"
        )));
    }
    let val = parts
        .next()
        .ok_or_else(|| ParseError::Malformed(format!("missing value for '{expected_key}'")))?;
    Ok(val.parse()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cli;

    const SAMPLE: &str = include_str!("../test_data/smallcoeff_pisinger/knapPI_1_50_1000.csv");

    #[test]
    fn parses_multiple_instances() {
        let instances = parse_str(SAMPLE).unwrap();
        assert_eq!(instances.len(), 100);
        assert_eq!(instances[0].name, "knapPI_1_50_1000_1");
        assert_eq!(instances[0].capacity, 995);
        assert_eq!(instances[0].optimal, 8373);
        assert_eq!(instances[1].name, "knapPI_1_50_1000_2");
        assert_eq!(instances[1].capacity, 997);
        assert_eq!(instances[1].optimal, 5847);
        // we dont care about the rest for now...
    }

    #[test]
    fn all_instances_self_consistent() {
        for inst in parse_str(SAMPLE).unwrap() {
            assert_eq!(
                inst.known_optimal_from_flags(),
                inst.optimal,
                "{} flag/z mismatch",
                inst.name
            );
            assert!(
                inst.known_weight_from_flags() <= inst.capacity,
                "{} exceeds capacity",
                inst.name
            );
        }
    }

    #[test]
    fn all_instances_solved_correctly() {
        let args = Cli::default();
        for inst in parse_str(SAMPLE).unwrap() {
            let result = crate::solver(inst.items, inst.capacity, &args).best;
            assert_eq!(result, inst.optimal, "{} solver mismatch", inst.name);
        }
    }
}
