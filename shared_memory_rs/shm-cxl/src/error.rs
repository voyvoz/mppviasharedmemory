use std::fmt::Display;

#[derive(Debug)]
pub enum ShmError {
    GenericError(String),
}

impl Display for ShmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShmError::GenericError(err) => {
                write!(f, " {}", err)
            }
        }
    }
}

impl std::error::Error for ShmError {}

impl From<std::io::Error> for ShmError {
    fn from(value: std::io::Error) -> ShmError {
        // Extract the information from io::Error
        // For example, `match`-ing on ErrorKind and turning that into a string
        ShmError::GenericError(value.to_string())
    }
}

pub type Result<T> = std::result::Result<T, ShmError>;
