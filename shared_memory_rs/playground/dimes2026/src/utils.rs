
use clap::Parser;

/// Search for a pattern in a file and display the lines that contain it.
#[derive(Debug, Parser)]
pub struct Cli {
    /// 0 => baseline, 1 => CXL with 1 Host, n => CXL with n Hosts
    #[arg(short, long, default_value_t = 0)]
    pub num_hosts: usize,

    /// Capacity parameter when creating the hashtable
    #[arg(long, default_value_t = 256)]
    pub ht_buckets: usize,

    /// Capacity parameter when creating a bucket within the hashtable
    #[arg(long, default_value_t = 16)]
    pub ht_bucket_capacity: usize,

    /// buildfile
    #[arg(short,long, default_value_t = String::from("none"))]
    pub buildfile : String,

    /// probefile
    #[arg(short,long, default_value_t = String::from("none"))]
    pub probefile : String,
}


#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct RowID(pub u64);
impl RowID{
    pub fn from_value(v:u64) -> Self {RowID(v)}
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct JoinKey(pub u64);

// Always used as the build side -> should be smaller than the other table
#[derive(Debug)]
#[repr(C)]
pub struct TableRow {
    pub row: RowID,
    pub join_key: JoinKey,
}


pub fn get_build_table_simple() -> Vec<TableRow> {
    let start_row_id: u64 = 1000_0000;
    vec![
        TableRow {
            join_key: JoinKey(1),
            row: RowID(start_row_id),
        },
        TableRow {
            join_key: JoinKey(2),
            row: RowID(start_row_id+1),
        },
        TableRow {
            join_key: JoinKey(2),
            row: RowID(start_row_id+2),
        },
        TableRow {
            join_key: JoinKey(4),
            row: RowID(start_row_id+3),
        },
        TableRow {
            join_key: JoinKey(1),
            row: RowID(start_row_id+4),
        },
        TableRow {
            join_key: JoinKey(7),
            row: RowID(start_row_id+5),
        },
    ]
}

pub fn get_probe_table_simple() -> Vec<TableRow> {
    let start_row_id: u64 = 8000_0000;
    vec![
        TableRow {
            join_key: JoinKey(1),
            row: RowID(start_row_id+1),
        },
        TableRow {
            join_key: JoinKey(2),
            row: RowID(start_row_id+2),
        },
        TableRow {
            join_key: JoinKey(3),
            row: RowID(start_row_id+3),
        },
        TableRow {
            join_key: JoinKey(3),
            row: RowID(start_row_id+4),
        },
        TableRow {
            join_key: JoinKey(3),
            row: RowID(start_row_id+5),
        },
        TableRow {
            join_key: JoinKey(7),
            row: RowID(start_row_id+6),
        },
    ]
}
