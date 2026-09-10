use clap::Parser;

#[derive(Clone, Debug, Parser)]
pub struct Cli {
    /// 0 => baseline, n => CXL with 1 Host
    #[arg(short, long, default_value_t = 0)]
    pub num_hosts: usize,

    /// How many messages to send to each host from each host
    #[arg(short, long, default_value_t = 1000)]
    pub msgs: usize,

    ///How many consumer threads should be used
    #[arg(short, long, default_value_t = 1)]
    pub consumer_threads: usize,

    ///How many producer threads should be used
    #[arg(short, long, default_value_t = 1)]
    pub producer_threads: usize,
}
