use mpi::ffi::MPI_DISTRIBUTE_CYCLIC;

pub fn idx(i: usize, j: usize, cols: usize) -> usize {
    i * cols + j
}

pub struct RowPartition {
    pub row_begin: usize,  // global inclusive
    pub row_end: usize,    // global exclusive
    pub local_rows: usize, // row_end - row_begin
    pub up_peer: Option<usize>,
    pub down_peer: Option<usize>,
}

impl RowPartition {
    pub fn new(rows: usize, part_idx: usize, workers: usize) -> Self {
        assert!(workers > 0);
        assert!(part_idx < workers);

        let interior_rows = rows.saturating_sub(2);
        let row_begin = 1 + part_idx * interior_rows / workers;
        let row_end = 1 + (part_idx + 1) * interior_rows / workers;
        let local_rows = row_end.saturating_sub(row_begin);

        let up_peer = if row_begin > 1 {
            Some(part_idx - 1)
        } else {
            None
        };

        let down_peer = if row_end < rows.saturating_sub(1) {
            Some(part_idx + 1)
        } else {
            None
        };

        RowPartition {
            row_begin,
            row_end,
            local_rows,
            up_peer,
            down_peer,
        }
    }

    pub fn owns_global_row(&self, global_row: usize) -> bool {
        self.row_begin <= global_row && global_row < self.row_end
    }

    pub fn global_to_local(&self, global_row: usize) -> usize {
        assert!(self.owns_global_row(global_row));
        (global_row - self.row_begin) + 1
    }
}

// The second piece is LocalGrid. Its only job is to provide a convenient local
// array interface for "owned rows + top/bottom halo".
// This can live in src/dist_exchange.rs right after RowPartition.

// local row 0              = top halo
// local row 1..local_rows  = owned rows
// local row local_rows+1   = bottom halo
pub struct LocalGrid {
    buf: Vec<f64>,
    local_rows: usize,
    cols: usize,
}

impl LocalGrid {
    pub fn new(local_rows: usize, cols: usize) -> Self {
        LocalGrid {
            buf: vec![0.0; (local_rows + 2) * cols],
            local_rows,
            cols,
        }
    }

    pub fn idx(&self, local_row: usize, col: usize) -> usize {
        local_row * self.cols + col
    }

    pub fn row(&self, local_row: usize) -> &[f64] {
        let start = self.idx(local_row, 0);
        let  end = start + self.cols;
        &self.buf[start..end]
    }

    pub fn row_mut(&mut self, local_row: usize) -> &mut [f64] {
           let start = self.idx(local_row, 0);
        let  end = start + self.cols;
        &mut self.buf[start..end]

    }

    pub fn top_halo_mut(&mut self) -> &mut [f64] {
        self.row_mut(0)
    }

    pub fn bottom_halo_mut(&mut self) -> &mut [f64] {
        self.row_mut(self.local_rows + 1)
    }

    pub fn first_owned_row(&self) -> &[f64] {
        self.row(1)
    }

    #[allow(dead_code)]
    pub fn first_owned_row_mut(&mut self) -> &mut [f64] {
        self.row_mut(1)
    }

    pub fn last_owned_row(&self) -> &[f64] {
        self.row(self.local_rows)
    }

    #[allow(dead_code)]
    pub fn last_owned_row_mut(&mut self) -> &mut [f64] {
        self.row_mut(self.local_rows)
    }

    /// Splits the buffer into the four disjoint slices a halo exchange needs
    /// at once: (first_owned_row, last_owned_row, top_halo_mut, bottom_halo_mut).
    /// Borrowing all four through one call lets the caller post every send/recv
    /// in a single non-blocking batch instead of two sequential phases.
    /// Only valid when `local_rows > 0`, same precondition as the other
    /// owned-row accessors above.
    pub fn split_for_exchange(&mut self) -> (&[f64], &[f64], &mut [f64], &mut [f64]) {
        let cols = self.cols;
        let local_rows = self.local_rows;

        let (top, rest) = self.buf.split_at_mut(cols);
        let (owned, bottom) = rest.split_at_mut(local_rows * cols);

        let first = &owned[..cols];
        let last = &owned[(local_rows - 1) * cols..];

        (first, last, top, bottom)
    }

    pub fn local_rows(&self) -> usize {
        self.local_rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn owned_rows(&self) -> &[f64] {
        let start = self.cols;
        let end = start + self.local_rows * self.cols;
        &self.buf[start..end]
    }
}

pub fn init_local_grids(rows: usize, cols: usize, part: &RowPartition) -> (LocalGrid, LocalGrid) {
    let mut old = LocalGrid::new(part.local_rows, cols);
    let mut new = LocalGrid::new(part.local_rows, cols);


    let center_row = rows / 2;
    let center_col = cols / 2;

    if part.owns_global_row(center_row) {
        let local_i = part.global_to_local(center_row);
        old.row_mut(local_i)[center_col] = 100.0;
        new.row_mut(local_i)[center_col] = 100.0;
    }

    (old, new)
}

pub fn update_local_block(old: &LocalGrid, new: &mut LocalGrid) {
    for local_i in 1..=old.local_rows() {
        let top = old.row(local_i - 1);
        let mid = old.row(local_i);
        let bottom = old.row(local_i + 1);
        let dst = new.row_mut(local_i);

        // Copy the whole row first so boundary columns stay unchanged.
        //dst.copy_from_slice(mid);
        if !dst.is_empty() {
            let last = dst.len() - 1;
            dst[0] = mid[0];
            dst[last] = mid[last];
        }
        
        for j in 1..old.cols().saturating_sub(1) {
            dst[j] = 0.2 * (mid[j] + top[j] + bottom[j] + mid[j - 1] + mid[j + 1]);
        }
    }
}

// pub fn copy_owned_rows_into_dense(
//     part: &RowPartition,
//     local: &LocalGrid,
//     dense: &mut [f64],
//     cols: usize,
// ) {

// }

pub fn checksum(grid: &[f64]) -> f64 {
    grid.iter().sum()
}
