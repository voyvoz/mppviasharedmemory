use std::any::Any;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use datafusion::arrow::array::{Array, Int32Array};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::catalog::{MemTable, Session};
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::Result;
use datafusion::execution::context::SessionContext;
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown};
use datafusion::physical_plan::ExecutionPlan;

use query_driven_index::index::Index;

// ── TableProvider ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct IndexedTable {
    schema: SchemaRef,
    ///
    data: Arc<Vec<RecordBatch>>,
    /// Built lazily on first qualifying query.
    index: Arc<RwLock<Option<Index>>>,
    /// Which column is indexed (if any).
    indexed_column: String,
}

impl IndexedTable {
    fn new(schema: SchemaRef, data: Vec<RecordBatch>, indexed_column: &str) -> Self {
        Self {
            schema,
            data: Arc::new(data),
            index: Arc::new(RwLock::new(None)),
            indexed_column: indexed_column.to_string(),
        }
    }

    ///
    fn build_index(&self) {
        let mut idx = Index::init("/dev/dax0.0", 8 * 1024 * 1024);

        for (batch_i, batch) in self.data.iter().enumerate() {
            let col_i = batch.schema().index_of(&self.indexed_column).unwrap();
            let col = batch
                .column(col_i)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();

            for row_i in 0..col.len() {
                //idx.entries
                //    .entry(col.value(row_i).to_string())
                //    .or_default()
                //    .push(row_i);
                idx.insert(col.value(row_i) as u64, row_i as u64);
            }
        }

        let mut g = self.index.write().unwrap();
        (*g) = Some(idx);
        println!("[index] built on column '{}'", self.indexed_column);
    }

    ///
    fn extract_index_lookup<'a>(&self, filters: &'a [Expr]) -> Option<&'a i32> {
        for expr in filters {
            if let Expr::BinaryExpr(bin) = expr {
                use datafusion::logical_expr::Operator;
                if bin.op == Operator::Eq {
                    // col = lit
                    if let (Expr::Column(col), Expr::Literal(scalar, _)) =
                        (bin.left.as_ref(), bin.right.as_ref())
                    {
                        if col.name == self.indexed_column {
                            if let datafusion::scalar::ScalarValue::Int32(Some(v)) = scalar {
                                return Some(v);
                            }
                        }
                    }
                }
            }
        }
        None
    }
}

#[async_trait]
impl TableProvider for IndexedTable {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    ///
    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        Ok(filters
            .iter()
            .map(|f| {
                if self.extract_index_lookup(std::slice::from_ref(f)).is_some() {
                    TableProviderFilterPushDown::Exact
                } else {
                    TableProviderFilterPushDown::Unsupported
                }
            })
            .collect())
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // Create index if not exists
        if self.index.read().unwrap().is_none() {
            self.build_index();
        }

        //
        let batches = if let Some(value) = self.extract_index_lookup(filters) {
            let mut g = self.index.write().unwrap();
            let idx = g.as_mut().unwrap();

            println!("[index] lookup '{}' = '{}'", self.indexed_column, value);

            match idx.lookup(*value as u64) {
                None => vec![],
                Some(row_indices) => {
                    // Reconstruct matching rows into a RecordBatch.
                    // Simplified: assumes single batch and StringArray + Int32Array layout.
                    let src = &self.data[0];
                    let name_col = src.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
                    let age_col = src.column(1).as_any().downcast_ref::<Int32Array>().unwrap();

                    let names: Int32Array = row_indices
                        .iter()
                        .map(|&i| Some(name_col.value(i as usize)))
                        .collect();
                    let ages: Int32Array = row_indices
                        .iter()
                        .map(|&i| Some(age_col.value(i as usize)))
                        .collect();

                    vec![vec![RecordBatch::try_new(
                        self.schema.clone(),
                        vec![Arc::new(names), Arc::new(ages)],
                    )?]]
                }
            }
        } else {
            println!("[index] no index match");
            vec![(*self.data).clone()]
        };

        let mem_table = MemTable::try_new(self.schema.clone(), batches)?;
        mem_table.scan(state, projection, filters, limit).await
    }
}

// --------
// main
// --------

#[tokio::main]
async fn main() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("name", DataType::Int32, false),
        Field::new("age", DataType::Int32, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
            Arc::new(Int32Array::from(vec![30, 25, 35, 28])),
        ],
    )?;

    let table = IndexedTable::new(schema, vec![batch], "name");

    let ctx = SessionContext::new();
    ctx.register_table("people", Arc::new(table))?;

    // First query: triggers index build + index lookup.
    println!("=== query 1: indexed lookup ===");
    let df = ctx.sql("SELECT * FROM people WHERE name = 1").await?;
    df.show().await?;

    // Second query: index already built, direct lookup.
    println!("=== query 2: different value ===");
    let df = ctx.sql("SELECT * FROM people WHERE name = 2").await?;
    df.show().await?;

    // Third query: no filter on indexed column → full scan.
    println!("=== query 3: full scan ===");
    let df = ctx.sql("SELECT * FROM people WHERE age > 28").await?;
    df.show().await?;

    Ok(())
}
