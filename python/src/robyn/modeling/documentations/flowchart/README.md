
```mermaid
graph TD
    A[Start] --> B[Load Data]
    B --> C[robyn_inputs]
    C --> D[robyn_run]
    D --> E[robyn_outputs]
    E --> F[robyn_allocator]
    F --> G[End]

    C --> H[Feature Engineering]
    H --> I[Adstock Transformations]
    I --> J[Saturation Transformations]

    D --> K[Hyperparameter Optimization]
    K --> L[Model Training]
    L --> M[Model Evaluation]

    E --> N[Pareto Front Analysis]
    N --> O[Model Selection]

    F --> P[Budget Allocation]
    P --> Q[Response Curves]
