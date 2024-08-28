graph TD
    A[Start] --> B[Load Data]
    B --> C[robyn_inputs]
    C --> D[robyn_run]
    D --> E[robyn_outputs]
    E --> F[robyn_allocator]
    F --> G[End]

    subgraph "robyn_inputs"
        C1[Feature Engineering]
        C2[Adstock Transformations]
        C3[Saturation Transformations]
    end

    subgraph "robyn_run"
        D1[Hyperparameter Optimization]
        D2[Model Training]
        D3[Model Evaluation]
    end

    subgraph "robyn_outputs"
        E1[Pareto Front Analysis]
        E2[Model Selection]
    end

    subgraph "robyn_allocator"
        F1[Budget Allocation]
        F2[Response Curves]
    end

    C --> C1 --> C2 --> C3
    D --> D1 --> D2 --> D3
    E --> E1 --> E2
    F --> F1 --> F2

    subgraph "Inputs"
        H1[dt_input]
        H2[dt_holidays]
        H3[paid_media_spends]
        H4[paid_media_vars]
        H5[organic_vars]
        H6[prophet_vars]
        H7[hyperparameters]
    end

    subgraph "Outputs"
        I1[InputCollect]
        I2[OutputModels]
        I3[OutputCollect]
        I4[AllocatorCollect]
    end

    H1 --> B
    H2 --> B
    H1 --> C
    H2 --> C
    H3 --> C
    H4 --> C
    H5 --> C
    H6 --> C
    H7 --> C
    C --> I1
    I1 --> D
    D --> I2
    I1 --> E
    I2 --> E
    E --> I3
    I1 --> F
    I3 --> F
    F --> I4
