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
        C4[check_inputs]
        C5[robyn_engineering]
    end

    subgraph "robyn_run"
        D1[Hyperparameter Optimization]
        D2[Model Training]
        D3[Model Evaluation]
        D4[robyn_train]
        D5[robyn_mmm]
    end

    subgraph "robyn_outputs"
        E1[Pareto Front Analysis]
        E2[Model Selection]
        E3[robyn_pareto]
        E4[robyn_clusters]
    end

    subgraph "robyn_allocator"
        F1[Budget Allocation]
        F2[Response Curves]
        F3[robyn_response]
    end

    C --> C1 --> C2 --> C3
    C --> C4 --> C5
    D --> D1 --> D2 --> D3
    D --> D4 --> D5
    E --> E1 --> E2
    E --> E3 --> E4
    F --> F1 --> F2
    F --> F3

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

    subgraph "Auxiliary Functions"
        J1[checks.R]
        J2[transformations.R]
        J3[model.R]
        J4[pareto.R]
        J5[clusters.R]
        J6[plots.R]
        J7[auxiliary.R]
        J8[json.R]
    end

    J1 --> C4
    J2 --> C2
    J2 --> C3
    J3 --> D4
    J3 --> D5
    J4 --> E3
    J5 --> E4
    J6 --> E
    J6 --> F
    J7 --> C
    J7 --> D
    J7 --> E
    J7 --> F
    J8 --> C
    J8 --> D
    J8 --> E
    J8 --> F

    subgraph "External Libraries"
        K1[nevergrad]
        K2[reticulate]
        K3[prophet]
        K4[glmnet]
    end

    K1 --> D1
    K2 --> D
    K3 --> C1
    K4 --> D2
