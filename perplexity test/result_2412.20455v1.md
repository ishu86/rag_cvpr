# Perplexity Test Results (8 Threads)

## System Configuration
- Threads: 8 (batch threads: 8/20)
- CPU Features: SSE3, SSSE3, AVX, AVX2, F16C, FMA
- Additional: LLAMAFILE, OPENMP, AARCH64_REPACK

## Test Parameters
- Context Size: 512
- Batch Size: 2048
- Sequences: 4
- Chunks: 27
- Tokenization Time: 44.809 ms

## Chunk-wise Perplexity Results
```
Token   PPL     Log2    StdErr
0       5.4722  1.6997  0.1462
0       5.6790  1.7368  0.1050
0       6.4731  1.8677  0.0901
0       6.1602  1.8181  0.0762
2048    6.5054  1.8726  0.0702
...     ...     ...     ...
12288   6.6024  1.8874  0.0307
12288   6.5095  1.8733  0.0299
```

## Final Results
- **PPL**: 6.5095 ± 0.19488

## Performance Metrics
- Load Time: 19,296.89 ms
- Prompt Evaluation:
  - Time: 1,394,637.51 ms
  - Tokens: 13,824
  - Per Token: 100.89 ms
  - Tokens/Second: 9.91
- Total Time: 1,402,180.25 ms (13,825 tokens)
