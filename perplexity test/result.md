# Perplexity Test Results

## System Information
- Threads: 4 (batch threads: 4/20)
- CPU Features: SSE3, SSSE3, AVX, AVX2, F16C, FMA
- Additional Features: LLAMAFILE, OPENMP, AARCH64_REPACK

## Test Configuration
- Context Size (n_ctx): 1096
- Batch Size: 1096
- Sequence Count: 1
- Chunks: 7

## Performance Metrics
### Tokenization
- Time: 28.345 ms

### Processing Time
- Time per Pass: 141.67 seconds
- Estimated Total Time: 16.52 minutes

### Perplexity Scores by Chunk
1. 6.8017
2. 7.2383
3. 7.1005
4. 6.4421
5. 6.4294
6. 5.8594
7. 5.9457

### Final Results
- **Final PPL**: 5.9457 ± 0.23848

## Performance Summary
- Load Time: 23,277.88 ms
- Prompt Evaluation:
  - Total Time: 1,087,511.87 ms
  - Tokens Processed: 7,672
  - Per Token: 141.75 ms
  - Tokens per Second: 7.05
- Total Performance:
  - Total Time: 1,088,130.56 ms
  - Total Tokens: 7,673
