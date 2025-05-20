# Instruction description

## Memory

| Assembly                        | Description              |
|---------------------------------|--------------------------|
| LDG Ry, [Rx+0xa]                | Load from global memory  |
| STG [Rx+0xa], Ry                | Store to global memory   |
| LDL Ry, [Rx+0xa]                | Load from local memory   |
| STL [Rx+0xa], Ry                | Store to local memory    |
| LDS Ry, [Rx+0xa]                | Load from shared memory  |
| STS [Rx+0xa], Ry                | Store to shared memory   |
| LDC Ry, c[0xa][Rx+0xa]          | Load from constant memory|
| TEX Ry, Rx, 0xa, 0xa, 0xa       | Texture fetch            |