# GPT Prompt (O3)


You need to write a decompiler for CUDA SASS code. 
There are something you should konw:

cubin：CUDA binary，只有GPU代码
host binary：ELF里嵌入了cubin
fatbin：编译器会为不同的计算能力（例如 sm_30、sm_50、sm_70 等）生成相应的代码，并将它们全部打包进一个 fat binary 文件中。这样，在程序运行时，CUDA 运行时系统会根据当前 GPU 的具体架构，从 fat binary 中选取最合适的那一段设备代码进行执行

`cuobjdump  -lelf  example_fatbin` 列出fatbin中所有的cubin, 例如:
```sh
$ cuobjdump  -lelf  example_ml_dsa
ELF file    1: lto.sm_89.cubin
```

`cuobjdump  -xelf all  example_fatbin` 可以提取出所有的cubin, 例如:
```sh
$ cuobjdump  -xelf all  example_ml_dsa
Extracting ELF file    1: lto.sm_89.cubin
```

`cuobjdump -sass  example_binary` 可以从fatbin中得到sass assembly, like this:
```
Fatbin elf code:
================
arch = sm_89
code version = [1,7]
host = linux
compile_size = 64bit
identifier = lto

        code for sm_89
                Function : _Z13verify_kernelPbPKhS1_mS1_Ph
        .headerflags    @"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS EF_CUDA_SM89 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM89)"
        /*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;                   /* 0x00000a00ff017624 */
                                                                                             /* 0x000fe400078e00ff */
        /*0010*/                   S2R R26, SR_TID.X ;                                       /* 0x00000000001a7919 */
                                                                                             /* 0x000e220000002100 */
        /*0020*/                   IMAD.MOV.U32 R2, RZ, RZ, 0x978 ;                          /* 0x00000978ff027424 */
                                                                                             /* 0x000fe200078e00ff */
        /*0030*/                   ULDC.64 UR8, c[0x0][0x118] ;                              /* 0x0000460000087ab9 */
                                                                                             /* 0x000fe20000000a00 */
        /*0040*/                   IADD3 R1, R1, -0x130, RZ ;                                /* 0xfffffed001017810 */
                                                                                             /* 0x000fe20007ffe0ff */
        /*0050*/                   S2R R27, SR_CTAID.X ;                                     /* 0x00000000001b7919 */
                                                                                             /* 0x000e620000002500 */
        /*0060*/                   ISETP.GT.U32.AND P0, PT, R26, 0x3, PT ;                   /* 0x000000031a00780c */
                                                                                             /* 0x001fe20003f04070 */
        /*0070*/                   IMAD.WIDE R2, R27, R2, c[0x0][0x168] ;                    /* 0x00005a001b027625 */
                                                                                             /* 0x002fd800078e0202 */
        /*0080*/              @!P0 IMAD.WIDE.U32 R4, R26, 0x8, R2 ;                          /* 0x000000081a048825 */
                                                                                             /* 0x000fcc00078e0002 */
        /*0090*/              @!P0 LDG.E.64 R4, [R4.64] ;                                    /* 0x0000000804048981 */
                                                                                             /* 0x000ea2000c1e1b00 */
        /*00a0*/                   IMAD.MOV.U32 R18, RZ, RZ, 0x520 ;                         /* 0x00000520ff127424 */
                                                                                             /* 0x000fe200078e00ff */
        /*00b0*/                   SHF.R.S32.HI R16, RZ, 0x1f, R27 ;                         /* 0x0000001fff107819 */
                                                                                             /* 0x000fe2000001141b */
        /*00c0*/                   IMAD.MOV.U32 R52, RZ, RZ, 0x4000 ;                        /* 0x00004000ff347424 */
                                                                                             /* 0x000fe400078e00ff */
        /*00d0*/                   IMAD.MOV.U32 R17, RZ, RZ, 0x1 ;                           /* 0x00000001ff117424 */
                                                                                             /* 0x000fe400078e00ff */
```

`cuobjdump -sass example.cubin` 也可以从提取后的cubin中得到sass assembly, like this:
```
$ cuobjdump -sass lto.sm_89.cubin | head -n 20

        code for sm_89
                Function : _Z13verify_kernelPbPKhS1_mS1_Ph
        .headerflags    @"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS EF_CUDA_SM89 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM89)"
        /*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;                   /* 0x00000a00ff017624 */
                                                                                             /* 0x000fe400078e00ff */
        /*0010*/                   S2R R26, SR_TID.X ;                                       /* 0x00000000001a7919 */
                                                                                             /* 0x000e220000002100 */
        /*0020*/                   IMAD.MOV.U32 R2, RZ, RZ, 0x978 ;                          /* 0x00000978ff027424 */
                                                                                             /* 0x000fe200078e00ff */
        /*0030*/                   ULDC.64 UR8, c[0x0][0x118] ;                              /* 0x0000460000087ab9 */
                                                                                             /* 0x000fe20000000a00 */
        /*0040*/                   IADD3 R1, R1, -0x130, RZ ;                                /* 0xfffffed001017810 */
                                                                                             /* 0x000fe20007ffe0ff */
        /*0050*/                   S2R R27, SR_CTAID.X ;                                     /* 0x00000000001b7919 */
                                                                                             /* 0x000e620000002500 */
        /*0060*/                   ISETP.GT.U32.AND P0, PT, R26, 0x3, PT ;                   /* 0x000000031a00780c */
                                                                                             /* 0x001fe20003f04070 */
        /*0070*/                   IMAD.WIDE R2, R27, R2, c[0x0][0x168] ;                    /* 0x00005a001b027625 */
                                                                                             /* 0x002fd800078e0202 */
```

`nvdisasm -bbcfg lto.sm_89.cubin` 可以获取到 graphviz 格式的控制流图, 其中每个node是一个basic block. 

`cu++filt` 可以拿demangle name,例如:
```
$ cu++filt "_Z13verify_kernelPbPKhS1_mS1_Ph"
verify_kernel(bool *, const unsigned char *, const unsigned char *, unsigned long, const unsigned char *, unsigned char *)
```

值得注意的是, SASS的指令是没有开源而且每一代都不完全相同的, 你可以根据 https://docs.nvidia.com/cuda/parallel-thread-execution/index.html 参考一下ptx中的同名指令的格式.
我希望使用rust编写整个项目. 这个项目整体可能比较复杂, 你可以先分析从宏观上如何设计和实现整个项目, 比如包含哪些模块和阶段. 我们可以先从简单的开始, 比如只关注反编译单个函数的SASS, 只关注一个特定的架构(sm_89, 4090). 

请你根据以上要求和信息, 结合你自身的知识和网络上的信息, 给出详细的代码和操作指引, 谢谢.
