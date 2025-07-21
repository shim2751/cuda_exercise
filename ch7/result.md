N[1024,1024], F[7, 7]

Basic GPU:        2.745 ms\n
Constant Memory:  2.422 ms\n
Tiled:            2.497 ms\n
Cached Tiled:     2.722 ms\n

Speedup (const):  1.13x\n
Speedup (tiled):  1.10x\n
Speedup (cached_tiled):  1.01x\n

The Tiled kernel is slow because not all threads within a block are utilized for the final computation, as some are only used for loading the larger input tile.
The Cached Tiled kernel is slow because of severe branch divergence within the main calculation loop