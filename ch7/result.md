N[1024,1024], F[7, 7]
BLOCK_SIZE 32  
IN_TILE_WIDTH 32  
OUT_TILE_WIDTH (IN_TILE_WIDTH - 2*FILTER_RADIUS)  

Basic GPU:        2.745 ms  
Constant Memory:  2.422 ms  
Tiled:            2.497 ms  
Cached Tiled:     2.722 ms  

Speedup (const):  1.13x  
Speedup (tiled):  1.10x  
Speedup (cached_tiled):  1.01x  

The Tiled kernel is slow because not all threads within a block are utilized for the final computation, as some are only used for loading the larger input tile.
The Cached Tiled kernel is slow because of severe branch divergence within the main calculation loop