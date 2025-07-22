N[1024,1024], F[7, 7]  
BLOCK_SIZE 32  
IN_TILE_WIDTH 32  
OUT_TILE_WIDTH (IN_TILE_WIDTH - 2*FILTER_RADIUS)  

CPU: 1155.92 ms

[Basic GPU] Kernel execution time: 124.258 ms
[Constant Memory] Kernel execution time: 0.168 ms
[Tiled] Kernel execution time: 0.253 ms
[Cached Tiled] Kernel execution time: 0.458 ms

The Tiled kernel is slow because not all threads within a block are utilized for the final computation, as some are only used for loading the larger input tile.
The Cached Tiled kernel is slow because of severe branch divergence within the main calculation loop