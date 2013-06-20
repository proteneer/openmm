#define GROUP_SIZE 256

//crashes with BUFFER_GROUPS = 1?
#define BUFFER_GROUPS 3
#define BUFFER_SIZE BUFFER_GROUPS*GROUP_SIZE
#define WARP_SIZE 32
#define INVALID 0xFFFF

/**
 * Find a bounding box for the atoms in each block.
 */
extern "C" __global__ void findBlockBounds(int numAtoms, real4 periodicBoxSize, real4 invPeriodicBoxSize, const real4* __restrict__ posq,
        real4* __restrict__ blockCenter, real4* __restrict__ blockBoundingBox, int* __restrict__ rebuildNeighborList, real2* __restrict__ sortedBlocks) {
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    int base = index*TILE_SIZE;
    while (base < numAtoms) {
        real4 pos = posq[base];
#ifdef USE_PERIODIC
        pos.x -= floor(pos.x*invPeriodicBoxSize.x)*periodicBoxSize.x;
        pos.y -= floor(pos.y*invPeriodicBoxSize.y)*periodicBoxSize.y;
        pos.z -= floor(pos.z*invPeriodicBoxSize.z)*periodicBoxSize.z;
#endif
        real4 minPos = pos;
        real4 maxPos = pos;
        int last = min(base+TILE_SIZE, numAtoms);
        for (int i = base+1; i < last; i++) {
            pos = posq[i];
#ifdef USE_PERIODIC
            real4 center = 0.5f*(maxPos+minPos);
            pos.x -= floor((pos.x-center.x)*invPeriodicBoxSize.x+0.5f)*periodicBoxSize.x;
            pos.y -= floor((pos.y-center.y)*invPeriodicBoxSize.y+0.5f)*periodicBoxSize.y;
            pos.z -= floor((pos.z-center.z)*invPeriodicBoxSize.z+0.5f)*periodicBoxSize.z;
#endif
            minPos = make_real4(min(minPos.x,pos.x), min(minPos.y,pos.y), min(minPos.z,pos.z), 0);
            maxPos = make_real4(max(maxPos.x,pos.x), max(maxPos.y,pos.y), max(maxPos.z,pos.z), 0);
        }
        real4 blockSize = 0.5f*(maxPos-minPos);
        blockBoundingBox[index] = blockSize;
        blockCenter[index] = 0.5f*(maxPos+minPos);
        sortedBlocks[index] = make_real2(blockSize.x+blockSize.y+blockSize.z, index);
        index += blockDim.x*gridDim.x;
        base = index*TILE_SIZE;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0)
        rebuildNeighborList[0] = 0;
}

/**
 * Sort the data about bounding boxes so it can be accessed more efficiently in the next kernel.
 */
extern "C" __global__ void sortBoxData(const real2* __restrict__ sortedBlock, const real4* __restrict__ blockCenter,
        const real4* __restrict__ blockBoundingBox, real4* __restrict__ sortedBlockCenter,
        real4* __restrict__ sortedBlockBoundingBox, const real4* __restrict__ posq, const real4* __restrict__ oldPositions,
        /*unsigned int* __restrict__ interactionCount,*/ int* __restrict__ rebuildNeighborList) {
    for (int i = threadIdx.x+blockIdx.x*blockDim.x; i < NUM_BLOCKS; i += blockDim.x*gridDim.x) {
        int index = (int) sortedBlock[i].y;
        sortedBlockCenter[i] = blockCenter[index];
        sortedBlockBoundingBox[i] = blockBoundingBox[index];
    }
    
    // Also check whether any atom has moved enough so that we really need to rebuild the neighbor list.

    bool rebuild = false;
    for (int i = threadIdx.x+blockIdx.x*blockDim.x; i < NUM_ATOMS; i += blockDim.x*gridDim.x) {
        real4 delta = oldPositions[i]-posq[i];
        if (delta.x*delta.x + delta.y*delta.y + delta.z*delta.z > 0.25f*PADDING*PADDING)
            rebuild = true;
    }
    if (rebuild) {
        rebuildNeighborList[0] = 1;
        //interactionCount[0] = 0;
    }
}

/**
 * Perform a parallel prefix sum over an array.  The input values are all assumed to be 0 or 1.
 */
__device__ void prefixSum(short* sum, ushort2* temp) {
#if __CUDA_ARCH__ >= 300
    const int indexInWarp = threadIdx.x%WARP_SIZE;
    const int warpMask = (2<<indexInWarp)-1;
    for (int base = 0; base < BUFFER_SIZE; base += blockDim.x)
        temp[base+threadIdx.x].x = __popc(__ballot(sum[base+threadIdx.x])&warpMask);
    __syncthreads();
    if (threadIdx.x < BUFFER_SIZE/WARP_SIZE) {
        int multiWarpSum = temp[(threadIdx.x+1)*WARP_SIZE-1].x;
        for (int offset = 1; offset < BUFFER_SIZE/WARP_SIZE; offset *= 2) {
            short n = __shfl_up(multiWarpSum, offset, WARP_SIZE);
            if (indexInWarp >= offset)
                multiWarpSum += n;
        }
        temp[threadIdx.x].y = multiWarpSum;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < BUFFER_SIZE; i += blockDim.x)
        sum[i] = temp[i].x+(i < WARP_SIZE ? 0 : temp[i/WARP_SIZE-1].y);
    __syncthreads();
#else
    for (int i = threadIdx.x; i < BUFFER_SIZE; i += blockDim.x)
        temp[i].x = sum[i];
    __syncthreads();
    int whichBuffer = 0;
    for (int offset = 1; offset < BUFFER_SIZE; offset *= 2) {
        if (whichBuffer == 0)
            for (int i = threadIdx.x; i < BUFFER_SIZE; i += blockDim.x)
                temp[i].y = (i < offset ? temp[i].x : temp[i].x+temp[i-offset].x);
        else
            for (int i = threadIdx.x; i < BUFFER_SIZE; i += blockDim.x)
                temp[i].x = (i < offset ? temp[i].y : temp[i].y+temp[i-offset].y);
        whichBuffer = 1-whichBuffer;
        __syncthreads();
    }
    if (whichBuffer == 0)
        for (int i = threadIdx.x; i < BUFFER_SIZE; i += blockDim.x)
            sum[i] = temp[i].x;
    else
        for (int i = threadIdx.x; i < BUFFER_SIZE; i += blockDim.x)
            sum[i] = temp[i].y;
    __syncthreads();
#endif
}

/**
 * Compare the bounding boxes for each pair of atom blocks (comprised of 32 atoms each), forming a tile. If the two
 * atom blocks are sufficiently far apart, mark them as non-interacting. There are two stages in the algorithm.
 *
 * STAGE 1:
 *
 * A coarse grain atomblock against interacting atomblock neighbourlist is constructed. 
 *
 * Each threadblock first loads in some block X of interest. Each thread within the threadblock then loads 
 * in a different atomblock Y. If Y has exclusions with X, then Y is not processed.  If the bounding boxes 
 * of the two atomblocks are within the cutoff distance, then the two atomblocks are considered to be
 * interacting and Y is added to the buffer for X. If during any given iteration an atomblock (or thread) 
 * finds BUFFER_GROUP interacting blocks, the entire buffer is sent for compaction by storeInteractionData().
 *
 * STAGE 2:
 *
 * A fine grain atomblock against interacting atoms neighbourlist is constructed.
 *
 * The input is an atomblock list detailing the interactions with other atomblocks. The list of interacting 
 * atom blocks are initially stored in the buffer array in shared memory. buffer is then compacted using 
 * prefixSum. Afterwards, each threadblock processes one contiguous atomblock X. Each warp in a threadblock 
 * processes a block Y to find the atoms that interact with any given atom in X. Once BUFFER_SIZE/WARP_SIZE 
 * (eg. 16) atomblocks have been processed for a given X, the list of interacting atoms in these 16 blocks 
 * are subsequently compacted. The process repeats until all atomblocks that interact with X are computed.
 *
 * [in] periodicBoxSize        - size of the rectangular periodic box
 * [in] invPeriodicBoxSize     - inverse of the periodic box
 * [in] blockCenter            - the center of each bounding box
 * [in] blockBoundingBox       - bounding box of each atom block
 * [out] interactionCount      - total number of tiles that have interactions
 * [out] interactingTiles      - set of blocks that have interactions
 * [out] interactingAtoms      - a list of atoms that interact with each atom block
 * [in] posq                   - x,y,z coordinates of each atom and charge q
 * [in] maxTiles               - maximum number of tiles to process, used for multi-GPUs
 * [in] startBlockIndex        - first block to process by this GPU
 * [in] numBlocks              - total number of atom blocks processed by this GPU
 * [in] sortedBlocks           - a sorted list of atom blocks based on volume
 * [in] sortedBlockCenter      - sorted centers, duplicated for fast access to avoid indexing
 * [in] sortedBlockBoundingBox - sorted bounding boxes, duplicated for fast access
 * [in] exclusionIndices       - maps into exclusionRowIndices with the starting position for a given atom
 * [in] exclusionRowIndices    - stores the a continuous list of exclusions
 *           eg: block 0 is excluded from atom 3,5,6
 *               block 1 is excluded from atom 3,4
 *               block 2 is excluded from atom 1,3,5,6
 *              exclusionIndices[0][3][5][8]
 *           exclusionRowIndices[3][5][6][3][4][1][3][5][6]
 *                         index 0  1  2  3  4  5  6  7  8 
 * [out] oldPos                - stores the positions of the atoms in which this neighbourlist was built on
 *                             - this is used to decide when to rebuild a neighbourlist
 * [in] rebuildNeighbourList   - whether or not to execute this kernel
 *
 */
extern "C" __global__ void findBlocksWithInteractions(
    real4 periodicBoxSize, 
    real4 invPeriodicBoxSize,
    unsigned int* __restrict__ interactions, // size num_blocks*maxInteractionsPerBlock
    unsigned int* __restrict__ interactionBits, // size num_blocks*maxInteractionsPerBlock
    unsigned int* __restrict__ interactionsPerBlock, // size num_blocks
    const unsigned int interactionsAllocatedPerBlock, 
    const real4* __restrict__ posq, 
    const unsigned int startBlockIndex,
    const unsigned int numBlocks, 
    real2* __restrict__ sortedBlocks, 
    const real4* __restrict__ sortedBlockCenter, 
    const real4* __restrict__ sortedBlockBoundingBox,
    const unsigned int* __restrict__ exclusionIndices, 
    const unsigned int* __restrict__ exclusionRowIndices, 
    real4* __restrict__ oldPositions,
    const int* __restrict__ rebuildNeighborList) {

    __shared__ unsigned short buffer[BUFFER_SIZE];
    __shared__ short sum[BUFFER_SIZE];
    __shared__ ushort2 temp[BUFFER_SIZE];
    __shared__ unsigned int atoms[BUFFER_SIZE];
    __shared__ unsigned int interactionBitsBuffer[BUFFER_SIZE];
    __shared__ real3 posBuffer[TILE_SIZE];
    __shared__ int exclusionsForX[MAX_EXCLUSIONS];
    __shared__ int bufferFull;
    __shared__ int numAtoms;
    
    if (rebuildNeighborList[0] == 0)
        return; // The neighbor list doesn't need to be rebuilt.
    
    int valuesInBuffer = 0;
    if (threadIdx.x == 0)
        bufferFull = false;
    for (int i = 0; i < BUFFER_GROUPS; ++i)
        buffer[i*GROUP_SIZE+threadIdx.x] = INVALID;
    __syncthreads();
    
    // Loop over all blocks sorted by size.
    for (int i = startBlockIndex+blockIdx.x; i < startBlockIndex+numBlocks; i += gridDim.x) {
        if (threadIdx.x == blockDim.x-1)
            numAtoms = 0;
        const real2 sortedKey = sortedBlocks[i];
        const unsigned int x = (int) sortedKey.y;
        const real4 blockCenterX = sortedBlockCenter[i];
        const real4 blockSizeX = sortedBlockBoundingBox[i];

        // Load exclusion data for block x.
        
        const int exclusionStart = exclusionRowIndices[x];
        const int exclusionEnd = exclusionRowIndices[x+1];
        const int numExclusions = exclusionEnd-exclusionStart;
        for (int j = threadIdx.x; j < numExclusions; j += blockDim.x)
            exclusionsForX[j] = exclusionIndices[exclusionStart+j];

        const bool singlePeriodicCopy = (0.5f*periodicBoxSize.x-blockSizeX.x >= PADDED_CUTOFF &&
                                0.5f*periodicBoxSize.y-blockSizeX.y >= PADDED_CUTOFF &&
                                0.5f*periodicBoxSize.z-blockSizeX.z >= PADDED_CUTOFF);

        // TODO: try shuffles again
        // initialize the pos buffer (could use shuffles later on for this if need be)

        if (threadIdx.x < TILE_SIZE) {
            real3 pos = trimTo3(posq[x*TILE_SIZE+threadIdx.x]);
            posBuffer[threadIdx.x] = pos;
#ifdef USE_PERIODIC
            if (singlePeriodicCopy) {
                // The box is small enough that we can just translate all the atoms into a single periodic
                // box, then skip having to apply periodic boundary conditions later.
                pos.x -= floor((pos.x-blockCenterX.x)*invPeriodicBoxSize.x+0.5f)*periodicBoxSize.x;
                pos.y -= floor((pos.y-blockCenterX.y)*invPeriodicBoxSize.y+0.5f)*periodicBoxSize.y;
                pos.z -= floor((pos.z-blockCenterX.z)*invPeriodicBoxSize.z+0.5f)*periodicBoxSize.z;
                posBuffer[threadIdx.x] = pos;
            }
#endif
        }
        __syncthreads();

        // to do sort in smem?

        // Compare block x to other blocks after this one in sorted order.
        for (int base = i+1; base < NUM_BLOCKS; base += blockDim.x) {
            const int j = base+threadIdx.x;
            const real2 sortedKey2 = (j < NUM_BLOCKS ? sortedBlocks[j] : make_real2(0));
            const real4 blockCenterY = (j < NUM_BLOCKS ? sortedBlockCenter[j] : make_real4(0));
            const real4 blockSizeY = (j < NUM_BLOCKS ? sortedBlockBoundingBox[j] : make_real4(0));
            const unsigned short y = (unsigned short) sortedKey2.y;
            real4 delta = blockCenterX-blockCenterY;
#ifdef USE_PERIODIC
            delta.x -= floor(delta.x*invPeriodicBoxSize.x+0.5f)*periodicBoxSize.x;
            delta.y -= floor(delta.y*invPeriodicBoxSize.y+0.5f)*periodicBoxSize.y;
            delta.z -= floor(delta.z*invPeriodicBoxSize.z+0.5f)*periodicBoxSize.z;
#endif
            delta.x = max(0.0f, fabs(delta.x)-blockSizeX.x-blockSizeY.x);
            delta.y = max(0.0f, fabs(delta.y)-blockSizeX.y-blockSizeY.y);
            delta.z = max(0.0f, fabs(delta.z)-blockSizeX.z-blockSizeY.z);
            bool hasExclusions = false;
            for (int k = 0; k < numExclusions; k++)
                hasExclusions |= (exclusionsForX[k] == y);

            if (j < NUM_BLOCKS && delta.x*delta.x+delta.y*delta.y+delta.z*delta.z < PADDED_CUTOFF_SQUARED && !hasExclusions) {
                // Add this tile to the buffer.
                const int bufferIndex = valuesInBuffer*GROUP_SIZE+threadIdx.x;
                buffer[bufferIndex] = y;
                valuesInBuffer++;

                // cuda-memcheck --tool racecheck will throw errors about this as 
                // RAW/WAW/WAR race condition errors. But this is safe in all instances
                if (!bufferFull && valuesInBuffer == BUFFER_GROUPS)
                    bufferFull = true;
            }
            __syncthreads();

            // this is known at compile time!
            //const int lastGroup = (GROUP_SIZE < NUM_BLOCKS) ? NUM_BLOCKS - GROUP_SIZE : 0;

            // If the buffer is full, or if we are in the threadblock responsible for processing the last set of atomblocks,
            // we compact the buffer to write out a more fine grained neighbor list, note that NUM_BLOCKS - GROUP_SIZE
            // may be negative but that's ok. 
            if (bufferFull || base >= NUM_BLOCKS - GROUP_SIZE) {

                for (int k = threadIdx.x; k < BUFFER_SIZE; k += blockDim.x) {
                    sum[k] = (buffer[k] == INVALID ? 0 : 1);
                    interactionBitsBuffer[k] = 0;
                }
                __syncthreads();
                prefixSum(sum, temp);

                // Number of interacting blocks
                const unsigned int numValid = sum[BUFFER_SIZE-1];

                // Compact the buffer.
                for (int k = threadIdx.x; k < BUFFER_SIZE; k += blockDim.x)
                    if (buffer[k] != INVALID)
                        temp[sum[k]-1].x = buffer[k];
                __syncthreads();
                for (int k = threadIdx.x; k < BUFFER_SIZE; k += blockDim.x)
                    buffer[k] = temp[k].x;
                __syncthreads();

                // Loop over the tiles and find specific interactions in them.
                const int indexInWarp = threadIdx.x%WARP_SIZE;
                for (int base2 = 0; base2 < numValid; base2 += BUFFER_SIZE/WARP_SIZE) {
                    for (int k = threadIdx.x/WARP_SIZE; k < BUFFER_SIZE/WARP_SIZE && base2+k < numValid; k += GROUP_SIZE/WARP_SIZE) {
                        // Check each atom in block Y for interactions.
                        real3 pos = trimTo3(posq[buffer[base2+k]*TILE_SIZE+indexInWarp]);
#ifdef USE_PERIODIC
                        if (singlePeriodicCopy) {
                            pos.x -= floor((pos.x-blockCenterX.x)*invPeriodicBoxSize.x+0.5f)*periodicBoxSize.x;
                            pos.y -= floor((pos.y-blockCenterX.y)*invPeriodicBoxSize.y+0.5f)*periodicBoxSize.y;
                            pos.z -= floor((pos.z-blockCenterX.z)*invPeriodicBoxSize.z+0.5f)*periodicBoxSize.z;
                        }
#endif
                        unsigned int ixnBits = 0;
#ifdef USE_PERIODIC
                        if (!singlePeriodicCopy) {
                            for (int m = 0; m < TILE_SIZE; m++) {
                                real3 delta = pos-posBuffer[m];
                                delta.x -= floor(delta.x*invPeriodicBoxSize.x+0.5f)*periodicBoxSize.x;
                                delta.y -= floor(delta.y*invPeriodicBoxSize.y+0.5f)*periodicBoxSize.y;
                                delta.z -= floor(delta.z*invPeriodicBoxSize.z+0.5f)*periodicBoxSize.z;
                                //interacts |= (delta.x*delta.x+delta.y*delta.y+delta.z*delta.z < PADDED_CUTOFF_SQUARED);
                                ixnBits |= (delta.x*delta.x+delta.y*delta.y+delta.z*delta.z < PADDED_CUTOFF_SQUARED) << m;
                            }
                        } else {
#endif
                            for (int m = 0; m < TILE_SIZE; m++) {
                                const real3 delta = pos-posBuffer[m];
                                //interacts |= (delta.x*delta.x+delta.y*delta.y+delta.z*delta.z < PADDED_CUTOFF_SQUARED);
                                ixnBits |= (delta.x*delta.x+delta.y*delta.y+delta.z*delta.z < PADDED_CUTOFF_SQUARED) << m;
                            }
#ifdef USE_PERIODIC
                        }
#endif
                        sum[k*WARP_SIZE+indexInWarp] = (ixnBits ? 1 : 0);
                        interactionBitsBuffer[k*WARP_SIZE+indexInWarp] = ixnBits;
                    }
                    // Is this necessary? Why not just initialize sum to start with 0s
                    for (int k = numValid-base2+threadIdx.x/WARP_SIZE; k < BUFFER_SIZE/WARP_SIZE; k += GROUP_SIZE/WARP_SIZE)
                        sum[k*WARP_SIZE+indexInWarp] = 0;

                    // Compact the list of atoms and interactionBits
                    __syncthreads();
                    prefixSum(sum, temp);

                    // Store the list of atoms and interactionBits to global memory.
                    const unsigned int atomsToStore = sum[BUFFER_SIZE-1];
                    const unsigned int offset = numAtoms;
                    __syncthreads();

                    if(offset + atomsToStore < interactionsAllocatedPerBlock) {
                        for(int k = threadIdx.x; k < BUFFER_SIZE; k+= blockDim.x) {
                            if (sum[k] != (k == 0 ? 0 : sum[k-1])) {
                                const unsigned int gatherIndex = sum[k]-1;               
                                interactions[x*interactionsAllocatedPerBlock+offset+gatherIndex] = buffer[base2+k/WARP_SIZE]*TILE_SIZE+indexInWarp;
                                interactionBits[x*interactionsAllocatedPerBlock+offset+gatherIndex] = interactionBitsBuffer[k];
                            }
                        }
                    }

                    if(threadIdx.x == 0)
                        numAtoms += atomsToStore;

                    __syncthreads();
                } // finished a set of interacting blocks

                // Reset the buffer for processing more tiles.

                for (int k = threadIdx.x; k < BUFFER_SIZE; k += blockDim.x)
                    buffer[k] = INVALID;
                valuesInBuffer = 0;
                if(threadIdx.x == 0)
                    bufferFull = false;

                __syncthreads();
            } // buffer full for block x

        } // finished processing every block against block x
        if(threadIdx.x == 0)
            interactionsPerBlock[x] = numAtoms;
        __syncthreads();
    } // finished all atomblocks this threadblock is responsible for
    
    // Record the positions the neighbor list is based on.
    
    for (int i = threadIdx.x+blockIdx.x*blockDim.x; i < NUM_ATOMS; i += blockDim.x*gridDim.x)
        oldPositions[i] = posq[i];
}
