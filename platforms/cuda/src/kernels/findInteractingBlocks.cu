#define GROUP_SIZE 256
#define BUFFER_GROUPS 2
#define BUFFER_SIZE BUFFER_GROUPS*GROUP_SIZE
#define WARP_SIZE 32
#define INVALID 0xFFFF
#define STORE_FRACTION 0.1
#define ATOM_THRESHOLD 3

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
        unsigned int* __restrict__ tileInteractionCount, int* __restrict__ rebuildNeighborList) {
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
        tileInteractionCount[0] = 0;
    }
}


// perform an inclusive parallel prefix sum over an array in shared memory, signature:
// __device__ template<typename T> void(T* sum, T* buffer, int size);

/**
 * Perform an inclusive parallel prefix sum over an array.  The input values are all assumed to be 0 or 1.
 */
_device__ void prefixSum(short* sum, ushort2* temp) {
//__device__ template<typename T, typename T2>
//    void prefixSum(T* sum, T2* temp) {
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
 * This is called by findBlocksWithInteractions(). It compacts the list of blocks, identifies interactions
 * in them, and writes the result to global memory. For tiles with less than 0.1*1024 interactions, atomInteractions is used.
 *
 * A new neighbourlist algorithm. Consider the following input of two tiles that interact:
 *
 *             warp 0                   warp 1
 *    0  1  2  3  4  5  6  7   8  9 10 11 12 13 14 15
 * 8  1  0  1  0  1  0  0  0   1  0  0  0  0  0  0  1
 * 9  0  1  1  0  0  1  0  0   0  1  0  0  0  0  0  0
 * 10 1  1  0  1  0  1  0  0   0  1  0  0  0  0  0  0
 * 11 0  1  1  1  0  0  0  0   0  1  1  0  0  0  0  0         
 * 12 0  1  1  0  0  0  0  1   1  1  0  0  0  0  1  0
 * 13 0  0  1  0  1  0  0  0   0  0  0  1  0  0  0  0
 * 14 0  0  1  0  0  1  0  0   0  0  0  0  0  1  0  0
 * 15 0  1  1  0  0  0  0  0   0  1  1  0  0  1  0  1
 *    P  A  A  P  P  P  E  P   P  A  P  P  E  P  P  P
 *
 * Overall, compacts into three arrays: PPPPPPPPPPP, AAA, EE
 * Each column stores a bit pattern, read from bottom to top: 
 * col1 = 00000101, col2 = 10011110, col3 = 11111011, etc.
 *
 * The bit pattern per column decomposes into:
 * a. Almost Full (A) 3>interactions per atom
 * b. Partial     (P) 0<interactions per atom<=3
 * c. Empty       (E) 0=interactions per atom
 *
 * A,P compaction proceeds as before. For Almost Full, only a 1 or a 0 is ultimately compacted. 
 * For Partial, the column bit pattern is also compacted.
 *
 * Partial Results:
 *
 *    0  5  9  12 13 25 78 92 10 24 33 52 98 4  92 53
 * 0  0  1  0  0  0  0  0  0  0  1  0  0  0  0  0  0
 * 1  0  0  1  1  0  0  0  0  0  0  1  1  0  0  0  0
 * 2  1  0  0  0  1  0  0  0  1  0  0  0  1  0  0  0
 * 3  0  0  0  0  0  0  1  0  0  0  0  0  0  0  1  0
 * 4  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 * 5  0  1  0  0  0  1  1  0  0  1  0  0  0  1  1  0
 * 6  0  0  0  0  0  1  0  1  0  0  0  0  0  1  0  1
 * 7  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 *
 *  0  0  0  5  5  5  9  9  9  
 * [2][x][x][0][5][x][1][x][x] ...
 *
 * Guarantee: No column has more than 3 1's. So, we only need a buffer 
 * of size 3*sizeof(ushort2)*threadsPerBlock. 
 *
 * __shared__ ushort2 partialBuffer[3*sizeof(ushort2)*threadsPerBlock];
 *
 * unsigned int x; // x-dimension of the tile we're in
 * unsigned int pattern = 10000100; // bitwise
 * unsigned int atom2 = shared[threadIdx.x];
 * for(int i=0;i<3;i++) {
 *    unsigned int pos = __ffs(pattern);
 *    unsigned int atom1 = pos;
 *    partialBuffer[i*sizeof(ushort2)*threadsPerBlock] = make_ushort2(atom1, atom2);
 *    pattern -= 1 << __ffs(pattern);
 * }
 * __syncthreads();
 *
 * compactPartialBuffer();
 *
 * __syncthreads();
 *
 * appendToEnd();
 */
__device__ void storeInteractionData(unsigned short x, unsigned short* buffer, short* sum, short* sum2, unsigned int *interactionBits, ushort2* temp, 
    int* denseAtoms, unsigned int *sparseAtoms, int& numDenseAtoms, int& numSparseAtoms, unsigned int* sparseAtomsCompactionBuffer,
    int& baseIndex, unsigned int* tileInteractionCount, ushort2* interactingTiles, unsigned int* interactingAtoms, uint2* sparseInteractions, unsigned int* numSparseInteractions
    real4 periodicBoxSize, real4 invPeriodicBoxSize, const real4* posq, real3* posBuffer, real4 blockCenterX, real4 blockSizeX, unsigned int maxTiles, bool finish) {

    const bool singlePeriodicCopy = (0.5f*periodicBoxSize.x-blockSizeX.x >= PADDED_CUTOFF &&
                                     0.5f*periodicBoxSize.y-blockSizeX.y >= PADDED_CUTOFF &&
                                     0.5f*periodicBoxSize.z-blockSizeX.z >= PADDED_CUTOFF);
    
    // posBuffer stores the atoms in the X component of the tile
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
    
    // The buffer is full, so we need to compact it and write out results.  Start by doing a parallel prefix sum.

    for (int i = threadIdx.x; i < BUFFER_SIZE; i += blockDim.x)
        sum[i] = (buffer[i] == INVALID ? 0 : 1);
    __syncthreads();
    prefixSum(sum, temp);
    int numValid = sum[BUFFER_SIZE-1]; // number of valid tiles

    // Compact the buffer containing block indices
    for (int i = threadIdx.x; i < BUFFER_SIZE; i += blockDim.x)
        if (buffer[i] != INVALID)
            temp[sum[i]-1].x = buffer[i];
    __syncthreads();
    for (int i = threadIdx.x; i < BUFFER_SIZE; i += blockDim.x)
        buffer[i] = temp[i].x;
    __syncthreads();
   
    const int indexInWarp = threadIdx.x % WARP_SIZE;
    for (int base = 0; base < numValid; base += BUFFER_SIZE/WARP_SIZE) {
        for (int i = threadIdx.x/WARP_SIZE; i < BUFFER_SIZE/WARP_SIZE && base+i < numValid; i += GROUP_SIZE/WARP_SIZE) {
            // Check each atom in block Y for interactions.

            // pos is the position of atoms along the Y direction, and buffer[base+i] holds starting index of atomblocks
            real3 pos = trimTo3(posq[buffer[base+i]*TILE_SIZE+indexInWarp]);
#ifdef USE_PERIODIC
            if (singlePeriodicCopy) {
                pos.x -= floor((pos.x-blockCenterX.x)*invPeriodicBoxSize.x+0.5f)*periodicBoxSize.x;
                pos.y -= floor((pos.y-blockCenterX.y)*invPeriodicBoxSize.y+0.5f)*periodicBoxSize.y;
                pos.z -= floor((pos.z-blockCenterX.z)*invPeriodicBoxSize.z+0.5f)*periodicBoxSize.z;
            }
#endif
            //bool interacts = false;
            unsigned int interactionFlags = 0;
            // load atoms from the X component
#ifdef USE_PERIODIC
            if (!singlePeriodicCopy) {
                for (int j = 0; j < TILE_SIZE; j++) {
                    real3 delta = pos-posBuffer[j];
                    delta.x -= floor(delta.x*invPeriodicBoxSize.x+0.5f)*periodicBoxSize.x;
                    delta.y -= floor(delta.y*invPeriodicBoxSize.y+0.5f)*periodicBoxSize.y;
                    delta.z -= floor(delta.z*invPeriodicBoxSize.z+0.5f)*periodicBoxSize.z;
                    //interacts |= (delta.x*delta.x+delta.y*delta.y+delta.z*delta.z < PADDED_CUTOFF_SQUARED);
                    interactionFlags |= 1 << j;
                }
            }
            else {
#endif
            // set bits starting from the lsb
            for (int j = 0; j < TILE_SIZE; j++) {
                real3 delta = pos-posBuffer[j];
                //interacts |= (delta.x*delta.x+delta.y*delta.y+delta.z*delta.z < PADDED_CUTOFF_SQUARED);
                interactionFlags |= 1 << j;
            }
#ifdef USE_PERIODIC
            }
#endif
            int flagCount = __popc(interactionFlags);
            sum[i*WARP_SIZE+indexInWarp] = flagCount> ATOM_THRESHOLD ? 1 : 0; // For dense atoms
            sum2[i*WARP_SIZE+indexInWarp] = (flagCount <= ATOM_THRESHOLD && flagCount > 0) ? 1 : 0; // For sparse atoms
            interactionBits[i*WARP_SIZE+indexInWarp] = (flagCount <= ATOM_THRESHOLD && flagCount > 0) ? interactionFlags : 0; // Interaction bitflags in sparse atoms
        }

        for (int i = numValid-base+threadIdx.x/WARP_SIZE; i < BUFFER_SIZE/WARP_SIZE; i += GROUP_SIZE/WARP_SIZE) {
            sum[i*WARP_SIZE+indexInWarp] = 0;
            sum2[i*WARP_SIZE+indexInWarp] = 0;
            interactionBits[i*WARP_SIZE+indexInWarp] = 0;
        }

        // Part 1. Build neighborlist for densely interacting atoms
        // Compact
        __syncthreads();
        prefixSum(sum, temp);
        for (int i = threadIdx.x; i < BUFFER_SIZE; i += blockDim.x)
            if (sum[i] != (i == 0 ? 0 : sum[i-1]))
                denseAtoms[numDenseAtoms+sum[i]-1] = buffer[base+i/WARP_SIZE]*TILE_SIZE+indexInWarp;
        // Store to global memory
        int atomsToStore = numDenseAtoms+sum[BUFFER_SIZE-1];
        // finish is set to true in the final call to this function. 
        bool storePartialTile = (finish && base >= numValid-BUFFER_SIZE/WARP_SIZE);
        int tilesToStore = (storePartialTile ? (atomsToStore+TILE_SIZE-1)/TILE_SIZE : atomsToStore/TILE_SIZE);
        if (tilesToStore > 0) {
            // this is a trick used to "allocate" some space in the final output array so we can have on the fly compaction 
            if (threadIdx.x == 0)
                baseIndex = atomicAdd(tileInteractionCount, tilesToStore);
            __syncthreads();
            if (threadIdx.x == 0)
                numDenseAtoms = atomsToStore-tilesToStore*TILE_SIZE;
            if (baseIndex+tilesToStore <= maxTiles) {
                if (threadIdx.x < tilesToStore)
                    interactingTiles[baseIndex+threadIdx.x] = make_ushort2(x, singlePeriodicCopy);
                for (int i = threadIdx.x; i < tilesToStore*TILE_SIZE; i += blockDim.x)
                    interactingAtoms[baseIndex*TILE_SIZE+i] = (i < atomsToStore ? denseAtoms[i] : NUM_ATOMS);
            }
        } else { // less than TILE_SIZE number of atoms need be added
            __syncthreads();
            if (threadIdx.x == 0)
                numDenseAtoms += sum[BUFFER_SIZE-1];
        }
        __syncthreads();
        // Store leftover atoms
        if(threadIdx.x < numDenseAtoms && !storePartialTile)
            denseAtoms[threadIdx.x] = denseAtoms[tilesToStore*TILE_SIZE+threadIdx.x];

        // Part 2. Build neighborlist for sparsely interacting atoms
        // Compact atom indices
        prefixSum(sum2, temp);
        numValid = sum2[BUFFER_SIZE-1];
        for (int i = threadIdx.x; i < BUFFER_SIZE; i += blockDim.x) {
            if (sum2[i] != (i == 0 ? 0 : sum2[i-1])) {
                sparseAtoms[numSparseAtoms+sum2[i]-1] = buffer[base+i/WARP_SIZE]*TILE_SIZE+indexInWarp;
            }
        }
 
        // Compact interaction bitflags
        // sizeof ushort2 == sizeof unsigned int, and they are both of BUFFER_SIZE
        __syncthreads();
        unsigned int* uintBuffer = reinterpret_cast<int*>(temp);
        for(int i = threadIdx.x; i < BUFFER_SIZE; i += blockDim.x) {
            uintBuffer[i] = interactionBits[i];
        }
        __syncthreads();
        for (int i = threadIdx.x; i < BUFFER_SIZE; i += blockDim.x) {
            if (sum2[i] != (i == 0 ? 0 : sum2[i-1])) {
                interactionBits[sum2[i]-1] = uintBuffer[i];
            }
        }
        __syncthreads();
        
        // sum, sum2, temp, are now free for use
        // can't use atoms! it is used for overflow
        unsigned int *spareBufferStorage = reinterpret_cast<unsigned int*> atoms;

        // k might be indexing into numValid components?
        
        // TODO: Try on the fly compaction writing to global mem? would need to do atomicAdd here
        // but would save some flops and use less shared memory
        // for each of the possible bits that are set

        // TODO: Try pragma unroll
        // TOOD: Try registers

        // offset stores number of elements in buffer
        unsigned int offset = 0;

        for(int bitcounter = 0; bitcounter < ATOM_THRESHOLD; bitcounter++) {
            // fill up to BUFFER_SIZE (and could probably set bitflags directly to registers)
            for(int k = threadIdx.x; k < BUFFER_SIZE; k += blockDim.x) {
                unsigned int atom2 = sparseAtoms[k];
                unsigned int bitpos = __ffs(interactionBits[k]);
                if(bitpos > 0) {
                    unsigned int atom1 = x*TILE_SIZE+bitpos;
                    sparseAtomsPrefixSumBuffer[k] = make_uint2(atom1, atom2);
                }
                sum[k] = (bitpos > 0) ? 1 : 0;
                interactionBits[k] -= 1 << bitpos;
            }
            __syncthreads();
            prefixSum(sum,temp);
            offset += sum[BUFFER_SIZE-1];
            // gather the atoms using prefix sum
            for (int i = threadIdx.x; i < BUFFER_SIZE; i += blockDim.x) {
                // sum[i]-1 is the scatter index as its an inclusive scan
                if (sum[i] != (i == 0 ? 0 : sum[i-1])) {
                    unsigned int pos = offset+sum[i]-1;
                    sparseAtomsCompactionBuffer[pos] = sparseAtomsPrefixSumBuffer[i];
                    //interactionBits[sum2[i]-1] = uintBuffer[i];
                }
            }
            __syncthreads();
        }

        unsigned int atomsToAdd = offset;

        if (threadIdx.x == 0)
            numSparseAtoms = atomicAdd(numSparseAtoms, atomsToAdd);

        __syncthreads();

        // write out to global memory
        for(int k = threadIdx.x; k < offset; k += blockDim.x) {
            sparseInteractions[numSparseAtoms+k] = sparseCompactionBuffer[k];
        }

        // need to compact large buffer and write buffer to disk
        
        // will NEVER have leftovers!
        // ffs time to identify interactingTiles
        /*
         * __shared__ ushort2 partialBuffer[3*sizeof(ushort2)*threadsPerBlock];
         *
         * unsigned int x; // x-dimension of the tile we're in
         * unsigned int pattern = 10000100; // bitwise
         * unsigned int atom2 = shared[threadIdx.x];
         * for(int i=0;i<3;i++) {
         *    unsigned int pos = __ffs(pattern);
         *    unsigned int atom1 = pos;
         *    partialBuffer[i*threadsPerBlock] = make_ushort2(atom1, atom2);
         *    pattern -= 1 << __ffs(pattern);
         * }
         * __syncthreads();
         *
         * compactPartialBuffer();
         *
         * __syncthreads();
         */




    }

    // no need to worry about this for sparseTiles
    if (numValid == 0 && numDenseAtoms > 0 && finish) {
        // We don't have any more tiles to process, but there were some atoms left over from a
        // previous call to this function.  Save them now.

        if (threadIdx.x == 0)
            baseIndex = atomicAdd(tileInteractionCount, 1);
        __syncthreads();
        if (baseIndex < maxTiles) {
            if (threadIdx.x == 0)
                interactingTiles[baseIndex] = make_ushort2(x, singlePeriodicCopy);
            if (threadIdx.x < TILE_SIZE)
                interactingAtoms[baseIndex*TILE_SIZE+threadIdx.x] = (threadIdx.x < numAtoms ? atoms[threadIdx.x] : NUM_ATOMS);
        }
    }





    // Reset the buffer for processing more tiles.

    for (int i = threadIdx.x; i < BUFFER_SIZE; i += blockDim.x)
        buffer[i] = INVALID;
    __syncthreads();
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
 * [out] tileInteractionCount      - total number of tiles that have interactions
 * [out] interactingTiles      - set of tiles that have interactions
 * [out] interactingAtoms      - a list of atoms that interact with each atom block
 *            eg: block 5 interacts with atoms 1,4,5,6,9,12,19,20
 *                block 1 interacts with atoms 5,88,2,12,4,13,10
 *                block 4 interacts with atoms 3,90,95,45,1,2,5,9
 *              interactingTiles[5][1][4]
 *              interactingAtoms[1,4,5,6,9,12,19,20][5,88,2,12,4,13,10][3,90,95,45,12,5,9]
 * [in] posq                   - x,y,z coordinates of each atom and charge q
 * [in] maxTiles               - maximum number of tiles to process, used for multi-GPUs
 * [in] startBlockIndex        - first block to process, used for multi-GPUs,
 * [in] numBlocks              - total number of atom blocks
 * [in] sortedBlocks           - a sorted list of atom blocks based on volume
 * [in] sortedBlockCenter      - sorted centers, duplicated for fast access to avoid indexing
 * [in] sortedBlockBoundingBox - sorted bounding boxes, duplicated for fast access
 * [in] exclusionIndices       - maps into exclusionRowIndices with the starting position for a given block
 * [in] exclusionRowIndices    - stores the a continuous list of exclusions
 *           eg: block 0 is excluded from block 3,5,6
 *               block 1 is excluded from block 3,4
 *               block 2 is excluded from block 1,3,5,6
 *              exclusionIndices[0][3][5][8]
 *           exclusionRowIndices[3][5][6][3][4][1][3][5][6]
 *                         index 0  1  2  3  4  5  6  7  8 
 * [out] oldPos                - stores the positions of the atoms in which this neighbourlist was built on
 *                             - this is used to decide when to rebuild a neighbourlist
 * [in] rebuildNeighbourList   - whether or not to execute this kernel
 *
 */
extern "C" __global__ void findBlocksWithInteractions(real4 periodicBoxSize, real4 invPeriodicBoxSize, unsigned int* __restrict__ tileInteractionCount,
        ushort2* __restrict__ interactingTiles, unsigned int* __restrict__ interactingAtoms, const real4* __restrict__ posq, unsigned int maxTiles, unsigned int startBlockIndex,
        unsigned int numBlocks, real2* __restrict__ sortedBlocks, const real4* __restrict__ sortedBlockCenter, const real4* __restrict__ sortedBlockBoundingBox,
        const unsigned int* __restrict__ exclusionIndices, const unsigned int* __restrict__ exclusionRowIndices, real4* __restrict__ oldPositions,
        const int* __restrict__ rebuildNeighborList) {
    __shared__ unsigned short buffer[BUFFER_SIZE];
    __shared__ short sum[BUFFER_SIZE]; // for almost full
    __shared__ short sum2[BUFFER_SIZE]; // for partials
    __shared__ ushort2 temp[BUFFER_SIZE];
    __shared__ unsigned int partialList[BUFFER_SIZE];
    __shared__ int denseAtoms[BUFFER_SIZE+TILE_SIZE];
    __shared__ int sparseAtoms[BUFFER_SIZE+TILE_SIZE];

    // TODO: Use spareAtomCompactionBuffer for everything that isn't overlapping. 
    __shared__ unsigned int sparseAtomsPrefixSumBuffer[BUFFER_SIZE];
    __shared__ unsigned int sparseAtomsCompactionBuffers[BUFFER_SIZE*ATOM_THRESHOLD]; 

    __shared__ real3 posBuffer[TILE_SIZE];
    __shared__ int exclusionsForX[MAX_EXCLUSIONS];
    __shared__ int bufferFull;
    __shared__ int globalIndex;
    __shared__ int numDenseAtoms;
    __shared__ int numSparseAtoms;
    
    if (rebuildNeighborList[0] == 0)
        return; // The neighbor list doesn't need to be rebuilt.
    
    int valuesInBuffer = 0;
    if (threadIdx.x == 0)
        bufferFull = false;
    for (int i = 0; i < BUFFER_GROUPS; ++i)
        buffer[i*GROUP_SIZE+threadIdx.x] = INVALID;
    __syncthreads();
    
    // Loop over blocks sorted by size.
    
    for (int i = startBlockIndex+blockIdx.x; i < startBlockIndex+numBlocks; i += gridDim.x) {
        if (threadIdx.x == blockDim.x-1) {
            numDenseAtoms = 0;
            numSparseAtoms = 0;
        }
        real2 sortedKey = sortedBlocks[i];
        unsigned short x = (unsigned short) sortedKey.y;
        real4 blockCenterX = sortedBlockCenter[i];
        real4 blockSizeX = sortedBlockBoundingBox[i];

        // Load exclusion data for block x.
        
        const int exclusionStart = exclusionRowIndices[x];
        const int exclusionEnd = exclusionRowIndices[x+1];
        const int numExclusions = exclusionEnd-exclusionStart;
        for (int j = threadIdx.x; j < numExclusions; j += blockDim.x)
            exclusionsForX[j] = exclusionIndices[exclusionStart+j];
        __syncthreads();
        
        // Compare it to other blocks after this one in sorted order.
        
        for (int base = i+1; base < NUM_BLOCKS; base += blockDim.x) {
            int j = base+threadIdx.x;
            real2 sortedKey2 = (j < NUM_BLOCKS ? sortedBlocks[j] : make_real2(0));
            real4 blockCenterY = (j < NUM_BLOCKS ? sortedBlockCenter[j] : make_real4(0));
            real4 blockSizeY = (j < NUM_BLOCKS ? sortedBlockBoundingBox[j] : make_real4(0));
            unsigned short y = (unsigned short) sortedKey2.y;
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

                int bufferIndex = valuesInBuffer*GROUP_SIZE+threadIdx.x;
                buffer[bufferIndex] = y;
                valuesInBuffer++;

                // cuda-memcheck --tool racecheck will throw errors about this as 
                // RAW/WAW/WAR race condition errors. But this is safe in all instances
                if (!bufferFull && valuesInBuffer == BUFFER_GROUPS)
                    bufferFull = true;
            }
            __syncthreads();
            if (bufferFull) {
                storeInteractionData(x, buffer, sum, sum2, temp, atoms, numDenseAtoms, numSparseAtoms, globalIndex, tileInteractionCount, interactingTiles, interactingAtoms, periodicBoxSize, invPeriodicBoxSize, posq, posBuffer, blockCenterX, blockSizeX, maxTiles, false);
                valuesInBuffer = 0;
                if (threadIdx.x == 0)
                    bufferFull = false;
            }
            __syncthreads();
        }
        storeInteractionData(x, buffer, sum, sum2, temp, atoms, numDenseAtoms, numSparseAtoms, globalIndex, tileInteractionCount, interactingTiles, interactingAtoms, periodicBoxSize, invPeriodicBoxSize, posq, posBuffer, blockCenterX, blockSizeX, maxTiles, true);
    }
    
    // Record the positions the neighbor list is based on.
    
    for (int i = threadIdx.x+blockIdx.x*blockDim.x; i < NUM_ATOMS; i += blockDim.x*gridDim.x)
        oldPositions[i] = posq[i];
}
