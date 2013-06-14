#define WARPS_PER_GROUP (THREAD_BLOCK_SIZE/TILE_SIZE)

#ifndef ENABLE_SHUFFLE
typedef struct {
    real x, y, z;
    real q;
    real fx, fy, fz;
    ATOM_PARAMETER_DATA
#ifndef PARAMETER_SIZE_IS_EVEN
    real padding;
#endif
} AtomData;
#endif

#ifdef ENABLE_SHUFFLE
//support for 64 bit shuffles
static __inline__ __device__ float real_shfl(float var, int srcLane) {
    return __shfl(var, srcLane);
}

static __inline__ __device__ double real_shfl(double var, int srcLane) {
    int hi, lo;
    asm volatile("mov.b64 { %0, %1 }, %2;" : "=r"(lo), "=r"(hi) : "d"(var));
    hi = __shfl(hi, srcLane);
    lo = __shfl(lo, srcLane);
    return __hiloint2double( hi, lo );
}

static __inline__ __device__ long long real_shfl(long long var, int srcLane) {
    int hi, lo;
    asm volatile("mov.b64 { %0, %1 }, %2;" : "=r"(lo), "=r"(hi) : "l"(var));
    hi = __shfl(hi, srcLane);
    lo = __shfl(lo, srcLane);
    // unforunately there isn't an __nv_hiloint2long(hi,lo) intrinsic cast
    int2 fuse; fuse.x = lo; fuse.y = hi;
    return *reinterpret_cast<long long*>(&fuse);
}
#endif

/**
 * Compute nonbonded interactions. The kernel is separated into two parts,
 * tiles with exclusions and tiles without exclusions. It relies heavily on 
 * implicit warp-level synchronization. A tile is defined by two atom blocks 
 * each of warpsize. Each warp computes a range of tiles.
 * 
 * Tiles with exclusions compute the entire set of interactions across
 * atom blocks, equal to warpsize*warpsize. In order to avoid access conflicts 
 * the forces are computed and accumulated diagonally in the manner shown below
 * where, suppose
 *
 * [a-h] comprise atom block 1, [i-p] comprise atom block 2
 *
 * 1 denotes the first set of calculations within the warp
 * 2 denotes the second set of calculations within the warp
 * ... etc.
 * 
 *        threads
 *     0 1 2 3 4 5 6 7
 *         atom1 
 * L    a b c d e f g h 
 * o  i 1 2 3 4 5 6 7 8
 * c  j 8 1 2 3 4 5 6 7
 * a  k 7 8 1 2 3 4 5 6
 * l  l 6 7 8 1 2 3 4 5
 * D  m 5 6 7 8 1 2 3 4 
 * a  n 4 5 6 7 8 1 2 3
 * t  o 3 4 5 6 7 8 1 2
 * a  p 2 3 4 5 6 7 8 1
 *
 * Tiles without exclusions read off directly from the neighbourlist interactingAtoms
 * and follows the same force accumulation method. If more there are more interactingTiles
 * than the size of the neighbourlist initially allocated, the neighbourlist is rebuilt
 * and the full tileset is computed. This should happen on the first step, and very rarely 
 * afterwards.
 *
 * On CUDA devices that support the shuffle intrinsic, on diagonal exclusion tiles use
 * __shfl to broadcast. For all other types of tiles __shfl is used to pass around the 
 * forces, positions, and parameters when computing the forces. 
 *
 * [out]forceBuffers    - forces on each atom to eventually be accumulated
 * [out]energyBuffer    - energyBuffer to eventually be accumulated
 * [in]posq             - x,y,z,charge 
 * [in]exclusions       - 1024-bit flags denoting atom-atom exclusions for each tile
 * [in]exclusionTiles   - x,y denotes the indices of tiles that have an exclusion
 * [in]startTileIndex   - index into first tile to be processed
 * [in]numTileIndices   - number of tiles this context is responsible for processing
 * [in]int tiles        - the atom block for each tile
 * [in]interactionCount - total number of tiles that have an interaction
 * [in]maxTiles         - stores the size of the neighbourlist in case it needs 
 *                      - to be expanded
 * [in]periodicBoxSize  - size of the Periodic Box, last dimension (w) not used
 * [in]invPeriodicBox   - inverse of the periodicBoxSize, pre-computed for speed
 * [in]blockCenter      - the center of each block in euclidean coordinates
 * [in]blockSize        - size of the each block, radiating from the center
 *                      - x is half the distance of total length
 *                      - y is half the distance of total width
 *                      - z is half the distance of total height
 *                      - w is not used
 * [in]interactingAtoms - a list of interactions within a given tile     
 *
 */

__device__ void printBuffer(float *buffer, int gid) {
    if(threadIdx.x == gid) {
        for(int i=0; i<8; i++) {
            for(int j=0; j<32; j++) {
                printf("%f ", buffer[i*32+j]);
            }
            printf("\n");
        }
    }
}

extern "C" __global__ void computeNonbonded(
        unsigned long long* __restrict__ forceBuffers, real* __restrict__ energyBuffer, const real4* __restrict__ posq, const tileflags* __restrict__ exclusions,
        const ushort2* __restrict__ exclusionTiles, unsigned int startTileIndex, unsigned int numTileIndices
#ifdef USE_CUTOFF
        , const int* __restrict__ tiles, const unsigned int* __restrict__ interactionCount, real4 periodicBoxSize, real4 invPeriodicBoxSize, 
        unsigned int maxTiles, const real4* __restrict__ blockCenter, const real4* __restrict__ blockSize, const unsigned int* __restrict__ interactingAtoms
#endif
        PARAMETER_ARGUMENTS) {
    const unsigned int totalWarps = (blockDim.x*gridDim.x)/TILE_SIZE;
    const unsigned int warp = (blockIdx.x*blockDim.x+threadIdx.x)/TILE_SIZE; // global warpIndex
    const unsigned int tgx = threadIdx.x & (TILE_SIZE-1); // index within the warp
    const unsigned int tbx = threadIdx.x - tgx;           // block warpIndex
    real energy = 0.0f;
    // used shared memory if the device cannot shuffle
#ifndef ENABLE_SHUFFLE
    __shared__ AtomData localData[THREAD_BLOCK_SIZE];
#endif
    // First loop: process tiles that contain exclusions.
    const unsigned int firstExclusionTile = FIRST_EXCLUSION_TILE+warp*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    const unsigned int lastExclusionTile = FIRST_EXCLUSION_TILE+(warp+1)*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    for (int pos = firstExclusionTile; pos < lastExclusionTile; pos++) {
        const ushort2 tileIndices = exclusionTiles[pos];
        const unsigned int x = tileIndices.x;
        const unsigned int y = tileIndices.y;
        real3 force = make_real3(0);
        unsigned int atom1 = x*TILE_SIZE + tgx;
        real4 posq1 = posq[atom1];
        LOAD_ATOM1_PARAMETERS
#ifdef USE_EXCLUSIONS
        tileflags excl = exclusions[pos*TILE_SIZE+tgx];
#endif
        const bool hasExclusions = true;
        if (x == y) {
            // This tile is on the diagonal.
#ifdef ENABLE_SHUFFLE
            real4 shflPosq = posq1;
#else
            localData[threadIdx.x].x = posq1.x;
            localData[threadIdx.x].y = posq1.y;
            localData[threadIdx.x].z = posq1.z;
            localData[threadIdx.x].q = posq1.w;
            LOAD_LOCAL_PARAMETERS_FROM_1
#endif

            // we do not need to fetch parameters from global since this is a symmetric tile
            // instead we can broadcast the values using shuffle
            for (unsigned int j = 0; j < TILE_SIZE; j++) {
                int atom2 = tbx+j;
                real4 posq2;
#ifdef ENABLE_SHUFFLE
                BROADCAST_WARP_DATA
#else   
                posq2 = make_real4(localData[atom2].x, localData[atom2].y, localData[atom2].z, localData[atom2].q);
#endif
                real3 delta = make_real3(posq2.x-posq1.x, posq2.y-posq1.y, posq2.z-posq1.z);
#ifdef USE_PERIODIC
                delta.x -= floor(delta.x*invPeriodicBoxSize.x+0.5f)*periodicBoxSize.x;
                delta.y -= floor(delta.y*invPeriodicBoxSize.y+0.5f)*periodicBoxSize.y;
                delta.z -= floor(delta.z*invPeriodicBoxSize.z+0.5f)*periodicBoxSize.z;
#endif
                real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                real invR = RSQRT(r2);
                real r = RECIP(invR);
                LOAD_ATOM2_PARAMETERS
                atom2 = y*TILE_SIZE+j;
#ifdef USE_SYMMETRIC
                real dEdR = 0.0f;
#else
                real3 dEdR1 = make_real3(0);
                real3 dEdR2 = make_real3(0);
#endif
#ifdef USE_EXCLUSIONS
                bool isExcluded = (atom1 >= NUM_ATOMS || atom2 >= NUM_ATOMS || !(excl & 0x1));
#endif
                real tempEnergy = 0.0f;
                COMPUTE_INTERACTION
                energy += 0.5f*tempEnergy;
#ifdef USE_SYMMETRIC
                force.x -= delta.x*dEdR;
                force.y -= delta.y*dEdR;
                force.z -= delta.z*dEdR;
#else
                force.x -= dEdR1.x;
                force.y -= dEdR1.y;
                force.z -= dEdR1.z;
#endif
#ifdef USE_EXCLUSIONS
                excl >>= 1;
#endif
            }
        }
        else {
            // This is an off-diagonal tile.
            unsigned int j = y*TILE_SIZE + tgx;
            real4 shflPosq = posq[j];
#ifdef ENABLE_SHUFFLE
            real3 shflForce;
            shflForce.x = 0.0f;
            shflForce.y = 0.0f;
            shflForce.z = 0.0f;
#else
            localData[threadIdx.x].x = shflPosq.x;
            localData[threadIdx.x].y = shflPosq.y;
            localData[threadIdx.x].z = shflPosq.z;
            localData[threadIdx.x].q = shflPosq.w;
            localData[threadIdx.x].fx = 0.0f;
            localData[threadIdx.x].fy = 0.0f;
            localData[threadIdx.x].fz = 0.0f;
#endif
            DECLARE_LOCAL_PARAMETERS
            LOAD_LOCAL_PARAMETERS_FROM_GLOBAL
#ifdef USE_EXCLUSIONS
            excl = (excl >> tgx) | (excl << (TILE_SIZE - tgx));
#endif
            unsigned int tj = tgx;
            for (j = 0; j < TILE_SIZE; j++) {
                int atom2 = tbx+tj;
#ifdef ENABLE_SHUFFLE
                real4 posq2 = shflPosq;
#else
                real4 posq2 = make_real4(localData[atom2].x, localData[atom2].y, localData[atom2].z, localData[atom2].q);
#endif
                real3 delta = make_real3(posq2.x-posq1.x, posq2.y-posq1.y, posq2.z-posq1.z);
#ifdef USE_PERIODIC
                delta.x -= floor(delta.x*invPeriodicBoxSize.x+0.5f)*periodicBoxSize.x;
                delta.y -= floor(delta.y*invPeriodicBoxSize.y+0.5f)*periodicBoxSize.y;
                delta.z -= floor(delta.z*invPeriodicBoxSize.z+0.5f)*periodicBoxSize.z;
#endif
                real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
#ifdef USE_CUTOFF
                if (r2 < CUTOFF_SQUARED) {
#endif
                    real invR = RSQRT(r2);
                    real r = RECIP(invR);
                    LOAD_ATOM2_PARAMETERS
                    atom2 = y*TILE_SIZE+tj;
#ifdef USE_SYMMETRIC
                    real dEdR = 0.0f;
#else
                    real3 dEdR1 = make_real3(0);
                    real3 dEdR2 = make_real3(0);
#endif
#ifdef USE_EXCLUSIONS
                    bool isExcluded = (atom1 >= NUM_ATOMS || atom2 >= NUM_ATOMS || !(excl & 0x1));
#endif
                    real tempEnergy = 0.0f;
                    COMPUTE_INTERACTION
                    energy += tempEnergy;
#ifdef USE_SYMMETRIC
                    delta *= dEdR;
                    force.x -= delta.x;
                    force.y -= delta.y;
                    force.z -= delta.z;
#ifdef ENABLE_SHUFFLE
                    shflForce.x += delta.x;
                    shflForce.y += delta.y;
                    shflForce.z += delta.z;

#else
                    localData[tbx+tj].fx += delta.x;
                    localData[tbx+tj].fy += delta.y;
                    localData[tbx+tj].fz += delta.z;
#endif
#else // !USE_SYMMETRIC
                    force.x -= dEdR1.x;
                    force.y -= dEdR1.y;
                    force.z -= dEdR1.z;
#ifdef ENABLE_SHUFFLE
                    shflForce.x += dEdR2.x;
                    shflForce.y += dEdR2.y;
                    shflForce.z += dEdR2.z;
#else
                    localData[tbx+tj].fx += dEdR2.x;
                    localData[tbx+tj].fy += dEdR2.y;
                    localData[tbx+tj].fz += dEdR2.z;
#endif 
#endif // end USE_SYMMETRIC
#ifdef USE_CUTOFF
                }
#endif
#ifdef USE_EXCLUSIONS
                excl >>= 1;
#endif
#ifdef ENABLE_SHUFFLE
                SHUFFLE_WARP_DATA
#endif
                // cycles the indices
                // 0 1 2 3 4 5 6 7 -> 1 2 3 4 5 6 7 0
                tj = (tj + 1) & (TILE_SIZE - 1);
            }
            const unsigned int offset = y*TILE_SIZE + tgx;
            // write results for off diagonal tiles
#ifdef ENABLE_SHUFFLE
            atomicAdd(&forceBuffers[offset], static_cast<unsigned long long>((long long) (shflForce.x*0x100000000)));
            atomicAdd(&forceBuffers[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (shflForce.y*0x100000000)));
            atomicAdd(&forceBuffers[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (shflForce.z*0x100000000)));
#else
            atomicAdd(&forceBuffers[offset], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fx*0x100000000)));
            atomicAdd(&forceBuffers[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fy*0x100000000)));
            atomicAdd(&forceBuffers[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fz*0x100000000)));
#endif
        }
        // Write results for on and off diagonal tiles
        const unsigned int offset = x*TILE_SIZE + tgx;
        atomicAdd(&forceBuffers[offset], static_cast<unsigned long long>((long long) (force.x*0x100000000)));
        atomicAdd(&forceBuffers[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (force.y*0x100000000)));
        atomicAdd(&forceBuffers[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (force.z*0x100000000)));
    }

    // Second loop: tiles without exclusions, either from the neighbor list (with cutoff) or just enumerating all
    // of them (no cutoff).
#ifdef USE_CUTOFF
    const unsigned int numTiles = interactionCount[0];
    int pos = (numTiles > maxTiles ? startTileIndex+warp*numTileIndices/totalWarps : warp*numTiles/totalWarps);
    int end = (numTiles > maxTiles ? startTileIndex+(warp+1)*numTileIndices/totalWarps : (warp+1)*numTiles/totalWarps);
#else
    const unsigned int numTiles = numTileIndices;
    int pos = startTileIndex+warp*numTiles/totalWarps;
    int end = startTileIndex+(warp+1)*numTiles/totalWarps;
#endif
    int skipBase = 0;
    int currentSkipIndex = tbx;
    // atomIndices can probably be shuffled as well
    // but it probably wouldn't make things any faster



    // used for single precision
    // with cutoffs
    // with periodic boundary conditions

    __shared__ volatile float ffsReductionBuffer[32*8];

    //__shared__ int atomIndices[THREAD_BLOCK_SIZE];
    //__shared__ volatile int skipTiles[THREAD_BLOCK_SIZE];
    
    volatile int* skipTiles = reinterpret_cast<volatile int*>(ffsReductionBuffer);
    
    skipTiles[threadIdx.x] = -1;

    while (pos < end) {
        const bool hasExclusions = false;
        real3 force = make_real3(0);
        bool includeTile = true;

        // Extract the coordinates of this tile.
        unsigned int x, y;
        bool singlePeriodicCopy = false;

#ifdef USE_CUTOFF
        if (numTiles <= maxTiles) {
            x = tiles[pos];
            real4 blockSizeX = blockSize[x];
            singlePeriodicCopy = (0.5f*periodicBoxSize.x-blockSizeX.x >= CUTOFF &&
                                  0.5f*periodicBoxSize.y-blockSizeX.y >= CUTOFF &&
                                  0.5f*periodicBoxSize.z-blockSizeX.z >= CUTOFF);
        }
        else
#endif
        {
            y = (unsigned int) floor(NUM_BLOCKS+0.5f-SQRT((NUM_BLOCKS+0.5f)*(NUM_BLOCKS+0.5f)-2*pos));
            x = (pos-y*NUM_BLOCKS+y*(y+1)/2);
            if (x < y || x >= NUM_BLOCKS) { // Occasionally happens due to roundoff error.
                y += (x < y ? -1 : 1);
                x = (pos-y*NUM_BLOCKS+y*(y+1)/2);
            }

            // Skip over tiles that have exclusions, since they were already processed.

            while (skipTiles[tbx+TILE_SIZE-1] < pos) {
                if (skipBase+tgx < NUM_TILES_WITH_EXCLUSIONS) {
                    ushort2 tile = exclusionTiles[skipBase+tgx];
                    skipTiles[threadIdx.x] = tile.x + tile.y*NUM_BLOCKS - tile.y*(tile.y+1)/2;
                }
                else
                    skipTiles[threadIdx.x] = end;
                skipBase += TILE_SIZE;            
                currentSkipIndex = tbx;
            }
            while (skipTiles[currentSkipIndex] < pos)
                currentSkipIndex++;
            includeTile = (skipTiles[currentSkipIndex] != pos);
        }
        if (includeTile) {
            // everything in atom1 is up for shuffling later
            //real3 force1;
            const real4 shflPosq1 = posq[x*TILE_SIZE+tgx]; // used for shuffling   
            const real2 shflSigmaEpsilon1 = global_sigmaEpsilon[x*TILE_SIZE+tgx];
            // writing directly to atomics every 8x32 intermittently may be faster?
            // as it would save another 4 registers
            real3 force1;
            force1.x = 0; force1.y = 0; force1.z = 0;

            // we can do some ninja register optimization if need be to reduce register pressure.
            // at any given moment in the 8x32 computation, we only need the posq of 8 atoms, yet we store 32 of them
            // in every thread. Instead we can:
            // threads 0-7   hold register components for posq1x for atoms i to i+8
            // threads 8-15  hold register components for posq1y for atoms i to i+8
            // threads 16-23 hold register components for posq1z for atoms i to i+8
            // threads 24-31 hold register components for posq1q for atoms i to i+8

            // we can do the same for sigmaEpsilon as well
            // since we only need 8 posqs at any given moment, we can break sigmaEpsilon into two parts
            // so that threads 0-7 save the .x component and 8-15 save the .y component
            // when shuffling we load from different registers depending on component
            // const shflSigmaEpsilon1 = global_sigmaEpsilon[atom1];

            // in total, 4 registers can be saved here

            const unsigned int atom2 = interactingAtoms[pos*TILE_SIZE+tgx];
            const real4 posq2 = posq[atom2];

            // check and see register usage later
            real3 force2;
            force2.x = 0; force2.y = 0; force2.z = 0;
            const float2 sigmaEpsilon2 = global_sigmaEpsilon[atom2];
            //LOAD_ATOM2_PARAMETERS;
            //DECLARE_LOCAL_PARAMETERS;
            //exactly 4 loops, try unrolling later

            //verify results
            /*
            unsigned int ixnBits = 0;
            for(int j=0; j<32; j++) {
                    const int atom1 = x*TILE_SIZE+j;
                    // load posq1 from registers;
                    real3 tmpPosq1;
                    tmpPosq1.x = __shfl(shflPosq1.x,j);
                    tmpPosq1.y = __shfl(shflPosq1.y,j);
                    tmpPosq1.z = __shfl(shflPosq1.z,j);
                    real3 delta = make_real3(posq2.x-tmpPosq1.x,posq2.y-tmpPosq1.y,posq2.z-tmpPosq1.z);
                    // need to take advantage of single periodic copy later, would save us a lot of flops since 
                    // we recompute distances very often
    #ifdef USE_PERIODIC
                    delta.x -= floor(delta.x*invPeriodicBoxSize.x+0.5f)*periodicBoxSize.x;
                    delta.y -= floor(delta.y*invPeriodicBoxSize.y+0.5f)*periodicBoxSize.y;
                    delta.z -= floor(delta.z*invPeriodicBoxSize.z+0.5f)*periodicBoxSize.z;
    #endif
                    real r2 = delta.x*delta.x+delta.y*delta.y+delta.z*delta.z;
                    ixnBits |= ((r2 < CUTOFF_SQUARED) << j);
            }

            if(blockIdx.x==34 && threadIdx.x < 32) {
                char buffer[32+1];
                buffer[32]='\0';
                for(int j=0;j<32;j++) {
                    bool bitset = 0;
                    if((ixnBits & 1 << j) > 0) {    
                        bitset = 1;
                    }

                    buffer[j] = bitset+48;
                }
                printf("thread %d bid %d: %s\n", threadIdx.x, blockIdx.x, buffer);
            }

            if(blockIdx.x == 34 && threadIdx.x == 0)
                printf("==============================================================\n");
            */

            // bits set are correct!



            for(int rowOffset = 0; rowOffset < 32; rowOffset+=8) {


                // clear reduction buffer

                const int clearStart = tgx*8;
                const int clearEnd = clearStart+8; 
                for(int i=clearStart; i<clearEnd; i++) {
                    ffsReductionBuffer[i] = 0;
                }
                
                unsigned int interactionBits = 0;
                const unsigned int startBit = rowOffset;
                const unsigned int endBit = rowOffset+8;
                // mark in advance the points that interact
                for(unsigned int i = startBit; i < endBit; i++) {
                    const int atom1 = x*TILE_SIZE+i;
                    // load posq1 from registers;
                    real3 tmpPosq1;
                    tmpPosq1.x = __shfl(shflPosq1.x,i);
                    tmpPosq1.y = __shfl(shflPosq1.y,i);
                    tmpPosq1.z = __shfl(shflPosq1.z,i);
                    real3 delta = make_real3(posq2.x-tmpPosq1.x,posq2.y-tmpPosq1.y,posq2.z-tmpPosq1.z);
                    // need to take advantage of single periodic copy later, would save us a lot of flops since 
                    // we recompute distances very often
    #ifdef USE_PERIODIC
                    delta.x -= floor(delta.x*invPeriodicBoxSize.x+0.5f)*periodicBoxSize.x;
                    delta.y -= floor(delta.y*invPeriodicBoxSize.y+0.5f)*periodicBoxSize.y;
                    delta.z -= floor(delta.z*invPeriodicBoxSize.z+0.5f)*periodicBoxSize.z;
    #endif
                    real r2 = delta.x*delta.x+delta.y*delta.y+delta.z*delta.z;
                    interactionBits |= ((r2 < CUTOFF_SQUARED) << i);
                }            

                //print out interaction bits
                /*             
                if(blockIdx.x==34 && threadIdx.x < 32) {
                    char buffer[32+1];
                    buffer[32]='\0';
                    for(int j=0;j<32;j++) {
                        bool bitset = 0;
                        if((interactionBits & 1 << j) > 0) {    
                            bitset = 1;
                        }

                        buffer[j] = bitset+48;
                    }
                    printf("thread %d bid %d: %s\n", threadIdx.x, blockIdx.x, buffer);
                }
                */

                // compute interactions in 32x8 chunks
                for(int i=0; i<8; i++) {
                //while((unsigned int bitPos = __ffs(interactionBits)) > 0) {
                    //unsigned int offset = bitPos-1;
                    // initialize ffsReductionBuffer to zero
                    // this must be int, not unsigned, as __ffs(interactionBits) can return 0!
                    const int offset = __ffs(interactionBits)-1;

                    // clear the set bit so ffs can move on to the next set bit
                    interactionBits &= 0 << offset;

                    const unsigned int atom1 = x*TILE_SIZE+offset;
                    const unsigned int srcLane = (offset >= 0) ? offset : tgx;

                    // if any threads has offset > 0
                    // all threads in a warp must issue __shfl, else result is undefined behavior
                    if(__any(offset>=0)) {
                        real2 sigmaEpsilon1;
                        sigmaEpsilon1.x = __shfl(shflSigmaEpsilon1.x, srcLane);
                        sigmaEpsilon1.y = __shfl(shflSigmaEpsilon1.y, srcLane);
                        real4 posq1;
                        posq1.x = __shfl(shflPosq1.x, srcLane);
                        posq1.y = __shfl(shflPosq1.y, srcLane);
                        posq1.z = __shfl(shflPosq1.z, srcLane);
                        posq1.w = __shfl(shflPosq1.w, srcLane);
                    
                        // can remove this later?
                        if(offset >= 0) {
                            real3 delta = make_real3(posq2.x-posq1.x,posq2.y-posq1.y,posq2.z-posq1.z);
#ifdef USE_PERIODIC
                            delta.x -= floor(delta.x*invPeriodicBoxSize.x+0.5f)*periodicBoxSize.x;
                            delta.y -= floor(delta.y*invPeriodicBoxSize.y+0.5f)*periodicBoxSize.y;
                            delta.z -= floor(delta.z*invPeriodicBoxSize.z+0.5f)*periodicBoxSize.z;
#endif
                            real r2 = delta.x*delta.x+delta.y*delta.y+delta.z*delta.z;
                            real invR = RSQRT(r2);
                            real r = RECIP(invR);
#ifdef USE_SYMMETRIC
                            real dEdR = 0.0f;
#else
                            real3 dEdR1 = make_real3(0);
                            real3 dEdR2 = make_real3(0);
#endif
#ifdef USE_EXCLUSIONS
                            bool isExcluded = (atom1 >= NUM_ATOMS || atom2 >= NUM_ATOMS);
#endif
                            real tempEnergy = 0.0f;
                            COMPUTE_INTERACTION
                            energy += tempEnergy;
#ifdef USE_SYMMETRIC
                            delta *= dEdR;
                            force2.x -= delta.x;
                            force2.y -= delta.y;
                            force2.z -= delta.z;
                            ffsReductionBuffer[32*(offset%8)+tgx] = dEdR;
#else // !USE_SYMMETRIC
                            force.x -= dEdR1.x;
                            force.y -= dEdR1.y;
                            force.z -= dEdR1.z;
                            ffsReductionBuffer[32*(offset%8)+tgx] = dEdR2;
#endif // end USE_SYMMETRIC
                        }
                    }
                }
                // reduce the buffer
                //---t0-->---t1-->---t2-->---t3-->0
                //---t4-->---t5-->---t6-->---t7-->1
                //------->------->------->------->2
                //------->------->------->------->3
                //------->------->------->------->4
                //------->------->------->------->5
                //------->------->------->------->6
                //------->------->------->------->7
                // shuffle add 1
                //-------t0------>-------t2------>0
                //-------t4------>-------t6------>1
                //--------------->--------------->2
                //--------------->--------------->3
                //--------------->--------------->4
                //--------------->--------------->5
                //--------------->--------------->6
                //--------------->--------------->7
                // shuffle add 2
                //---------------t0-------------->0
                //---------------t4-------------->1
                //---------------t8-------------->2
                //---------------t12------------->3
                //---------------t16------------->4
                //---------------t20------------->5
                //---------------t24------------->6
                //---------------t28------------->7
                // t0: start 0
                // t1: start 8
                // t2: start 16
                // t3: start 24
                // t4: start 32

                real3 force1Accum;
                force1Accum.x = 0;
                force1Accum.y = 0;
                force1Accum.z = 0;

                // TODO: Guarantee that no boundary edge conditions can occur here
                // since this is pretty bloody ugly already

                // cycle offset so accumulation is convenient at the end

                // note unsigned int despite possible mod 32 of a negative number

                // less than 20 flops .. and mods are expensive? change to tgx & (8-1) (mod 8)
                //const unsigned int start = (tgx%8)*32+(tgx-(tgx%8)-rowOffset)%32;
                const int start = tgx*8;
                const unsigned int end = start+8;
                for(unsigned int jj = start; jj < end; jj++) {
                    const unsigned int atomRow = tgx/4+rowOffset;
                    // recompute delta
                    real3 posq1;
                    posq1.x = __shfl(shflPosq1.x, atomRow);
                    posq1.y = __shfl(shflPosq1.y, atomRow);
                    posq1.z = __shfl(shflPosq1.z, atomRow);
                    real3 delta = make_real3(posq2.x-posq1.x,posq2.y-posq1.y,posq2.z-posq1.z);
    #ifdef USE_PERIODIC
                    delta.x -= floor(delta.x*invPeriodicBoxSize.x+0.5f)*periodicBoxSize.x;
                    delta.y -= floor(delta.y*invPeriodicBoxSize.y+0.5f)*periodicBoxSize.y;
                    delta.z -= floor(delta.z*invPeriodicBoxSize.z+0.5f)*periodicBoxSize.z;
    #endif
                    const real energy = ffsReductionBuffer[jj];
                    force1Accum.x += delta.x*energy;
                    force1Accum.y += delta.y*energy;
                    force1Accum.z += delta.z*energy;
                }

                // can we ninja this so it just last set of threads to accumulate just happen to match


                // block0, threads 0 1 2  3  4  5  6  7,  16 17 18 19 20 21 22 23 partake
                // block1, threads 8 9 10 11 12 13 14 15, 24 25 26 27 28 29 30 31 partake
                // .. etc

                //all threads in warp must call __shfl if they are a srcLane

                // 7 flops....v 
                //unsigned int srcLane = ( ((tgx-rowOffset)%32)/8 == 0 || ((tgx-rowOffset)%32)/8 == 2 ) ? (tgx+8)%32 : tgx;
                unsigned int srcLane = (tgx % 2 == 0) ? tgx+1 : tgx;
                force1Accum.x += __shfl(force1Accum.x, srcLane);
                force1Accum.y += __shfl(force1Accum.y, srcLane);
                force1Accum.z += __shfl(force1Accum.z, srcLane);

                //srcLane = (tgx-rowOffset)/8 == 0 ? (tgx+16)%32 : tgx;
                srcLane = (tgx % 4 == 0) ? tgx+2 : tgx;
                force1Accum.x += __shfl(force1Accum.x, srcLane);
                force1Accum.y += __shfl(force1Accum.y, srcLane);
                force1Accum.z += __shfl(force1Accum.z, srcLane);

                if( tgx % 4 != 0 ) {
                    force1Accum.x = 0;
                    force1Accum.y = 0;
                    force1Accum.z = 0;
                }
                // update in registers
                // this is wrong!!!!
                // threads 0,4,8,12,16,20,24,28 holds accumulated force values for
                // f0,f1,f2, f3, f4, f5, f6, f7  <-- loop 0
                // f8,f9,f10,f11,f12,f13,f14,f15 <-- loop 1
                // etc.

                srcLane = (tgx >= start+rowOffset) && (tgx < end+rowOffset) ? (tgx-rowOffset)*4: tgx;
                // 3 shuffles = 6 cycles
                // 6 cycles
                force1.x += __shfl(force1Accum.x, srcLane);
                force1.y += __shfl(force1Accum.y, srcLane);
                force1.z += __shfl(force1Accum.z, srcLane);
            }
            const int atom1 = x*TILE_SIZE+tgx;
            atomicAdd(&forceBuffers[atom1], static_cast<unsigned long long>((long long) (force1.x*0x100000000)));
            atomicAdd(&forceBuffers[atom1+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (force1.y*0x100000000)));
            atomicAdd(&forceBuffers[atom1+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (force1.z*0x100000000)));          
//#ifdef USE_CUTOFF
//            unsigned int atom2 = atomIndices[threadIdx.x];
//#else
//            unsigned int atom2 = y*TILE_SIZE + tgx;
//#endif
            if (atom2 < PADDED_NUM_ATOMS) {
                atomicAdd(&forceBuffers[atom2], static_cast<unsigned long long>((long long) (force2.x*0x100000000)));
                atomicAdd(&forceBuffers[atom2+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (force2.y*0x100000000)));
                atomicAdd(&forceBuffers[atom2+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (force2.z*0x100000000)));
            }
        }
        pos++;
        /*
            unsigned int atom1 = x*TILE_SIZE + tgx;
            // Load atom data for this tile.
            real4 posq1 = posq[atom1];
            LOAD_ATOM1_PARAMETERS
            //const unsigned int localAtomIndex = threadIdx.x;
#ifdef USE_CUTOFF
            unsigned int j = (numTiles <= maxTiles ? interactingAtoms[pos*TILE_SIZE+tgx] : y*TILE_SIZE + tgx);
#else
            unsigned int j = y*TILE_SIZE + tgx;
#endif
            atomIndices[threadIdx.x] = j;
#ifdef ENABLE_SHUFFLE
            DECLARE_LOCAL_PARAMETERS
            real4 shflPosq;
            real3 shflForce;
            shflForce.x = 0.0f;
            shflForce.y = 0.0f;
            shflForce.z = 0.0f;
#endif
            if (j < PADDED_NUM_ATOMS) {
                // Load position of atom j from from global memory
#ifdef ENABLE_SHUFFLE
                shflPosq = posq[j];
#else
                localData[threadIdx.x].x = posq[j].x;
                localData[threadIdx.x].y = posq[j].y;
                localData[threadIdx.x].z = posq[j].z;
                localData[threadIdx.x].q = posq[j].w;
                localData[threadIdx.x].fx = 0.0f;
                localData[threadIdx.x].fy = 0.0f;
                localData[threadIdx.x].fz = 0.0f;
#endif                
                LOAD_LOCAL_PARAMETERS_FROM_GLOBAL
            }
#ifdef USE_PERIODIC
            if (0 && singlePeriodicCopy) {
                // The box is small enough that we can just translate all the atoms into a single periodic
                // box, then skip having to apply periodic boundary conditions later.
                real4 blockCenterX = blockCenter[x];
                posq1.x -= floor((posq1.x-blockCenterX.x)*invPeriodicBoxSize.x+0.5f)*periodicBoxSize.x;
                posq1.y -= floor((posq1.y-blockCenterX.y)*invPeriodicBoxSize.y+0.5f)*periodicBoxSize.y;
                posq1.z -= floor((posq1.z-blockCenterX.z)*invPeriodicBoxSize.z+0.5f)*periodicBoxSize.z;
#ifdef ENABLE_SHUFFLE
                shflPosq.x -= floor((shflPosq.x-blockCenterX.x)*invPeriodicBoxSize.x+0.5f)*periodicBoxSize.x;
                shflPosq.y -= floor((shflPosq.y-blockCenterX.y)*invPeriodicBoxSize.y+0.5f)*periodicBoxSize.y;
                shflPosq.z -= floor((shflPosq.z-blockCenterX.z)*invPeriodicBoxSize.z+0.5f)*periodicBoxSize.z;
#else
                localData[threadIdx.x].x -= floor((localData[threadIdx.x].x-blockCenterX.x)*invPeriodicBoxSize.x+0.5f)*periodicBoxSize.x;
                localData[threadIdx.x].y -= floor((localData[threadIdx.x].y-blockCenterX.y)*invPeriodicBoxSize.y+0.5f)*periodicBoxSize.y;
                localData[threadIdx.x].z -= floor((localData[threadIdx.x].z-blockCenterX.z)*invPeriodicBoxSize.z+0.5f)*periodicBoxSize.z;
#endif
                unsigned int tj = tgx;
                for (j = 0; j < TILE_SIZE; j++) {
                    int atom2 = tbx+tj;
#ifdef ENABLE_SHUFFLE
                    real4 posq2 = shflPosq; 
#else
                    real4 posq2 = make_real4(localData[atom2].x, localData[atom2].y, localData[atom2].z, localData[atom2].q);
#endif
                    real3 delta = make_real3(posq2.x-posq1.x, posq2.y-posq1.y, posq2.z-posq1.z);
                    real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                    if (r2 < CUTOFF_SQUARED) {
                        real invR = RSQRT(r2);
                        real r = RECIP(invR);
                        LOAD_ATOM2_PARAMETERS
                        atom2 = atomIndices[tbx+tj];
#ifdef USE_SYMMETRIC
                        real dEdR = 0.0f;
#else
                        real3 dEdR1 = make_real3(0);
                        real3 dEdR2 = make_real3(0);
#endif
#ifdef USE_EXCLUSIONS
                        bool isExcluded = (atom1 >= NUM_ATOMS || atom2 >= NUM_ATOMS);
#endif
                        real tempEnergy = 0.0f;
                        COMPUTE_INTERACTION
                        energy += tempEnergy;
#ifdef USE_SYMMETRIC
                        delta *= dEdR;
                        force.x -= delta.x;
                        force.y -= delta.y;
                        force.z -= delta.z;
#ifdef ENABLE_SHUFFLE
                        shflForce.x += delta.x;
                        shflForce.y += delta.y;
                        shflForce.z += delta.z;

#else
                        localData[tbx+tj].fx += delta.x;
                        localData[tbx+tj].fy += delta.y;
                        localData[tbx+tj].fz += delta.z;
#endif
#else // !USE_SYMMETRIC
                        force.x -= dEdR1.x;
                        force.y -= dEdR1.y;
                        force.z -= dEdR1.z;
#ifdef ENABLE_SHUFFLE
                        shflForce.x += dEdR2.x;
                        shflForce.y += dEdR2.y;
                        shflForce.z += dEdR2.z;
#else
                        localData[tbx+tj].fx += dEdR2.x;
                        localData[tbx+tj].fy += dEdR2.y;
                        localData[tbx+tj].fz += dEdR2.z;
#endif 
#endif // end USE_SYMMETRIC
                    }
#ifdef ENABLE_SHUFFLE
                    SHUFFLE_WARP_DATA
#endif
                    tj = (tj + 1) & (TILE_SIZE - 1);
                }
            }
            else
#endif
            {
                // We need to apply periodic boundary conditions separately for each interaction.
                unsigned int tj = tgx;
                for (j = 0; j < TILE_SIZE; j++) {
                    int atom2 = tbx+tj;
#ifdef ENABLE_SHUFFLE
                    real4 posq2 = shflPosq;
#else
                    real4 posq2 = make_real4(localData[atom2].x, localData[atom2].y, localData[atom2].z, localData[atom2].q);
#endif
                    real3 delta = make_real3(posq2.x-posq1.x, posq2.y-posq1.y, posq2.z-posq1.z);
#ifdef USE_PERIODIC
                    delta.x -= floor(delta.x*invPeriodicBoxSize.x+0.5f)*periodicBoxSize.x;
                    delta.y -= floor(delta.y*invPeriodicBoxSize.y+0.5f)*periodicBoxSize.y;
                    delta.z -= floor(delta.z*invPeriodicBoxSize.z+0.5f)*periodicBoxSize.z;
#endif
                    real r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
#ifdef USE_CUTOFF
                    if (r2 < CUTOFF_SQUARED) {
#endif
                        real invR = RSQRT(r2);
                        real r = RECIP(invR);
                        LOAD_ATOM2_PARAMETERS
                        atom2 = atomIndices[tbx+tj];
#ifdef USE_SYMMETRIC
                        real dEdR = 0.0f;
#else
                        real3 dEdR1 = make_real3(0);
                        real3 dEdR2 = make_real3(0);
#endif
#ifdef USE_EXCLUSIONS
                        bool isExcluded = (atom1 >= NUM_ATOMS || atom2 >= NUM_ATOMS);
#endif
                        real tempEnergy = 0.0f;
                        COMPUTE_INTERACTION
                        energy += tempEnergy;
#ifdef USE_SYMMETRIC
                        delta *= dEdR;
                        force.x -= delta.x;
                        force.y -= delta.y;
                        force.z -= delta.z;
#ifdef ENABLE_SHUFFLE
                        shflForce.x += delta.x;
                        shflForce.y += delta.y;
                        shflForce.z += delta.z;

#else
                        localData[tbx+tj].fx += delta.x;
                        localData[tbx+tj].fy += delta.y;
                        localData[tbx+tj].fz += delta.z;
#endif
#else // !USE_SYMMETRIC
                        force.x -= dEdR1.x;
                        force.y -= dEdR1.y;
                        force.z -= dEdR1.z;
#ifdef ENABLE_SHUFFLE
                        shflForce.x += dEdR2.x;
                        shflForce.y += dEdR2.y;
                        shflForce.z += dEdR2.z;
#else
                        localData[tbx+tj].fx += dEdR2.x;
                        localData[tbx+tj].fy += dEdR2.y;
                        localData[tbx+tj].fz += dEdR2.z;
#endif 
#endif // end USE_SYMMETRIC
#ifdef USE_CUTOFF
                    }
#endif
#ifdef ENABLE_SHUFFLE
                    SHUFFLE_WARP_DATA
#endif
                    tj = (tj + 1) & (TILE_SIZE - 1);
                }
            }

            // Write results.
            atomicAdd(&forceBuffers[atom1], static_cast<unsigned long long>((long long) (force.x*0x100000000)));
            atomicAdd(&forceBuffers[atom1+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (force.y*0x100000000)));
            atomicAdd(&forceBuffers[atom1+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (force.z*0x100000000)));
#ifdef USE_CUTOFF
            unsigned int atom2 = atomIndices[threadIdx.x];
#else
            unsigned int atom2 = y*TILE_SIZE + tgx;
#endif
            if (atom2 < PADDED_NUM_ATOMS) {
#ifdef ENABLE_SHUFFLE
                atomicAdd(&forceBuffers[atom2], static_cast<unsigned long long>((long long) (shflForce.x*0x100000000)));
                atomicAdd(&forceBuffers[atom2+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (shflForce.y*0x100000000)));
                atomicAdd(&forceBuffers[atom2+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (shflForce.z*0x100000000)));
#else
                atomicAdd(&forceBuffers[atom2], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fx*0x100000000)));
                atomicAdd(&forceBuffers[atom2+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fy*0x100000000)));
                atomicAdd(&forceBuffers[atom2+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fz*0x100000000)));
#endif
            }
        }
        pos++;
        */
    }
    energyBuffer[blockIdx.x*blockDim.x+threadIdx.x] += energy;
}