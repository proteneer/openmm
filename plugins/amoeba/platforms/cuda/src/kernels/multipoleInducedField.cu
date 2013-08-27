#define WARPS_PER_GROUP (THREAD_BLOCK_SIZE/TILE_SIZE)

typedef struct {
    real3 pos;
    real3 field, fieldPolar, inducedDipole, inducedDipolePolar;
#ifdef USE_GK
    real3 fieldS, fieldPolarS, inducedDipoleS, inducedDipolePolarS;
    real bornRadius;
#endif
    float thole, damp;
} AtomData;



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

#ifdef USE_GK
inline __device__ void loadAtomData(AtomData& data, int atom, const real4* __restrict__ posq, const real* __restrict__ inducedDipole,
        const real* __restrict__ inducedDipolePolar, const float2* __restrict__ dampingAndThole, const real* __restrict__ inducedDipoleS,
        const real* __restrict__ inducedDipolePolarS, const real* __restrict__ bornRadii) {
#else
inline __device__ void loadAtomData(AtomData& data, int atom, const real4* __restrict__ posq, const real* __restrict__ inducedDipole,
        const real* __restrict__ inducedDipolePolar, const float2* __restrict__ dampingAndThole) {
#endif
    real4 atomPosq = posq[atom];
    data.pos = make_real3(atomPosq.x, atomPosq.y, atomPosq.z);
    data.inducedDipole.x = inducedDipole[atom*3];
    data.inducedDipole.y = inducedDipole[atom*3+1];
    data.inducedDipole.z = inducedDipole[atom*3+2];
    data.inducedDipolePolar.x = inducedDipolePolar[atom*3];
    data.inducedDipolePolar.y = inducedDipolePolar[atom*3+1];
    data.inducedDipolePolar.z = inducedDipolePolar[atom*3+2];
    float2 temp = dampingAndThole[atom];
    data.damp = temp.x;
    data.thole = temp.y;
#ifdef USE_GK
    data.inducedDipoleS.x = inducedDipoleS[atom*3];
    data.inducedDipoleS.y = inducedDipoleS[atom*3+1];
    data.inducedDipoleS.z = inducedDipoleS[atom*3+2];
    data.inducedDipolePolarS.x = inducedDipolePolarS[atom*3];
    data.inducedDipolePolarS.y = inducedDipolePolarS[atom*3+1];
    data.inducedDipolePolarS.z = inducedDipolePolarS[atom*3+2];
    data.bornRadius = bornRadii[atom];
#endif
}

inline __device__ void zeroAtomData(AtomData& data) {
    data.field = make_real3(0);
    data.fieldPolar = make_real3(0);
#ifdef USE_GK
    data.fieldS = make_real3(0);
    data.fieldPolarS = make_real3(0);
#endif
}

#ifdef USE_EWALD
__device__ void computeOneInteraction(AtomData& atom1, AtomData& atom2, real3 deltaR, bool isSelfInteraction) {
    if (isSelfInteraction)
        return;
    real scale1, scale2;
    real r2 = dot(deltaR, deltaR);
    if (r2 < CUTOFF_SQUARED) {
        real rI = RSQRT(r2);
        real r = RECIP(rI);

        // calculate the error function damping terms

        real ralpha = EWALD_ALPHA*r;
        real bn0 = erfc(ralpha)*rI;
        real alsq2 = 2*EWALD_ALPHA*EWALD_ALPHA;
        real alsq2n = RECIP(SQRT_PI*EWALD_ALPHA);
        real exp2a = EXP(-(ralpha*ralpha));
        alsq2n *= alsq2;
        real bn1 = (bn0+alsq2n*exp2a)*rI*rI;

        alsq2n *= alsq2;
        real bn2 = (3*bn1+alsq2n*exp2a)*rI*rI;

        // compute the error function scaled and unscaled terms

        real scale3 = 1;
        real scale5 = 1;
        real damp = atom1.damp*atom2.damp;
        if (damp != 0) {
            real ratio = (r/damp);
            ratio = ratio*ratio*ratio;
            float pgamma = atom1.thole < atom2.thole ? atom1.thole : atom2.thole;
            damp = -pgamma*ratio;
            if (damp > -50) {
                real expdamp = EXP(damp);
                scale3 = 1 - expdamp;
                scale5 = 1 - expdamp*(1-damp);
            }
        }
        real dsc3 = scale3;
        real dsc5 = scale5;
        real r3 = (r*r2);
        real r5 = (r3*r2);
        real rr3 = (1-dsc3)/r3;
        real rr5 = 3*(1-dsc5)/r5;

        scale1 = rr3 - bn1;
        scale2 = bn2 - rr5;
    }
    else {
        scale1 = 0;
        scale2 = 0;
    }
    real dDotDelta = scale2*dot(deltaR, atom2.inducedDipole);
    atom1.field += scale1*atom2.inducedDipole + dDotDelta*deltaR;
    dDotDelta = scale2*dot(deltaR, atom2.inducedDipolePolar);
    atom1.fieldPolar += scale1*atom2.inducedDipolePolar + dDotDelta*deltaR;
    dDotDelta = scale2*dot(deltaR, atom1.inducedDipole);
    atom2.field += scale1*atom1.inducedDipole + dDotDelta*deltaR;
    dDotDelta = scale2*dot(deltaR, atom1.inducedDipolePolar);
    atom2.fieldPolar += scale1*atom1.inducedDipolePolar + dDotDelta*deltaR;
}
#elif defined USE_GK
__device__ void computeOneInteraction(AtomData& atom1, AtomData& atom2, real3 deltaR, bool isSelfInteraction) {
    real r2 = dot(deltaR, deltaR);
    real r = SQRT(r2);
    if (!isSelfInteraction) {
        real rI = RECIP(r);
        real r2I = rI*rI;
        real rr3 = -rI*r2I;
        real rr5 = -3*rr3*r2I;

        real dampProd = atom1.damp*atom2.damp;
        real ratio = (dampProd != 0 ? r/dampProd : 1);
        float pGamma = (atom1.thole > atom2.thole ? atom2.thole: atom1.thole);
        real damp = ratio*ratio*ratio*pGamma;
        real dampExp = (dampProd != 0 ? EXP(-damp) : 0); 

        rr3 *= 1-dampExp;
        rr5 *= 1-(1+damp)*dampExp;

        real dDotDelta = rr5*dot(deltaR, atom2.inducedDipole);
        atom1.field += rr3*atom2.inducedDipole + dDotDelta*deltaR;
        dDotDelta = rr5*dot(deltaR, atom2.inducedDipolePolar);
        atom1.fieldPolar += rr3*atom2.inducedDipolePolar + dDotDelta*deltaR;
        dDotDelta = rr5*dot(deltaR, atom1.inducedDipole);
        atom2.field += rr3*atom1.inducedDipole + dDotDelta*deltaR;
        dDotDelta = rr5*dot(deltaR, atom1.inducedDipolePolar);
        atom2.fieldPolar += rr3*atom1.inducedDipolePolar + dDotDelta*deltaR;
        dDotDelta = rr5*dot(deltaR, atom2.inducedDipoleS);
        atom1.fieldS += rr3*atom2.inducedDipoleS + dDotDelta*deltaR;
        dDotDelta = rr5*dot(deltaR, atom2.inducedDipolePolarS);
        atom1.fieldPolarS += rr3*atom2.inducedDipolePolarS + dDotDelta*deltaR;
        dDotDelta = rr5*dot(deltaR, atom1.inducedDipoleS);
        atom2.fieldS += rr3*atom1.inducedDipoleS + dDotDelta*deltaR;
        dDotDelta = rr5*dot(deltaR, atom1.inducedDipolePolarS);
        atom2.fieldPolarS += rr3*atom1.inducedDipolePolarS + dDotDelta*deltaR;
    }

    real rb2 = atom1.bornRadius*atom2.bornRadius;
    real expterm = EXP(-r2/(GK_C*rb2));
    real expc = expterm/GK_C; 
    real gf2 = RECIP(r2+rb2*expterm);
    real gf = SQRT(gf2);
    real gf3 = gf2*gf;
    real gf5 = gf3*gf2;
    real a10 = -gf3;
    real expc1 = 1 - expc;
    real a11 = expc1 * 3 * gf5;
    real3 gux = GK_FD*make_real3(a10+deltaR.x*deltaR.x*a11, deltaR.x*deltaR.y*a11, deltaR.x*deltaR.z*a11);
    real3 guy = make_real3(gux.y, GK_FD*(a10+deltaR.y*deltaR.y*a11), GK_FD*deltaR.y*deltaR.z*a11);
    real3 guz = make_real3(gux.z, guy.z, GK_FD*(a10+deltaR.z*deltaR.z*a11));
 
    atom1.fieldS += atom2.inducedDipoleS.x*gux+atom2.inducedDipoleS.y*guy+atom2.inducedDipoleS.z*guz;
    atom2.fieldS += atom1.inducedDipoleS.x*gux+atom1.inducedDipoleS.y*guy+atom1.inducedDipoleS.z*guz;
    atom1.fieldPolarS += atom2.inducedDipolePolarS.x*gux+atom2.inducedDipolePolarS.y*guy+atom2.inducedDipolePolarS.z*guz;
    atom2.fieldPolarS += atom1.inducedDipolePolarS.x*gux+atom1.inducedDipolePolarS.y*guy+atom1.inducedDipolePolarS.z*guz;
}
#else
__device__ void computeOneInteraction(AtomData& atom1, AtomData& atom2, real3 deltaR, bool isSelfInteraction) {
    if (isSelfInteraction)
        return;
    real rI = RSQRT(dot(deltaR, deltaR));
    real r = RECIP(rI);
    real r2I = rI*rI;
    real rr3 = -rI*r2I;
    real rr5 = -3*rr3*r2I;
    real dampProd = atom1.damp*atom2.damp;
    real ratio = (dampProd != 0 ? r/dampProd : 1);
    float pGamma = (atom2.thole > atom1.thole ? atom1.thole: atom2.thole);
    real damp = ratio*ratio*ratio*pGamma;
    real dampExp = (dampProd != 0 ? EXP(-damp) : 0); 
    rr3 *= 1 - dampExp;
    rr5 *= 1 - (1+damp)*dampExp;
    real dDotDelta = rr5*dot(deltaR, atom2.inducedDipole);
    atom1.field += rr3*atom2.inducedDipole + dDotDelta*deltaR;
    dDotDelta = rr5*dot(deltaR, atom2.inducedDipolePolar);
    atom1.fieldPolar += rr3*atom2.inducedDipolePolar + dDotDelta*deltaR;
    dDotDelta = rr5*dot(deltaR, atom1.inducedDipole);
    atom2.field += rr3*atom1.inducedDipole + dDotDelta*deltaR;
    dDotDelta = rr5*dot(deltaR, atom1.inducedDipolePolar);
    atom2.fieldPolar += rr3*atom1.inducedDipolePolar + dDotDelta*deltaR;




}
#endif

/**
 * Build a preconditioner used in conjugate gradient
 * 
 * Construct a new neighborlist with a smaller cutoff distance than the cutoff used in the nonbonded neighborlist. The output is a tensorlist where each
 * interaction is represented by a real6 tensor (minv0 to minv5).
 
 */

/**
 * Initialize conjugate gradient
 * 
 * 1) Compute new direct dipoles from direct field
 * 2) Set induced dipoles equal to direct dipoles
 * K> Compute a new induced field based on induced dipoles
 */

extern "C" __global__ void initializeInducedDipoles() {

}


/*
* Initialize rsd and rsdp using induced field, direct dipoles, and induced dipoles. 
* rsd, rsdp, zrsd, zrsdp are store in dimension major format:
* rsd[0*PADDED_NUM_ATOMS+i] for x coordinates in atom i
* rsd[1*PADDED_NUM_ATOMS+i] for y coordinates in atom i
* this ensures nicely coalesced memory access patterns
*/ 
extern "C" __global__ void initializeCG(real* __restrict__ rsd, real* __restrict__ rsdp, 
    real* __restrict__zrsd, real3* __restrict__ zrsdp, const float* __restrict__ polarizability,
    const real* __restrict__ inducedDipole, const real* __restrict__ inducedDipolePolar,
    const real* __restrict__ directDipole, const real* __restrict__ directDipolePolar,
    unsigned long long* __restrict__ inducedField, unsigned long long* __restrict__ inducedFieldPolar) {
    const float udiag = 2.0f;
    for(int i = threadIdx.x; i < NUM_ATOMS; i += blockDim.x) { 
            real rsdx, rsdy, rsdz, rsdpx, rsdpy, rsdpz;
            // inducedField is stored in dimension major format
            // directDipole and inducedDipole are stored in atom major format
            rsdx = (directDipole[i*3+0] - inducedDipole[i*3+0])/polarizability[i] + inducedField[0*PADDED_NUM_ATOMS+i]/0x100000000;
            rsdy = (directDipole[i*3+1] - inducedDipole[i*3+1])/polarizability[i] + inducedField[1*PADDED_NUM_ATOMS+i]/0x100000000;
            rsdz = (directDipole[i*3+2] - inducedDipole[i*3+2])/polarizability[i] + inducedField[2*PADDED_NUM_ATOMS+i]/0x100000000;
            rsdpx = (directDipolePolar[i*3+0] - inducedDipolePolar[i*3+0])/polarizability[i] + inducedFieldPolar[0*PADDED_NUM_ATOMS+i]/0x100000000;
            rsdpy = (directDipolePolar[i*3+1] - inducedDipolePolar[i*3+1])/polarizability[i] + inducedFieldPolar[1*PADDED_NUM_ATOMS+i]/0x100000000;
            rsdpz = (directDipolePolar[i*3+2] - inducedDipolePolar[i*3+2])/polarizability[i] + inducedFieldPolar[2*PADDED_NUM_ATOMS+i]/0x100000000;

            rsd[0*PADDED_NUM_ATOMS+i] = rsdx;
            rsd[1*PADDED_NUM_ATOMS+i] = rsdy;
            rsd[2*PADDED_NUM_ATOMS+i] = rsdz;
            rsdp[0*PADDED_NUM_ATOMS+i] = rsdx;
            rsdp[1*PADDED_NUM_ATOMS+i] = rsdy;
            rsdp[2*PADDED_NUM_ATOMS+i] = rsdz;

            zrsd[0*PADDED_NUM_ATOMS+i] = udiag * polarizability[i] * rsdx;
            zrsd[1*PADDED_NUM_ATOMS+i] = udiag * polarizability[i] * rsdy;
            zrsd[2*PADDED_NUM_ATOMS+i] = udiag * polarizability[i] * rsdz;
            zrsdp[0*PADDED_NUM_ATOMS+i] = udiag * polarizability[i] * rsdpx;
            zrsdp[1*PADDED_NUM_ATOMS+i] = udiag * polarizability[i] * rsdpy;
            zrsdp[2*PADDED_NUM_ATOMS+i] = udiag * polarizability[i] * rsdpz;
    }
}

// We use the neighborlist constructed by findInteractingBlocks to apply the pre-conditioner on residuals
// Three types of interactions:
// 1) On diagonal tiles
// --horizontal accumulation
// 2) Neighborlist tiles
// --staggered accumulation
extern "C" __global__ void applyPreconditioner(
    unsigned int numTileIndices, const int* __restrict__ interactingTiles, const unsigned int* __restrict__ interactingAtoms, 
    //const tileflags* __restrict__ exclusions, const ushort2* __restrict__ exclusionTiles,
    const float* __restrict__ polarizability, const float2* __restrict__ dampingAndThole,
    const real4* posq, real4 periodicBoxSize, real4 invPeriodicBoxSize,
    const real* __restrict__ rsd, const real* __restrict__ rsdp,
    real* __restrict__ zrsd, real* __restrict__ zrsdp) {

    // initialize new residuals to zero
    for(unsigned int gid = threadIdx.x+blockDim.x*blockIdx.x; gid < PADDED_NUM_ATOMS; gid += blockDim.x*gridDim.x) {
        zrsd[gid] = 0;
        zrsdp[gid] = 0;
    }

    const unsigned int totalWarps = (blockDim.x*gridDim.x)/TILE_SIZE;
    const unsigned int warp = (blockIdx.x*blockDim.x+threadIdx.x)/TILE_SIZE; // global warpIndex
    const unsigned int tgx = threadIdx.x & (TILE_SIZE-1); // index within the warp
    const unsigned int tbx = threadIdx.x - tgx;           // block warpIndex

    // used shared memory if the device cannot shuffle
    
    const real udiag = 2.0;
    
    // first look
    const unsigned int lastWarp = (NUM_ATOMS+TILE_SIZE-1)/TILE_SIZE;

    // first loop: diagonal tiles
    for(int pos = warp; pos < lastWarp; pos += totalWarps) {
        // suppose shuffles are available
        const unsigned int atom1 = pos*TILE_SIZE + tgx;

        // shuffled
        real3 posq1 = trimTo3(posq[atom1]);

        real3 rsd1, rsdp1;
        rsd1.x = rsd[0*PADDED_NUM_ATOMS+atom1];
        rsd1.y = rsd[1*PADDED_NUM_ATOMS+atom1];
        rsd1.z = rsd[2*PADDED_NUM_ATOMS+atom1];
        rsdp1.x = rsdp[0*PADDED_NUM_ATOMS+atom1];
        rsdp1.y = rsdp[1*PADDED_NUM_ATOMS+atom1];
        rsdp1.z = rsdp[2*PADDED_NUM_ATOMS+atom1];
        
        // used for accumulation
        real3 zrsd1, zrsdp1;

        zrsd1 = make_real3(0, 0, 0);
        zrsdp1 = make_real3(0, 0, 0);

        const float pdi1 = dampingAndThole[atom1].x;
        const float pti1 = dampingAndThole[atom1].y;
        const float poli1 = polarizability[atom1];

        for(unsigned int j=0; (j < TILE_SIZE) && (pos + j < NUM_ATOMS) ; j++) {
            const unsigned int atom2 = pos+j;

            // all threads in a warp must partake in broadcast shuffle
            real3 posq2;
            posq2.x = real_shfl(posq1.x, j);
            posq2.y = real_shfl(posq1.y, j);
            posq2.z = real_shfl(posq1.z, j);
            real3 rsd2, rsdp2;
            rsd2.x = real_shfl(rsd1.x, j);
            rsd2.y = real_shfl(rsd1.y, j);
            rsd2.z = real_shfl(rsd1.z, j);

            const float pdi2 = real_shfl(pdi1, j);
            const float pti2 = real_shfl(pti1, j);
            const float poli2 = real_shfl(poli1, j);
            
            // do a check after shuffling to see if atom1 < NUM_ATOMS && atom2 < NUM_ATOMS

            const real xr = posq1.x - posq2.x;
            const real yr = posq1.y - posq2.y;
            const real zr = posq1.z - posq2.z;
            const real r2 = xr*xr + yr*yr + zr*zr;

            if(atom1 == atom2) {
                zrsd1.x += udiag * polarizability[i] * rsd[i].x;
                zrsd1.y += udiag * polarizability[i] * rsd[i].y;
                zrsd1.z += udiag * polarizability[i] * rsd[i].z;
            } else if(r2 < CUTOFF2_SQUARED) {
                // load parameters for CG
                const real r = SQRT(r2);
                const real scale3 = 1.0;
                const real scale5 = 1.0;
                const real damp = pdi1*pdi2;
                if(damp != 0) {
                    const real pgamma = min(pti1, pti2);
                    damp = -pgamma*pow((r/damp),3);
                    if(damp > -50.0) { // not really needed
                        const real expdamp = exp(damp);
                        scale3 = scale3 * (1.0-expdamp);
                        scale5 = scale5 * (1.0-expdamp*(1.0-expdamp));
                    }
                }
                const real polik = poli1*poli2;
                const real rr3 = scale3*polik/(r*r2);
                const real rr5 = 3.0f*scale5*polik/(r*r2*r2);
      
                // load minv
                real m0,m1,m2,m3,m4,m5;
                const real m0 = rr5*xr*xr-rr3;
                const real m1 = rr5*xr*yr;
                const real m2 = rr5*xr*zr;
                const real m3 = rr5*yr*yr-rr3;
                const real m4 = rr5*yr*zr;
                const real m5 = rr5*zr*zr-rr3;
              
                zrsd1.x += m0*rsd2.x+m1*rsd2.y+m2*rsd2.z;
                zrsd1.y += m1*rsd2.x+m3*rsd2.y+m4*rsd2.z;
                zrsd1.z += m2*rsd2.x+m4*rsd2.y+m5*rsd2.z;
                zrsdp1.x += m0*rsdp2.x+m1*rsdp2.y+m2*rsdp2.z;
                zrsdp1.y += m1*rsdp2.x+m3*rsdp2.y+m4*rsdp2.z;
                zrsdp1.z += m2*rsdp2.x+m4*rsdp2.y+m5*rsdp2.z;
            }
        }
        if(atom1 < NUM_ATOMS) {
            atomicAdd(&zrsd[0*PADDED_NUM_ATOMS+atom1], zrsd1.x);
            atomicAdd(&zrsd[1*PADDED_NUM_ATOMS+atom1], zrsd1.y);
            atomicAdd(&zrsd[2*PADDED_NUM_ATOMS+atom1], zrsd1.z);
            atomicAdd(&zrsdp[0*PADDED_NUM_ATOMS+atom1], zrsdp1.x);
            atomicAdd(&zrsdp[1*PADDED_NUM_ATOMS+atom1], zrsdp1.y);
            atomicAdd(&zrsdp[2*PADDED_NUM_ATOMS+atom1], zrsdp1.z);
        }
    }

    // second loop: neighborlist tiles

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

    // compute off diagonal neighborlist tiles
    while(pos < end) {
        unsigned int x = tiles[pos];
        unsigned int j = interactingAtoms[pos*TILE_SIZE+tgx];



        pos++;
    }

}

/**
 * Compute the mutual induced field.
 */
extern "C" __global__ void computeInducedField(
        unsigned long long* __restrict__ field, unsigned long long* __restrict__ fieldPolar, const real4* __restrict__ posq, const ushort2* __restrict__ exclusionTiles, 
        const real* __restrict__ inducedDipole, const real* __restrict__ inducedDipolePolar, unsigned int startTileIndex, unsigned int numTileIndices,
#ifdef USE_CUTOFF
        const int* __restrict__ tiles, const unsigned int* __restrict__ interactionCount, real4 periodicBoxSize, real4 invPeriodicBoxSize, unsigned int maxTiles, const real4* __restrict__ blockCenter, const unsigned int* __restrict__ interactingAtoms,
#elif defined USE_GK
        unsigned long long* __restrict__ fieldS, unsigned long long* __restrict__ fieldPolarS, const real* __restrict__ inducedDipoleS,
        const real* __restrict__ inducedDipolePolarS, const real* __restrict__ bornRadii,
#endif
        const float2* __restrict__ dampingAndThole) {
    const unsigned int totalWarps = (blockDim.x*gridDim.x)/TILE_SIZE;
    const unsigned int warp = (blockIdx.x*blockDim.x+threadIdx.x)/TILE_SIZE;
    const unsigned int tgx = threadIdx.x & (TILE_SIZE-1);
    const unsigned int tbx = threadIdx.x - tgx;
    __shared__ AtomData localData[THREAD_BLOCK_SIZE];

    // First loop: process tiles that contain exclusions.
    
    const unsigned int firstExclusionTile = FIRST_EXCLUSION_TILE+warp*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    const unsigned int lastExclusionTile = FIRST_EXCLUSION_TILE+(warp+1)*(LAST_EXCLUSION_TILE-FIRST_EXCLUSION_TILE)/totalWarps;
    for (int pos = firstExclusionTile; pos < lastExclusionTile; pos++) {
        const ushort2 tileIndices = exclusionTiles[pos];
        const unsigned int x = tileIndices.x;
        const unsigned int y = tileIndices.y;
        AtomData data;
        zeroAtomData(data);
        unsigned int atom1 = x*TILE_SIZE + tgx;
#ifdef USE_GK
        loadAtomData(data, atom1, posq, inducedDipole, inducedDipolePolar, dampingAndThole, inducedDipoleS, inducedDipolePolarS, bornRadii);
#else
        loadAtomData(data, atom1, posq, inducedDipole, inducedDipolePolar, dampingAndThole);
#endif
        if (x == y) {
            // This tile is on the diagonal.
            localData[threadIdx.x].pos = data.pos;
            localData[threadIdx.x].inducedDipole = data.inducedDipole;
            localData[threadIdx.x].inducedDipolePolar = data.inducedDipolePolar;
            localData[threadIdx.x].thole = data.thole;
            localData[threadIdx.x].damp = data.damp;
#ifdef USE_GK
            localData[threadIdx.x].inducedDipoleS = data.inducedDipoleS;
            localData[threadIdx.x].inducedDipolePolarS = data.inducedDipolePolarS;
            localData[threadIdx.x].bornRadius = data.bornRadius;
#endif
            for (unsigned int j = 0; j < TILE_SIZE; j++) {
                real3 delta = localData[tbx+j].pos-data.pos;
#ifdef USE_PERIODIC
                delta.x -= floor(delta.x*invPeriodicBoxSize.x+0.5f)*periodicBoxSize.x;
                delta.y -= floor(delta.y*invPeriodicBoxSize.y+0.5f)*periodicBoxSize.y;
                delta.z -= floor(delta.z*invPeriodicBoxSize.z+0.5f)*periodicBoxSize.z;
#endif
                int atom2 = y*TILE_SIZE+j;
                if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS)
                    computeOneInteraction(data, localData[tbx+j], delta, atom1 == atom2);
            }
        }
        else {
            // This is an off-diagonal tile.

#ifdef USE_GK
            loadAtomData(localData[threadIdx.x], y*TILE_SIZE+tgx, posq, inducedDipole, inducedDipolePolar, dampingAndThole, inducedDipoleS, inducedDipolePolarS, bornRadii);
#else
            loadAtomData(localData[threadIdx.x], y*TILE_SIZE+tgx, posq, inducedDipole, inducedDipolePolar, dampingAndThole);
#endif
            zeroAtomData(localData[threadIdx.x]);
            unsigned int tj = tgx;
            for (unsigned int j = 0; j < TILE_SIZE; j++) {
                real3 delta = localData[tbx+tj].pos-data.pos;
#ifdef USE_PERIODIC
                delta.x -= floor(delta.x*invPeriodicBoxSize.x+0.5f)*periodicBoxSize.x;
                delta.y -= floor(delta.y*invPeriodicBoxSize.y+0.5f)*periodicBoxSize.y;
                delta.z -= floor(delta.z*invPeriodicBoxSize.z+0.5f)*periodicBoxSize.z;
#endif
                int atom2 = y*TILE_SIZE+j;
                if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS)
                    computeOneInteraction(data, localData[tbx+tj], delta, false);
                tj = (tj + 1) & (TILE_SIZE - 1);
            }
        }

        // Write results.

        unsigned int offset = x*TILE_SIZE + tgx;
        atomicAdd(&field[offset], static_cast<unsigned long long>((long long) (data.field.x*0x100000000)));
        atomicAdd(&field[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.field.y*0x100000000)));
        atomicAdd(&field[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.field.z*0x100000000)));
        atomicAdd(&fieldPolar[offset], static_cast<unsigned long long>((long long) (data.fieldPolar.x*0x100000000)));
        atomicAdd(&fieldPolar[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.fieldPolar.y*0x100000000)));
        atomicAdd(&fieldPolar[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.fieldPolar.z*0x100000000)));
#ifdef USE_GK
        atomicAdd(&fieldS[offset], static_cast<unsigned long long>((long long) (data.fieldS.x*0x100000000)));
        atomicAdd(&fieldS[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.fieldS.y*0x100000000)));
        atomicAdd(&fieldS[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.fieldS.z*0x100000000)));
        atomicAdd(&fieldPolarS[offset], static_cast<unsigned long long>((long long) (data.fieldPolarS.x*0x100000000)));
        atomicAdd(&fieldPolarS[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.fieldPolarS.y*0x100000000)));
        atomicAdd(&fieldPolarS[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.fieldPolarS.z*0x100000000)));
#endif
        if (x != y) {
            offset = y*TILE_SIZE + tgx;
            atomicAdd(&field[offset], static_cast<unsigned long long>((long long) (localData[threadIdx.x].field.x*0x100000000)));
            atomicAdd(&field[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].field.y*0x100000000)));
            atomicAdd(&field[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].field.z*0x100000000)));
            atomicAdd(&fieldPolar[offset], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fieldPolar.x*0x100000000)));
            atomicAdd(&fieldPolar[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fieldPolar.y*0x100000000)));
            atomicAdd(&fieldPolar[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fieldPolar.z*0x100000000)));
#ifdef USE_GK
            atomicAdd(&fieldS[offset], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fieldS.x*0x100000000)));
            atomicAdd(&fieldS[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fieldS.y*0x100000000)));
            atomicAdd(&fieldS[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fieldS.z*0x100000000)));
            atomicAdd(&fieldPolarS[offset], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fieldPolarS.x*0x100000000)));
            atomicAdd(&fieldPolarS[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fieldPolarS.y*0x100000000)));
            atomicAdd(&fieldPolarS[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fieldPolarS.z*0x100000000)));
#endif
        }
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
    __shared__ int atomIndices[THREAD_BLOCK_SIZE];
    __shared__ volatile int skipTiles[THREAD_BLOCK_SIZE];
    skipTiles[threadIdx.x] = -1;
    
    while (pos < end) {
        bool includeTile = true;

        // Extract the coordinates of this tile.
        
        unsigned int x, y;
#ifdef USE_CUTOFF
        if (numTiles <= maxTiles)
            x = tiles[pos];
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
            unsigned int atom1 = x*TILE_SIZE + tgx;

            // Load atom data for this tile.

            AtomData data;
            zeroAtomData(data);
#ifdef USE_GK
            loadAtomData(data, atom1, posq, inducedDipole, inducedDipolePolar, dampingAndThole, inducedDipoleS, inducedDipolePolarS, bornRadii);
#else
            loadAtomData(data, atom1, posq, inducedDipole, inducedDipolePolar, dampingAndThole);
#endif
#ifdef USE_CUTOFF
            unsigned int j = (numTiles <= maxTiles ? interactingAtoms[pos*TILE_SIZE+tgx] : y*TILE_SIZE + tgx);
#else
            unsigned int j = y*TILE_SIZE + tgx;
#endif
            atomIndices[threadIdx.x] = j;
#ifdef USE_GK
            loadAtomData(localData[threadIdx.x], j, posq, inducedDipole, inducedDipolePolar, dampingAndThole, inducedDipoleS, inducedDipolePolarS, bornRadii);
#else
            loadAtomData(localData[threadIdx.x], j, posq, inducedDipole, inducedDipolePolar, dampingAndThole);
#endif
            zeroAtomData(localData[threadIdx.x]);

            // Compute the full set of interactions in this tile.

            unsigned int tj = tgx;
            for (j = 0; j < TILE_SIZE; j++) {
                real3 delta = localData[tbx+tj].pos-data.pos;
#ifdef USE_PERIODIC
                delta.x -= floor(delta.x*invPeriodicBoxSize.x+0.5f)*periodicBoxSize.x;
                delta.y -= floor(delta.y*invPeriodicBoxSize.y+0.5f)*periodicBoxSize.y;
                delta.z -= floor(delta.z*invPeriodicBoxSize.z+0.5f)*periodicBoxSize.z;
#endif
                int atom2 = atomIndices[tbx+tj];
                if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS)
                    computeOneInteraction(data, localData[tbx+tj], delta, false);
                tj = (tj + 1) & (TILE_SIZE - 1);
            }

            // Write results.

            unsigned int offset = x*TILE_SIZE + tgx;
            atomicAdd(&field[offset], static_cast<unsigned long long>((long long) (data.field.x*0x100000000)));
            atomicAdd(&field[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.field.y*0x100000000)));
            atomicAdd(&field[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.field.z*0x100000000)));
            atomicAdd(&fieldPolar[offset], static_cast<unsigned long long>((long long) (data.fieldPolar.x*0x100000000)));
            atomicAdd(&fieldPolar[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.fieldPolar.y*0x100000000)));
            atomicAdd(&fieldPolar[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.fieldPolar.z*0x100000000)));
#ifdef USE_GK
            atomicAdd(&fieldS[offset], static_cast<unsigned long long>((long long) (data.fieldS.x*0x100000000)));
            atomicAdd(&fieldS[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.fieldS.y*0x100000000)));
            atomicAdd(&fieldS[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.fieldS.z*0x100000000)));
            atomicAdd(&fieldPolarS[offset], static_cast<unsigned long long>((long long) (data.fieldPolarS.x*0x100000000)));
            atomicAdd(&fieldPolarS[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.fieldPolarS.y*0x100000000)));
            atomicAdd(&fieldPolarS[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (data.fieldPolarS.z*0x100000000)));
#endif
#ifdef USE_CUTOFF
            offset = atomIndices[threadIdx.x];
#else
            offset = y*TILE_SIZE + tgx;
#endif
            atomicAdd(&field[offset], static_cast<unsigned long long>((long long) (localData[threadIdx.x].field.x*0x100000000)));
            atomicAdd(&field[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].field.y*0x100000000)));
            atomicAdd(&field[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].field.z*0x100000000)));
            atomicAdd(&fieldPolar[offset], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fieldPolar.x*0x100000000)));
            atomicAdd(&fieldPolar[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fieldPolar.y*0x100000000)));
            atomicAdd(&fieldPolar[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fieldPolar.z*0x100000000)));
#ifdef USE_GK
            atomicAdd(&fieldS[offset], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fieldS.x*0x100000000)));
            atomicAdd(&fieldS[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fieldS.y*0x100000000)));
            atomicAdd(&fieldS[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fieldS.z*0x100000000)));
            atomicAdd(&fieldPolarS[offset], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fieldPolarS.x*0x100000000)));
            atomicAdd(&fieldPolarS[offset+PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fieldPolarS.y*0x100000000)));
            atomicAdd(&fieldPolarS[offset+2*PADDED_NUM_ATOMS], static_cast<unsigned long long>((long long) (localData[threadIdx.x].fieldPolarS.z*0x100000000)));
#endif
        }
        pos++;
    }
}

extern "C" __global__ void updateInducedFieldBySOR(const long long* __restrict__ fixedField, const long long* __restrict__ fixedFieldPolar,
        const long long* __restrict__ fixedFieldS, const long long* __restrict__ inducedField, const long long* __restrict__ inducedFieldPolar,
        real* __restrict__ inducedDipole, real* __restrict__ inducedDipolePolar, const float* __restrict__ polarizability, float2* __restrict__ errors) {
    extern __shared__ real2 buffer[];
    const float polarSOR = 0.55f;
#ifdef USE_EWALD
    const real ewaldScale = (4/(real) 3)*(EWALD_ALPHA*EWALD_ALPHA*EWALD_ALPHA)/SQRT_PI;
#else
    const real ewaldScale = 0;
#endif
    const real fieldScale = 1/(real) 0x100000000;
    real sumErrors = 0;
    real sumPolarErrors = 0;
    for (int atom = blockIdx.x*blockDim.x + threadIdx.x; atom < NUM_ATOMS; atom += blockDim.x*gridDim.x) {
        real scale = polarizability[atom];
        for (int component = 0; component < 3; component++) {
            int dipoleIndex = 3*atom+component;
            int fieldIndex = atom+component*PADDED_NUM_ATOMS;
            real previousDipole = inducedDipole[dipoleIndex];
            real previousDipolePolar = inducedDipolePolar[dipoleIndex];
            long long fixedS = (fixedFieldS == NULL ? (long long) 0 : fixedFieldS[fieldIndex]);
            real newDipole = scale*((fixedField[fieldIndex]+fixedS+inducedField[fieldIndex])*fieldScale+ewaldScale*previousDipole);
            real newDipolePolar = scale*((fixedFieldPolar[fieldIndex]+fixedS+inducedFieldPolar[fieldIndex])*fieldScale+ewaldScale*previousDipolePolar);
            newDipole = previousDipole + polarSOR*(newDipole-previousDipole);
            newDipolePolar = previousDipolePolar + polarSOR*(newDipolePolar-previousDipolePolar);
            inducedDipole[dipoleIndex] = newDipole;
            inducedDipolePolar[dipoleIndex] = newDipolePolar;
            sumErrors += (newDipole-previousDipole)*(newDipole-previousDipole);
            sumPolarErrors += (newDipolePolar-previousDipolePolar)*(newDipolePolar-previousDipolePolar);
        }
    }
    
    // Sum the errors over threads and store the total for this block.
    
    buffer[threadIdx.x] = make_real2(sumErrors, sumPolarErrors);
    __syncthreads();
    for (int offset = 1; offset < blockDim.x; offset *= 2) {   
        if (threadIdx.x+offset < blockDim.x && (threadIdx.x&(2*offset-1)) == 0) {
            buffer[threadIdx.x].x += buffer[threadIdx.x+offset].x;
            buffer[threadIdx.x].y += buffer[threadIdx.x+offset].y;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        errors[blockIdx.x] = make_float2((float) buffer[0].x, (float) buffer[0].y);
}