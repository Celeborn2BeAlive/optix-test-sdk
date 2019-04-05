#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "pbrMetallicRoughness.h"

using namespace optix;

rtDeclareVariable(float3, baseColorFactor, , );

rtDeclareVariable(float, metallicFactor, , );
rtDeclareVariable(float, roughnessFactor, , );

rtDeclareVariable(float, normalScale, , );

rtDeclareVariable(float, occlusionStrength, , );

rtDeclareVariable(float3, emissiveFactor, , );