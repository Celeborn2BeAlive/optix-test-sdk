#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "pbrMetallicRoughness.h"

using namespace optix;

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float3, shading_tangent, attribute shading_tangent, );
rtDeclareVariable(float3, shading_bitangent, attribute shading_bitangent, );
rtDeclareVariable(float3, texcoord, attribute texcoord, );

rtDeclareVariable(float3, baseColorFactor, , );
rtTextureSampler<uchar4, 2, cudaReadModeNormalizedFloat> baseColorTexture;

rtDeclareVariable(float, metallicFactor, , );
rtDeclareVariable(float, roughnessFactor, , );
rtTextureSampler<uchar4, 2, cudaReadModeNormalizedFloat> metallicRoughnessTexture;

rtDeclareVariable(float, normalScale, , );
rtTextureSampler<uchar4, 2, cudaReadModeNormalizedFloat> normalTexture;

rtDeclareVariable(float, occlusionStrength, , );
rtTextureSampler<uchar4, 2, cudaReadModeNormalizedFloat> occlusionTexture;

rtDeclareVariable(float3, emissiveFactor, , );
rtTextureSampler<uchar4, 2, cudaReadModeNormalizedFloat> emissiveTexture;

RT_PROGRAM void any_hit_shadow()
{
    pbrMetallicRoughnessShadowed();
}

RT_PROGRAM void closest_hit_radiance()
{
    float3 tangent_space_normal = normalScale * (make_float3(tex2D(normalTexture, texcoord.x, texcoord.y)) * 2.f - make_float3(1.f));
    float3 local_normal = tangent_space_normal.x * shading_tangent + tangent_space_normal.y * shading_bitangent + tangent_space_normal.z * shading_normal;

    float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, local_normal));
    float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));

    float3 world_shading_tangent = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_tangent));
    float3 world_shading_bitangent = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_bitangent));

    float3 ffnormal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);

    const float3 baseColor = baseColorFactor * make_float3(tex2D(baseColorTexture, texcoord.x, texcoord.y));
    const float3 nothingMetallicRoughness = make_float3(tex2D(baseColorTexture, texcoord.x, texcoord.y));

    const float roughness = roughnessFactor * nothingMetallicRoughness.y;
    const float metallic = metallicFactor * nothingMetallicRoughness.z;

    const float3 emission = emissiveFactor * make_float3(tex2D(emissiveTexture, texcoord.x, texcoord.y));

    pbrMetallicRoughnessShade(baseColor, metallic, roughness, emission, world_shading_tangent, world_shading_bitangent, ffnormal);
}