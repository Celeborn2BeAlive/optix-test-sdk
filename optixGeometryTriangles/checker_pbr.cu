/* 
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include "pbrMetallicRoughness.h" 

using namespace optix;

rtDeclareVariable(float3,       baseColor1, , );
rtDeclareVariable(float3,       baseColor2, , );
rtDeclareVariable(float3,       emission1, , );
rtDeclareVariable(float3,       emission2, , );
rtDeclareVariable(float,       metallic1, , );
rtDeclareVariable(float, metallic2, , );
rtDeclareVariable(float,       roughness1, , );
rtDeclareVariable(float,       roughness2, , );
rtDeclareVariable(float3,       inv_checker_size, , );  // Inverse checker height, width and depth in texture space

rtDeclareVariable(float3, texcoord, attribute texcoord, ); 
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 


RT_PROGRAM void any_hit_shadow()
{
    pbrMetallicRoughnessShadowed();
}

// Building an orthonormal basis, Revisited [Duff et al. 2017]
void branchlessONB(const float3 & n, float3 & b1, float3 & b2)
{
    float sign = copysignf(1.0f, n.z);
    const float a = -1.f / (sign + n.z);
    const float b = n.x * n.y * a;
    b1 = make_float3(1.f + sign * n.x * n.x * a, sign * b, -sign * n.x);
    b2 = make_float3(b, sign + n.y * n.y * a, -n.y);
}

RT_PROGRAM void closest_hit_radiance()
{
    float3 baseColor, emission;
    float metallic, roughness;

  float3 t  = texcoord * inv_checker_size;
  t.x = floorf(t.x);
  t.y = floorf(t.y);
  t.z = floorf(t.z);

  int which_check = ( static_cast<int>( t.x ) +
                      static_cast<int>( t.y ) +
                      static_cast<int>( t.z ) ) & 1;

  if ( !which_check ) {
      baseColor = baseColor1; emission = emission1; metallic = metallic1; roughness = roughness1;
  } else {
      baseColor = baseColor2; emission = emission2; metallic = metallic2; roughness = roughness2;
  }

  float3 world_shading_normal   = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
  float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
  float3 ffnormal  = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

  float3 tangent, bitangent;
  branchlessONB(ffnormal, tangent, bitangent);

  //prd.result = make_float3(1, 0, 0);

  pbrMetallicRoughnessShade(baseColor, metallic, roughness, emission, tangent, bitangent, ffnormal);
}
