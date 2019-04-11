#pragma once

#include <optix_world.h>
#include "common.h"
#include "helpers.h"
#include "random.h"

struct PerRayData_radiance
{
    float3 result;
    float importance;
    int depth;
    unsigned int seed;
};

struct PerRayData_shadow
{
    float3 attenuation;
};

rtDeclareVariable(int, max_depth, , );
rtBuffer<BasicLight>                 lights;
rtDeclareVariable(float3, ambient_light_color, , );
rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(rtObject, top_shadower, , );
rtDeclareVariable(int, use_substance_fresnel, , );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_radiance, prd, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );

static __device__ void pbrMetallicRoughnessShadowed()
{
    // this material is opaque, so it fully attenuates all shadow rays
    prd_shadow.attenuation = optix::make_float3(0.0f);
    rtTerminateRay();
}

//// CODE FROM SUBSTANCE ////

float normal_distrib(
    float ndh,
    float Roughness)
{
    // use GGX / Trowbridge-Reitz, same as Disney and Unreal 4
    // cf http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf p3
    float alpha = Roughness * Roughness;
    float tmp = alpha / max(1e-8, (ndh*ndh*(alpha*alpha - 1.0) + 1.0));
    return tmp * tmp * M_1_PIf;
}

float3 fresnel(
    float vdh,
    float3 F0)
{
    // Schlick with Spherical Gaussian approximation
    // cf http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf p3
    float sphg = pow(2.0, (-5.55473*vdh - 6.98316) * vdh);
    return F0 + (make_float3(1.0, 1.0, 1.0) - F0) * sphg;
}

float G1(
    float ndw, // w is either Ln or Vn
    float k)
{
    // One generic factor of the geometry function divided by ndw
    // NB : We should have k > 0
    return 1.0 / (ndw*(1.0 - k) + k);
}

float visibility(
    float ndl,
    float ndv,
    float Roughness)
{
    // Schlick with Smith-like choice of k
    // cf http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf p3
    // visibility is a Cook-Torrance geometry function divided by (n.l)*(n.v)
    float k = fmaxf(Roughness * Roughness * 0.5, 1e-5);
    return G1(ndl, k)*G1(ndv, k);
}

float3 cook_torrance_contrib(
    float vdh,
    float ndh,
    float ndl,
    float ndv,
    float3 Ks,
    float Roughness)
{
    // This is the contribution when using importance sampling with the GGX based
    // sample distribution. This means ct_contrib = ct_brdf / ggx_probability
    // #c2ba note: ndl is the cosTheta of the integral, it has nothing to do with the brdf.
    // #c2ba npte: if you multiply with ggx_probability you obtain the brdf, which is F * D * vis / 4 (D and 4 are contained in the pdf, the other factors cancel)
    return fresnel(vdh, Ks) * (visibility(ndl, ndv, Roughness) * vdh * ndl / ndh);
}

float3 importanceSampleGGX(float2 Xi, float3 T, float3 B, float3 N, float roughness)
{
    float a = roughness*roughness;
    float cosT = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0)*Xi.y));
    float sinT = sqrt(1.0 - cosT*cosT);
    float phi = 2.f * M_PIf * Xi.x;
    return
        T * (sinT*cos(phi)) +
        B * (sinT*sin(phi)) +
        N *  cosT;
}

float probabilityGGX(float ndh, float vdh, float Roughness)
{
    return normal_distrib(ndh, Roughness) * ndh / (4.0*vdh);
}

// Horizon fading trick from http ://marmosetco.tumblr.com/post/81245981087
float horizonFading(float ndl, float horizonFade)
{
    float horiz = optix::clamp(1.f + horizonFade * ndl, 0.0f, 1.0f);
    return horiz * horiz;
}

///// MY CODE /////

// pdf is cosTheta / pi
float3 sampleCosineLobe(float2 Xi, float3 T, float3 B, float3 N, float & pdf)
{
    const float phi = 2.f * M_PIf * Xi.x;
    const float cosT = Xi.y;
    const float sinT = sqrtf(1 - cosT * cosT);
    pdf = cosT * M_1_PIf;
    return
        T * (sinT*cos(phi)) +
        B * (sinT*sin(phi)) +
        N *  cosT;
}

//// CODE FROM GLTF SPEC ////


float3 computeLightAttenuation(const float3 hit_point, const float3 Ldir)
{
    PerRayData_shadow shadow_prd;
    shadow_prd.attenuation = make_float3(1.0f);
    optix::Ray shadow_ray = optix::make_Ray(hit_point, Ldir, SHADOW_RAY_TYPE, scene_epsilon, RT_DEFAULT_MAX);
    rtTrace(top_shadower, shadow_ray, shadow_prd);
    return shadow_prd.attenuation;
}

static
__device__ void pbrMetallicRoughnessShade(
    float3 baseColor,
    float metallic,
    float roughness,
    float3 emission,
    float3 tangent,
    float3 bitangent,
    float3 p_normal)
{
    const float3 dielectricSpecular = make_float3(0.04, 0.04, 0.04);
    const float3 black = make_float3(0, 0, 0);

    const float3 hit_point = ray.origin + t_hit * ray.direction;

    const float3 V = -ray.direction;
    const float NdotV = optix::clamp(optix::dot(p_normal, V), 0.f, 1.f);
    //const float NdotV2 = NdotV * NdotV;
    const float alpha = roughness * roughness;
    const float alpha2 = alpha * alpha;
    const float oneMinusAlpha2 = 1 - alpha2;

    const float3 c_diff = optix::lerp(baseColor * (1 - dielectricSpecular.x), black, metallic);
    const float3 specularColor = optix::lerp(dielectricSpecular, baseColor, metallic);
    const float3 diffuse = c_diff * M_1_PIf;

    float3 result = emission;

    // Directional lights
    unsigned int num_lights = lights.size();
    for (int i = 0; i < num_lights; ++i) 
    {
        const BasicLight light = lights[i];
        const float3 L = light.dir;
        float NdotL = optix::clamp(optix::dot(p_normal, L), 0.f, 1.f);

        if (NdotL == 0.f)
            continue;

        float3 light_attenuation = make_float3(1.f);
        if (light.casts_shadow) 
            light_attenuation = computeLightAttenuation(hit_point, L);

        if (fmaxf(light_attenuation) == 0.f)
            continue;

        // If not completely shadowed, light the hit point
        float3 Lc = light.color * light_attenuation;

        const float3 H = optix::normalize(L + V);
        const float NdotH = optix::dot(p_normal, H);
        const float NdotH2 = NdotH * NdotH;

        const float VdotH = optix::dot(V, H);

        const float baseSlickFactor = optix::clamp(1.f - VdotH, 0.f , 1.f);
        float slickFactor = baseSlickFactor * baseSlickFactor; // ^2
        slickFactor *= slickFactor; // ^4
        slickFactor *= baseSlickFactor; // ^5

        // Smith Joint GGX
        //const float vis = 0.5f / (NdotL * sqrt(NdotV2 * oneMinusAlpha2 + alpha2) + NdotV * sqrt(NdotL2 * oneMinusAlpha2 + alpha2)); // glTF specification version
        const float vis = 0.25f * G1(NdotL, alpha2) * G1(NdotV, alpha2); // substance version

        const float Ddenom = (NdotH2 * (-oneMinusAlpha2) + 1);
        const float D = Ddenom > 0.f ? alpha2 * M_1_PIf / (Ddenom * Ddenom) : 0.f; // Trowbridge-Reitz

        // const float3 F = specularColor + (1 - specularColor) * slickFactor; // Fresnel Schlick
        const float3 F = fresnel(VdotH, specularColor);

        const float3 f_specular = (F * vis * D);
        const float3 f_diffuse = optix::clamp(1 - F, 0.f, 1.f) * diffuse;

        const float horizonFade = 1.3;
        float fade = horizonFading(optix::dot(p_normal, L), horizonFade);

        const float3 fresnel_factor = use_substance_fresnel ? fresnel(VdotH, specularColor) : F;

        //result += (f_diffuse + f_specular) * Lc * NdotL;
        //result += fresnel(VdotH, specularColor) * (visibility(NdotL, NdotV, roughness) * VdotH * NdotL / NdotH);

        // Substance style:
        result += f_diffuse * NdotL * Lc;
        result += cook_torrance_contrib(VdotH, NdotH, NdotL, NdotV, specularColor, roughness) * probabilityGGX(NdotH, VdotH, roughness) * Lc;
    }



    if (fmaxf(specularColor) > 0) 
    {
        // diffuse reflection
        {
            PerRayData_radiance new_prd;
            new_prd.importance = prd.importance * optix::luminance(diffuse);
            new_prd.depth = prd.depth + 1;

            if (new_prd.importance >= 0.01f && new_prd.depth <= max_depth)
            {
                const float2 Xi = make_float2(rnd(prd.seed), rnd(prd.seed));
                float pdf = 0.f;
                const float3 R = sampleCosineLobe(Xi, tangent, bitangent, p_normal, pdf);

                if (pdf > 0.f)
                {
                    optix::Ray refl_ray = optix::make_Ray(hit_point, R, RADIANCE_RAY_TYPE, scene_epsilon, RT_DEFAULT_MAX);
                    rtTrace(top_object, refl_ray, new_prd);

                    const float3 H = optix::normalize(V + R);

                    const float VdotH = optix::clamp(optix::dot(V, H), 0.f, 1.f);

                    const float3 F = fresnel(VdotH, specularColor);
                    const float3 f_diffuse = optix::clamp(1 - F, 0.f, 1.f) * diffuse;
                    result += f_diffuse * M_PIf * new_prd.result; // cancel NdotR and add Pi again by dividing with de pdf
                }
            }
        }

        // specular reflection
        {
            PerRayData_radiance new_prd;
            new_prd.importance = prd.importance * optix::luminance(specularColor);
            new_prd.depth = prd.depth + 1;

            if (new_prd.importance >= 0.01f && new_prd.depth <= max_depth)
            {
                if (roughness > 0.f)
                {
                    const float2 Xi = make_float2(rnd(prd.seed), rnd(prd.seed));
                    const float3 H = importanceSampleGGX(Xi, tangent, bitangent, p_normal, roughness);
                    const float3 R = optix::reflect(ray.direction, H);

                    const float VdotH = optix::clamp(optix::dot(V, H), 0.f, 1.f);
                    const float NdotH = optix::clamp(optix::dot(p_normal, H), 0.f, 1.f);

                    const float ggxPdf = probabilityGGX(NdotH, VdotH, roughness);

                    if (ggxPdf > 0.005f)
                    {
                        optix::Ray refl_ray = optix::make_Ray(hit_point, R, RADIANCE_RAY_TYPE, scene_epsilon, RT_DEFAULT_MAX);
                        rtTrace(top_object, refl_ray, new_prd);

                        const float3 F = fresnel(VdotH, specularColor);
                        const float3 f_diffuse = optix::clamp(1 - F, 0.f, 1.f) * diffuse;
                        const float NdotR = optix::clamp(optix::dot(p_normal, R), 0.f, 1.f);

                        result += cook_torrance_contrib(VdotH, NdotH, NdotR, NdotV, specularColor, roughness) * new_prd.result;
                    }
                }
                else
                {
                    float3 R = optix::reflect(ray.direction, p_normal);
                    optix::Ray refl_ray = optix::make_Ray(hit_point, R, RADIANCE_RAY_TYPE, scene_epsilon, RT_DEFAULT_MAX);
                    rtTrace(top_object, refl_ray, new_prd);

                    const float3 H = p_normal;
                    const float VdotH = optix::clamp(optix::dot(V, H), 0.f, 1.f);

                    const float3 F = fresnel(VdotH, specularColor);
                    const float NdotR = optix::clamp(optix::dot(p_normal, R), 0.f, 1.f);

                    result += F * NdotR * new_prd.result;
                }
            }
        }
    }

    // pass the color back up the tree
    prd.result = result;
}