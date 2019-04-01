/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

//-----------------------------------------------------------------------------
//
// optixGeometryTriangles: Demonstrates the GeometryTriangles API.
//
//-----------------------------------------------------------------------------

#ifdef __APPLE__
#  include <GLUT/glut.h>
#else
#  include <GL/glew.h>
#  if defined( _WIN32 )
#  include <GL/wglew.h>
#  include <GL/freeglut.h>
#  else
#  include <GL/glut.h>
#  endif
#endif

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

#include <sutil.h>
#include "common.h"
#include <Arcball.h>

#include <cstring>
#include <iostream>
#include <stdint.h>

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tinygltf/tiny_gltf.h>

using namespace optix;

const char* const SAMPLE_NAME = "optixGeometryTriangles";

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

Context  context;
uint32_t width   = 768u;
uint32_t height  = 768u;
bool     use_pbo = true;

// Camera state
//float3         camera_up;
//float3         camera_lookat;
//float3         camera_eye;
//Matrix4x4      camera_rotate;
//bool           camera_dirty = true;  // Do camera params need to be copied to OptiX context
//sutil::Arcball arcball;

float3 camera_lookat;
float camera_phi = 0, camera_theta = 0;
float camera_dist = 1.f;
bool camera_dirty = true;

void computeCameraVectors(float3 & camera_eye, float3 & camera_up)
{
    const auto cos_theta = cos(camera_theta);
    const float3 cam_z = make_float3(cos_theta * sin(camera_phi), sin(camera_theta), cos_theta * cos(camera_phi));

    camera_eye = camera_lookat + cam_z * camera_dist;
    camera_up = make_float3(0.f, 1.f, 0.f);
}

float3 bboxMin = make_float3(std::numeric_limits<float>::max());
float3 bboxMax = make_float3(std::numeric_limits<float>::lowest());

// Mouse state
int2 mouse_prev_pos;
int  mouse_button;

// Materials
Material phong_matl;
Material tex_matl;
Material checker_matl;
Material normal_matl;
Material bary_matl;

// Geometry
GeometryInstance tri_gi;
GeometryInstance sphere_gi;
GeometryInstance plane_gi;


//------------------------------------------------------------------------------
//
// Forward decls
//
//------------------------------------------------------------------------------

Buffer getOutputBuffer();
void destroyContext();
void registerExitHandler();
void createContext();
void createMaterials();
GeometryGroup createGeometry();
GeometryGroup createGeometryTriangles();
void setupScene();
void setupCamera();
void setupLights();
void updateCamera();
void glutInitialize( int* argc, char** argv );
void glutRun();

void glutDisplay();
void glutKeyboardPress( unsigned char k, int x, int y );
void glutMousePress( int button, int state, int x, int y );
void glutMouseMotion( int x, int y);
void glutResize( int w, int h );


//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

struct Tetrahedron
{
    float3   vertices [12];
    float3   normals  [12];
    float2   texcoords[12];
    unsigned indices  [12];

    Tetrahedron( const float H, const float3 trans )
    {
        const float a = ( 3.0f * H ) / sqrtf( 6.0f ); // Side length
        const float d = a * sqrtf( 3.0f ) / 6.0f;     // Offset for base vertices from apex

        // There are only four vertex positions, but we will duplicate vertices
        // instead of sharing them among faces.
        const float3 v0 = trans + make_float3( 0.0f, 0, H - d );
        const float3 v1 = trans + make_float3( a / 2.0f, 0, -d );
        const float3 v2 = trans + make_float3( -a / 2.0f, 0, -d );
        const float3 v3 = trans + make_float3( 0.0f, H, 0.0f );

        // Bottom face
        vertices[0] = v0;
        vertices[1] = v1;
        vertices[2] = v2;

        // Duplicate the face normals across the vertices.
        float3 n = optix::normalize( optix::cross( v2-v0, v1-v0 ) );
        normals[0] = n;
        normals[1] = n;
        normals[2] = n;

        texcoords[0] = make_float2( 0.5f, 1.0f );
        texcoords[1] = make_float2( 1.0f, 0.0f );
        texcoords[2] = make_float2( 0.0f, 0.0f );

        // Left face
        vertices[3] = v3;
        vertices[4] = v2;
        vertices[5] = v0;

        n = optix::normalize( optix::cross( v2-v3, v0-v3 ) );
        normals[3] = n;
        normals[4] = n;
        normals[5] = n;

        texcoords[3] = make_float2( 0.5f, 1.0f );
        texcoords[4] = make_float2( 0.0f, 0.0f );
        texcoords[5] = make_float2( 1.0f, 0.0f );

        // Right face
        vertices[6] = v3;
        vertices[7] = v0;
        vertices[8] = v1;

        n = optix::normalize( optix::cross( v0-v3, v1-v3 ) );
        normals[6] = n;
        normals[7] = n;
        normals[8] = n;

        texcoords[6] = make_float2( 0.5f, 1.0f );
        texcoords[7] = make_float2( 0.0f, 0.0f );
        texcoords[8] = make_float2( 1.0f, 0.0f );

        // Back face
        vertices[9]  = v3;
        vertices[10] = v1;
        vertices[11] = v2;

        n = optix::normalize( optix::cross( v1-v3, v2-v3 ) );
        normals[9]  = n;
        normals[10] = n;
        normals[11] = n;

        texcoords[9]  = make_float2( 0.5f, 1.0f );
        texcoords[10] = make_float2( 0.0f, 0.0f );
        texcoords[11] = make_float2( 1.0f, 0.0f );

        for( int i = 0; i < 12; ++i )
            indices[i]  = i;
    }
};


Buffer getOutputBuffer()
{
    return context[ "output_buffer" ]->getBuffer();
}


void destroyContext()
{
    if( context )
    {
        context->destroy();
        context = 0;
    }
}


void registerExitHandler()
{
    // register shutdown handler
#ifdef _WIN32
    glutCloseFunc( destroyContext );  // this function is freeglut-only
#else
    atexit( destroyContext );
#endif
}


void createContext()
{
    int v = 1;
    rtGlobalSetAttribute(RT_GLOBAL_ATTRIBUTE_ENABLE_RTX, sizeof(int), &v);

    // Set up context
    context = Context::create();
    context->setRayTypeCount( 2 );
    context->setEntryPointCount( 1 );
    context->setStackSize( 2800 );
    context->setMaxTraceDepth( 12 );

    // Note: high max depth for reflection and refraction through glass
    context["max_depth"]->setInt( 10 );
    context["frame"]->setUint( 0u );
    context["scene_epsilon"]->setFloat( 1.e-4f );
    context["ambient_light_color"]->setFloat( 0.4f, 0.4f, 0.4f );

    Buffer buffer = sutil::createOutputBuffer( context, RT_FORMAT_UNSIGNED_BYTE4, width, height, use_pbo );
    context["output_buffer"]->set( buffer );

    // Accumulation buffer.
    Buffer accum_buffer = context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
            RT_FORMAT_FLOAT4, width, height );
    context["accum_buffer"]->set( accum_buffer );

    // Ray generation program
    const char *ptx = sutil::getPtxString( SAMPLE_NAME, "accum_camera.cu" );
    Program ray_gen_program = context->createProgramFromPTXString( ptx, "pinhole_camera" );
    context->setRayGenerationProgram( 0, ray_gen_program );

    // Exception program
    Program exception_program = context->createProgramFromPTXString( ptx, "exception" );
    context->setExceptionProgram( 0, exception_program );
    context["bad_color"]->setFloat( 1.0f, 0.0f, 1.0f );

    // Miss program
    context->setMissProgram( 0, context->createProgramFromPTXString( sutil::getPtxString( SAMPLE_NAME, "constantbg.cu" ), "miss" ) );
    context["bg_color"]->setFloat( 0.34f, 0.55f, 0.85f );
}


void createMaterials()
{
    // Normal shader material
    const char* ptx    = sutil::getPtxString( SAMPLE_NAME, "normal_shader.cu" );
    Program     tri_ch = context->createProgramFromPTXString( ptx, "closest_hit_radiance" );
    Program     tri_ah = context->createProgramFromPTXString( ptx, "any_hit_shadow" );

    normal_matl = context->createMaterial();
    normal_matl->setClosestHitProgram( 0, tri_ch );
    normal_matl->setAnyHitProgram( 1, tri_ah );

    // Metal material
    ptx              = sutil::getPtxString( SAMPLE_NAME, "phong.cu" );
    Program phong_ch = context->createProgramFromPTXString( ptx, "closest_hit_radiance" );
    Program phong_ah = context->createProgramFromPTXString( ptx, "any_hit_shadow" );

    phong_matl = context->createMaterial();
    phong_matl->setClosestHitProgram( 0, phong_ch );
    phong_matl->setAnyHitProgram( 1, phong_ah );
    phong_matl["Ka"]->setFloat( 0.2f, 0.5f, 0.5f );
    phong_matl["Kd"]->setFloat( 1.f, 1.f, 1.f );
    phong_matl["Ks"]->setFloat( 0.9f, 0.9f, 0.9f );
    phong_matl["phong_exp"]->setFloat( 64 );
    phong_matl["Kr"]->setFloat( 0.5f, 0.5f, 0.5f );

    // Texture material
    ptx            = sutil::getPtxString( SAMPLE_NAME, "phong.cu" );
    Program tex_ch = context->createProgramFromPTXString( ptx, "closest_hit_radiance_textured" );
    Program tex_ah = context->createProgramFromPTXString( ptx, "any_hit_shadow" );

    tex_matl = context->createMaterial();
    tex_matl->setClosestHitProgram( 0, tex_ch );
    tex_matl->setAnyHitProgram( 1, tex_ah );

    tex_matl["Kd_map"]->setTextureSampler(
        sutil::loadTexture( context, std::string( sutil::samplesDir() ) + "/data/nvidia_logo.ppm",
                            optix::make_float3(0.0f, 0.0f, 1.0f) ) );

    // Checker material
    ptx              = sutil::getPtxString( SAMPLE_NAME, "checker.cu" );
    Program check_ch = context->createProgramFromPTXString( ptx, "closest_hit_radiance" );
    Program check_ah = context->createProgramFromPTXString( ptx, "any_hit_shadow" );

    checker_matl = context->createMaterial();
    checker_matl->setClosestHitProgram( 0, check_ch );
    checker_matl->setAnyHitProgram( 1, check_ah );

    checker_matl["Kd1"]->setFloat( 0.5f, 0.9f, 0.4f );
    checker_matl["Ka1"]->setFloat( 0.5f, 0.9f, 0.4f );
    checker_matl["Ks1"]->setFloat( 0.5f, 0.5f, 0.5f );
    checker_matl["Kd2"]->setFloat( 0.9f, 0.4f, 0.2f );
    checker_matl["Ka2"]->setFloat( 0.9f, 0.4f, 0.2f );
    checker_matl["Ks2"]->setFloat( 0.0f, 0.0f, 0.0f );
    checker_matl["inv_checker_size"]->setFloat( 16.0f, 16.0f, 1.0f );
    checker_matl["phong_exp1"]->setFloat( 64.0f );
    checker_matl["phong_exp2"]->setFloat( 0.0f );
    checker_matl["Kr1"]->setFloat(0.5f, 0.5f, 0.5f);
    checker_matl["Kr2"]->setFloat( 0.0f, 0.0f, 0.0f );

    // Barycentric material
    ptx             = sutil::getPtxString( SAMPLE_NAME, "optixGeometryTriangles.cu" );
    Program bary_ch = context->createProgramFromPTXString( ptx, "closest_hit_radiance" );
    Program bary_ah = context->createProgramFromPTXString( ptx, "any_hit_shadow" );

    bary_matl = context->createMaterial();
    bary_matl->setClosestHitProgram( 0, bary_ch );
    bary_matl->setAnyHitProgram( 1, bary_ah );
}


GeometryGroup createGeometryTriangles()
{
    // Create a tetrahedron using four triangular faces.  First We will create
    // vertex and index buffers for the faces, and then create a
    // GeometryTriangles object.
    const unsigned num_faces    = 4;
    const unsigned num_vertices = num_faces * 3;

    // Define a regular tetrahedron of height 2, translated 1.5 units from the origin.
    Tetrahedron tet( 2.0f, make_float3( 1.5f, 0.0f, 0.0f ) );

    // Create Buffers for the triangle vertices, normals, texture coordinates, and indices.
    Buffer vertex_buffer   = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, num_vertices );
    Buffer normal_buffer   = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, num_vertices );
    Buffer texcoord_buffer = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, num_vertices );
    Buffer index_buffer    = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3, num_faces );

    // Copy the tetrahedron geometry into the device Buffers.
    memcpy( vertex_buffer->map(), tet.vertices, sizeof( tet.vertices ) );
    memcpy( normal_buffer->map(), tet.normals, sizeof( tet.normals ) );
    memcpy( texcoord_buffer->map(), tet.texcoords, sizeof( tet.texcoords ) );
    memcpy( index_buffer->map(), tet.indices, sizeof( tet.indices ) );

    vertex_buffer->unmap();
    normal_buffer->unmap();
    texcoord_buffer->unmap();
    index_buffer->unmap();

    // Create a GeometryTriangles object.
    optix::GeometryTriangles geom_tri = context->createGeometryTriangles();

    geom_tri->setPrimitiveCount( num_faces );
    geom_tri->setTriangleIndices( index_buffer, RT_FORMAT_UNSIGNED_INT3 );
    geom_tri->setVertices( num_vertices, vertex_buffer, RT_FORMAT_FLOAT3 );
    geom_tri->setBuildFlags( RTgeometrybuildflags( 0 ) );

    // Set an attribute program for the GeometryTriangles, which will compute
    // things like normals and texture coordinates based on the barycentric
    // coordindates of the intersection.
    const char* ptx = sutil::getPtxString( SAMPLE_NAME, "optixGeometryTriangles.cu" );
    geom_tri->setAttributeProgram( context->createProgramFromPTXString( ptx, "triangle_attributes" ) );

    geom_tri["index_buffer"]->setBuffer( index_buffer );
    geom_tri["vertex_buffer"]->setBuffer( vertex_buffer );
    geom_tri["normal_buffer"]->setBuffer( normal_buffer );
    geom_tri["texcoord_buffer"]->setBuffer( texcoord_buffer );

    // Bind a Material to the GeometryTriangles.  Materials can be shared
    // between GeometryTriangles objects and other Geometry types, as long as
    // all of the attributes needed by the attached hit programs are produced in
    // the attribute program.
    tri_gi = context->createGeometryInstance( geom_tri, phong_matl );

    GeometryGroup tri_gg = context->createGeometryGroup();
    tri_gg->addChild( tri_gi );
    tri_gg->setAcceleration( context->createAcceleration( "Trbvh" ) );

    return tri_gg;
}

GeometryGroup createGround(float groundHeight = 0)
{
    // Parallelogram geometry for ground plane.
    Geometry parallelogram = context->createGeometry();
    parallelogram->setPrimitiveCount(1u);

    const char * ptx = sutil::getPtxString(SAMPLE_NAME, "parallelogram.cu");
    parallelogram->setBoundingBoxProgram(context->createProgramFromPTXString(ptx, "bounds"));
    parallelogram->setIntersectionProgram(context->createProgramFromPTXString(ptx, "intersect"));

    float3 anchor = make_float3(-8.0f, groundHeight, -8.0f);
    float3 v1 = make_float3(16.0f, 0.f, 0.0f);
    float3 v2 = make_float3(0.0f, 0.f, 16.0f);
    float3 normal = normalize(cross(v1, v2));

    float d = dot(normal, anchor);
    v1 *= 1.0f / dot(v1, v1);
    v2 *= 1.0f / dot(v2, v2);

    float4 plane = make_float4(normal, d);
    parallelogram["plane"]->setFloat(plane);
    parallelogram["v1"]->setFloat(v1);
    parallelogram["v2"]->setFloat(v2);
    parallelogram["anchor"]->setFloat(anchor);

    // Greate GIs to bind Materials to the Geometry objects.
    plane_gi = context->createGeometryInstance(parallelogram, &checker_matl, &checker_matl + 1);

    // Create a GeometryGroup for the non-GeometryTriangles objects.
    GeometryGroup gg = context->createGeometryGroup();
    gg->addChild(plane_gi);
    gg->setAcceleration(context->createAcceleration("Trbvh"));

    return gg;
}

GeometryGroup createGeometry()
{
    // Sphere geometry
    Geometry sphere = context->createGeometry();
    sphere->setPrimitiveCount( 1u );

    const char* ptx = sutil::getPtxString( SAMPLE_NAME, "sphere.cu" );
    sphere->setBoundingBoxProgram( context->createProgramFromPTXString( ptx, "bounds" ) );
    sphere->setIntersectionProgram( context->createProgramFromPTXString( ptx, "robust_intersect" ) );
    sphere["sphere"]->setFloat( -1.5f, 1.0f, 0.0f, 1.0f );

    // Parallelogram geometry for ground plane.
    Geometry parallelogram = context->createGeometry();
    parallelogram->setPrimitiveCount( 1u );

    ptx = sutil::getPtxString( SAMPLE_NAME, "parallelogram.cu" );
    parallelogram->setBoundingBoxProgram( context->createProgramFromPTXString( ptx, "bounds" ) );
    parallelogram->setIntersectionProgram( context->createProgramFromPTXString( ptx, "intersect" ) );

    float3 anchor = make_float3( -8.0f, -1e-3f, -8.0f );
    float3 v1     = make_float3( 16.0f, 0.0f, 0.0f );
    float3 v2     = make_float3( 0.0f, 0.0f, 16.0f );
    float3 normal = normalize( cross( v1, v2 ) );

    float d = dot( normal, anchor );
    v1 *= 1.0f / dot( v1, v1 );
    v2 *= 1.0f / dot( v2, v2 );

    float4 plane = make_float4( normal, d );
    parallelogram["plane"]->setFloat( plane );
    parallelogram["v1"]->setFloat( v1 );
    parallelogram["v2"]->setFloat( v2 );
    parallelogram["anchor"]->setFloat( anchor );

    // Greate GIs to bind Materials to the Geometry objects.
    sphere_gi = context->createGeometryInstance( sphere, &phong_matl, &phong_matl + 1 );
    plane_gi  = context->createGeometryInstance( parallelogram, &checker_matl, &checker_matl + 1 );

    // Create a GeometryGroup for the non-GeometryTriangles objects.
    GeometryGroup gg = context->createGeometryGroup();
    gg->addChild( sphere_gi );
    gg->addChild( plane_gi );
    gg->setAcceleration( context->createAcceleration( "Trbvh" ) );

    return gg;
}

std::string prettyName(std::string str)
{
    if (str.empty())
        return "[undefined]";
    return str;
}

const std::unordered_map<int, size_t> glType = {
    { TINYGLTF_TYPE_VEC2, 2 },
    { TINYGLTF_TYPE_VEC3, 3 },
    { TINYGLTF_TYPE_VEC4, 4 },
    { TINYGLTF_TYPE_MAT2, 4 },
    { TINYGLTF_TYPE_MAT3, 9 },
    { TINYGLTF_TYPE_MAT4, 16 },
    { TINYGLTF_TYPE_SCALAR, 1 },
    { TINYGLTF_TYPE_VECTOR, 4 },
    { TINYGLTF_TYPE_MATRIX, 16 },
};

const std::unordered_map<int, size_t> glParamTypeToByteSize = {
    { TINYGLTF_PARAMETER_TYPE_BYTE, sizeof(char) },
    { TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE, sizeof(unsigned char) },
    { TINYGLTF_PARAMETER_TYPE_SHORT, sizeof(short) },
    { TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT, sizeof(unsigned short) },
    { TINYGLTF_PARAMETER_TYPE_INT, sizeof(int32_t) },
    { TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT, sizeof(uint32_t) },
    { TINYGLTF_PARAMETER_TYPE_FLOAT, sizeof(float) }
};

template <typename T, size_t C>
size_t loadGltfBuffer(const size_t accesorIndex, const std::string & bufferName, const tinygltf::Model & model, std::vector<T> & outBuffer)
{
    const size_t componentCount = C;
    const tinygltf::Accessor & accessor = model.accessors[accesorIndex];
    const tinygltf::BufferView & bufferView = model.bufferViews[accessor.bufferView];
    const tinygltf::Buffer & buffer = model.buffers[bufferView.buffer];

    if (glType.at(accessor.type) != componentCount)
    {
        std::cerr << "[GLTF] Buffer \'" << bufferName << "\': found component count (" << glType.at(accessor.type) << ") is not as expected (" << componentCount << "). Cancel buffer import";
        return -1;
    }

    outBuffer.resize(accessor.count * componentCount);
    const size_t offset = accessor.byteOffset + bufferView.byteOffset;
    const size_t byteStride = (bufferView.byteStride == 0) ? glParamTypeToByteSize.at(accessor.componentType) * componentCount : bufferView.byteStride;

    for (size_t i = 0; i < accessor.count; ++i)
    {
        const size_t index = offset + i * byteStride;
        for (size_t c = 0; c < componentCount; ++c)
        {
            const size_t componentIndex = index + c * glParamTypeToByteSize.at(accessor.componentType);
            T data;
            switch (accessor.componentType)
            {
            case TINYGLTF_PARAMETER_TYPE_BYTE:
                data = (T)*(char*)(&buffer.data[componentIndex]);
                break;
            case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE:
                data = (T)*(unsigned char*)(&buffer.data[componentIndex]);
                break;
            case TINYGLTF_PARAMETER_TYPE_SHORT:
                data = (T)*(short*)(&buffer.data[componentIndex]);
                break;
            case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT:
                data = (T)*(unsigned short*)(&buffer.data[componentIndex]);
                break;
            case TINYGLTF_PARAMETER_TYPE_INT:
                data = (T)*(int32_t*)(&buffer.data[componentIndex]);
                break;
            case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT:
                data = (T)*(uint32_t*)(&buffer.data[componentIndex]);
                break;
            case TINYGLTF_PARAMETER_TYPE_FLOAT:
                data = (T)*(float*)(&buffer.data[componentIndex]);
                break;
            default:
                std::cerr << "[GLTF] Buffer \'" << bufferName << "\': data type is not supported (" << accessor.componentType << "). Cancel buffer import";
                return -1;
                break;
            }
            outBuffer[i * componentCount + c] = data;
        }
    }

    return accessor.count;
}

size_t findVertexAttributeIndex(const tinygltf::Primitive & prim, const std::string & attrName)
{
    const auto & found = prim.attributes.find(attrName);
    if (found != prim.attributes.end())
        return found->second;
    return -1;
}

using M44f = optix::Matrix4x4;
using V3f = optix::float3;
using V4f = optix::float4;

// convert quaternions to euler angles (https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Quaternion_to_Euler_Angles_Conversion)
inline
V3f toEulerAngle(const V4f & q)
{
    V3f eulerAngles;

    // roll (x-axis rotation)
    double sinr_cosp = 2.f * (q.w * q.x + q.y * q.z);
    double cosr_cosp = 1.f - 2.f * (q.x * q.x + q.y * q.y);
    eulerAngles.x = atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    double sinp = +2.0 * (q.w * q.y - q.z * q.x);
    if (fabs(sinp) >= 1)
        eulerAngles.y = copysign(M_PI_2f, sinp); // use 90 degrees if out of range
    else
        eulerAngles.y = asin(sinp);

    // yaw (z-axis rotation)
    double siny_cosp = 2.f * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1.f - 2.f * (q.y * q.y + q.z * q.z);
    eulerAngles.z = atan2(siny_cosp, cosy_cosp);

    return eulerAngles;
}

struct GltfInstance
{
    GltfInstance(std::string inName, M44f inTransform, int inMesh)
        : name(inName)
        , transform(inTransform)
        , mesh(inMesh) {}

    std::string name;
    M44f transform;
    int mesh = -1;
};

void recursivelyFindInstances(const tinygltf::Model & model, const tinygltf::Node & currentNode, const M44f& parentMatrix, std::vector<GltfInstance> & outInstances)
{
    M44f currentMatrix = M44f::identity();
    if (currentNode.matrix.size() == 16)
    {
        float floatMatrixData[16];
        for (size_t i = 0; i < 16; ++i)
            floatMatrixData[i] = (float) currentNode.matrix[i];

        currentMatrix = M44f(&floatMatrixData[0]).transpose(); // transpose because gltf matrices are column major while optix matrices are row major
    }
    else
    {
        if (currentNode.translation.size() == 3)
        {
            V3f translation = optix::make_float3((float)currentNode.translation[0], (float)currentNode.translation[1], (float)currentNode.translation[2]);
            currentMatrix = currentMatrix * M44f::translate(translation);
        }
        if (currentNode.rotation.size() == 4)
        {
            const V4f quaternion = optix::make_float4((float)currentNode.rotation[0], (float)currentNode.rotation[1], (float)currentNode.rotation[2], (float)currentNode.rotation[3]);
            const V3f scaledVec = make_float3(quaternion.x, quaternion.y, quaternion.z);
            const float sinTheta_2 = optix::length(scaledVec);
            if (sinTheta_2 > 0.f)
            {
                const auto direction = scaledVec / sinTheta_2;
                const auto theta_2 = std::atan2(sinTheta_2, quaternion.w);
                const auto theta = 2 * theta_2;

                currentMatrix = currentMatrix * M44f::rotate(theta, direction);
            }
        }
        if (currentNode.scale.size() == 3)
        {
            V3f scale = optix::make_float3((float)currentNode.scale[0], (float)currentNode.scale[1], (float)currentNode.scale[2]);
            currentMatrix = currentMatrix * M44f::scale(scale);
        }
    }

    currentMatrix = parentMatrix * currentMatrix;

    if (currentNode.mesh != -1)
    {
        outInstances.emplace_back(currentNode.name, currentMatrix, currentNode.mesh);
    }
    else
    {
        for (const auto & childIndex : currentNode.children)
        {
            recursivelyFindInstances(model, model.nodes[childIndex], currentMatrix, outInstances);
        }
    }
}

struct OptixInstance
{
    GeometryGroup gg;
    Transform transform;
    size_t meshIdx;
    M44f localToWorld;

    OptixInstance(GeometryGroup gg, Transform t, size_t meshIdx, const M44f & mat) : gg(gg), transform(t), meshIdx{ meshIdx }, localToWorld{ mat } {}
};

V3f min(const V3f & lhs, const V3f & rhs)
{
    return make_float3(std::min(lhs.x, rhs.x), std::min(lhs.y, rhs.y), std::min(lhs.z, rhs.z));
}

V3f max(const V3f & lhs, const V3f & rhs)
{
    return make_float3(std::max(lhs.x, rhs.x), std::max(lhs.y, rhs.y), std::max(lhs.z, rhs.z));
}

std::vector<OptixInstance> createGLTFGeometry(const std::string & input_gltf)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    const auto ret = loader.LoadASCIIFromFile(&model, &err, &warn, input_gltf);

    if (!warn.empty())
        std::cerr << "tinygltf warning: " << warn << "\n";

    if (!err.empty())
        std::cerr << "tinygltf error: " << err << "\n";

    if (!ret)
    {
        const auto msg = "Failed to parse glTF";
        std::cerr << msg << "\n";
        throw std::runtime_error(msg);
    }

    std::vector<GeometryTriangles> geometries;
    std::vector<size_t> perMeshGeometryOffset(model.meshes.size(), 0);

    size_t geometryOffset = 0;
    for (size_t meshIdx = 0; meshIdx < model.meshes.size(); ++meshIdx)
    {
        const auto & mesh = model.meshes[meshIdx];

        for (size_t primIdx = 0; primIdx < mesh.primitives.size(); ++primIdx)
        {
            geometries.emplace_back();
            auto & geometry = geometries.back();

            const auto & prim = mesh.primitives[primIdx];
            if (prim.mode != TINYGLTF_MODE_TRIANGLES)
            {
                std::cerr << "Only triangle geometry is supported, skipping mesh " << meshIdx << " named " << prettyName(mesh.name) << "\n";
                continue;
            }

            // topology
            std::vector<int32_t> indexBuffer;
            const size_t indexCount = loadGltfBuffer<int32_t, 1>(prim.indices, "indices", model, indexBuffer);
            if (indexCount == -1)
                continue;
            const auto triangleCount = indexCount / 3;

            if (indexCount == 0)
            {
                std::clog << "[GLTF] Warning: indexCount == 0 for mesh " << meshIdx;
            }

            // mandatory vertex attributes : positions and normals
            const size_t positionIndex = findVertexAttributeIndex(prim, "POSITION");
            if (positionIndex == -1)
            {
                std::cerr << "[GLTF] Unable to find POSITION vertex attributes. Cancel mesh import";
                continue;
            }

            const size_t normalIndex = findVertexAttributeIndex(prim, "NORMAL");
            if (normalIndex == -1)
            {
                std::cerr << "[GLTF] Unable to find NORMAL vertex attributes. Cancel mesh import";
                continue;
            }

            std::vector<float> positionData;
            const size_t positionCount = loadGltfBuffer<float, 3>(positionIndex, "position", model, positionData);

            if (positionCount == 0)
            {
                std::clog << "[GLTF] Warning: positionCount == 0 for mesh " << meshIdx;
            }

            std::vector<float> normalData;
            const size_t normalCount = loadGltfBuffer<float, 3>(normalIndex, "normal", model, normalData);

            if (normalCount == 0)
            {
                std::clog << "[GLTF] Warning: normalCount == 0 for mesh " << meshIdx;
            }

            if (positionCount != normalCount)
            {
                std::clog << "[GLTF] Warning: positionCount != normalCount for mesh " << meshIdx;
            }

            for (size_t i = 0; i < indexCount; ++i)
            {
                if (indexBuffer[i] < 0 || indexBuffer[i] >= positionCount)
                    std::clog << "[GLTF] Warning: invalid index " << indexBuffer[i] << " for mesh " << meshIdx << " having " << positionCount << " positions.\n";
            }

            std::vector<float> texcoordData;
            const size_t texcoordCount = [&]() -> size_t {
                const size_t texcoordIndex = findVertexAttributeIndex(prim, "TEXCOORD_0");
                if (texcoordIndex == -1)
                {
                    //std::clog << "[GLTF] Unable to find TEXCOORD_0 vertex attributes.\n";
                    return 0;
                }
                return loadGltfBuffer<float, 2>(texcoordIndex, "texcoord", model, texcoordData);
            }();

            // Create Buffers for the triangle vertices, normals, texture coordinates, and indices.
            Buffer vertex_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, positionCount);
            Buffer normal_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, normalCount);
            Buffer texcoord_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, texcoordCount);
            Buffer index_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3, triangleCount);

            // Copy the tetrahedron geometry into the device Buffers.
            memcpy(vertex_buffer->map(), positionData.data(), positionData.size() * sizeof(positionData[0]));
            memcpy(normal_buffer->map(), normalData.data(), normalData.size() * sizeof(normalData[0]));
            if (texcoordCount > 0)
                memcpy(texcoord_buffer->map(), texcoordData.data(), texcoordData.size() * sizeof(texcoordData[0]));
            memcpy(index_buffer->map(), indexBuffer.data(), indexBuffer.size() * sizeof(indexBuffer[0]));

            vertex_buffer->unmap();
            normal_buffer->unmap();
            if (texcoordCount > 0)
                texcoord_buffer->unmap();
            index_buffer->unmap();

            // Create a GeometryTriangles object.
            optix::GeometryTriangles geom_tri = context->createGeometryTriangles();

            geom_tri->setPrimitiveCount(triangleCount);
            geom_tri->setTriangleIndices(index_buffer, RT_FORMAT_UNSIGNED_INT3);
            geom_tri->setVertices(positionCount, vertex_buffer, RT_FORMAT_FLOAT3);
            geom_tri->setBuildFlags(RTgeometrybuildflags(0));

            // Set an attribute program for the GeometryTriangles, which will compute
            // things like normals and texture coordinates based on the barycentric
            // coordindates of the intersection.
            const char* ptx = sutil::getPtxString(SAMPLE_NAME, "optixGeometryTriangles.cu");
            geom_tri->setAttributeProgram(context->createProgramFromPTXString(ptx, "triangle_attributes"));

            geom_tri["index_buffer"]->setBuffer(index_buffer);
            geom_tri["vertex_buffer"]->setBuffer(vertex_buffer);
            geom_tri["normal_buffer"]->setBuffer(normal_buffer);
            geom_tri["texcoord_buffer"]->setBuffer(texcoord_buffer);

            geometry = geom_tri;
        }
        perMeshGeometryOffset[meshIdx] = geometryOffset;
        geometryOffset += mesh.primitives.size();
    }

    std::clog << "[GLTF] Load " << geometries.size() << " meshes. \n";

    std::vector<OptixInstance> instances;

    std::clog << "[GLTF] Load " << model.scenes.size() << " scene. \n";

    std::vector<Acceleration> meshToAccel(geometries.size()); // Acceleration structure can be shared between instances having the same mesh

    for (const auto & gltfScene : model.scenes)
    {
        std::clog << "[GLTF] Load " << gltfScene.nodes.size() << " nodes. \n";
        for (const auto & nodeIndex : gltfScene.nodes)
        {
            std::vector<GltfInstance> toImportInstances;
            recursivelyFindInstances(model, model.nodes[nodeIndex], M44f::identity(), toImportInstances);
            std::clog << "[GLTF] Load " << toImportInstances.size() << " instances for node " << nodeIndex << ". \n";
            for (const auto & gltfInst : toImportInstances)
            {
                if (gltfInst.mesh >= model.meshes.size() || gltfInst.mesh < 0)
                {
                    std::cerr << "[GLTF] Error: geometry index " << gltfInst.mesh << std::endl;
                }

                const auto & mesh = model.meshes[gltfInst.mesh];
                const auto geometryOffset = perMeshGeometryOffset[gltfInst.mesh];

                for (size_t primIdx = 0; primIdx < mesh.primitives.size(); ++primIdx)
                {
                    const auto & prim = mesh.primitives[primIdx];
                    if (prim.mode != TINYGLTF_MODE_TRIANGLES)
                    {
                        continue;
                    }

                    const auto geomIdx = geometryOffset + primIdx;

                    if (!meshToAccel[geomIdx])
                        meshToAccel[geomIdx] = context->createAcceleration("Trbvh");

                    GeometryGroup gg = context->createGeometryGroup();
                    gg->setAcceleration(meshToAccel[geomIdx]);

                    GeometryInstance gi = context->createGeometryInstance(geometries[geomIdx], phong_matl);
                    gg->addChild(gi);

                    Transform transform = context->createTransform();

                    transform->setMatrix(false, gltfInst.transform.getData(), nullptr);
                    transform->setChild(gg);

                    instances.emplace_back(gg, transform, gltfInst.mesh, gltfInst.transform);
                }
            }
        }
    }

    for (const auto & inst : instances)
    {
        const auto & mesh = model.meshes[inst.meshIdx];
        for (size_t primIdx = 0; primIdx < mesh.primitives.size(); ++primIdx)
        {
            const auto & prim = mesh.primitives[primIdx];
            if (prim.mode != TINYGLTF_MODE_TRIANGLES)
            {
                std::cerr << "Only triangle geometry is supported, skipping mesh " << inst.meshIdx << " named " << prettyName(mesh.name) << "\n";
                continue;
            }

            const size_t positionIndex = findVertexAttributeIndex(prim, "POSITION");
            if (positionIndex == -1)
            {
                std::cerr << "[GLTF] Unable to find POSITION vertex attributes. Cancel mesh import";
                continue;
            }

            std::vector<float> positionData;
            const size_t positionCount = loadGltfBuffer<float, 3>(positionIndex, "position", model, positionData);

            // topology
            std::vector<int32_t> indexBuffer;
            const size_t indexCount = loadGltfBuffer<int32_t, 1>(prim.indices, "indices", model, indexBuffer);

            for (size_t i = 0; i < indexCount; ++i)
            {
                const float * position = positionData.data() + indexBuffer[i] * 3;
                const auto pos = inst.localToWorld * make_float4(position[0], position[1], position[2], 1.f);
                bboxMin = min(bboxMin, make_float3(pos.x, pos.y, pos.z));
                bboxMax = max(bboxMax, make_float3(pos.x, pos.y, pos.z));
            }
        }
    }

    return instances;
}

void setupScene(const std::string & input_gltf)
{
    // Create a GeometryGroup for the GeometryTriangles instances and a separate
    // GeometryGroup for all other primitives.
    GeometryGroup tri_gg = createGeometryTriangles();
    GeometryGroup gg     = createGeometry();
    auto gltf_instances = createGLTFGeometry(input_gltf);

    // Create a top-level Group to contain the two GeometryGroups.
    Group top_group = context->createGroup();
    top_group->setAcceleration( context->createAcceleration( "Trbvh" ) );

    //top_group->addChild( tri_gg );

    for (const auto & inst : gltf_instances)
    {
        top_group->addChild(inst.transform);
    }

    const auto ground = createGround(bboxMin.y);
    top_group->addChild(ground);

    context["top_object"]->set( top_group );
    context["top_shadower"]->set( top_group );
}


void setupCamera()
{
    const auto bboxDiag = bboxMax - bboxMin;

    const auto eye = bboxMax + bboxDiag;
    camera_lookat = (bboxMax + bboxMin) * 0.5f;

    auto cam_z = eye - camera_lookat;
    camera_dist = length(cam_z);
    cam_z /= camera_dist;

    const auto cos_theta = length(make_float2(cam_z.x, cam_z.z));
    const auto sin_theta = cam_z.y;
    const auto cos_phi = cam_z.z / cos_theta;
    const auto sin_phi = cam_z.x / cos_theta;

    camera_theta = atan2(sin_theta, cos_theta);
    camera_phi = atan2(sin_phi, cos_phi);

    //camera_lookat = (bboxMax + bboxMin) * 0.5f;
    //camera_lookat.y = bboxMin.y;
    //camera_dist = 10.f;
    //camera_theta = 0.f;
    //camera_phi = 0.f;

    camera_dirty  = true;
}


void setupLights()
{

    BasicLight lights[] = {
        { make_float3( 60.0f, 40.0f, 0.0f ), make_float3( 1.0f, 1.0f, 1.0f ), 1 }
    };

    Buffer light_buffer = context->createBuffer( RT_BUFFER_INPUT );
    light_buffer->setFormat( RT_FORMAT_USER );
    light_buffer->setElementSize( sizeof( BasicLight ) );
    light_buffer->setSize( sizeof(lights)/sizeof(lights[0]) );
    memcpy(light_buffer->map(), lights, sizeof(lights));
    light_buffer->unmap();

    context[ "lights" ]->set( light_buffer );
}


void updateCamera()
{
    const float vfov  = 30.0f;
    const float aspect_ratio = static_cast<float>(width) /
        static_cast<float>(height);

    float3 cam_eye;
    float3 cam_up;
    computeCameraVectors(cam_eye, cam_up);

    float3 camera_u, camera_v, camera_w;
    sutil::calculateCameraVariables(
        cam_eye, camera_lookat, cam_up, vfov, aspect_ratio,
        camera_u, camera_v, camera_w, /*fov_is_vertical*/ true );

    const Matrix4x4 frame = Matrix4x4::fromBasis(
            normalize( camera_u ),
            normalize( camera_v ),
            normalize( -camera_w ),
            camera_lookat);



    //const Matrix4x4 frame_inv = frame.inverse();

    //// Apply camera rotation twice to match old SDK behavior
    //const Matrix4x4 trans   = frame*camera_rotate*frame_inv;

    //camera_eye    = make_float3( trans*make_float4( camera_eye,    1.0f ) );
    //camera_lookat = make_float3( trans*make_float4( camera_lookat, 1.0f ) );
    //camera_up     = make_float3( trans*make_float4( camera_up,     0.0f ) );

    //sutil::calculateCameraVariables(
    //        camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
    //        camera_u, camera_v, camera_w, true );

    //camera_rotate = Matrix4x4::identity();

    context["eye"]->setFloat(cam_eye);
    context["U"  ]->setFloat( camera_u );
    context["V"  ]->setFloat( camera_v );
    context["W"  ]->setFloat( camera_w );

    camera_dirty = false;
}


void glutInitialize( int* argc, char** argv )
{
    glutInit( argc, argv );
    glutInitDisplayMode( GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE );
    glutInitWindowSize( width, height );
    glutInitWindowPosition( 100, 100 );
    glutCreateWindow( SAMPLE_NAME );
    glutHideWindow();
}


void glutRun()
{
    // Initialize GL state
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, -1, 1 );
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glViewport(0, 0, width, height);

    glutShowWindow();
    glutReshapeWindow( width, height);

    // register glut callbacks
    glutDisplayFunc( glutDisplay );
    glutIdleFunc( glutDisplay );
    glutReshapeFunc( glutResize );
    glutKeyboardFunc( glutKeyboardPress );
    glutMouseFunc( glutMousePress );
    glutMotionFunc( glutMouseMotion );

    registerExitHandler();

    glutMainLoop();
}


//------------------------------------------------------------------------------
//
//  GLUT callbacks
//
//------------------------------------------------------------------------------

void glutDisplay()
{
    static unsigned int accumulation_frame = 0;
    if( camera_dirty ) {
        updateCamera();
        accumulation_frame = 0;
    }

    context["frame"]->setUint( accumulation_frame++ );
    context->launch( 0, width, height );

    sutil::displayBufferGL( getOutputBuffer() );

    {
        static unsigned frame_count = 0;
        sutil::displayFps( frame_count++ );
    }

    glutSwapBuffers();
}


void glutKeyboardPress( unsigned char k, int x, int y )
{
    switch( k )
    {
        case( 'm' ):
        {
            // Cycle through a few different materials.
            static int count = 0;
            if( ++count == 1 )
                tri_gi->setMaterial( 0, normal_matl );
            else if( count == 2 )
                tri_gi->setMaterial( 0, bary_matl );
            else if( count == 3 )
                tri_gi->setMaterial( 0, tex_matl );
            else if( count == 4 )
                tri_gi->setMaterial( 0, checker_matl );
            else
                tri_gi->setMaterial( 0, phong_matl );

            count %= 5;
            camera_dirty = true;
            break;
        }
        case( 'q' ):
        case( 27 ): // ESC
        {
            destroyContext();
            exit(0);
        }
        case( 's' ):
        {
            const std::string outputImage = std::string(SAMPLE_NAME) + ".ppm";
            std::cerr << "Saving current frame to '" << outputImage << "'\n";
            sutil::displayBufferPPM( outputImage.c_str(), getOutputBuffer() );
            break;
        }
    }
}


void glutMousePress( int button, int state, int x, int y )
{
    if( state == GLUT_DOWN )
    {
        mouse_button = button;
        mouse_prev_pos = make_int2( x, y );
    }
    else
    {
        // nothing
    }
}


void glutMouseMotion( int x, int y)
{
    if( mouse_button == GLUT_RIGHT_BUTTON )
    {
        const float dx = static_cast<float>( x - mouse_prev_pos.x ) /
                         static_cast<float>( width );
        const float dy = static_cast<float>( y - mouse_prev_pos.y ) /
                         static_cast<float>( height );
        const float dmax = fabsf( dx ) > fabs( dy ) ? dx : dy;
        const float scale = fminf( dmax, 0.9f );
        if (scale != 0.f)
        {
            camera_dist = camera_dist * (1 + scale);
            camera_dirty = true;
        }
    }
    else if( mouse_button == GLUT_LEFT_BUTTON )
    {
        const float2 from = { static_cast<float>(mouse_prev_pos.x),
                              static_cast<float>(mouse_prev_pos.y) };
        const float2 to   = { static_cast<float>(x),
                              static_cast<float>(y) };

        const float2 a = { from.x / width, from.y / height };
        const float2 b = { to.x   / width, to.y   / height };

        camera_phi += -(b.x - a.x);
        camera_theta += (b.y - a.y);
        camera_dirty = true;

        //camera_rotate = M44f::rotate(b.y - a.y, make_float3(1.f, 1.f, 0.f)) * M44f::rotate(a.x - b.x, make_float3(0.f, 1.f, 0.f));

        //const auto v = camera_eye - camera_lookat;
        //const auto rotated = M44f::rotate(a.x - b.x, camera_up) * make_float4(v.x, v.y, v.z, 0);
        //const auto rotated_3 = make_float3(rotated.x, rotated.y, rotated.z);
        //camera_eye = camera_lookat + rotated_3;

        //const auto v2 = camera_eye - camera_lookat;
        //const auto camera_left = optix::cross(rotated_3, camera_up);
        //const auto rotated_again = M44f::rotate(b.y - a.y, camera_left) * make_float4(v2.x, v2.y, v2.z, 0);
        //camera_eye = camera_lookat + make_float3(rotated_again.x, rotated_again.y, rotated_again.z);

        camera_dirty = true;
    }

    mouse_prev_pos = make_int2( x, y );
}


void glutResize( int w, int h )
{
    width  = w;
    height = h;
    sutil::ensureMinimumSize(width, height);

    camera_dirty = true;

    sutil::resizeBuffer( getOutputBuffer(), width, height );
    sutil::resizeBuffer( context[ "accum_buffer" ]->getBuffer(), width, height );

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, -1, 1);
    glViewport(0, 0, width, height);
    glutPostRedisplay();
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0 )
{
    std::cerr << "\nUsage: " << argv0 << " [options]\n";
    std::cerr <<
        "App Options:\n"
        "  -h | --help         Print this usage message and exit.\n"
        "  -f | --file         Save single frame to file and exit.\n"
        "  -i | --input        Input gltf file.\n"
        "  -n | --nopbo        Disable GL interop for display buffer.\n"
        "App Keystrokes:\n"
        "  q  Quit\n"
        "  s  Save image to '" << SAMPLE_NAME << ".ppm'\n"
        "  m  Toggle the material on the GeometryTriangles object.\n"
        << std::endl;

    exit(1);
}

int main( int argc, char** argv )
{
    std::string out_file;
    std::string input_file;
    for( int i=1; i<argc; ++i )
    {
        const std::string arg( argv[i] );

        if( arg == "-h" || arg == "--help" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "-f" || arg == "--file"  )
        {
            if( i == argc-1 )
            {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsageAndExit( argv[0] );
            }
            out_file = argv[++i];
        }
        else if( arg == "-n" || arg == "--nopbo"  )
        {
            use_pbo = false;
        }
        else if (arg == "-i" || arg == "--input")
        {
            if (i == argc - 1)
            {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsageAndExit(argv[0]);
            }
            input_file = argv[++i];
        }
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    try
    {
        glutInitialize( &argc, argv );

#ifndef __APPLE__
        glewInit();
#endif

        createContext();
        createMaterials();
        setupScene(input_file);
        setupCamera();
        setupLights();

        context->validate();

        if ( out_file.empty() )
        {
            glutRun();
        }
        else
        {
            updateCamera();
            context->launch( 0, width, height );
            sutil::displayBufferPPM( out_file.c_str(), getOutputBuffer() );
            destroyContext();
        }
        return 0;
    }
    SUTIL_CATCH( context->get() )
}

