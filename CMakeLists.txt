cmake_minimum_required (VERSION 3.10)

# if(POLICY CMP0025)
#     cmake_policy(SET CMP0025 NEW)
# endif()

project(gte_sample)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE openvino::runtime)