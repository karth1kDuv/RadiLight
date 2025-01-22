cmake_minimum_required(VERSION 3.15)
project(opencvTest LANGUAGES C CXX CUDA)
# See cmake policy CMP0146 for exact details on this language specification,
# but in short using find_package(CUDA) is deprecated and should no longer
# be used. Instead, by specifying CUDA as one of the languages, you can use
# add_executable as normal and it will automatically figure out what to do
# with any .cu files.

set(CMAKE_CXX_STANDARD 20) # Make sure nvcc knows to use c++20 otherwise spans will break
#set(glm_DIR /usr/local/include/glm)
# Not sure if the above is really needed. I left it in here in case you find otherwise

find_package(OpenCV REQUIRED)

# All the wpilib stuff that needs to be turned off. We just want to build ntcore and nothing else
set(WITH_WPILIB OFF)
set(WITH_GUI OFF)
set(WITH_SIMULATION_MODULES OFF)
set(WITH_PROTOBUF OFF)
set(WITH_WPIMATH OFF)
set(WITH_CSCORE OFF)
set(WITH_TESTS OFF)
add_subdirectory(allwpilib)
# The allwpilib repo is submoduled in ./allwpilib

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(visionTest main.cu) # See above commend on CMP0146
#cuda_add_executable(visionTest main.cu)

# I adjusted the paths for the apriltag library since I have no idea why you are using an absolute
# path and not a relative one. Relative paths make this much more portable. It would be even better
# if this was compiled from source, but I dont know where you got libapril_tagging.so from. If you
# send me the repo you got it from I'll help you add it to cmake as well so you dont have a precompiled
# binary checked in to your version control lmao
add_library(libapril_tagging STATIC IMPORTED)
set_target_properties(libapril_tagging PROPERTIES IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/nvapriltags/libapril_tagging.a)
set_target_properties(libapril_tagging PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/nvapriltags/include)
target_link_libraries(visionTest libapril_tagging ntcore)

target_link_libraries(visionTest ${CUDA_LIBRARIES} ${OpenCV_LIBS})

# Not sure if this is entirely needed since we specify the standard above but its a sanity check. You
# can try removing it and see if stuff still works. You'll know if its not working because when you go
# to build the project you'll get a bunch of warnings when compiling the nt header about `span` not
# being in the `std` namespace, since spans were not introduced until c++20.
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++20")

# If you need any other help cleaning up your code or creating a more organized file structure let me know.
