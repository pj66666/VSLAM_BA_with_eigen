# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/pj/pj/VSLAM_BA_with_eigen

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/pj/pj/VSLAM_BA_with_eigen/build

# Include any dependencies generated for this target.
include src/CMakeFiles/myslam.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/myslam.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/myslam.dir/flags.make

src/CMakeFiles/myslam.dir/frame.cpp.o: src/CMakeFiles/myslam.dir/flags.make
src/CMakeFiles/myslam.dir/frame.cpp.o: ../src/frame.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pj/pj/VSLAM_BA_with_eigen/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/myslam.dir/frame.cpp.o"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/myslam.dir/frame.cpp.o -c /home/pj/pj/VSLAM_BA_with_eigen/src/frame.cpp

src/CMakeFiles/myslam.dir/frame.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/myslam.dir/frame.cpp.i"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pj/pj/VSLAM_BA_with_eigen/src/frame.cpp > CMakeFiles/myslam.dir/frame.cpp.i

src/CMakeFiles/myslam.dir/frame.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/myslam.dir/frame.cpp.s"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pj/pj/VSLAM_BA_with_eigen/src/frame.cpp -o CMakeFiles/myslam.dir/frame.cpp.s

src/CMakeFiles/myslam.dir/mappoint.cpp.o: src/CMakeFiles/myslam.dir/flags.make
src/CMakeFiles/myslam.dir/mappoint.cpp.o: ../src/mappoint.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pj/pj/VSLAM_BA_with_eigen/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/CMakeFiles/myslam.dir/mappoint.cpp.o"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/myslam.dir/mappoint.cpp.o -c /home/pj/pj/VSLAM_BA_with_eigen/src/mappoint.cpp

src/CMakeFiles/myslam.dir/mappoint.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/myslam.dir/mappoint.cpp.i"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pj/pj/VSLAM_BA_with_eigen/src/mappoint.cpp > CMakeFiles/myslam.dir/mappoint.cpp.i

src/CMakeFiles/myslam.dir/mappoint.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/myslam.dir/mappoint.cpp.s"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pj/pj/VSLAM_BA_with_eigen/src/mappoint.cpp -o CMakeFiles/myslam.dir/mappoint.cpp.s

src/CMakeFiles/myslam.dir/map.cpp.o: src/CMakeFiles/myslam.dir/flags.make
src/CMakeFiles/myslam.dir/map.cpp.o: ../src/map.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pj/pj/VSLAM_BA_with_eigen/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/CMakeFiles/myslam.dir/map.cpp.o"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/myslam.dir/map.cpp.o -c /home/pj/pj/VSLAM_BA_with_eigen/src/map.cpp

src/CMakeFiles/myslam.dir/map.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/myslam.dir/map.cpp.i"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pj/pj/VSLAM_BA_with_eigen/src/map.cpp > CMakeFiles/myslam.dir/map.cpp.i

src/CMakeFiles/myslam.dir/map.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/myslam.dir/map.cpp.s"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pj/pj/VSLAM_BA_with_eigen/src/map.cpp -o CMakeFiles/myslam.dir/map.cpp.s

src/CMakeFiles/myslam.dir/camera.cpp.o: src/CMakeFiles/myslam.dir/flags.make
src/CMakeFiles/myslam.dir/camera.cpp.o: ../src/camera.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pj/pj/VSLAM_BA_with_eigen/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/CMakeFiles/myslam.dir/camera.cpp.o"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/myslam.dir/camera.cpp.o -c /home/pj/pj/VSLAM_BA_with_eigen/src/camera.cpp

src/CMakeFiles/myslam.dir/camera.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/myslam.dir/camera.cpp.i"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pj/pj/VSLAM_BA_with_eigen/src/camera.cpp > CMakeFiles/myslam.dir/camera.cpp.i

src/CMakeFiles/myslam.dir/camera.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/myslam.dir/camera.cpp.s"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pj/pj/VSLAM_BA_with_eigen/src/camera.cpp -o CMakeFiles/myslam.dir/camera.cpp.s

src/CMakeFiles/myslam.dir/config.cpp.o: src/CMakeFiles/myslam.dir/flags.make
src/CMakeFiles/myslam.dir/config.cpp.o: ../src/config.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pj/pj/VSLAM_BA_with_eigen/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object src/CMakeFiles/myslam.dir/config.cpp.o"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/myslam.dir/config.cpp.o -c /home/pj/pj/VSLAM_BA_with_eigen/src/config.cpp

src/CMakeFiles/myslam.dir/config.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/myslam.dir/config.cpp.i"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pj/pj/VSLAM_BA_with_eigen/src/config.cpp > CMakeFiles/myslam.dir/config.cpp.i

src/CMakeFiles/myslam.dir/config.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/myslam.dir/config.cpp.s"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pj/pj/VSLAM_BA_with_eigen/src/config.cpp -o CMakeFiles/myslam.dir/config.cpp.s

src/CMakeFiles/myslam.dir/feature.cpp.o: src/CMakeFiles/myslam.dir/flags.make
src/CMakeFiles/myslam.dir/feature.cpp.o: ../src/feature.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pj/pj/VSLAM_BA_with_eigen/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object src/CMakeFiles/myslam.dir/feature.cpp.o"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/myslam.dir/feature.cpp.o -c /home/pj/pj/VSLAM_BA_with_eigen/src/feature.cpp

src/CMakeFiles/myslam.dir/feature.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/myslam.dir/feature.cpp.i"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pj/pj/VSLAM_BA_with_eigen/src/feature.cpp > CMakeFiles/myslam.dir/feature.cpp.i

src/CMakeFiles/myslam.dir/feature.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/myslam.dir/feature.cpp.s"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pj/pj/VSLAM_BA_with_eigen/src/feature.cpp -o CMakeFiles/myslam.dir/feature.cpp.s

src/CMakeFiles/myslam.dir/frontend.cpp.o: src/CMakeFiles/myslam.dir/flags.make
src/CMakeFiles/myslam.dir/frontend.cpp.o: ../src/frontend.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pj/pj/VSLAM_BA_with_eigen/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object src/CMakeFiles/myslam.dir/frontend.cpp.o"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/myslam.dir/frontend.cpp.o -c /home/pj/pj/VSLAM_BA_with_eigen/src/frontend.cpp

src/CMakeFiles/myslam.dir/frontend.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/myslam.dir/frontend.cpp.i"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pj/pj/VSLAM_BA_with_eigen/src/frontend.cpp > CMakeFiles/myslam.dir/frontend.cpp.i

src/CMakeFiles/myslam.dir/frontend.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/myslam.dir/frontend.cpp.s"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pj/pj/VSLAM_BA_with_eigen/src/frontend.cpp -o CMakeFiles/myslam.dir/frontend.cpp.s

src/CMakeFiles/myslam.dir/backend.cpp.o: src/CMakeFiles/myslam.dir/flags.make
src/CMakeFiles/myslam.dir/backend.cpp.o: ../src/backend.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pj/pj/VSLAM_BA_with_eigen/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object src/CMakeFiles/myslam.dir/backend.cpp.o"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/myslam.dir/backend.cpp.o -c /home/pj/pj/VSLAM_BA_with_eigen/src/backend.cpp

src/CMakeFiles/myslam.dir/backend.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/myslam.dir/backend.cpp.i"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pj/pj/VSLAM_BA_with_eigen/src/backend.cpp > CMakeFiles/myslam.dir/backend.cpp.i

src/CMakeFiles/myslam.dir/backend.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/myslam.dir/backend.cpp.s"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pj/pj/VSLAM_BA_with_eigen/src/backend.cpp -o CMakeFiles/myslam.dir/backend.cpp.s

src/CMakeFiles/myslam.dir/viewer.cpp.o: src/CMakeFiles/myslam.dir/flags.make
src/CMakeFiles/myslam.dir/viewer.cpp.o: ../src/viewer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pj/pj/VSLAM_BA_with_eigen/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object src/CMakeFiles/myslam.dir/viewer.cpp.o"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/myslam.dir/viewer.cpp.o -c /home/pj/pj/VSLAM_BA_with_eigen/src/viewer.cpp

src/CMakeFiles/myslam.dir/viewer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/myslam.dir/viewer.cpp.i"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pj/pj/VSLAM_BA_with_eigen/src/viewer.cpp > CMakeFiles/myslam.dir/viewer.cpp.i

src/CMakeFiles/myslam.dir/viewer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/myslam.dir/viewer.cpp.s"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pj/pj/VSLAM_BA_with_eigen/src/viewer.cpp -o CMakeFiles/myslam.dir/viewer.cpp.s

src/CMakeFiles/myslam.dir/visual_odometry.cpp.o: src/CMakeFiles/myslam.dir/flags.make
src/CMakeFiles/myslam.dir/visual_odometry.cpp.o: ../src/visual_odometry.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pj/pj/VSLAM_BA_with_eigen/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object src/CMakeFiles/myslam.dir/visual_odometry.cpp.o"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/myslam.dir/visual_odometry.cpp.o -c /home/pj/pj/VSLAM_BA_with_eigen/src/visual_odometry.cpp

src/CMakeFiles/myslam.dir/visual_odometry.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/myslam.dir/visual_odometry.cpp.i"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pj/pj/VSLAM_BA_with_eigen/src/visual_odometry.cpp > CMakeFiles/myslam.dir/visual_odometry.cpp.i

src/CMakeFiles/myslam.dir/visual_odometry.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/myslam.dir/visual_odometry.cpp.s"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pj/pj/VSLAM_BA_with_eigen/src/visual_odometry.cpp -o CMakeFiles/myslam.dir/visual_odometry.cpp.s

src/CMakeFiles/myslam.dir/dataset.cpp.o: src/CMakeFiles/myslam.dir/flags.make
src/CMakeFiles/myslam.dir/dataset.cpp.o: ../src/dataset.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pj/pj/VSLAM_BA_with_eigen/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object src/CMakeFiles/myslam.dir/dataset.cpp.o"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/myslam.dir/dataset.cpp.o -c /home/pj/pj/VSLAM_BA_with_eigen/src/dataset.cpp

src/CMakeFiles/myslam.dir/dataset.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/myslam.dir/dataset.cpp.i"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pj/pj/VSLAM_BA_with_eigen/src/dataset.cpp > CMakeFiles/myslam.dir/dataset.cpp.i

src/CMakeFiles/myslam.dir/dataset.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/myslam.dir/dataset.cpp.s"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pj/pj/VSLAM_BA_with_eigen/src/dataset.cpp -o CMakeFiles/myslam.dir/dataset.cpp.s

src/CMakeFiles/myslam.dir/vertex.cpp.o: src/CMakeFiles/myslam.dir/flags.make
src/CMakeFiles/myslam.dir/vertex.cpp.o: ../src/vertex.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pj/pj/VSLAM_BA_with_eigen/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object src/CMakeFiles/myslam.dir/vertex.cpp.o"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/myslam.dir/vertex.cpp.o -c /home/pj/pj/VSLAM_BA_with_eigen/src/vertex.cpp

src/CMakeFiles/myslam.dir/vertex.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/myslam.dir/vertex.cpp.i"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pj/pj/VSLAM_BA_with_eigen/src/vertex.cpp > CMakeFiles/myslam.dir/vertex.cpp.i

src/CMakeFiles/myslam.dir/vertex.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/myslam.dir/vertex.cpp.s"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pj/pj/VSLAM_BA_with_eigen/src/vertex.cpp -o CMakeFiles/myslam.dir/vertex.cpp.s

src/CMakeFiles/myslam.dir/edge.cpp.o: src/CMakeFiles/myslam.dir/flags.make
src/CMakeFiles/myslam.dir/edge.cpp.o: ../src/edge.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pj/pj/VSLAM_BA_with_eigen/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building CXX object src/CMakeFiles/myslam.dir/edge.cpp.o"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/myslam.dir/edge.cpp.o -c /home/pj/pj/VSLAM_BA_with_eigen/src/edge.cpp

src/CMakeFiles/myslam.dir/edge.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/myslam.dir/edge.cpp.i"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pj/pj/VSLAM_BA_with_eigen/src/edge.cpp > CMakeFiles/myslam.dir/edge.cpp.i

src/CMakeFiles/myslam.dir/edge.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/myslam.dir/edge.cpp.s"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pj/pj/VSLAM_BA_with_eigen/src/edge.cpp -o CMakeFiles/myslam.dir/edge.cpp.s

src/CMakeFiles/myslam.dir/problem.cpp.o: src/CMakeFiles/myslam.dir/flags.make
src/CMakeFiles/myslam.dir/problem.cpp.o: ../src/problem.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pj/pj/VSLAM_BA_with_eigen/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Building CXX object src/CMakeFiles/myslam.dir/problem.cpp.o"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/myslam.dir/problem.cpp.o -c /home/pj/pj/VSLAM_BA_with_eigen/src/problem.cpp

src/CMakeFiles/myslam.dir/problem.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/myslam.dir/problem.cpp.i"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pj/pj/VSLAM_BA_with_eigen/src/problem.cpp > CMakeFiles/myslam.dir/problem.cpp.i

src/CMakeFiles/myslam.dir/problem.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/myslam.dir/problem.cpp.s"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pj/pj/VSLAM_BA_with_eigen/src/problem.cpp -o CMakeFiles/myslam.dir/problem.cpp.s

# Object files for target myslam
myslam_OBJECTS = \
"CMakeFiles/myslam.dir/frame.cpp.o" \
"CMakeFiles/myslam.dir/mappoint.cpp.o" \
"CMakeFiles/myslam.dir/map.cpp.o" \
"CMakeFiles/myslam.dir/camera.cpp.o" \
"CMakeFiles/myslam.dir/config.cpp.o" \
"CMakeFiles/myslam.dir/feature.cpp.o" \
"CMakeFiles/myslam.dir/frontend.cpp.o" \
"CMakeFiles/myslam.dir/backend.cpp.o" \
"CMakeFiles/myslam.dir/viewer.cpp.o" \
"CMakeFiles/myslam.dir/visual_odometry.cpp.o" \
"CMakeFiles/myslam.dir/dataset.cpp.o" \
"CMakeFiles/myslam.dir/vertex.cpp.o" \
"CMakeFiles/myslam.dir/edge.cpp.o" \
"CMakeFiles/myslam.dir/problem.cpp.o"

# External object files for target myslam
myslam_EXTERNAL_OBJECTS =

../lib/libmyslam.so: src/CMakeFiles/myslam.dir/frame.cpp.o
../lib/libmyslam.so: src/CMakeFiles/myslam.dir/mappoint.cpp.o
../lib/libmyslam.so: src/CMakeFiles/myslam.dir/map.cpp.o
../lib/libmyslam.so: src/CMakeFiles/myslam.dir/camera.cpp.o
../lib/libmyslam.so: src/CMakeFiles/myslam.dir/config.cpp.o
../lib/libmyslam.so: src/CMakeFiles/myslam.dir/feature.cpp.o
../lib/libmyslam.so: src/CMakeFiles/myslam.dir/frontend.cpp.o
../lib/libmyslam.so: src/CMakeFiles/myslam.dir/backend.cpp.o
../lib/libmyslam.so: src/CMakeFiles/myslam.dir/viewer.cpp.o
../lib/libmyslam.so: src/CMakeFiles/myslam.dir/visual_odometry.cpp.o
../lib/libmyslam.so: src/CMakeFiles/myslam.dir/dataset.cpp.o
../lib/libmyslam.so: src/CMakeFiles/myslam.dir/vertex.cpp.o
../lib/libmyslam.so: src/CMakeFiles/myslam.dir/edge.cpp.o
../lib/libmyslam.so: src/CMakeFiles/myslam.dir/problem.cpp.o
../lib/libmyslam.so: src/CMakeFiles/myslam.dir/build.make
../lib/libmyslam.so: /usr/local/lib/libopencv_dnn.so.3.4.16
../lib/libmyslam.so: /usr/local/lib/libopencv_highgui.so.3.4.16
../lib/libmyslam.so: /usr/local/lib/libopencv_ml.so.3.4.16
../lib/libmyslam.so: /usr/local/lib/libopencv_objdetect.so.3.4.16
../lib/libmyslam.so: /usr/local/lib/libopencv_shape.so.3.4.16
../lib/libmyslam.so: /usr/local/lib/libopencv_stitching.so.3.4.16
../lib/libmyslam.so: /usr/local/lib/libopencv_superres.so.3.4.16
../lib/libmyslam.so: /usr/local/lib/libopencv_videostab.so.3.4.16
../lib/libmyslam.so: /usr/local/lib/libopencv_viz.so.3.4.16
../lib/libmyslam.so: /usr/local/lib/libpango_glgeometry.so
../lib/libmyslam.so: /usr/local/lib/libpango_python.so
../lib/libmyslam.so: /usr/local/lib/libpango_scene.so
../lib/libmyslam.so: /usr/local/lib/libpango_tools.so
../lib/libmyslam.so: /usr/local/lib/libpango_video.so
../lib/libmyslam.so: /usr/lib/x86_64-linux-gnu/libgtest.a
../lib/libmyslam.so: /usr/lib/x86_64-linux-gnu/libgtest_main.a
../lib/libmyslam.so: /usr/local/lib/libglog.a
../lib/libmyslam.so: /usr/local/lib/libgflags.so.2.2.2
../lib/libmyslam.so: /usr/lib/x86_64-linux-gnu/libcxsparse.so
../lib/libmyslam.so: /usr/lib/x86_64-linux-gnu/libfmt.a
../lib/libmyslam.so: /usr/local/lib/libopencv_calib3d.so.3.4.16
../lib/libmyslam.so: /usr/local/lib/libopencv_features2d.so.3.4.16
../lib/libmyslam.so: /usr/local/lib/libopencv_flann.so.3.4.16
../lib/libmyslam.so: /usr/local/lib/libopencv_photo.so.3.4.16
../lib/libmyslam.so: /usr/local/lib/libopencv_video.so.3.4.16
../lib/libmyslam.so: /usr/local/lib/libopencv_videoio.so.3.4.16
../lib/libmyslam.so: /usr/local/lib/libopencv_imgcodecs.so.3.4.16
../lib/libmyslam.so: /usr/local/lib/libopencv_imgproc.so.3.4.16
../lib/libmyslam.so: /usr/local/lib/libopencv_core.so.3.4.16
../lib/libmyslam.so: /usr/local/lib/libpango_geometry.so
../lib/libmyslam.so: /usr/local/lib/libtinyobj.so
../lib/libmyslam.so: /usr/local/lib/libpango_plot.so
../lib/libmyslam.so: /usr/local/lib/libpango_display.so
../lib/libmyslam.so: /usr/local/lib/libpango_vars.so
../lib/libmyslam.so: /usr/local/lib/libpango_windowing.so
../lib/libmyslam.so: /usr/local/lib/libpango_opengl.so
../lib/libmyslam.so: /usr/lib/x86_64-linux-gnu/libGLEW.so
../lib/libmyslam.so: /usr/lib/x86_64-linux-gnu/libOpenGL.so
../lib/libmyslam.so: /usr/lib/x86_64-linux-gnu/libGLX.so
../lib/libmyslam.so: /usr/lib/x86_64-linux-gnu/libGLU.so
../lib/libmyslam.so: /usr/local/lib/libpango_image.so
../lib/libmyslam.so: /usr/local/lib/libpango_packetstream.so
../lib/libmyslam.so: /usr/local/lib/libpango_core.so
../lib/libmyslam.so: src/CMakeFiles/myslam.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/pj/pj/VSLAM_BA_with_eigen/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_15) "Linking CXX shared library ../../lib/libmyslam.so"
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/myslam.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/myslam.dir/build: ../lib/libmyslam.so

.PHONY : src/CMakeFiles/myslam.dir/build

src/CMakeFiles/myslam.dir/clean:
	cd /home/pj/pj/VSLAM_BA_with_eigen/build/src && $(CMAKE_COMMAND) -P CMakeFiles/myslam.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/myslam.dir/clean

src/CMakeFiles/myslam.dir/depend:
	cd /home/pj/pj/VSLAM_BA_with_eigen/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/pj/pj/VSLAM_BA_with_eigen /home/pj/pj/VSLAM_BA_with_eigen/src /home/pj/pj/VSLAM_BA_with_eigen/build /home/pj/pj/VSLAM_BA_with_eigen/build/src /home/pj/pj/VSLAM_BA_with_eigen/build/src/CMakeFiles/myslam.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/myslam.dir/depend
