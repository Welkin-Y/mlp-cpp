# Compiler and flags
CXX = g++
CXX_VERSION = -std=c++2a
CXXFLAGS_DEBUG = -g -DDEBUG -DRANGE_CHECKING -Wall -Wextra
CXXFLAGS_RELEASE = -O3

# Targets and sources
TARGET_DEBUG = debug_executable
TARGET_RELEASE = executable
SOURCES = project2_a.cpp

# Build debug target
debug: $(SOURCES)
	$(CXX) $(CXX_VERSION) $(CXXFLAGS_DEBUG) -o $(TARGET_DEBUG) $(SOURCES)

# Build release target
release: $(SOURCES)
	$(CXX) $(CXX_VERSION) $(CXXFLAGS_RELEASE) -o $(TARGET_RELEASE) $(SOURCES)

# Clean build artifacts
clean:
	rm -f $(TARGET_DEBUG) $(TARGET_RELEASE)

# Default target
all: debug release
