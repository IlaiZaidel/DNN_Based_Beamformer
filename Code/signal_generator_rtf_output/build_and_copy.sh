#!/bin/bash
set -e

echo "🔧 Building signal_generator extension..."

# Step 1: Create and enter build directory
mkdir -p build
cd build

# Step 2: Run cmake + make
cmake ..
make -j$(nproc)

# Step 3: Find the generated .so file
SO_FILE=$(find . -name "signal_generator*.so" | head -n 1)

if [ -z "$SO_FILE" ]; then
    echo "❌ Failed to build .so file."
    exit 1
fi

# Step 4: Copy .so to parent dir
cp "$SO_FILE" ../
echo "✅ Copied $(basename "$SO_FILE") to ../"

# Done
echo "🎉 Build complete."
