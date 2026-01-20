#define _CRT_SECURE_NO_WARNINGS

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <immintrin.h>
#include <intrin.h>
#include <string>
#include <thread>
#include <vector>

static void avx_support() {
	int cpuInfo[4];
	__cpuid(cpuInfo, 7);

	std::cout << "AVX2 Support: ";
	if ((cpuInfo[1] & 1 << 5) != 0) std::cout << "Supported!\n";
	else std::cout << "Not Supported!\n";

	std::cout << "AVX512 Support: ";
	if ((cpuInfo[1] & 1 << 16) != 0) std::cout << "Supported!\n";
	else std::cout << "Not Supported!\n";
}

static void invert(const unsigned char* hImg, unsigned char* hOutImg, int width, int height, int channels) {
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
			for (int c = 0; c < channels; c++) {
				int idx = ((y * width) + x) * channels;
				hOutImg[idx + c] = 255 - hImg[idx + c];
			}
}

static void brightness(const unsigned char* hImg, unsigned char* hOutImg, int width, int height, int channels, int delta) {
	size_t totalBytes = width * height * channels;
	for (size_t i = 0; i < totalBytes; i++) {
		int y = static_cast<int>(hImg[i]) + delta;
		hOutImg[i] = static_cast<unsigned char>(std::clamp(y, 0, 255));
	}
}

static void avx2_invert(const unsigned char* hImg, unsigned char* hOutImg, int width, int height, int channels) {
	// Vector filled with 255 in every byte (byte specific)
	const __m256i v255 = _mm256_set1_epi8(static_cast<char>(255));

	// Calculate total bytes
	size_t totalBytes = width * height * channels;
	size_t i = 0;

	for (; i + 31 < totalBytes; i += 32) {
		// Load from hImg at offset i into AVX register v
		__m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(hImg + i));

		// Arithmetic Substract: 255 - x (unsigned saturating substract, byte specific)
		__m256i out = _mm256_subs_epu8(v255, v);

		// Store from AVX register out to hOutImg and offset i
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(hOutImg + i), out);
	}

	// Handle Remaining Bytes (if there are any...)
	for (; i < totalBytes; i++)
		hOutImg[i] = 255 - hImg[i];
}

static void multithread_invert(const unsigned char* hImg, unsigned char* hOutImg, int width, int height, int channels) {
	unsigned int numThreads = std::thread::hardware_concurrency();
	std::vector<std::thread> threads(numThreads);

	size_t totalBytes = width * height * channels;
	size_t chunkSize = totalBytes / numThreads;  // work per thread

	for (size_t i = 0; i < threads.size(); i++) {
		size_t startIdx = i * chunkSize;
		size_t endIdx = (i == threads.size() - 1) ? totalBytes : startIdx + chunkSize;
		threads[i] = std::thread([=]() {
			for (size_t j = startIdx; j < endIdx; j++)
				hOutImg[j] = 255 - hImg[j];
			});
	}

	for (auto& t : threads)
		t.join();
}

static void multithread_avx2_invert(const unsigned char* hImg, unsigned char* hOutImg, int width, int height, int channels) {
	// TO:DO
}

int main(int argc, char** argv) {
	// Display AVX2/AVX512 Support
	avx_support();

	// Define the image paths
	std::string evening{ "resources/evening.jpg" };
	std::string inverted_evening{ "resources/inverted_evening.jpg" };
	std::string pexels{ "resources/pexelsChristianHeitz.jpg" };
	std::string inverted_pexels{ "resources/inverted_pexelsChristianHeitz.jpg" };

	// Define image variables/attributes
	int width, height, channels;

	// Load the image
	unsigned char* hImg = stbi_load(pexels.c_str(), &width, &height, &channels, 3);

	// Check if the load was successfull
	if (!hImg) {
		std::cerr << "Error: Failed to load the image\n";
		return -1;
	}
	unsigned char* hOutImg = new unsigned char[width * height * channels];

	// Invert image and calculate time function with chrono
	auto start = std::chrono::high_resolution_clock::now();
	invert(hImg, hOutImg, width, height, channels);
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "Time taken (No SIMD, No Multithreading): " << std::chrono::duration<double, std::milli>(end - start).count() << "ms.\n";
	stbi_write_jpg(inverted_pexels.c_str(), width, height, channels, hOutImg, 95);

	// Invert image (Multithreading) and calculate time function with chrono
	start = std::chrono::high_resolution_clock::now();
	multithread_invert(hImg, hOutImg, width, height, channels);
	end = std::chrono::high_resolution_clock::now();
	std::cout << "Time taken (No SIMD, Multithreading): " << std::chrono::duration<double, std::milli>(end - start).count() << "ms.\n";
	stbi_write_jpg(inverted_pexels.c_str(), width, height, channels, hOutImg, 95);

	// Invert image (AVX2) and calculate time function with chrono
	start = std::chrono::high_resolution_clock::now();
	avx2_invert(hImg, hOutImg, width, height, channels);
	end = std::chrono::high_resolution_clock::now();
	std::cout << "Time taken (SIMD, No Multithreading): " << std::chrono::duration<double, std::milli>(end - start).count() << "ms.\n";
	stbi_write_jpg(inverted_pexels.c_str(), width, height, channels, hOutImg, 95);

	// Deallocate Memory
	delete[] hOutImg;
	stbi_image_free(hImg);

	// Terminate successfully
	return 0;
}