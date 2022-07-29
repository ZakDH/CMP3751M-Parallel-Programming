/*A combination of atomic functions, parallel scan, and map functions are used to equalise the input images using parallel programming and functions.

The first function, which looks to compute the initial intensity histogram using the input image and output intensity histogram as parameters, 
assumes the bins are initialised at zero, sets the current bin index to the pixel value of each pixel in the input image, and then uses the 
'atomic_inc' function to add this pixel to the intensity histogram.

Instead of using the atomic add function for the second function, an inclusive scan function written by Hillis-Steele was used. 
To create a double-buffered variant of this inclusive scan, this function takes the intensity histogram, the output cumulative histogram, and two local buffers.

Because the cumulative histogram isn't scaled to 8-bit images when it's computed, 
the function 'norm_hist' is used to scale and normalise the cumulative histogram to 0-255 for 8-bit images. 
The normalisation is carried out by dividing each element/pixel in the cumulative histogram by the result of the image size (total number of pixels) 
divided by the bin size (256).

The cumulative histogram is now utilised as a look-up table for mapping the original image intensities onto the equalised output image, 
as it has been scaled for 8-bit images. The 'back project' function attempts to accomplish this by taking as parameters the original image, the output image, 
and the normalised cumulative histogram. The function here aims to use the original image pixel intensities as an index to the look-up table, 
assigning that new intensity value to each output pixel.

The user is eventually presented with a new image, as well as information on memory transfer, kernel execution time, and total program execution time.*/

#include <iostream>
#include <vector>
#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.ppm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char** argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test.pgm";

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	} 

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		int bin_count;
		std::cout << "Enter number of bins - 256 for 8-bit image" << std::endl;
		cin >> bin_count;
		CImg<unsigned char> image_input(image_filename.c_str());
		CImgDisplay disp_input(image_input, "input");
		const int image_size = image_input.size();

		std::vector<int> H(bin_count); //number of bins (length of buffer B)

		size_t local_size = H.size(); //local size set to number of bins
		
		//Part 3 - host operations
		//3.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runing on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//3.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}
		std::vector<int> int_histogram_buffer(H.size());
		size_t int_hist_size = int_histogram_buffer.size() * sizeof(int);
		//Part 4 - device operations

		//device - buffers
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size()); //should be the same as input image
		cl::Buffer int_histogram(context, CL_MEM_READ_WRITE, int_hist_size * sizeof(int));
		cl::Buffer cum_histogram(context, CL_MEM_READ_WRITE, int_hist_size * sizeof(int));
		cl::Buffer norm_histogram(context, CL_MEM_READ_WRITE, int_hist_size * sizeof(int));

		cl::Event event;
		cl::Event eventB;
		cl::Event eventC;
		cl::Event eventD;

		//4.1 Copy images to device memory
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0], NULL, &event);
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0], NULL, &eventB);
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0], NULL, &eventC);
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0], NULL, &eventD);


		//4.2 Setup and execute the kernel (i.e. device code)
		cl::Kernel kernel = cl::Kernel(program, "int_hist");
		kernel.setArg(0, dev_image_input);
		kernel.setArg(1, int_histogram);

		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NDRange(local_size), NULL, &event);
		//4.3 Copy the result from device to host
		queue.enqueueReadBuffer(int_histogram, CL_TRUE, 0, int_hist_size, &int_histogram_buffer.data()[0]);

		//std::cout << "Int_Histogram = " << int_histogram_buffer << std::endl;

		std::vector<int> cum_histogram_buffer(H.size());
		size_t cum_hist_size = cum_histogram_buffer.size() * sizeof(int);

		cl::Kernel kernel_cum = cl::Kernel(program, "cum_hist");
		kernel_cum.setArg(0, int_histogram);
		kernel_cum.setArg(1, cum_histogram);
		kernel_cum.setArg(2, cl::Local(local_size * sizeof(size_t)));//local memory size - arguments for kernel function!!
		kernel_cum.setArg(3, cl::Local(local_size * sizeof(size_t)));//local memory size - arguments for kernel function!! - for scan_add second buffer

		queue.enqueueNDRangeKernel(kernel_cum, cl::NullRange, cl::NDRange(cum_hist_size), cl::NDRange(local_size), NULL, &event);
		queue.enqueueReadBuffer(cum_histogram, CL_TRUE, 0, cum_hist_size, &cum_histogram_buffer[0]);

		//std::cout << "Cumulative_Histogram = " << cum_histogram_buffer << std::endl;

		std::vector<int> norm_histogram_buffer(H.size());
		size_t norm_hist_size = norm_histogram_buffer.size() * sizeof(int);

		cl::Kernel kernel_norm = cl::Kernel(program, "norm_hist");
		kernel_norm.setArg(0, cum_histogram);
		kernel_norm.setArg(1, norm_histogram);
		kernel_norm.setArg(2, int(image_input.size()));
		kernel_norm.setArg(3, int(H.size()));

		queue.enqueueNDRangeKernel(kernel_norm, cl::NullRange, cl::NDRange(norm_hist_size), cl::NDRange(local_size), NULL, &event);
		queue.enqueueReadBuffer(norm_histogram, CL_TRUE, 0, norm_hist_size, &norm_histogram_buffer[0]);

		//std::cout << "Norm_Histogram = " << norm_histogram_buffer << std::endl;

		vector<unsigned char> output_buffer(image_input.size());

		cl::Kernel kernel_output = cl::Kernel(program, "back_project");
		kernel_output.setArg(0, dev_image_input);
		kernel_output.setArg(1, norm_histogram);
		kernel_output.setArg(2, dev_image_output);

		queue.enqueueNDRangeKernel(kernel_output, cl::NullRange, cl::NDRange(image_input.size()), cl::NDRange(local_size), NULL, &event);
		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);

		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		
		std::cout << "Kernel execution time [ns]:" << event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << "Memory transfer:" << (event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>() +
			event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>() + event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>()
			+ event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) << std::endl;

		std::cout << GetFullProfilingInfo(event, ProfilingResolution::PROF_US) << std::endl;

		CImgDisplay disp_output(output_image, "output");

		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
			disp_input.wait(1);
			disp_output.wait(1);
		}
	}

	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}
