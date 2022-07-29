kernel void int_hist(global const uchar* A, global int* B) { //takes the input image, and output intensity histogram
	int id = get_global_id(0); // gets global id - current input value
	int bin_index = A[id]; //takes the current pixel intensity value as a bin index
	atomic_inc(&B[bin_index]); //stores each value of the current bin index to the intensity histogram
}

kernel void cum_hist(__global const int* A, global int* B, local int* scratch_1, local int* scratch_2) { // takes the intensity histogram, output cumulative histogram, and two local size buffers
	int id = get_global_id(0); // gets global id - current input value
	int lid = get_local_id(0); //gets local id
	int N = get_local_size(0); // gets local size
	local int* scratch_3;//used for buffer swap
	
	scratch_1[lid] = A[id]; //cache all N values from global memory to local memory

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (lid >= i)
			scratch_2[lid] = scratch_1[lid] + scratch_1[lid - i];
		else
			scratch_2[lid] = scratch_1[lid];

		barrier(CLK_LOCAL_MEM_FENCE); //clears the scratch bins

		//swaps the buffers
		scratch_3 = scratch_2;
		scratch_2 = scratch_1;
		scratch_1 = scratch_3;
	}
	B[id] = scratch_1[lid]; //copy the cache to output array
}

kernel void norm_hist(global const int* A, global int* B, const int image_size, const int bin_size) { // takes the cumulative histogram, output normalised histogram, image size and bin size
	int id = get_global_id(0); // gets global id - current input value
	int scale = image_size / bin_size; //calculates the scale by dividing the image size (total number of pixels) divided by the bin size (256)
	B[id] = A[id] / scale; // assigns normalised histogram by mapping the result of the cumulative histogram value divided by the scale variable
}

kernel void back_project(global const uchar* A, global const int* B, global uchar* C) { // takes the original image, normalised histogram and output image
	int id = get_global_id(0); // gets global id - current input value
	C[id] = B[A[id]]; // assigns the output image with the value of the original image pixel as the index to the look-up table (normalised cumulative histogram)
}