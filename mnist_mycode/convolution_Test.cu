//#include <iostream>
//#include<cuda.h>
//#include <cudnn.h>
//#include <Windows.h>
//#include <cublas.h>
//#include <cublas_v2.h>
//
//using namespace std;
//
//void checkCUDNN(cudnnStatus_t status)
//{
//	if (status != CUDNN_STATUS_SUCCESS)
//		cout << "[ERROR] CUDNN" << status << endl;
//}
//
//void checkCUDA(cudaError_t error)
//{
//	if (error != CUDA_SUCCESS)
//		cout << "[ERROR] CUDA" << error << endl;
//}
//
//void print(char* title, float* src, int filter_num, int h, int w)
//{
//	cout << title << endl;
//	for (int i = 0; i < filter_num; i++) {
//		for (int y = 0; y < h; y++) {
//			for (int x = 0; x < w; x++) {
//				printf("%.0f ", src[i*h*w + y*w + x]);
//			}
//			cout << endl;
//		}
//		cout << endl;
//	}
//}
//
//int main()
//{
//	const int batch_count = 1; //입력 데이터 개수, 배치사이즈
//	const int in_channel = 2; //입력 데이터 채널 수
//	const int in_height = 4;// 입력 데이터 세로 길이
//	const int in_width = 4;// 입력 데이터 가로 길이
//	const int out_channel = 2; //출력 클래스 수
//	const int filter_width = 3; //컨벌루션 필터 가로 길이
//	const int filter_height = 3; // 컨벌루션 필터 세로 길이
//	const int filter_num = 1; //컨벌루션 필터 개수
//	const int padding_w = 1; //컨벌루션 패딩.
//	const int padding_h = 1;
//	const int stride_horizontal = 1;
//	const int stride_vertical = 1;
//	const int pool_window_w = 2;
//	const int pool_window_h = 2;
//	const int pool_padding_horizontal = 0;
//	const int pool_padding_vertical = 0;
//	const int pool_stride_horizontal = 2;
//	const int pool_stride_vertical = 2;
//	const int pool_w = in_width / pool_stride_horizontal;
//	const int pool_h = in_height / pool_stride_vertical;
//	const int src_len = batch_count*filter_num*in_height*in_width;
//	const int pool_len = batch_count*filter_num*pool_h*pool_w;
//	float inData_NCHW[batch_count][in_channel][in_height][in_width];
//	float inData_NHWC[batch_count][in_height][in_width][in_channel];
//	float outData[batch_count][filter_num][in_height][in_width];
//	float outDataFC[out_channel * 1];
//	float *inData_d;
//	float *outData_d, *outData1_d, *outDataFC_d;
//	float *filterData_d; // device 컨벌루션 필터 데이터
//	float *filterData2_d; // device FCN 필터 데이터
//	float *biasData_d;
//	float *hostArray = new float[src_len];
//	void* workSpace; //cuDNN 작업 중 사용할 버퍼 메모리.
//
//
//
//					 //입력 데이터 셋팅
//	for (int i = 0; i < in_channel; i++) {
//		for (int y = 0; y < in_height; y++) {
//			for (int x = 0; x < in_width; x++) {
//				inData_NCHW[0][i][y][x] = i * in_channel * in_width*in_height + y*in_height + x;
//			}
//		}
//	}
//
//	//입력 데이터 형태 변환
//	for (int i = 0; i < in_channel; i++) {
//		for (int y = 0; y < in_height; y++) {
//			for (int x = 0; x < in_width; x++) {
//				inData_NHWC[0][y][x][i] = inData_NCHW[0][i][y][x];
//			}
//		}
//	}
//
//	//필터 셋팅
//	float filterData[filter_num][in_channel][filter_height][filter_width] = {
//		{ { { 0.0f, 0.0f, 0.0f },{ 0.0f, 1.0f, 0.0f },{ 0.0f, 0.0f, 0.0f } },
//		{ { 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f, 1.0f },{ 0.0f, 0.0f, 0.0f } }
//		}
//	};
//
//	//FCN 가중치
//	//float filterData2[out_channel][filter_num][pool_h][pool_w] = { { { { 0.1f, 0.1f },{ 0.1f, 0.1f } } } ,{ { { 0.2f, 0.2f },{ 0.2f, 0.2f } } } };
//	
//	float filterData2[out_channel * pool_h * pool_w] = { 0.1f, 0.1f, 0.1f, 0.1f, 0.2f, 0.2f, 0.2f, 0.2f };
//
//	float biasData[filter_num] = { -20 };
//
//	cout << "in_NCHW" << endl;
//	for (int i = 0; i < in_channel; i++) {
//		for (int y = 0; y < in_height; y++) {
//			for (int x = 0; x < in_width; x++) {
//				printf("%.1f ", inData_NCHW[0][i][y][x]);
//			} cout << endl;
//		} cout << endl;
//	}
//
//	cout << "in_NHWC" << endl;
//	for (int y = 0; y < in_height; y++) {
//		for (int x = 0; x < in_width; x++) {
//			for (int i = 0; i < in_channel; i++) {
//				printf("%.1f ", inData_NHWC[0][y][x][i]);
//			} cout << endl;
//		} cout << endl;
//	}
//
//	cout << "weights" << endl;
//	for (int i = 0; i < in_channel; i++) {
//		for (int y = 0; y < filter_height; y++) {
//			for (int x = 0; x < filter_width; x++) {
//				printf("%.1f ", filterData[0][i][y][x]);
//			} cout << endl;
//		}cout << endl;
//	}
//
//	//GPU 메모리 할당
//	checkCUDA(cudaMalloc((void**)&inData_d, sizeof(inData_NHWC)));
//	checkCUDA(cudaMalloc((void**)&outData_d, sizeof(outData)));
//	checkCUDA(cudaMalloc((void**)&filterData_d, sizeof(filterData)));
//	checkCUDA(cudaMalloc((void**)&filterData2_d, sizeof(filterData2)));
//	checkCUDA(cudaMalloc((void**)&outData1_d, sizeof(outData)));
//	checkCUDA(cudaMalloc((void**)&biasData_d, sizeof(biasData)));
//	checkCUDA(cudaMalloc((void**)&outDataFC_d, sizeof(outDataFC)));
//
//	//CPU에서 GPU로 복사
//	checkCUDA(cudaMemcpy(inData_d, inData_NHWC, sizeof(inData_NHWC), cudaMemcpyHostToDevice));
//	checkCUDA(cudaMemcpy(filterData_d, filterData, sizeof(filterData), cudaMemcpyHostToDevice));
//	checkCUDA(cudaMemcpy(filterData2_d, filterData2, sizeof(filterData2), cudaMemcpyHostToDevice));
//	checkCUDA(cudaMemcpy(biasData_d, biasData, sizeof(biasData), cudaMemcpyHostToDevice));
//
//	
//
//	//CUDNN 배열
//	cudnnHandle_t cudnnHandle; // cuDNN 핸들러
//	cudnnTensorDescriptor_t inTensorDesc, outTensorDesc, biasTensorDesc, poolOutTensorDesc, sftTensorDesc; //데이터 구조체 선언
//	cudnnFilterDescriptor_t filterDesc, filterDesc2;
//	cudnnConvolutionDescriptor_t convDesc, convDesc2;
//	cudnnPoolingDescriptor_t poolDesc;
//	cudnnActivationDescriptor_t actDesc; //활성함수 구조체 선언
//
//	cublasHandle_t cublasHandle;
//
//										 //할당
//	checkCUDNN(cudnnCreate(&cudnnHandle));
//	checkCUDNN(cudnnCreateTensorDescriptor(&inTensorDesc));
//	checkCUDNN(cudnnCreateTensorDescriptor(&outTensorDesc));
//	checkCUDNN(cudnnCreateTensorDescriptor(&biasTensorDesc));
//	checkCUDNN(cudnnCreateTensorDescriptor(&poolOutTensorDesc));
//	checkCUDNN(cudnnCreateTensorDescriptor(&sftTensorDesc));
//	checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
//	checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc2));
//	checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
//	checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc2));
//	checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));
//	checkCUDNN(cudnnCreateActivationDescriptor(&actDesc));
//
//	cublasCreate(&cublasHandle);
//
//	cublasSetMatrix(out_channel, pool_h * pool_w, sizeof(*filterData2), filterData2, out_channel, filterData2_d, out_channel);
//
//	//초기화
//	//inData_NHWC정보
//	checkCUDNN(cudnnSetTensor4dDescriptor(inTensorDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, batch_count, in_channel, in_height, in_width));
//	checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, filter_num, in_channel, filter_height, filter_width));
//	checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc2, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_channel, filter_num, pool_h, pool_w));
//	checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, padding_h, padding_w, stride_vertical, stride_horizontal, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
//	//FCN 정보
//	checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc2, 0, 0, 2, 2, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
//	checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, pool_window_h, pool_window_w, pool_padding_vertical, pool_padding_horizontal, pool_stride_vertical
//		, pool_stride_horizontal));
//	checkCUDNN(cudnnSetTensor4dDescriptor(biasTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, filter_num, 1, 1));
//	checkCUDNN(cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0));
//
//	int out_n, out_c, out_h, out_w;
//	//입력데이터를 위 셋팅대로 컨벌루션을 했을 때 출력 구조
//	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, inTensorDesc, filterDesc, &out_n, &out_c, &out_h, &out_w));
//	printf("conv out shape (n x c x h x w) = (%d x %d x %d x %d)\n", out_n, out_c, out_h, out_w);
//	checkCUDNN(cudnnSetTensor4dDescriptor(outTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w));
//
//	//풀링 결과 구조 확인
//	checkCUDNN(cudnnGetPooling2dForwardOutputDim(poolDesc, outTensorDesc, &out_n, &out_c, &out_h, &out_w));
//	printf("pool out shape (n x c x h x w) = (%d x %d x %d x %d)\n", out_n, out_c, out_h, out_w);
//	//풀링 결과 구조 셋업
//	checkCUDNN(cudnnSetTensor4dDescriptor(poolOutTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w));
//
//	//FCN 결과 구조 확인
//	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc2, poolOutTensorDesc, filterDesc2, &out_n, &out_c, &out_h, &out_w));
//	printf("FCN out shape (n x c x h x w)= (%d x %d x %d x %d)\n", out_n, out_c, out_h, out_w);
//	checkCUDNN(cudnnSetTensor4dDescriptor(sftTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w));
//
//	//입력과 필터, 컨벌루션 패딩, 스트라이드가 주어졌을 때 가장 빠른 알고리즘 찾기
//	cudnnConvolutionFwdAlgo_t algo;
//	checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
//		inTensorDesc,
//		filterDesc,
//		convDesc,
//		outTensorDesc,
//		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
//		0,
//		&algo));
//
//	cout << "Fatest algorithm for conv0 = " << algo << endl;
//
//	cudnnConvolutionFwdAlgo_t algo2;
//	checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
//		poolOutTensorDesc,
//		filterDesc2,
//		convDesc2,
//		sftTensorDesc,
//		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
//		0,
//		&algo2));
//
//	cout << "Fatest algorithm for conv1 = " << algo2 << endl;
//
//	//위의 빠른 알고리즘을 사용할 경우 필요한 버퍼 데이터 크기 알아내기
//	size_t sizeinBytes = 0;
//	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
//		inTensorDesc,
//		filterDesc,
//		convDesc,
//		outTensorDesc,
//		algo,
//		&sizeinBytes));
//
//	cout << "workspace size (sizeinbytes):" << sizeinBytes << endl;
//	//계산 과정에서 버퍼 메모리가 필요한 경우 메모리 할당
//	if (sizeinBytes != 0) checkCUDA(cudaMalloc(&workSpace, sizeinBytes));
//
//	float alpha = 1.0f;
//	float beta = 0.0f;
//	//컨벌루션 시작
//	//"output = alpha * Op(input) + beta* output
//	//컨벌루션은 "output = 1 * Op(inpuit) + 0 * output"
//
//
//	checkCUDNN(cudnnConvolutionForward(cudnnHandle,
//		&alpha,
//		inTensorDesc,
//		inData_d,
//		filterDesc,
//		filterData_d,
//		convDesc,
//		algo,
//		workSpace,
//		sizeinBytes,
//		&beta,
//		outTensorDesc,
//		outData_d));
//
//	checkCUDA(cudaMemcpy(hostArray, outData_d, sizeof(float)* src_len, cudaMemcpyDeviceToHost));
//	print("conv out", hostArray, filter_num, in_height, in_width);
//
//	//add bias
//	beta = 1.0f;
//	checkCUDNN(cudnnAddTensor(cudnnHandle,
//		&alpha,
//		biasTensorDesc,
//		biasData_d,
//		&beta,
//		outTensorDesc,
//		outData_d));
//	checkCUDA(cudaMemcpy(hostArray, outData_d, sizeof(float)* src_len, cudaMemcpyDeviceToHost));
//	print("add bias out", hostArray, filter_num, in_height, in_width);
//
//	//activation_RELU
//	beta = 0.0f;
//	checkCUDNN(cudnnActivationForward(cudnnHandle,
//		actDesc,
//		&alpha,
//		outTensorDesc,
//		outData_d,
//		&beta,
//		outTensorDesc,
//		outData1_d));
//	//checkCUDA(cudaMemcpy(hostArray, outData1_d, sizeof(float)* src_len, cudaMemcpyDeviceToHost));
//	//print("RELU out", hostArray, filter_num, in_height, in_width);
//
//	//pooling
//	checkCUDNN(cudnnPoolingForward(cudnnHandle,
//		poolDesc,
//		&alpha,
//		outTensorDesc,
//		outData1_d,
//		&beta,
//		poolOutTensorDesc,
//		outData_d));
//	//checkCUDA(cudaMemcpy(hostArray, outData_d, sizeof(float)*pool_len, cudaMemcpyDeviceToHost));
//	//print("POOLING Out", hostArray, filter_num, pool_h, pool_w);
//
//	
//	alpha = 1.0f;
//	beta = 1.0f;
//	cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 1, out_channel, 4, &alpha, outData_d, 4, filterData2_d, 4, &beta, outDataFC_d, 1);
//	cublasGetMatrix(1, out_channel , sizeof(*outDataFC), outDataFC_d, out_channel, outDataFC ,out_channel);
//
//	checkCUDA(cudaMemcpy(hostArray, outDataFC_d, sizeof(float)*out_channel, cudaMemcpyDeviceToHost));
//	print("FC Out:", hostArray, out_channel, 1, 1);
//
//	beta = 0.0f;
//	//softmax
//	checkCUDNN(cudnnSoftmaxForward(cudnnHandle,
//		CUDNN_SOFTMAX_ACCURATE,
//		CUDNN_SOFTMAX_MODE_CHANNEL,
//		&alpha,
//		sftTensorDesc,
//		outDataFC_d,
//		&beta,
//		sftTensorDesc,
//		outData_d));
//
//	checkCUDA(cudaMemcpy(hostArray, outData_d, sizeof(float)*out_channel, cudaMemcpyDeviceToHost));
//	print("softmax out", hostArray, out_channel, 1, 1);
//
//
//	Sleep(1000000);
//
//	//메모리 해제
//	checkCUDNN(cudnnDestroyTensorDescriptor(inTensorDesc));
//	checkCUDNN(cudnnDestroyTensorDescriptor(outTensorDesc));
//	checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
//	checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc2));
//	checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
//	checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc2));
//	checkCUDNN(cudnnDestroyTensorDescriptor(biasTensorDesc));
//	checkCUDNN(cudnnDestroyTensorDescriptor(poolOutTensorDesc));
//	checkCUDNN(cudnnDestroyTensorDescriptor(sftTensorDesc));
//	checkCUDNN(cudnnDestroyPoolingDescriptor(poolDesc));
//	checkCUDNN(cudnnDestroyActivationDescriptor(actDesc));
//	checkCUDNN(cudnnDestroy(cudnnHandle));
//
//	checkCUDA(cudaFree(outDataFC_d));
//	checkCUDA(cudaFree(inData_d));
//	checkCUDA(cudaFree(outData_d));
//	checkCUDA(cudaFree(outData1_d));
//	checkCUDA(cudaFree(filterData2_d));
//	checkCUDA(cudaFree(filterData_d));
//	checkCUDA(cudaFree(biasData_d));
//	cublasDestroy(cublasHandle);
//
//	checkCUDA(cudaThreadSynchronize());
//	return 0;
//
//}