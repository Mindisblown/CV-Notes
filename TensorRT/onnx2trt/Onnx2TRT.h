#pragma once
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include "npp.h"

#define MAX_WORKSPACE_SIZE (1UL << 31)
#define MIN_WORKSPACE_SIZE (1UL << 30)

const int max_batch = 4;
const Npp32f NppNormalizeScale[3] = { 0.0078125, 0.0078125, 0.0078125 };
const Npp32f NppShift[3] = { -1, -1, -1 };

// cuda api
#define CUDA_CHECK(status)                                             \
  {                                                                    \
    if (status != cudaSuccess) {                                       \
      printf("%s in %s at %d\n", cudaGetErrorString(status), __FILE__, \
             __LINE__);                                                \
      exit(-1);                                                        \
    }                                                                  \
  }


struct EngineInfo
{
	int Dimension;
	int Channel;
	int Width;
	int Height;
};

// 日志类
class Logger : public nvinfer1::ILogger
{
public:
	void set_verbosity(bool verbose)
	{
		_verbose = verbose;
	}

	// noexcept 保证函数不会抛出异常
	void log(nvinfer1::ILogger::Severity severity, char const* msg) noexcept override
	{
		if (_verbose)
		{
			switch (severity)
			{
			case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
				std::cerr << "Internal Error";
				break;
			case nvinfer1::ILogger::Severity::kERROR:
				std::cerr << "Error";
				break;
			case nvinfer1::ILogger::Severity::kWARNING:
				std::cerr << "Warning";
				break;
			case nvinfer1::ILogger::Severity::kINFO:
				std::cerr << "Info";
				break;
			default:
				std::cerr << "Unknown";
				break;
			}
		}

	}

private:
	bool _verbose = false;
};

class NetTensorRT
{
public:
	NetTensorRT(const std::string& model_path, std::vector<std::vector<float>>& calib_data, int mode, int batch_size);
	~NetTensorRT();

private:
	// 推理数据出路
	void PadInferData(const cv::Mat& img, int idx);
	// 推理
	void NetInference(int num);
	// 获取输入大小
	int GetBufferSize(nvinfer1::Dims d, nvinfer1::DataType t);

	// 加载engine
	int LoadEngine(const std::string& engine_path);
	// 加载engine信息
	void LoadEngineInfo();
	// 构建推理所需的空间大小
	void PrepareBufferSize();

	// 序列化
	void SerializeEngine(const std::string& engine_path);
	// 生成engine文件
	void GenerateEngine(const std::string& onnx_path, 
		const std::vector<std::vector<float>>& calib_data, int mode);

	void splitBgrDynamic(unsigned char* srcData, float* dstData, int w, int h, int num, float lower = -1.0f, float upper = 1.0f);
	void splitRgbDynamic(unsigned char* srcData, float* dstData, int w, int h, int num, float lower = -1.0f, float upper = 1.0f);

private:
	Logger _gLogger;
	nvinfer1::ICudaEngine* _engine;
	nvinfer1::IRuntime* _cuda_runtime;
	nvinfer1::IExecutionContext* _cuda_context;
	cudaStream_t _cuda_stream;
	EngineInfo _input;
	EngineInfo _output;


	std::vector<int> _inIdxs;
	std::vector<int> _inBufferSize;
	std::vector<int> _outIdxs;
	std::vector<int> _outBufferSize;

	unsigned int _idxInBind;
	unsigned int _idxOutBind;


	std::vector<std::vector<float> > ResultData;
	

	std::vector<float *> _hostBuffers;
	std::vector<void *> _deviceBuffers;

	void* gpu_img_buf;
	void* gpu_data_buf;

};

