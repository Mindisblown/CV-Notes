#include "Onnx2TRT.h"

NetTensorRT::NetTensorRT(const std::string& model_path, std::vector<std::vector<float>>& calib_data, int mode, int batch_size)
{

}

NetTensorRT::~NetTensorRT()
{
	if (_engine)
	{
		_engine->destroy();
		_engine = 0;
	}

	if (_cuda_runtime)
	{
		_cuda_runtime->destroy();
		_cuda_runtime = 0;
	}
}

int NetTensorRT::LoadEngine(const std::string& engine_path)
{
	std::fstream file_ifstream(engine_path.c_str(), std::ios::binary);

	if (!file_ifstream.is_open())
	{
		std::cout << "file not exist" << std::endl;
		return -1;
	}

	_cuda_runtime = nvinfer1::createInferRuntime(_gLogger);

	file_ifstream.seekg(0, std::ios::end);
	const int model_size = file_ifstream.tellg();

	file_ifstream.seekg(0, std::ios::end);
	std::vector<char> model_memory(model_size);
	file_ifstream.read(model_memory.data(), model_size);

	_engine = _cuda_runtime->deserializeCudaEngine(model_memory.data(), model_size, nullptr);

	file_ifstream.close();
	return 0;

}

void NetTensorRT::LoadEngineInfo()
{
	int n_bindings = _engine->getNbBindings();
	for (int i = 0; i < n_bindings; ++i)
	{
		if (_engine->bindingIsInput(i))
		{
			nvinfer1::Dims i_dim = _engine->getBindingDimensions(i);
			switch (i_dim.nbDims)
			{
			case 2:
				_input.Dimension = i_dim.d[0];
				_input.Channel = i_dim.d[1];
				_input.Width = 0;
				_input.Height = 0;
				break;
			case 4:
				_input.Dimension = i_dim.d[0];
				_input.Channel = i_dim.d[1];
				_input.Height = i_dim.d[2];
				_input.Width = i_dim.d[3];
				break;
			}
		}
		else
		{
			nvinfer1::Dims i_dim = _engine->getBindingDimensions(i);
			switch (i_dim.nbDims)
			{
			case 2:
				_output.Dimension = i_dim.d[0];
				_output.Channel = i_dim.d[1];
				_output.Width = 0;
				_output.Height = 0;
				break;
			case 4:
				_output.Dimension = i_dim.d[0];
				_output.Channel = i_dim.d[1];
				_output.Height = i_dim.d[2];
				_output.Width = i_dim.d[3];
				break;
			}
		}
	}
}

void NetTensorRT::GenerateEngine(const std::string& onnx_path, const std::vector<std::vector<float>>& calib_data, int mode)
{
	std::cout << "Trying to generate trt engine from : " << onnx_path << std::endl;

	// inference builder
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(_gLogger);
	// config builder
	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
	// model builder
	nvinfer1::INetworkDefinition* network;

	// TRT版本7与8加载方式不同
#if nvinfer1::NV_TENSORRT_MAJOR
	network = builder->createNetwork();
#else
	// kEXPLICIT_BATCH 显式batch
	network = builder->createNetworkV2(1U << static_cast<int>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
#endif

	// onnx解析
	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, _gLogger);

	if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
	{
		throw std::runtime_error("Not Parse ONNX File");
	}
	else
	{
		std::cout << "Success Parse ONNX File" << std::endl;
	}

	// 获取输入Tensor及维度
	nvinfer1::ITensor* input = network->getInput(0);
	nvinfer1::Dims input_dims = input->getDimensions();
	nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();

	// 第一个batch维度为-1表示动态batch
	const bool is_dynamic = std::any_of(input_dims.d, input_dims.d + input_dims.nbDims, [](int dim) {return dim == -1; });
	if (is_dynamic)
	{
		std::cout << "Set Dynamic Batch : " << max_batch << std::endl;
		const char* input_name = input->getName();
		nvinfer1::Dims4 profile_min_dimension = nvinfer1::Dims4{1, input_dims.d[1], input_dims.d[2], input_dims.d[3]};
		nvinfer1::Dims4 profile_opt_dimension = nvinfer1::Dims4{max_batch, input_dims.d[1], input_dims.d[2], input_dims.d[3]};
		nvinfer1::Dims4 profile_max_dimension = nvinfer1::Dims4{max_batch, input_dims.d[1], input_dims.d[2], input_dims.d[3]};

		profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMIN, profile_min_dimension);
		profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kOPT, profile_opt_dimension);
		profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMAX, profile_max_dimension);

		if (profile->isValid()) config->addOptimizationProfile(profile);
	}
	else
	{
		builder->setMaxBatchSize(max_batch);
	}

	// 依据mode来选择int8还是float16
	if (mode >= 2)
	{
		std::cout << "Set Int8 Inference" << std::endl;
		if (!builder->platformHasFastInt8()) 
		{
			std::cout << "platform doesn't support int8 inference";
		}
		// Int8需要数据量化
		//nvifer1::Int8EntropyCalibrator* calibrator = nullptr;
	}
	
	if (mode & 1)
	{
		std::cout << "Set Float16 Inference" << std::endl;
		if (!builder->platformHasFastFp16()) 
		{
			std::cout << "platform doesn't support float16 inference";
		}
		config->setFlag(nvinfer1::BuilderFlag::kFP16);
	}

	for (unsigned long work_size = MAX_WORKSPACE_SIZE; work_size >= MIN_WORKSPACE_SIZE; work_size /= 2)
	{
		// 设置最大内存空间
		config->setMaxWorkspaceSize(work_size);

		_engine = builder->buildEngineWithConfig(*network, *config);

		if (!_engine)
		{
			std::cerr << "Failure creating engine from ONNX model" << std::endl
				<< "Current size is " << work_size << std::endl;
			continue;
		}
		else
		{
			std::cout << "Success creating engine from ONNX model" << std::endl
				<< "Final size is " << work_size << std::endl;
			break;
		}
	}

	if (!_engine) {
		throw std::runtime_error("ERROR: could not create engine from ONNX.");
	}
	else {
		std::cout << "Success creating engine from ONNX model" << std::endl;
	}

}

void NetTensorRT::SerializeEngine(const std::string& engine_path)
{
	std::cout << "Trying to serialize engine and save to : " << engine_path << std::endl;
	if (!_engine)
	{
		nvinfer1::IHostMemory* engine_plan = _engine->serialize();
		std::ofstream stream(engine_path.c_str(), std::ios::binary);

		if (stream)
		{
			stream.write(static_cast<char*>(engine_plan->data()), engine_plan->size());
		}
	}
}

void NetTensorRT::PrepareBufferSize()
{
	_cuda_context = _engine->createExecutionContext();
	CUDA_CHECK(cudaStreamCreate(&_cuda_stream));

	int n_bindings = _engine->getNbBindings();
	_deviceBuffers.clear();
	_deviceBuffers.resize(n_bindings);
	_hostBuffers.clear();
	_hostBuffers.resize(n_bindings);

	for (int i = 0; i < n_bindings; ++i)
	{
		int buffer_size = GetBufferSize(_engine->getBindingDimensions(i), _engine->getBindingDataType(i)) * max_batch;
		CUDA_CHECK(cudaMalloc(&_deviceBuffers[i], buffer_size));
		CUDA_CHECK(cudaMalloc(&_hostBuffers[i], buffer_size));

		if (_engine->bindingIsInput(i))
		{
			_idxInBind = i;
			_inIdxs.push_back(i);

			_inBufferSize.push_back(GetBufferSize(_engine->getBindingDimensions(_idxInBind),
				_engine->getBindingDataType(_idxInBind)) * max_batch);

		}
		else
		{
			_idxOutBind = i;
			_outIdxs.push_back(i);

			int tmpSize = GetBufferSize(_engine->getBindingDimensions(_idxOutBind),
				_engine->getBindingDataType(_idxOutBind)) * max_batch;
			_outBufferSize.push_back(tmpSize);

			ResultData[_outIdxs.size() - 1].resize(tmpSize / sizeof(float), 0);
		}

	}
	CUDA_CHECK(cudaMalloc(&gpu_img_buf, _input.Width * _input.Height * 3 * sizeof(uchar)));
	CUDA_CHECK(cudaMalloc(&gpu_data_buf, _input.Width * _input.Height * 3 * sizeof(float)));
}

void NetTensorRT::PadInferData(const cv::Mat& img, int idx)
{
	int _w = _input.Width;
	int _h = _input.Height;
	cv::Mat _temp;

	cv::resize(img, _temp, cv::Size(_w, _h));
	splitBgrDynamic(_temp.data, (float *)(_deviceBuffers[_idxInBind]), _w, _h, idx);

}

int NetTensorRT::GetBufferSize(nvinfer1::Dims d, nvinfer1::DataType t)
{
	int size = 1;
	for (int i = 1; i < d.nbDims; i++)
		size *= d.d[i];

	switch (t) {
	case nvinfer1::DataType::kINT32:
	case nvinfer1::DataType::kFLOAT:
		return size * 4;
	case nvinfer1::DataType::kHALF:
		return size * 2;
	case nvinfer1::DataType::kINT8:
	case nvinfer1::DataType::kBOOL:
		return size * 1;
	}
	return 0;
}

void NetTensorRT::splitBgrDynamic(unsigned char* srcData, float* dstData, int w, int h, int num, float lower /*= -1.0f*/, float upper /*= 1.0f*/)
{
	NppiSize dstSize = { w, h };
	int len = w * h * 3, lineWidth = w * 3, lineFdataW = lineWidth * sizeof(float);
	cudaMemcpy((Npp8u*)gpu_img_buf, srcData, len, cudaMemcpyHostToDevice);
	static int bgrOrder[3] = { 2, 1, 0 };

	nppiSwapChannels_8u_C3IR((Npp8u*)gpu_img_buf, lineWidth, dstSize, bgrOrder);    // bgr2rgb
	nppiConvert_8u32f_C3R((Npp8u*)gpu_img_buf, lineWidth, (Npp32f*)gpu_data_buf, lineFdataW, dstSize);  // 2float
	nppiMulC_32f_C3IR(NppNormalizeScale, (Npp32f*)gpu_data_buf, lineFdataW, dstSize);
	nppiAddC_32f_C3IR(NppShift, (Npp32f*)gpu_data_buf, lineFdataW, dstSize);

	// 所有数据放入buffer中
	Npp32f* dst_planes[3] = { dstData + (num * w * h * 3), dstData + (num * w * h * 3) + (w * h) , dstData + (num * w * h * 3) + (w * h * 2) };
	nppiCopy_32f_C3P3R((Npp32f*)gpu_data_buf, lineFdataW, dst_planes, w * sizeof(float), dstSize);//注意此处

}

void NetTensorRT::splitRgbDynamic(unsigned char* srcData, float* dstData, int w, int h, int num, float lower /*= -1.0f*/, float upper /*= 1.0f*/)
{
	NppiSize dstSize = { w, h };
	int len = w * h * 3, lineWidth = w * 3, lineFdataW = lineWidth * sizeof(float);
	cudaMemcpy((Npp8u*)gpu_img_buf, srcData, len, cudaMemcpyHostToDevice);

	nppiConvert_8u32f_C3R((Npp8u*)gpu_img_buf, lineWidth, (Npp32f*)gpu_data_buf, lineFdataW, dstSize); 
	nppiMulC_32f_C3IR(NppNormalizeScale, (Npp32f*)gpu_data_buf, lineFdataW, dstSize);
	nppiAddC_32f_C3IR(NppShift, (Npp32f*)gpu_data_buf, lineFdataW, dstSize);

	Npp32f* dst_planes[3] = { dstData + (num * w * h * 3), dstData + (num * w * h * 3) + (w * h) , dstData + (num * w * h * 3) + (w * h * 2) };
	nppiCopy_32f_C3P3R((Npp32f*)gpu_data_buf, lineFdataW, dst_planes, w * sizeof(float), dstSize);
}

void NetTensorRT::NetInference(int num)
{
	_cuda_context->setBindingDimensions(_idxInBind, nvinfer1::Dims4(num, _input.Channel, _input.Height, _input.Width));
	_cuda_context->enqueueV2(&_deviceBuffers[_idxInBind], _cuda_stream, nullptr);

	for (int i = 0; i < _output.Dimension; ++i)
	{
		CUDA_CHECK(cudaMemcpyAsync(ResultData[i].data(), _deviceBuffers[_outIdxs[i]],
			_outBufferSize[i], cudaMemcpyDeviceToHost, _cuda_stream));
	}
	CUDA_CHECK(cudaStreamSynchronize(_cuda_stream));
}