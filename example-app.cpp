#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

#define kIMAGE_SIZE 224
#define kCHANNELS 3
#define kTOP_K 3

class CStopWatch
{
public:
  CStopWatch() : m_start(std::chrono::system_clock::now())
  {}
  
  void Reset()
  {
    m_start = std::chrono::system_clock::now();
  }

  void GetElapsed()
  {
    m_end = std::chrono::system_clock::now();
    // Time in double
    std::chrono::duration<double> elapsed = m_end - m_start;
    std::chrono::nanoseconds nano = m_end - m_start;

    // Default expression
    std::cout << elapsed.count() << " seconds..." << std::endl;
    /*
    // In nano seconds
    std::cout << nano.count() << " nano seconds..." << std::endl;
    // In micro seconds
    std::chrono::microseconds micro = std::chrono::duration_cast<std::chrono::microseconds>(nano);
    std::cout << micro.count() << " micro seconds..." << std::endl;
    // In mili seconds
    std::chrono::milliseconds milli = std::chrono::duration_cast<std::chrono::milliseconds>(nano);
    std::cout << milli.count() << " milli seconds..." << std::endl;
    // In seconds
    std::chrono::seconds sec = std::chrono::duration_cast<std::chrono::seconds>(nano);
    std::cout << sec.count() << " seconds..." << std::endl;
    */
  }

private:
  std::chrono::system_clock::time_point m_start;
  std::chrono::system_clock::time_point m_end;
};

bool LoadImage(std::string file_name, cv::Mat &image) {
  image = cv::imread(file_name);  // CV_8UC3
  if (image.empty() || !image.data) {
    return false;
  }
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  std::cout << "== image size: " << image.size() << " ==" << std::endl;

  // scale image to fit
  cv::Size scale(kIMAGE_SIZE, kIMAGE_SIZE);
  cv::resize(image, image, scale);
  std::cout << "== simply resize: " << image.size() << " ==" << std::endl;

  // convert [unsigned int] to [float]
  image.convertTo(image, CV_32FC3, 1.0f / 255.0f);

  return true;
}

bool LoadImageNetLabel(std::string file_name,
                       std::vector<std::string> &labels) {
  std::ifstream ifs(file_name);
  if (!ifs) {
    return false;
  }
  std::string line;
  while (std::getline(ifs, line)) {
    labels.push_back(line);
  }
  return true;
}

int RunModel(std::string strPathLabel, std::string strPathModel, std::string strPathImage)
{
  std::vector<std::string> vLabels;
  if (LoadImageNetLabel(strPathLabel, vLabels)) {
    std::cout << "== Label loaded! Let's try it\n";
  } else {
    std::cerr << "Please check your label file path." << std::endl;
    return -1;
  }

  torch::jit::script::Module module;

  try {
    module = torch::jit::load(strPathModel);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the module \n";
    return -1;
  }

  cv::Mat imgInput;
  if (LoadImage(strPathImage, imgInput)) {
    auto input_tensor = torch::from_blob(imgInput.data, {1, kIMAGE_SIZE, kIMAGE_SIZE, kCHANNELS});
    input_tensor = input_tensor.permute({0, 3, 1, 2});
    input_tensor[0][0] = input_tensor[0][0].sub_(0.485).div_(0.229);
    input_tensor[0][1] = input_tensor[0][1].sub_(0.456).div_(0.224);
    input_tensor[0][2] = input_tensor[0][2].sub_(0.406).div_(0.225);

    //input_tensor = input_tensor.to(at::kCUDA);

    CStopWatch sw;
    torch::Tensor out_tensor = module.forward({input_tensor}).toTensor();
    sw.GetElapsed();

    auto results = out_tensor.sort(-1, true);
    auto softmaxs = std::get<0>(results)[0].softmax(0);
    auto indexs = std::get<1>(results)[0];

    for (int i = 0; i < kTOP_K; ++i) {
      auto idx = indexs[i].item<int>();
      std::cout << "    ============= Top-" << i + 1 << " =============" << std::endl;
      std::cout << "    Label:  " << vLabels[idx] << std::endl;
      std::cout << "    With Probability:  "
                << softmaxs[i].item<float>() * 100.0f << "%" << std::endl;
    }
  } else {
      std::cout << "Can't load the image, please check your path." << std::endl;
  }

  return 0;
}

int main() {

  std::string strPathLabel = "C:/projects/libtorch-exam/label.txt";
  std::string strPathImage = "C:/projects/libtorch-exam/images/dog.jpg";
  std::string strPathModel;

  std::cout << "\n---------------------------------------\n";
  strPathModel = "C:/projects/quantization/data/mobilenet_v2_float_scripted.pth";
  std::cout << strPathModel << std::endl;
  RunModel(strPathLabel, strPathModel, strPathImage);
  // 0.259003 seconds...

  std::cout << "\n---------------------------------------\n";
  strPathModel = "C:/projects/quantization/data/mobilenet_v2_opt_float_scripted.pth";
  std::cout << strPathModel << std::endl;
  RunModel(strPathLabel, strPathModel, strPathImage);
  // 0.113913 seconds...

  std::cout << "\n---------------------------------------\n";
  strPathModel = "C:/projects/quantization/data/mobilenet_v2_quant_per_ch_scripted.pth";
  std::cout << strPathModel << std::endl;
  RunModel(strPathLabel, strPathModel, strPathImage);
  // 0.287992 seconds...

  std::cout << "\n---------------------------------------\n";
  strPathModel = "C:/projects/quantization/data/mobilenet_v2_opt_quant_per_ch_scripted.pth";
  std::cout << strPathModel << std::endl;
  RunModel(strPathLabel, strPathModel, strPathImage);
  // 0.0345136 seconds...
}