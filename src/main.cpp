#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <onnxruntime_cxx_api.h>

class SimplePublisher : public rclcpp::Node
{
public:
    SimplePublisher() : Node("simple_publisher")
    {
        /*initialize ONNX Runtime*/
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test_onnx");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        const char *model_path = "/home/humanoid/DanielWorkspace/test_onnx_runtime/policy.onnx";
        session_ = Ort::Session(env, model_path, session_options);
        std::cout << "Model loaded successfully!" << std::endl;

        /*initialize input, output tensor*/
        input_shape_ = {1, 25};
        output_shape_ = {1, 6};
        input_data_ = {0.0f};
        output_data_ = {0.0f};
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        input_tensor_ = Ort::Value::CreateTensor<float>(
            memory_info,
            input_data_.data(),
            input_data_.size(),
            input_shape_.data(),
            input_shape_.size()
        );
        output_tensor_ = Ort::Value::CreateTensor<float>(
            memory_info,
            output_data_.data(),
            output_data_.size(),
            output_shape_.data(),
            output_shape_.size()
        );

        /*create publisher*/
        joint_state_cmd_publisher_ = this->create_publisher<sensor_msgs::msg::JointState>("joint_state_cmds", 10);
        
        /*create timer*/
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&SimplePublisher::publish_message, this));
    }

private:
    void publish_message()
    {
        /*fill in the input data*/
        std::vector<float> v{0.0, 0.0, 0.0};
        std::vector<float> w{0.0, 0.0, 0.0};

        std::copy(v.begin(), v.end(), input_data_.begin());
        std::copy(w.begin(), w.end(), std::back_inserter(input_data_));


        /*run the model*/
        static const char *input_names[] = {"obs"};
        static const char *output_names[] = {"actions"};
        Ort::RunOptions run_options;
        session_.Run(
            run_options,
            input_names, &input_tensor_, 1,
            output_names, &output_tensor_, 1
        );

        /*create message*/
        auto message = sensor_msgs::msg::JointState();
        message.header.stamp = this->now();
        std::copy(output_data_.begin(), output_data_.end(), std::back_inserter(message.position));
        // message.effort.push_back(duration.count());

        /*publish message*/
        joint_state_cmd_publisher_->publish(message);
        // RCLCPP_INFO(this->get_logger(), "Publishing joint state commands");'
    }

    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_cmd_publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    Ort::Session session_{nullptr};
    std::array<float, 25> input_data_;
    std::array<float, 6> output_data_;
    std::array<int64_t, 2> input_shape_;
    std::array<int64_t, 2> output_shape_;
    Ort::Value input_tensor_;
    Ort::Value output_tensor_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SimplePublisher>());
    rclcpp::shutdown();
    return 0;
}