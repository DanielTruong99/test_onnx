#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <onnxruntime_cxx_api.h>
#define ALL_IS_FINITE(x) std::all_of(x.begin(), x.end(), [](float v) { return std::isfinite(v); })
#define ALL_IS_NAN(x) std::any_of(x.begin(), x.end(), [](float v) { return std::isnan(v); })

class SimplePublisher : public rclcpp::Node
{
public:
    SimplePublisher() : Node("simple_publisher")
    {
        /* initialize ONNX Runtime */
        this->initialize_onnx_runtime();

        /* initialize input, output tensor */
        this->initialize_input_output_tensors();

        /* initialize buffers */
        this->initialize_buffers();

        /* create publisher */
        joint_state_cmd_publisher_ = this->create_publisher<sensor_msgs::msg::JointState>("joint_state_cmds", 10);

        /* create subscriber */
        joint_state_subscriber_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "joint_feedback",
            10,
            std::bind(&SimplePublisher::joint_state_handler, this, std::placeholders::_1)
        );
        pose_cmd_subscriber_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "pose_cmd",
            10,
            std::bind(&SimplePublisher::pose_cmd_handler, this, std::placeholders::_1)
        );

        /* create timer */
        /* 50Hz */
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(20),
            std::bind(&SimplePublisher::control_loop, this)
        );
    }

private:
    void control_loop()
    {
        /* compute observation
            [joint_pos_rel, joint_vel, pose_cmd, previous_action]
        */
        std::copy(joint_pos_rel_.begin(), joint_pos_rel_.end(), input_data_.begin());
        std::copy(joint_vel_.begin(), joint_vel_.end(), input_data_.begin() + joint_pos_rel_.size());
        std::copy(pose_cmd_.begin(), pose_cmd_.end(), input_data_.begin() + joint_pos_rel_.size() + joint_vel_.size());
        std::copy(previous_action_.begin(), previous_action_.end(), input_data_.begin() + joint_pos_rel_.size() + joint_vel_.size() + pose_cmd_.size());


        /*run the model*/
        static const char *input_names[] = {"obs"};
        static const char *output_names[] = {"actions"};
        Ort::RunOptions run_options;
        session_.Run(
            run_options,
            input_names, &input_tensor_, 1,
            output_names, &output_tensor_, 1
        );

        /* cached the action computed from network */
        std::copy(output_data_.begin(), output_data_.end(), previous_action_.begin());

        /* post process action */
        static const float action_scale = 1.0f;
        for (size_t i = 0; i < output_data_.size(); ++i) 
        {
            joint_pos_cmds_[i] = output_data_[i] * action_scale + default_joint_pos_[i];
        }

        /*create and publish command message*/
        static sensor_msgs::msg::JointState message = sensor_msgs::msg::JointState();
        message.header.stamp = this->now();
        message.position.resize(6);
        std::copy(joint_pos_cmds_.begin(), joint_pos_cmds_.end(), message.position.begin());
        joint_state_cmd_publisher_->publish(message);
    }


    

    void joint_state_handler(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        /* check all isfinite and nan */
        // if(!ALL_IS_FINITE(msg->position) || !ALL_IS_NAN(msg->position) || 
        //    !ALL_IS_FINITE(msg->velocity) || !ALL_IS_NAN(msg->velocity))
        // {
        //     return;
        // }
        
        /* cache the topic data */
        std::copy(msg->position.begin(), msg->position.end(), joint_pos_.begin());
        std::copy(msg->velocity.begin(), msg->velocity.end(), joint_vel_.begin());

        /* calculate the relative joint position */
        for (size_t i = 0; i < joint_pos_.size(); ++i) 
        {
            joint_pos_rel_[i] = joint_pos_[i] - default_joint_pos_[i];
        }
    }

    void pose_cmd_handler(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        /* check all isfinite and nan */
        // if (!std::isfinite(msg->pose.position.x) || !std::isnan(msg->pose.orientation.x) ||
        //     !std::isfinite(msg->pose.position.y) || !std::isnan(msg->pose.orientation.y) ||
        //     !std::isfinite(msg->pose.position.z) || !std::isnan(msg->pose.orientation.z) ||
        //     !std::isfinite(msg->pose.orientation.w))
        // {
        //     return;
        // }

        /* cache the topic data */
        pose_cmd_[0] = msg->pose.position.x;
        pose_cmd_[1] = msg->pose.position.y;
        pose_cmd_[2] = msg->pose.position.z;
        pose_cmd_[3] = msg->pose.orientation.w;
        pose_cmd_[4] = msg->pose.orientation.x;
        pose_cmd_[5] = msg->pose.orientation.y;
        pose_cmd_[6] = msg->pose.orientation.z;
    }

    void initialize_onnx_runtime()
    {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test_onnx");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        const char *model_path = "/home/humanoid/DanielWorkspace/test_onnx_runtime/policy.onnx";
        session_ = Ort::Session(env, model_path, session_options);
        std::cout << "Model loaded successfully!" << std::endl;
    }

    void initialize_input_output_tensors()
    {
        /* initialize input, output tensor */
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
    }

    void initialize_buffers()
    {
        /* cached joint data */
        joint_pos_.resize(6, 0.0f);
        joint_vel_.resize(6, 0.0f);
        default_joint_pos_ = {0.0f, -1.712f, 1.712f, 0.0f, 0.0f, 0.0f};
        joint_pos_rel_.resize(6, 0.0f);

        /* cached pose data */
        pose_cmd_.resize(7, 0.0f);
        pose_cmd_ = {0.5161f, 0.1884f, 0.1705f, 0.6373f, 0.3063f, 0.6373f, -0.3063f};

        /* cached previous action */
        previous_action_.resize(6, 0.0f);

        /* cached joint position commands */
        joint_pos_cmds_.resize(6, 0.0f);
    }

    /* subscribers and publisher */
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_cmd_publisher_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_subscriber_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_cmd_subscriber_;

    /* cached topic data */
    std::vector<float> joint_pos_;
    std::vector<float> joint_vel_;
    std::vector<float> pose_cmd_;
    std::vector<float> default_joint_pos_;
    std::vector<float> joint_pos_rel_;
    std::vector<float> previous_action_;
    std::vector<float> joint_pos_cmds_;

    /* timer */
    rclcpp::TimerBase::SharedPtr timer_;

    /* ONNX Runtime */
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