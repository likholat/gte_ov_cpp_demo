
#include "openvino/openvino.hpp"
#include <fstream>

int main(int argc, char *argv[])
{
    try
    {
        std::string device = "CPU"; // CPU, GPU or NPU
        std::string encoder_path = "../gte-large-ov/openvino_model.xml";
        std::string tokenizer_path = "../gte-large-ov/openvino_tokenizer.xml";
        std::string test_str;

        std::string ov_cache_dir = "../ov_cache";
        size_t niter = 100;

        if (argc == 4)
        {
            encoder_path = argv[1];
            tokenizer_path = argv[2];
            device = argv[3];
        }
        else if (argc != 1 && argc != 4)
        {
            std::cout << "Usage : " << argv[0] << " <path_to_embedding_model> <path_to_tokenizer_model> <device_name>" << std::endl;
            return EXIT_FAILURE;
        }

        // bool static_model = true;
        bool static_model = (device == "NPU") ? true : false;

        ov::Core core;
        core.add_extension("openvino_tokenizers.dll");

        core.set_property(ov::cache_dir(ov_cache_dir));

        ov::CompiledModel compiled_tokenizer = core.compile_model(tokenizer_path, "CPU");
        ov::CompiledModel compiled_encoder = core.compile_model(encoder_path, device);

        ov::InferRequest tokenizer_req = compiled_tokenizer.create_infer_request();
        ov::InferRequest encoder_req = compiled_encoder.create_infer_request();

        // default for test: "how to implement quick sort in python?"
        std::cout << "Input prompt:\n";
        std::getline(std::cin, test_str);

        tokenizer_req.set_input_tensor(ov::Tensor{ov::element::string, {1}, &test_str});
        tokenizer_req.infer();

        const ov::Tensor input_ids = tokenizer_req.get_output_tensor(0);
        const ov::Tensor token_type_ids = tokenizer_req.get_output_tensor(1);
        const ov::Tensor attention_mask = tokenizer_req.get_output_tensor(2);

        ov::Tensor input_ids_input;
        ov::Tensor token_type_ids_input;
        ov::Tensor attention_mask_input;

        if (static_model)
        {
            const ov::Shape input_ids_shape({1, encoder_req.get_input_tensor(0).get_shape()[1]});
            input_ids_input = ov::Tensor(ov::element::i64, input_ids_shape);
            std::fill_n(input_ids_input.data<int64_t>(), input_ids_input.get_size(), 0);
            std::copy_n(input_ids.data<std::int64_t>(), input_ids.get_size(), input_ids_input.data<int64_t>());

            const ov::Shape attention_mask_shape({1, encoder_req.get_input_tensor(1).get_shape()[1]});
            attention_mask_input = ov::Tensor(ov::element::i64, attention_mask_shape);
            std::fill_n(attention_mask_input.data<int64_t>(), attention_mask_input.get_size(), 0);
            std::copy_n(attention_mask.data<std::int64_t>(), attention_mask.get_size(), attention_mask_input.data<int64_t>());

            const ov::Shape token_type_ids_shape({1, encoder_req.get_input_tensor(2).get_shape()[1]});
            token_type_ids_input = ov::Tensor(ov::element::i64, token_type_ids_shape);
            std::fill_n(token_type_ids_input.data<int64_t>(), token_type_ids_input.get_size(), 0);
            std::copy_n(token_type_ids.data<std::int64_t>(), token_type_ids.get_size(), token_type_ids_input.data<int64_t>());
        }
        else
        {
            const ov::Shape input_ids_shape({1, input_ids.get_size()});
            input_ids_input = ov::Tensor(ov::element::i64, input_ids_shape);
            std::copy_n(input_ids.data<std::int64_t>(), input_ids.get_size(), input_ids_input.data<int64_t>());

            const ov::Shape attention_mask_shape({1, attention_mask.get_size()});
            attention_mask_input = ov::Tensor(ov::element::i64, attention_mask_shape);
            std::copy_n(attention_mask.data<std::int64_t>(), attention_mask.get_size(), attention_mask_input.data<int64_t>());

            const ov::Shape token_type_ids_shape({1, token_type_ids.get_size()});
            token_type_ids_input = ov::Tensor(ov::element::i64, token_type_ids_shape);
            std::copy_n(token_type_ids.data<std::int64_t>(), token_type_ids.get_size(), token_type_ids_input.data<int64_t>());
        }

        encoder_req.set_tensor("input_ids", input_ids_input);
        encoder_req.set_tensor("attention_mask", attention_mask_input);
        encoder_req.set_tensor("token_type_ids", token_type_ids_input);
        // Warm up
        encoder_req.infer();

        ov::Tensor text_embeddings = encoder_req.get_output_tensor(0);
        ov::Tensor result(text_embeddings.get_element_type(), text_embeddings.get_shape());
        text_embeddings.copy_to(result);

        if (static_model)
        {
            const ov::Shape static_shape = {result.get_shape()[0], input_ids.get_shape()[1], result.get_shape()[2]};
            result.set_shape(static_shape);
        }
        float *result_data = result.data<float>();

        std::ofstream outfile("cpp_res.txt");
        if (outfile.is_open())
        {
            for (int i = 0; i < result.get_size(); ++i)
                outfile << result_data[i] << " ";
            outfile.close();
            std::cout << "\nCpp output saved to cpp_res.txt\n" << std::endl;
        }
        else
        {
            std::cerr << "Error opening file!" << std::endl;
        }

        // Benchmark
        auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < niter; ++i)
            encoder_req.infer();
        auto end = std::chrono::steady_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Model: " << encoder_path << std::endl;
        std::cout << "Device: " << device << std::endl;
        std::cout << "Num of iterations: " << niter << std::endl;
        std::cout << "Mean inference time: " << duration / niter << " ms" << std::endl;
    }
    catch (const std::exception &ex)
    {
        std::cerr << std::endl
                  << "Exception occurred: " << ex.what() << std::endl
                  << std::flush;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
