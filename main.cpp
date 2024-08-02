
#include "openvino/openvino.hpp"
#include <fstream>

const size_t TOKENIZER_MODEL_MAX_LENGTH = 32768;  // 'model_max_length' parameter from 'tokenizer_config.json'

int main(int argc, char* argv[]) {
    try {
        ov::Core core;
        core.add_extension("libopenvino_tokenizers.so");

        const std::string tokenizer_path = "/home/alikh/projects/alibaba_embeddings/gte-large-ov/openvino_tokenizer.xml";
        const std::string encoder_path = "/home/alikh/projects/alibaba_embeddings/gte-large-ov/openvino_model.xml";

        std::shared_ptr<ov::Model> tokenizer = core.read_model(tokenizer_path);
        std::shared_ptr<ov::Model> text_encoder = core.read_model(encoder_path);

        std::string test_str = "What is the capital of China?";

        ov::CompiledModel compiled_tokenizer = core.compile_model(tokenizer, "CPU");
        ov::CompiledModel compiled_encoder = core.compile_model(text_encoder, "CPU");

        ov::InferRequest tokenizer_req = compiled_tokenizer.create_infer_request();
        ov::InferRequest encoder_req = compiled_encoder.create_infer_request();

        tokenizer_req.set_input_tensor(ov::Tensor{ov::element::string, {1}, &test_str});
        tokenizer_req.infer();

        const ov::Tensor input_ids = tokenizer_req.get_output_tensor(0);
        const ov::Tensor token_type_ids = tokenizer_req.get_output_tensor(1);
        const ov::Tensor attention_mask = tokenizer_req.get_output_tensor(2);

        // int64_t* token_data = input_ids.data<int64_t>();
        // for (int i = 0; i < input_ids.get_size(); ++i)
        //     std::cout << token_data[i] << " ";
        // std::cout << std::endl;

        // int64_t* token_type_ids_data = token_type_ids.data<int64_t>();
        // for (int i = 0; i < input_ids.get_size(); ++i)
        //     std::cout << token_type_ids_data[i] << " ";
        // std::cout << std::endl;

        // int64_t* attention_mask_data = attention_mask.data<int64_t>();
        // for (int i = 0; i < attention_mask.get_size(); ++i)
        //     std::cout << attention_mask_data[i] << " ";
        // std::cout << std::endl;

        const ov::Shape input_ids_shape({1, input_ids.get_size()});
        ov::Tensor input_ids_input(ov::element::i64, input_ids_shape);
        std::copy_n(input_ids.data<std::int64_t>(), input_ids.get_size(), input_ids_input.data<int64_t>());

        const ov::Shape token_type_ids_shape({1, token_type_ids.get_size()});
        ov::Tensor token_type_ids_input(ov::element::i64, token_type_ids_shape);
        std::copy_n(token_type_ids.data<std::int64_t>(), token_type_ids.get_size(), token_type_ids_input.data<int64_t>());

        const ov::Shape attention_mask_shape({1, attention_mask.get_size()});
        ov::Tensor attention_mask_input(ov::element::i64, attention_mask_shape);
        std::copy_n(attention_mask.data<std::int64_t>(), attention_mask.get_size(), attention_mask_input.data<int64_t>());

        encoder_req.set_tensor("input_ids", input_ids_input);
        encoder_req.set_tensor("attention_mask", attention_mask_input);
        encoder_req.set_tensor("token_type_ids", token_type_ids_input);
        encoder_req.infer();
        const ov::Tensor text_embeddings = encoder_req.get_output_tensor(0);

        float* text_embeddings_data = text_embeddings.data<float>();

        std::ofstream outfile("cpp_res.txt");
        if (outfile.is_open()) {
            for (int i = 0; i < text_embeddings.get_size(); ++i)
                outfile << text_embeddings_data[i] << " ";
            outfile.close();
            std::cout << "Output saved to cpp_res.txt" << std::endl;
        } else {
            std::cerr << "Error opening file!" << std::endl;
        }

        return 0;


        // std::cout << text_embeddings.get_size() << std::endl;

        // for (int i = 0; i < text_embeddings.get_size(); ++i)
        //     std::cout << text_embeddings_data[i] << " ";
        // std::cout << std::endl;


        // std::cout << "Hello World" << std::endl;

    } catch (const std::exception& ex) {
        std::cerr << std::endl << "Exception occurred: " << ex.what() << std::endl << std::flush;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}