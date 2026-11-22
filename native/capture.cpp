#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

constexpr int context_size = 1024;
constexpr int hidden = 2304;
constexpr int positions = 4;
constexpr int max_visits = 204;
constexpr std::string_view layer_prefix = "l_out-";

struct Capture {
    int tokens = 0;
    int visits = 0;
    float *output = nullptr;
    bool invalid = false;
    std::array<bool, max_visits> seen{};
};

bool read_layer(ggml_tensor *tensor, bool ask, void *opaque) noexcept {
    auto &capture = *static_cast<Capture *>(opaque);
    std::string_view name(ggml_get_name(tensor));
    if (!capture.output || !name.starts_with(layer_prefix)) {
        return false;
    }
    int slot = -1;
    const char *end = name.data() + name.size();
    const auto parsed = std::from_chars(name.data() + layer_prefix.size(), end, slot);
    if (ask) {
        return parsed.ptr == end && parsed.ec == std::errc{};
    }
    if (slot < 0 || slot >= capture.visits || capture.seen[slot] ||
        tensor->type != GGML_TYPE_F32 || tensor->ne[0] != hidden ||
        tensor->ne[1] != capture.tokens || !ggml_is_contiguous(tensor)) {
        capture.invalid = true;
        return false;
    }
    for (int index = 0; index < positions; ++index) {
        const auto token = (capture.tokens - 1) * (index + 1) / positions;
        const auto offset = slot * positions * hidden + index * hidden;
        ggml_backend_tensor_get(tensor, capture.output + offset, token * tensor->nb[1],
                                hidden * sizeof(float));
    }
    capture.seen[slot] = true;
    return true;
}

struct Runtime {
    Capture capture;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model{nullptr, llama_model_free};
    std::unique_ptr<llama_context, decltype(&llama_free)> context{nullptr, llama_free};
    std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch{nullptr, llama_batch_ext_free};
};

const llama_vocab * vocabulary(const Runtime &runtime) {
    return llama_model_get_vocab(runtime.model.get());
}

void report_error(char *error, int capacity, const std::exception &exception) noexcept {
    std::snprintf(error, capacity, "%s", exception.what());
}

} // namespace

extern "C" {

void lm_init() {
    llama_backend_init();
}

void *lm_open(const char *path, int passes, int threads, char *error, int capacity) {
    try {
        if (passes < 1 || passes > 16 || threads < 1) {
            throw std::runtime_error("Invalid pass or thread count");
        }
        auto runtime = std::make_unique<Runtime>();
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = std::numeric_limits<int>::max();
        std::array<llama_model_kv_override, 2> overrides{};
        std::snprintf(overrides[0].key, sizeof(overrides[0].key), "inkling2_loop.loop.count");
        overrides[0].tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        overrides[0].val_i64 = passes;
        model_params.kv_overrides = overrides.data();
        runtime->model.reset(llama_model_load_from_file(path, model_params));
        if (!runtime->model) {
            throw std::runtime_error("Model load failed");
        }
        const int visits = 12 + 12 * passes;
        if (llama_model_n_layer(runtime->model.get()) != visits ||
            llama_model_n_embd(runtime->model.get()) != hidden) {
            throw std::runtime_error("Loaded graph has an unexpected layer count or hidden size");
        }
        runtime->capture.visits = visits;
        auto params = llama_context_default_params();
        params.n_ctx = context_size;
        params.n_batch = context_size;
        params.n_ubatch = context_size;
        params.n_threads = threads;
        params.n_threads_batch = threads;
        params.cb_eval = read_layer;
        params.cb_eval_user_data = &runtime->capture;
        runtime->context.reset(llama_init_from_model(runtime->model.get(), params));
        if (!runtime->context) {
            throw std::runtime_error("Failed to allocate the inference context");
        }
        runtime->batch.reset(llama_batch_ext_init(runtime->context.get()));
        if (!runtime->batch) {
            throw std::runtime_error("Failed to allocate the input batch");
        }
        return runtime.release();
    } catch (const std::exception &exception) {
        report_error(error, capacity, exception);
        return nullptr;
    }
}

void lm_close(void *handle) {
    delete static_cast<Runtime *>(handle);
}

int lm_tokenize(void *handle, const char *text, int num_bytes, llama_token *ids, int capacity) {
    return llama_tokenize(vocabulary(*static_cast<Runtime *>(handle)), text, num_bytes,
                          ids, capacity, false, true);
}

int lm_forward(void *handle, llama_token *ids, int size, const llama_token *choices,
               float *choice_logits, float *active, char *error, int capacity) {
    auto &runtime = *static_cast<Runtime *>(handle);
    auto &capture = runtime.capture;
    try {
        if (size < positions || size > context_size) {
            throw std::runtime_error("Token count is outside the supported context");
        }
        const auto context = runtime.context.get();
        llama_memory_clear(llama_get_memory(context), true);
        capture.output = active;
        capture.tokens = size;
        capture.seen.fill(false);
        capture.invalid = false;
        auto *batch = runtime.batch.get();
        llama_batch_ext_clear(batch);
        for (llama_pos pos = 0; pos < size; ++pos) {
            const int index = llama_batch_ext_add_token(batch, 0, ids[pos]);
            if (index < 0 || !llama_batch_ext_set_pos(batch, index, &pos) ||
                !llama_batch_ext_set_output_logits(batch, index, pos == size - 1)) {
                throw std::runtime_error("Could not add an input token to the batch");
            }
        }
        const int status = llama_process(context, LLAMA_PROCESS_TYPE_DECODE, batch);
        capture.output = nullptr;
        if (status != 0 || capture.invalid ||
            !std::all_of(capture.seen.begin(), capture.seen.begin() + capture.visits,
                         [](bool seen) { return seen; })) {
            throw std::runtime_error("Forward did not capture each expected graph slot exactly once");
        }
        if (choices) {
            const float *logits = llama_get_logits_ith(context, -1);
            if (!logits) {
                throw std::runtime_error("Forward did not return final-token logits");
            }
            for (int i = 0; i < 4; ++i) {
                choice_logits[i] = logits[choices[i]];
            }
        }
        return 0;
    } catch (const std::exception &exception) {
        capture.output = nullptr;
        report_error(error, capacity, exception);
        return -1;
    }
}

} // extern "C"
