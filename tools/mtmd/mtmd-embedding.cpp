#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

// Per-token embeddings of ONE unpadded sequence, evaluated in one pass (text tokens and media rows in one mixed batch,
// so a model's bidirectional layers see the whole sequence). The prompt uses mtmd's media marker; mtmd adds the
// vision delimiters. Output: row-major native float32, no header, one L2-normalized row per token.
//
//   llama-mtmd-embedding MODEL MMPROJ IMAGE_OR_VIDEO_OR_- PROMPT OUTPUT MAX_TOKENS
//   llama-mtmd-embedding MODEL MMPROJ --serve MAX_TOKENS
//     serve: one request per stdin line "MEDIA<TAB>PROMPT<TAB>OUTPUT" (MEDIA "-" for none; "\n", "\t" and "\\" escaped
//     in PROMPT); answers "OK <rows> <OUTPUT>" or "ERR <message>" on stdout, one line per request.

static std::string unescape(const std::string & s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            const char c = s[++i];
            out += c == 'n' ? '\n' : c == 't' ? '\t' : c;
        } else {
            out += s[i];
        }
    }
    return out;
}

static int embed_one(llama_context * lctx, const llama_model * model, mtmd_context * vision, const char * media_path,
                     const std::string & prompt, const char * out_path, std::string & err) {
    llama_memory_clear(llama_get_memory(lctx), true);
    auto opt = mtmd_helper_init_opt_default();
    opt.video_params.fps_target = 0; // preserve an already-sampled video
    mtmd_helper_bitmap_wrapper media = {};
    const bool has_media = std::strcmp(media_path, "-") != 0;
    if (has_media) media = mtmd_helper_bitmap_init_from_file(vision, media_path, false, opt);
    if (has_media && !media.bitmap) {
        if (media.video_ctx) mtmd_helper_video_free(media.video_ctx);
        err = "failed to open media";
        return -1;
    }
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const mtmd_input_text input = { prompt.c_str(), prompt.size(), false, true };
    const mtmd_bitmap * bitmaps[] = { media.bitmap };
    int rows = -1;
    llama_batch_ext * batch = nullptr;
    do {
        if (mtmd_tokenize(vision, chunks, &input, bitmaps, has_media ? 1 : 0)) { err = "tokenize failed"; break; }
        size_t total = 0;
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
            total += mtmd_input_chunk_get_n_tokens(mtmd_input_chunks_get(chunks, i));
        }
        if (!total || total > llama_n_ubatch(lctx) || total > llama_n_batch(lctx)) {
            err = "sequence has " + std::to_string(total) + " tokens; increase MAX_TOKENS";
            break;
        }
        batch = llama_batch_ext_init(lctx);
        llama_pos pos = 0;
        const int dim = llama_model_n_embd_inp(model);
        bool ok = true;
        for (size_t c = 0; ok && c < mtmd_input_chunks_size(chunks); ++c) {
            const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, c);
            const auto type = mtmd_input_chunk_get_type(chunk);
            const size_t count = mtmd_input_chunk_get_n_tokens(chunk);
            if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                size_t n = 0;
                const llama_token * tokens = mtmd_input_chunk_get_tokens_text(chunk, &n);
                for (size_t i = 0; ok && i < n; ++i) {
                    const int idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
                    const llama_pos p = pos++;
                    ok = idx >= 0 && llama_batch_ext_set_pos(batch, idx, &p) && llama_batch_ext_set_output_embd(batch, idx, true);
                }
            } else if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                const mtmd_image_tokens * image = mtmd_input_chunk_get_tokens_image(chunk);
                const float * embd = mtmd_encode_chunk(vision, chunk) ? nullptr : mtmd_get_output_embd(vision);
                ok = embd && image;
                for (size_t i = 0; ok && i < count; ++i) {
                    const int idx = llama_batch_ext_add_embd(batch, 0, { embd + i * dim, 1, (size_t) dim });
                    const auto p = mtmd_image_tokens_get_decoder_pos(image, pos, i);
                    const llama_pos positions[] = { (llama_pos) p.t, (llama_pos) p.y, (llama_pos) p.x, (llama_pos) p.z };
                    ok = idx >= 0 && llama_batch_ext_set_pos(batch, idx, positions) && llama_batch_ext_set_output_embd(batch, idx, true);
                }
                pos += mtmd_input_chunk_get_n_pos(chunk);
            } else {
                err = "unsupported media chunk";
                ok = false;
            }
        }
        if (!ok) { if (err.empty()) err = "batch build failed"; break; }
        if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch)) { err = "decode failed"; break; }
        FILE * out = std::fopen(out_path, "wb");
        if (!out) { err = "cannot open output"; break; }
        const int out_dim = llama_model_n_embd_out(model);
        bool wrote = true;
        for (size_t i = 0; wrote && i < total; ++i) {
            float * row = llama_get_embeddings_ith(lctx, (int32_t) i);
            if (!row) { wrote = false; break; }
            double norm = 0;
            for (int j = 0; j < out_dim; ++j) norm += (double) row[j] * row[j];
            const float scale = 1.0f / (float) std::fmax(std::sqrt(norm), 1e-12);
            for (int j = 0; j < out_dim; ++j) row[j] *= scale;
            wrote = std::fwrite(row, sizeof(float), out_dim, out) == (size_t) out_dim;
        }
        if (std::fclose(out) || !wrote) { err = "write failed"; break; }
        rows = (int) total;
    } while (false);
    if (batch) llama_batch_ext_free(batch);
    mtmd_input_chunks_free(chunks);
    if (media.bitmap) mtmd_bitmap_free(media.bitmap);
    if (media.video_ctx) mtmd_helper_video_free(media.video_ctx);
    return rows;
}

int main(int argc, char ** argv) {
    const bool serve = argc == 5 && std::strcmp(argv[3], "--serve") == 0;
    if (argc != 7 && !serve) {
        std::fprintf(stderr, "usage: %s MODEL MMPROJ IMAGE_OR_VIDEO_OR_- PROMPT OUTPUT MAX_TOKENS\n"
                             "       %s MODEL MMPROJ --serve MAX_TOKENS\n", argv[0], argv[0]);
        return 1;
    }
    const int limit = std::atoi(serve ? argv[4] : argv[6]);
    if (limit <= 0) return 1;
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = -1;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    auto cp = llama_context_default_params();
    cp.n_ctx = cp.n_batch = cp.n_ubatch = limit;
    cp.embeddings = true;
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cp.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL;
    llama_context * lctx = llama_init_from_model(model, cp);
    auto vp = mtmd_context_params_default();
    mtmd_context * vision = mtmd_init_from_file(argv[2], model, vp);
    if (!lctx || !vision) return 1;
    int rc = 0;
    if (!serve) {
        std::string err;
        const int rows = embed_one(lctx, model, vision, argv[3], argv[4], argv[5], err);
        if (rows < 0) { std::fprintf(stderr, "%s\n", err.c_str()); rc = 1; }
        else std::fprintf(stderr, "%d rows x %d float32\n", rows, llama_model_n_embd_out(model));
    } else {
        std::printf("READY %d\n", llama_model_n_embd_out(model));
        std::fflush(stdout);
        std::string line;
        while (std::getline(std::cin, line)) {
            const size_t a = line.find('\t'), b = a == std::string::npos ? a : line.find('\t', a + 1);
            if (b == std::string::npos) { std::printf("ERR malformed request\n"); std::fflush(stdout); continue; }
            const std::string media = line.substr(0, a), out = line.substr(b + 1);
            std::string err;
            const int rows = embed_one(lctx, model, vision, media.c_str(), unescape(line.substr(a + 1, b - a - 1)), out.c_str(), err);
            if (rows < 0) std::printf("ERR %s\n", err.c_str());
            else std::printf("OK %d %s\n", rows, out.c_str());
            std::fflush(stdout);
        }
    }
    mtmd_free(vision);
    llama_free(lctx);
    llama_model_free(model);
    llama_backend_free();
    return rc;
}
