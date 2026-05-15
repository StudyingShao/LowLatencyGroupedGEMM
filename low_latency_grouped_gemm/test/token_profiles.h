// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace mga::test {

struct TokenProfileStats {
    int G;
    int active;
    int M_total;
    int max_mg;
};

inline std::vector<int> make_sanity64_token_counts() {
    // Small deterministic distribution for quick correctness-shaped smoke tests.
    return {
        /*  0..15  */ 5, 0, 12, 3,  0, 21, 7, 0,  9, 0, 18, 4,  2, 0, 11, 6,
        /* 16..31  */ 0, 14, 8, 0,  3, 0, 19, 1,  0, 5, 0,  10, 7, 2, 0, 16,
        /* 32..47  */ 4, 0,  0, 13, 6, 8, 0,  2,  0, 11, 0, 5,  9, 0, 17, 3,
        /* 48..63  */ 1, 0, 12, 0,  7, 4, 0,  6,  10, 0, 8, 0,  15, 2, 0, 5,
    };
}

inline std::vector<int> make_target192_token_counts() {
    // Exact 192-expert activation-count trace supplied by the target problem.
    return {
        /*   0.. 15 */ 5, 3, 2, 5, 5, 0, 5, 0, 1, 1, 0, 0, 4, 17, 0, 3,
        /*  16.. 31 */ 1, 2, 2, 14, 2, 1, 2, 1, 0, 4, 4, 0, 1, 0, 1, 1,
        /*  32.. 47 */ 3, 1, 5, 9, 2, 3, 6, 14, 1, 1, 0, 7, 1, 1, 1, 2,
        /*  48.. 63 */ 1, 1, 6, 14, 1, 3, 6, 21, 2, 4, 0, 4, 0, 3, 11, 7,
        /*  64.. 79 */ 1, 2, 0, 3, 2, 0, 2, 6, 0, 3, 0, 1, 3, 1, 4, 4,
        /*  80.. 95 */ 0, 0, 2, 5, 2, 0, 3, 5, 1, 5, 4, 2, 1, 3, 2, 4,
        /*  96..111 */ 9, 3, 1, 0, 2, 0, 3, 14, 2, 0, 1, 1, 2, 11, 0, 0,
        /* 112..127 */ 2, 2, 0, 2, 0, 1, 1, 3, 1, 3, 5, 6, 2, 1, 7, 3,
        /* 128..143 */ 3, 0, 7, 2, 0, 0, 2, 2, 1, 1, 0, 16, 3, 2, 1, 2,
        /* 144..159 */ 7, 2, 1, 0, 1, 0, 1, 14, 1, 5, 4, 0, 6, 0, 0, 2,
        /* 160..175 */ 16, 0, 0, 16, 0, 0, 5, 0, 0, 1, 1, 1, 2, 1, 1, 1,
        /* 176..191 */ 1, 0, 2, 1, 0, 0, 10, 4, 4, 3, 2, 2, 0, 11, 2, 0,
    };
}

inline std::vector<int> make_uniform_token_counts(int G, int M_per_expert) {
    if (G <= 0 || M_per_expert < 0) {
        std::fprintf(stderr,
                     "Invalid uniform profile: G=%d M_per_expert=%d\n",
                     G,
                     M_per_expert);
        std::exit(1);
    }
    return std::vector<int>(G, M_per_expert);
}

inline std::vector<int> make_token_counts(const char* profile) {
    if (std::strcmp(profile, "target192") == 0) return make_target192_token_counts();
    if (std::strcmp(profile, "sanity64") == 0) return make_sanity64_token_counts();
    constexpr const char* kUniform128Prefix = "uniform128_m";
    constexpr int kUniform128PrefixLen = 12;
    if (std::strncmp(profile, kUniform128Prefix, kUniform128PrefixLen) == 0) {
        const int m = std::atoi(profile + kUniform128PrefixLen);
        return make_uniform_token_counts(128, m);
    }
    constexpr const char* kUniform256Prefix = "uniform256_m";
    constexpr int kUniform256PrefixLen = 12;
    if (std::strncmp(profile, kUniform256Prefix, kUniform256PrefixLen) == 0) {
        const int m = std::atoi(profile + kUniform256PrefixLen);
        return make_uniform_token_counts(256, m);
    }
    std::fprintf(stderr,
                 "Unknown token profile '%s' "
                 "(allowed: target192, sanity64, uniform128_mX, uniform256_mX)\n",
                 profile);
    std::exit(1);
}

inline TokenProfileStats compute_token_profile_stats(const std::vector<int>& counts) {
    TokenProfileStats stats{};
    stats.G = static_cast<int>(counts.size());
    for (int c : counts) {
        if (c < 0) {
            std::fprintf(stderr, "Token profile has negative M_g=%d\n", c);
            std::exit(1);
        }
        if (c > 0) ++stats.active;
        stats.M_total += c;
        if (c > stats.max_mg) stats.max_mg = c;
    }
    return stats;
}

inline void validate_token_profile(const char* profile,
                                   const std::vector<int>& counts) {
    const TokenProfileStats stats = compute_token_profile_stats(counts);

    TokenProfileStats expected{};
    if (std::strcmp(profile, "target192") == 0) {
        expected = {192, 147, 560, 21};
    } else if (std::strcmp(profile, "sanity64") == 0) {
        expected = {64, 41, 331, 21};
    } else if (std::strncmp(profile, "uniform128_m", 12) == 0 ||
               std::strncmp(profile, "uniform256_m", 12) == 0) {
        return;
    } else {
        return;
    }

    if (stats.G != expected.G ||
        stats.active != expected.active ||
        stats.M_total != expected.M_total ||
        stats.max_mg != expected.max_mg) {
        std::fprintf(stderr,
                     "Token profile '%s' stats mismatch: "
                     "G=%d active=%d M_total=%d max_M_g=%d "
                     "(expected G=%d active=%d M_total=%d max_M_g=%d)\n",
                     profile,
                     stats.G,
                     stats.active,
                     stats.M_total,
                     stats.max_mg,
                     expected.G,
                     expected.active,
                     expected.M_total,
                     expected.max_mg);
        std::exit(1);
    }
}

inline std::vector<int> make_offsets(const std::vector<int>& counts) {
    std::vector<int> offsets(counts.size() + 1, 0);
    for (size_t g = 0; g < counts.size(); ++g) {
        offsets[g + 1] = offsets[g] + counts[g];
    }
    return offsets;
}

}  // namespace mga::test
