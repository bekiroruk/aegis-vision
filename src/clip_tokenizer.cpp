#include "aegisvision/clip.hpp"
#include <nlohmann/json.hpp>
#include <unicode/normalizer2.h>
#include <unicode/regex.h>
#include <unicode/locid.h>
#include <unicode/utf8.h>
#include <fstream>
#include <map>
#include <limits>
#include <stdexcept>

namespace aegisvision {
namespace {
void check(UErrorCode code) {
    if(U_FAILURE(code)) throw std::runtime_error(std::string("ICU: ") + u_errorName(code));
}
std::string utf8(int codepoint) {
    icu::UnicodeString value(codepoint); std::string result;
    value.toUTF8String(result); return result;
}
}
struct ClipTokenizer::Impl {
    std::map<std::string, std::int64_t> vocab;
    std::map<std::pair<std::string,std::string>, std::size_t> ranks;
    std::array<std::string,256> bytes;
    std::unique_ptr<icu::RegexPattern> pattern;
};
ClipTokenizer::ClipTokenizer(const std::filesystem::path& path) : impl_(std::make_unique<Impl>()) {
    std::ifstream input(path);
    if(!input) throw std::runtime_error("Cannot read CLIP tokenizer.json");
    const auto config = nlohmann::json::parse(input);
    const auto& model = config.at("model");
    if(model.at("type") != "BPE" || model.at("end_of_word_suffix") != "</w>")
        throw std::invalid_argument("Expected CLIP BPE tokenizer");
    impl_->vocab = model.at("vocab").get<decltype(impl_->vocab)>();
    if(impl_->vocab.size()!=49408 || impl_->vocab.at("<|startoftext|>")!=49406 || impl_->vocab.at("<|endoftext|>")!=49407)
        throw std::invalid_argument("Unsupported CLIP vocabulary");
    std::size_t rank = 0;
    for(const auto& entry : model.at("merges")) {
        if(entry.is_array()) {
            impl_->ranks.emplace(std::make_pair(entry.at(0).get<std::string>(),entry.at(1).get<std::string>()),rank++);
            continue;
        }
        const auto merge = entry.get<std::string>();
        const auto separator = merge.find(' ');
        if(separator==std::string::npos) throw std::invalid_argument("Malformed BPE merge");
        impl_->ranks.emplace(std::make_pair(merge.substr(0,separator),merge.substr(separator+1)),rank++);
    }
    int extra = 256;
    for(int b=0; b<256; ++b) {
        const bool direct = (b>=33 && b<=126) || (b>=161 && b<=172) || b>=174;
        impl_->bytes[b] = utf8(direct ? b : extra++);
    }
    UErrorCode status = U_ZERO_ERROR;
    impl_->pattern.reset(icu::RegexPattern::compile(icu::UnicodeString::fromUTF8(
        R"(<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+)"), 0, status));
    check(status);
}
ClipTokenizer::~ClipTokenizer() = default;
ClipTokens ClipTokenizer::encode(std::string_view text) const {
    if(text.empty() || text.size()>16384) throw std::invalid_argument("Text must contain 1..16384 UTF-8 bytes");
    for(int32_t i=0; i<static_cast<int32_t>(text.size());) {
        UChar32 code; U8_NEXT(text.data(),i,static_cast<int32_t>(text.size()),code);
        if(code<0) throw std::invalid_argument("Text is not valid UTF-8");
    }
    // Explicit special tokens are not user text; prevent accidental early EOS pooling.
    if(text.find("<|startoftext|>")!=text.npos || text.find("<|endoftext|>")!=text.npos)
        throw std::invalid_argument("Reserved CLIP special token in query");
    UErrorCode status = U_ZERO_ERROR;
    const auto* nfc = icu::Normalizer2::getNFCInstance(status); check(status);
    icu::UnicodeString normalized;
    nfc->normalize(icu::UnicodeString::fromUTF8(icu::StringPiece(text.data(),static_cast<int32_t>(text.size()))),normalized,status);
    check(status); normalized.toLower(icu::Locale::getRoot());
    if(normalized.indexOf(icu::UnicodeString::fromUTF8("<|startoftext|>"))>=0 ||
       normalized.indexOf(icu::UnicodeString::fromUTF8("<|endoftext|>"))>=0)
        throw std::invalid_argument("Reserved CLIP special token in normalized query");
    std::unique_ptr<icu::RegexMatcher> matcher(impl_->pattern->matcher(normalized,status)); check(status);
    std::vector<std::int64_t> ids{49406};
    while(matcher->find(status)) {
        std::string word; matcher->group(status).toUTF8String(word); check(status);
        std::vector<std::string> pieces;
        for(unsigned char b : word) pieces.push_back(impl_->bytes[b]);
        if(pieces.empty()) continue;
        pieces.back()+="</w>";
        while(pieces.size()>1) {
            std::size_t best=std::numeric_limits<std::size_t>::max(), position=pieces.size();
            for(std::size_t i=0; i+1<pieces.size(); ++i) {
                const auto it=impl_->ranks.find({pieces[i],pieces[i+1]});
                if(it!=impl_->ranks.end() && it->second<best) { best=it->second; position=i; }
            }
            if(position==pieces.size()) break;
            pieces[position]+=pieces[position+1];
            pieces.erase(pieces.begin()+static_cast<std::ptrdiff_t>(position+1));
        }
        for(const auto& piece : pieces) {
            ids.push_back(impl_->vocab.at(piece));
            if(ids.size()>76) throw std::invalid_argument("Query exceeds CLIP's 75 content tokens; shorten it");
        }
    }
    check(status);
    if(ids.size()==1) throw std::invalid_argument("Query contains only whitespace");
    ids.push_back(49407);
    ClipTokens result; result.ids.fill(49407);
    for(std::size_t i=0; i<ids.size(); ++i) { result.ids[i]=ids[i]; result.mask[i]=1; }
    return result;
}
}
