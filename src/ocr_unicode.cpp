#include "aegisvision/ocr.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>
namespace aegisvision::vision {
std::vector<std::uint32_t> utf8_codepoints(const std::string& text) {
    if(text.size()>1024) throw std::invalid_argument("UTF-8 text exceeds 1024 bytes");
    std::vector<std::uint32_t> result;
    for(std::size_t i=0;i<text.size();) {
        const auto head=static_cast<unsigned char>(text[i++]);
        std::uint32_t value;int rest;std::uint32_t minimum;
        if(head<0x80) {value=head;rest=0;minimum=0;}
        else if(head>=0xc2&&head<=0xdf) {value=head&31;rest=1;minimum=0x80;}
        else if(head>=0xe0&&head<=0xef) {value=head&15;rest=2;minimum=0x800;}
        else if(head>=0xf0&&head<=0xf4) {value=head&7;rest=3;minimum=0x10000;}
        else throw std::invalid_argument("Invalid UTF-8 leading byte");
        for(int j=0;j<rest;++j) {
            if(i==text.size()) throw std::invalid_argument("Truncated UTF-8");
            const auto c=static_cast<unsigned char>(text[i++]);
            if((c&0xc0)!=0x80) throw std::invalid_argument("Invalid UTF-8 continuation");
            value=(value<<6)|(c&63);
        }
        if(value<minimum||value>0x10ffff||(value>=0xd800&&value<=0xdfff)) throw std::invalid_argument("Invalid Unicode scalar");
        result.push_back(value);
    }
    return result;
}
std::size_t unicode_edit_distance(const std::string& reference,const std::string& prediction) {
    const auto a=utf8_codepoints(reference),b=utf8_codepoints(prediction);
    std::vector<std::size_t> row(b.size()+1);std::iota(row.begin(),row.end(),0);
    for(std::size_t i=0;i<a.size();++i) {
        auto diagonal=row[0];row[0]=i+1;
        for(std::size_t j=0;j<b.size();++j) {
            auto above=row[j+1];row[j+1]=std::min({above+1,row[j]+1,diagonal+(a[i]!=b[j])});diagonal=above;
        }
    }
    return row.back();
}
}
