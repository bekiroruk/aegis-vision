#include "aegisvision/ocr_multilingual.hpp"
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace aegisvision::vision;
void require(bool yes,const char* what) {if(!yes) throw std::runtime_error(what);}
template<class F> void rejects(F action) {bool failed=false;try{action();}catch(const std::exception&){failed=true;}require(failed,"Expected rejection");}
int main(int argc,char* argv[]) {
    try {
        require(utf8_codepoints("çğıİöşü").size()==7,"Unicode scalar count");
        require(unicode_edit_distance("İstanbul","Istanbul")==1,"Do not byte-count Turkish substitutions");
        require(unicode_edit_distance("ş","s")==1&&unicode_edit_distance("","ğ")==1,"Unicode insert/substitute");
        require(unicode_edit_distance("é","e\xcc\x81")==2,"No implicit Unicode normalization");
        for(const auto& invalid:{std::string("\xc0\x80"),std::string("\xed\xa0\x80"),std::string("\xf4\x90\x80\x80"),std::string("\xc3"),std::string("\x80")})
            rejects([&]{utf8_codepoints(invalid);});
        const std::vector<std::string> dict{"ş","ı"};std::vector<float> p(7*3,0);
        const int tokens[]{1,1,0,1,0,2,2};for(int i=0;i<7;++i)p[3*i+tokens[i]]=1;
        auto text=decode_unicode_ctc(p,7,dict);require(text.text=="şşı"&&text.confidence==1,"Unicode CTC repeat/blank/confidence");
        p[0]=std::numeric_limits<float>::quiet_NaN();rejects([&]{decode_unicode_ctc(p,7,dict);});
        rejects([&]{decode_unicode_ctc(std::vector<float>(3,1),1,dict);});
        rejects([&]{decode_unicode_ctc({},0,dict);});
        rejects([&]{PpocrMultilingual missing("no-such-bundle");});
        if(argc==3) {
            cv::setNumThreads(1);PpocrMultilingual model(argv[1]);const auto image=cv::imread(argv[2]);
            require(!image.empty(),"Missing synthetic reference");std::atomic_bool cancel{true};rejects([&]{model.read(image,&cancel);});
            auto regions=model.read(image);
            require(std::any_of(regions.begin(),regions.end(),[](const auto& r){return r.recognition.text=="Türkçe İstanbul";}),"Expected Turkish synthetic phrase");
            require(model.read(cv::Mat(240,320,CV_8UC3,cv::Scalar(255,255,255))).empty(),"Blank must have no text");
            std::sort(regions.begin(),regions.end(),[](const auto& a,const auto& b){return a.polygon[1].y<b.polygon[1].y;});
            require(regions.size()==2,"Synthetic fixture should have two regions");
            const std::vector<std::string> expected{"Türkçe İstanbul","Çığ ŞÖLEN ıİ öü 123"};std::size_t edits=0,characters=0;
            for(std::size_t i=0;i<2;++i) {edits+=unicode_edit_distance(expected[i],regions[i].recognition.text);characters+=utf8_codepoints(expected[i]).size();}
            std::cout<<"Synthetic two-line codepoint CER="<<edits<<'/'<<characters<<" (case/space sensitive; no normalization)\n";
        } else require(argc==1,"Expected optional BUNDLE SYNTHETIC_IMAGE");
        std::cout<<"Multilingual OCR tests passed\n";return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
