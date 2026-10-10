#include "aegisvision/ocr_evaluation.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace aegisvision::vision;
void require(bool value) {if(!value) throw std::runtime_error("OCR evaluation assertion failed");}
template<class F> void rejects(F action) {bool failed=false;try{action();}catch(const std::invalid_argument&){failed=true;}require(failed);}
int main() {
    try {
        const OcrLabel a{{10,10,30,20},"canon"},b{{100,10,40,20},"welcome"};
        auto result=evaluate_ocr({a,b},{b,a});
        require(result.matches==2&&result.exact_matches==2&&result.reference_characters==12&&result.spatial_edits==0);
        auto wrong=a;wrong.text="can0n";result=evaluate_ocr({a,b},{wrong});
        require(result.matches==1&&result.exact_matches==0&&result.matched_edits==1&&result.spatial_edits==8);
        result=evaluate_ocr({a},{a,a});require(result.matches==1&&result.predictions==2&&result.spatial_edits==5);
        result=evaluate_ocr({a},{b});require(result.matches==0&&result.spatial_edits==12);
        result=evaluate_ocr({},{a});require(result.reference_characters==0&&result.spatial_edits==5);
        require(evaluate_ocr({},{}).matches==0);
        const OcrLabel half{{10,10,15,20},"canon"};
        require(evaluate_ocr({a},{half},.5).matches==1&&evaluate_ocr({a},{half},.5001).matches==0);
        require(evaluate_ocr({{{0,0,10,10},"a"},{{6,0,10,10},"b"}},
            {{{1,0,10,10},"b"},{{0,0,6,10},"a"}},.3).matches==2);
        auto empty=a;empty.text="";require(evaluate_ocr({a},{empty}).matched_edits==5);
        rejects([&]{evaluate_ocr({empty},{});});
        auto invalid=a;invalid.text="\xc3\xbc";rejects([&]{evaluate_ocr({invalid},{});});
        invalid=a;invalid.box.width=0;rejects([&]{evaluate_ocr({invalid},{});});
        invalid=a;invalid.box.x=std::numeric_limits<float>::quiet_NaN();rejects([&]{evaluate_ocr({invalid},{});});
        rejects([&]{evaluate_ocr({a},{a},0);});rejects([&]{evaluate_ocr(std::vector<OcrLabel>(65,a),{});});
        auto swapped=a;swapped.text="welcome";auto swapped2=b;swapped2.text="canon";
        require(evaluate_ocr({a,b},{swapped,swapped2}).exact_matches==0); // Must not match by text.
        auto aggregate=evaluate_ocr({a},{a});aggregate+=evaluate_ocr({b},{});
        require(aggregate.references==2&&aggregate.matches==1&&aggregate.spatial_edits==7);
        std::cout<<"OCR evaluation tests passed\n";return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
