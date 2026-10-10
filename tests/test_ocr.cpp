#include "aegisvision/ocr.hpp"
#include <opencv2/imgcodecs.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace aegisvision::vision;
void require(bool condition,const char* message) {if(!condition) throw std::runtime_error(message);}
template<class F> void rejects(F action) {bool failed=false;try {action();} catch(const std::exception&) {failed=true;}require(failed,"Expected rejection");}
void units() {
    const int shape[]={7,1,37};cv::Mat logits(3,shape,CV_32F,cv::Scalar(-20));
    const int labels[]={11,11,0,11,0,12,12};
    for(int t=0;t<7;++t) logits.ptr<float>(t)[labels[t]]=20;
    const auto text=decode_english_ctc(logits);
    require(text.text=="aab" && text.confidence>.999F,"CTC blank/repeat decoding failed");
    logits.setTo(0);auto blank=decode_english_ctc(logits);
    require(blank.text.empty() && blank.confidence==0,"Blank should have no invented text or confidence");
    logits.ptr<float>(0)[1]=std::numeric_limits<float>::infinity();rejects([&]{decode_english_ctc(logits);});
    rejects([]{decode_english_ctc(cv::Mat::zeros(2,37,CV_32F));});
    cv::Mat image(32,100,CV_8UC3,cv::Scalar(10,20,30));
    const TextQuad quad={cv::Point2f{0,31},{0,0},{99,0},{99,31}};
    const auto crop=rectify_text(image,quad);require(cv::norm(image,crop,cv::NORM_INF)==0,"Rectification identity failed");
    auto bad=quad;bad[1].x=-1;rejects([&]{rectify_text(image,bad);});
    bad=quad;bad[0]=bad[1];rejects([&]{rectify_text(image,bad);});
    bad=quad;bad[0].x=std::numeric_limits<float>::quiet_NaN();rejects([&]{rectify_text(image,bad);});
    rejects([&]{rectify_text(cv::Mat(),quad);});
    require(ascii_edit_distance("welcome","welcome")==0,"Exact CER failed");
    require(ascii_edit_distance("kitten","sitting")==3,"Edit distance failed");
    require(ascii_edit_distance("","abc")==3 && ascii_edit_distance("abc","")==3,"Empty reference failed");
    rejects([]{ascii_edit_distance("\xc3\xbc","u");});
    rejects([]{PpocrCrnn("missing","missing",{0,1000});});
    rejects([]{PpocrCrnn("missing","missing");});
}
int main(int argc,char* argv[]) {
    try {
        units();
        if(argc==3) {
            const std::filesystem::path bundle=argv[1];
            cv::setNumThreads(1);
            PpocrCrnn ocr(bundle/"text_detection_en_ppocrv3_2023may.onnx",bundle/"text_recognition_CRNN_EN_2021sep.onnx");
            const auto input=cv::imread(argv[2]);std::atomic_bool cancel{true};
            rejects([&]{ocr.read(input,&cancel);});
            const auto result=ocr.read(input);
            require(result.size()==1 && result[0].recognition.text=="canon","Expected Canon scene reference");
            require(result[0].recognition.confidence>0 && result[0].recognition.confidence<=1,"Invalid real model confidence");
            require(ascii_edit_distance("canon",result[0].recognition.text)==0,"Canon smoke CER nonzero");
            const auto empty=ocr.read(cv::Mat(240,320,CV_8UC3,cv::Scalar(255,255,255)));
            require(empty.empty(),"Blank image should have no text regions");
            std::cout<<"Real PP-OCR/CRNN Canon smoke CER=0/5; blank image empty; precancel passed\n";
        } else require(argc==1,"Usage: test_ocr [BUNDLE CANON_IMAGE]");
        std::cout<<"OCR tests passed\n";return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
