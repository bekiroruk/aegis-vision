#include "aegisvision/segmentation.hpp"
#include <iostream>
#include <limits>
using namespace aegisvision::vision;
void require(bool value,const char* message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejects(F f) { bool failed=false;try { f(); } catch(const std::invalid_argument&) { failed=true; } require(failed,"Invalid segmentation input accepted"); }
int main() {
    try {
        YoloConfig config;config.input_size=32;config.max_detections=10;
        auto transform=prepare_yolo(cv::Mat(16,32,CV_8UC3,cv::Scalar::all(0)),32);
        const int shape[]={1,37,2}, proto_shape[]={1,32,8,8};
        cv::Mat pred(3,shape,CV_32F,cv::Scalar(0)),proto(4,proto_shape,CV_32F,cv::Scalar(0));
        for(int a=0;a<2;++a) {
            pred.ptr<float>(0,0)[a]=16;pred.ptr<float>(0,1)[a]=16;
            pred.ptr<float>(0,2)[a]=16;pred.ptr<float>(0,3)[a]=8;
            pred.ptr<float>(0,4)[a]=a==0?.9F:.8F;
            pred.ptr<float>(0,5)[a]=a==0?1.F:-1.F;
        }
        for(int y=0;y<8;++y)for(int x=0;x<8;++x)proto.ptr<float>(0,0,y)[x]=2;
        auto masks=decode_segmentation(pred,proto,transform,{"person"},config);
        require(masks.size()==1,"Same-class NMS or coefficient association failed");
        require(masks[0].region==cv::Rect(8,4,16,8),"Letterbox inverse mapping failed");
        require(cv::countNonZero(masks[0].mask)==128,"Positive logits did not produce mask");
        pred.ptr<float>(0,4)[1]=.95F;
        require(cv::countNonZero(decode_segmentation(pred,proto,transform,{"person"},config)[0].mask)==0,
            "NMS score sorting lost anchor-to-coefficient correspondence");
        pred.ptr<float>(0,4)[1]=.8F;
        pred.ptr<float>(0,5)[0]=-1;
        require(cv::countNonZero(decode_segmentation(pred,proto,transform,{"person"},config)[0].mask)==0,"Negative logits did not produce empty mask");
        pred.ptr<float>(0,4)[0]=0;pred.ptr<float>(0,4)[1]=0;
        require(decode_segmentation(pred,proto,transform,{"person"},config).empty(),"Empty frame emitted instances");
        rejects([&]{decode_segmentation(proto,pred,transform,{"person"},config);});
        auto invalid=transform;invalid.scale_x=1e100;
        rejects([&]{decode_segmentation(pred,proto,invalid,{"person"},config);});
        proto.ptr<float>()[0]=std::numeric_limits<float>::quiet_NaN();
        rejects([&]{decode_segmentation(pred,proto,transform,{"person"},config);});
        auto image=cv::Mat(16,32,CV_8UC3,cv::Scalar::all(0));
        require(paint_masks(image,masks).size()==image.size(),"Painting changed dimensions");
        masks[0].region.x=-1;rejects([&]{paint_masks(image,masks);});
        std::cout << "Segmentation decoding, NMS coefficients, padding, masks and validation passed\n";
        return 0;
    } catch(const std::exception& e) {std::cerr << e.what() << '\n';return 1;}
}
