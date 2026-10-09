#include "aegisvision/segmentation.hpp"
#include <iostream>
#include <limits>
using namespace aegisvision::vision;
void require(bool value,const char* message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejects(F f) { bool failed=false;try { f(); } catch(const std::invalid_argument&) { failed=true; } require(failed,"Invalid segmentation input accepted"); }
class FixtureSegmenter final: public ISegmenter {
public:
    std::vector<InstanceMask> next;
    std::vector<InstanceMask> segment(const cv::Mat&) override { return next; }
};
InstanceMask example(int x,float score=.9F) {
    return {{{static_cast<float>(x),0,static_cast<float>(x+2),2},"person",score,{},{}},
        {x,0,2,2},cv::Mat(2,2,CV_8UC1,cv::Scalar(255)),0};
}
void tracking_and_rle() {
    FixtureSegmenter source;SegmentationPipeline pipeline(source,true,.3F,1);
    cv::Mat image(4,8,CV_8UC3,cv::Scalar::all(0));
    source.next={example(0,.8F),example(4,.9F)};
    const auto first=pipeline.analyze(image);
    require(first[0].track_id && first[1].track_id && first[0].track_id!=first[1].track_id,"IDs missing");
    source.next={example(4,.7F),example(0,.95F)};
    const auto second=pipeline.analyze(image);
    require(second[0].track_id==first[1].track_id && second[1].track_id==first[0].track_id,"Rank change swapped masks");
    source.next.clear();require(pipeline.analyze(image).empty(),"Missing detection hallucinated mask");
    source.next={example(0)};
    require(pipeline.analyze(image)[0].track_id==first[0].track_id,"Short gap lost identity");
    source.next.clear();(void)pipeline.analyze(image);(void)pipeline.analyze(image);
    source.next={example(0)};require(pipeline.analyze(image)[0].track_id!=first[0].track_id,"Expired ID reused");
    FixtureSegmenter duplicates;duplicates.next={example(0),example(0)};
    SegmentationPipeline exact(duplicates,true);
    const auto same=exact.analyze(image);
    require(same[0].track_id!=same[1].track_id,"Identical boxes share one ID");
    SegmentationPipeline disabled(duplicates,false);
    require(disabled.analyze(image)[0].track_id==0,"Disabled tracking emitted ID");
    rejects([&]{(void)pipeline.analyze(cv::Mat(8,8,CV_8UC3));});
    const auto mask=example(0);
    require(mask_rle(mask,{3,3})==std::vector<std::uint32_t>({0,2,1,2,4}),"COCO column-major RLE wrong");
    auto empty=example(0);empty.mask.setTo(0);
    require(mask_rle(empty,{3,3})==std::vector<std::uint32_t>({9}),"Empty RLE wrong");
    empty.mask.setTo(7);rejects([&]{(void)mask_rle(empty,{3,3});});
}
int main() {
    try {
        tracking_and_rle();
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
