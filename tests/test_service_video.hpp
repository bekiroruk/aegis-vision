// Included inside test_service.cpp's fixture namespace.
void video_checks(httplib::Client& client,const fs::path& root,Segmenter& segmenter) {
    const Json request{{"type","segment_video"},{"path","test.avi"},{"max_frames",0}};
    for(const auto& bad:std::vector<Json>{
        {{"type","segment_video"},{"path","test.avi"},{"max_frames",1001}},
        {{"type","segment_video"},{"path","test.avi"},{"max_frames",-1}},
        {{"type","segment_video"},{"path","test.avi"},{"max_frames",1.5}},
        {{"type","segment_video"},{"path","test.avi"},{"output","outside.mp4"}},
        {{"type","segment_video"},{"path","../outside.avi"}}})
        (void)response(client.Post("/api/jobs",bad.dump(),"application/json"),400);
    const auto movie=completed(client,request);
    if(movie.at("state")!="succeeded") throw std::runtime_error("Video export failed: "+movie.at("error").dump());
    require(movie.at("result").at("frames")==10 && movie.at("result").at("tracking")==true &&
        movie.at("result").at("track_ids")==1,"Video export frame/track count differs");
    const auto base="/api/jobs/"+movie.at("id").get<std::string>();
    const auto range=client.Get(base+"/video.mp4",{{"Range","bytes=0-99"}});
    require(range && range->status==206 && range->body.size()==100 &&
        range->get_header_value("Content-Type")=="video/mp4","Export MP4 byte ranges unavailable");
    status(client.Get(base+"/video.mp4",{{"Range","bytes=999999999-"}}),416);
    const auto download=client.Get(base+"/video.mp4?download=1");
    require(download && download->status==200 && download->get_header_value("Content-Disposition").starts_with("attachment;"),"Video download disposition missing");
    const auto export_root=root/"media/analysis-output";
    const auto directory=export_root/movie.at("result").at("export_id").get<std::string>();
    require(!fs::exists(directory/"raw.avi"),"Raw intermediate retained after success");
    cv::VideoCapture exported((directory/"video.mp4").string());cv::Mat decoded;int count=0;
    while(exported.read(decoded)) {++count;require(decoded.cols==160 && decoded.rows==120,"Export dimensions differ");}
    require(count==10,"Export video did not decode fully");exported.release();
    const auto csv=client.Get(base+"/tracks.csv");
    require(csv && csv->body.find("0,1,person,")!=std::string::npos && csv->body.find("9,1,person,")!=std::string::npos,"Track report missing");
    status(client.Get(base+"/preview.jpg"),200);
    status(client.Get("/media/analysis-output/"+directory.filename().string()+"/video.mp4"),404);
    auto recursive=request;recursive["path"]="analysis-output/"+directory.filename().string()+"/video.mp4";
    (void)response(client.Post("/api/jobs",recursive.dump(),"application/json"),400);
    require(response(client.Get("/api/media")).at("videos").size()==1,"Exports leaked into input catalog");
    auto shorter=request;shorter["max_frames"]=3;
    const auto short_movie=completed(client,shorter);
    require(short_movie.at("result").at("frames")==3 && short_movie.at("result").at("stop_reason")=="requested_frame_limit","Video limit ignored");
    segmenter.mode=3;
    const auto running=response(client.Post("/api/jobs",request.dump(),"application/json"),202).at("id").get<std::string>();
    until([&]{return response(client.Get("/api/jobs/"+running)).at("progress").value("processed_frames",0)>0;});
    status(client.Get("/api/jobs/"+running+"/video.mp4"),404);
    (void)response(client.Post("/api/jobs/"+running+"/cancel","{}","application/json"));
    until([&]{return response(client.Get("/api/jobs/"+running)).at("state")=="cancelled";});
    status(client.Get("/api/jobs/"+running+"/video.mp4"),404);segmenter.mode=0;
    require(std::distance(fs::directory_iterator(export_root),fs::directory_iterator{})==2,"Cancelled export retained partial files");
    segmenter.mode=1;
    require(completed(client,request).at("state")=="failed","Export model error hidden");segmenter.mode=0;
    require(std::distance(fs::directory_iterator(export_root),fs::directory_iterator{})==2,"Failed export retained partial files");
    // Never remove or overwrite unrelated content to make room for another job.
    for(int i=0;i<30;++i) fs::create_directory(export_root/("quota-fixture-"+std::to_string(i)));
    require(completed(client,request).at("error").get<std::string>().find("quota")!=std::string::npos,"Output count quota ignored");
    for(int i=0;i<30;++i) fs::remove(export_root/("quota-fixture-"+std::to_string(i)));
    const auto before=fs::file_size(directory/"video.mp4");
    require(before==movie.at("result").at("bytes"),"Quota check changed existing output");
    const auto long_path=root/"media/too-long.avi";
    cv::VideoWriter long_writer(long_path.string(),cv::CAP_OPENCV_MJPEG,cv::VideoWriter::fourcc('M','J','P','G'),10,{32,32});
    require(long_writer.isOpened(),"Long video fixture unavailable");
    for(int i=0;i<1001;++i) long_writer.write(cv::Mat(32,32,CV_8UC3,cv::Scalar(0,0,0)));long_writer.release();
    auto long_request=request;long_request["path"]="too-long.avi";
    require(completed(client,long_request).at("error").get<std::string>().find("1000 frames")!=std::string::npos,"Full video silently truncated");
    fs::remove(long_path); // Exact newly-created test fixture.
}
