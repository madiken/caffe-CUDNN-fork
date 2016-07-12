#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
using namespace cv;
using namespace std;

#include<fstream>

#include <caffe/caffe.hpp>
#include <caffe/util/io.hpp>
#include <caffe/blob.hpp>

using namespace caffe;



const int IMAGE_H = 160;
const int IMAGE_W = 60;
const int PARTS_NUMBER = 3;
const int PARTS_OVERLAP = 10;

vector<Mat> breakImage(Mat image, int stripes_num, int overlap = 0){
    vector<Mat> parts;

    int overlap_stripes_num = stripes_num - 1;
    int part_height = (image.size().height + overlap_stripes_num * overlap)/stripes_num;

    for(int i = 0; i<stripes_num; i++){
        int start = i * part_height - max((i) * overlap, 0);
        int end = min(image.size().height, start + part_height);
        parts.push_back(image(Range(start,end), Range::all()));            
    }
    return parts;
}



class Predictor{

  public : 
     Predictor(string model_file, string weights_file) : 
              image_h(IMAGE_H), image_w(IMAGE_W), parts_number(PARTS_NUMBER), parts_overlap(PARTS_OVERLAP){
          caffe_net.reset(new Net<float>(model_file, caffe::TEST));
          caffe_net->CopyTrainedLayersFrom(weights_file);
     }



     vector<vector<float> > makePredictionListForBatch(vector<cv::Mat> images){
         vector<vector<cv::Mat> > images_parts = preprocessImages(images);         

         Blob<float>* input_layer = caffe_net->input_blobs()[0];
        
         input_layer->Reshape(images.size(), 3*parts_number, images_parts[0][0].size().height, images_parts[0][0].size().width);
         caffe_net->Reshape();
         

         
      
         std::vector<cv::Mat> input_channels;
         WrapInputLayer(&input_channels); 
         writeToInputChannels(images_parts, &input_channels);

         Blob<float>* output_layer = caffe_net->ForwardPrefilled()[0]; 
        
         //get the result 
         vector<vector<float> >  result;
         for (int i = 0; i < images.size(); i++){
             const float* begin = output_layer->cpu_data() + output_layer->channels() * i ;
             const float* end = begin + output_layer->channels();
 
             vector<float> descriptor =  std::vector<float>(begin, end);
             result.push_back(descriptor);
         } 
         return result; 
     }

     vector<vector<float> > makePredictionList(vector<cv::Mat> images, int batch_size = 128){
         int batches_number =  images.size()/batch_size + 1;
         vector< vector<float> > descriptors;  
         for (int i = 0; i < batches_number; i++){
              int start = i*batch_size;
              int end = min((i+1)*batch_size, (int)images.size());
              if ( start >= end )
                  break;
              vector<cv::Mat> batch = vector<cv::Mat>(images.begin() + start, images.begin() + end);
              vector< vector<float> > batch_result =  makePredictionListForBatch(batch);
              descriptors.insert(descriptors.end(), batch_result.begin(), batch_result.end());
         } 
  

     
         return descriptors;
     }


     vector<vector<float> > makePredictionList(vector<string> filenames){
    //make images vector
          vector<cv::Mat> images; 
          for (vector<string>::iterator it = filenames.begin(); it!= filenames.end(); it++){
              Mat image = cv::imread(*it, CV_LOAD_IMAGE_COLOR);   // Read the file
              if(!image.data)                              // Check for invalid input
              {
                  string error = "Could not open or find the image ";
                  throw error.append(*it);
              }
              images.push_back(image);      
          }
          return makePredictionList(images);
     }



  private :
    shared_ptr<caffe::Net<float> > caffe_net;
    const int image_h;
    const int image_w;
    const int parts_number;
    const int parts_overlap;

    void WrapInputLayer(std::vector<cv::Mat>* input_channels) {
        Blob<float>* input_layer = caffe_net->input_blobs()[0];

        int width = input_layer->width();
        int height = input_layer->height();
        float* input_data = input_layer->mutable_cpu_data();
        for (int n = 0; n < input_layer->num(); ++n){
            for (int i = 0; i < input_layer->channels(); ++i) {
                cv::Mat channel(height, width, CV_32FC1, input_data);
                input_channels->push_back(channel);
                input_data += width * height;
            }
        }
    }
    void writeToInputChannels(const vector< vector<cv::Mat> >& images_parts, std::vector<cv::Mat>* input_channels){
        int image_counter = 0;
       
        for (vector< vector<cv::Mat> >::const_iterator it = images_parts.begin(); it != images_parts.end(); it++){
            int parts_counter = 0;
            for (vector<cv::Mat>::const_iterator itp = it->begin(); itp != it->end(); itp++){ 
                vector<Mat> p;    
                p.clear();
                cv::split(*itp, p);
              
                //3 is the number of rgb channels, there are 3 channels in each image part
                int ch_counter = 0;
                for ( vector<cv::Mat>::const_iterator itch = p.begin(); itch!=p.end(); itch++){
                   
                    itch->copyTo((*input_channels)[image_counter*3*parts_number + 3*parts_counter + ch_counter]);
                    ch_counter++;
                //std::copy(p.begin(), p.end(), input_channels->begin() + image_counter*3*parts_number + 3*parts_counter);
                } 
                parts_counter++;
                
            }
            image_counter++;
        }
    }

    vector< vector<cv::Mat> > preprocessImages(const vector<cv::Mat>& images) {
        vector< vector<cv::Mat> > result;
        for (vector<cv::Mat>::const_iterator it = images.begin(); it != images.end(); ++it){
            cv::Mat image = *it;
            cv::Mat sample_resized(cv::Size(image_w, image_h), CV_8UC3);
            cv::resize(image, sample_resized, sample_resized.size());
            cv::Mat sample_float(sample_resized.size(), CV_32FC3);
            sample_resized.convertTo(sample_float, CV_32FC3);
            cv::Mat sample_normalized(sample_resized.size(), CV_32FC3);
            cv::addWeighted(sample_float, 0.00390625, sample_float, 0., 0., sample_normalized);
            vector<cv::Mat> normalized_parts = breakImage(sample_normalized, parts_number, parts_overlap);   
            result.push_back(normalized_parts);   
            
        }
        return result;
    }

};
   

vector<string> parseImageNameFile(char * filename){
    vector<string> result;
    std::ifstream infile(filename);
    string line;

    if (!infile) {
         string error = "Could not open or find the file ";
         throw error.append(filename);
    }

    while (infile >> line)
    {
       std::size_t found = line.find("\n");
       result.push_back(line.substr(0, found));

    }
    infile.close();
    return result;
}

void saveResult(char * filename, vector<string> fileList, vector<vector<float> > descriptors){
    std::ofstream outfile (filename);

    for (int i = 0; i < fileList.size(); i++){
        string imagefile = fileList[i];
        vector<float> descr = descriptors[i];
        outfile << imagefile << " ";
        for (vector<float>::iterator it = descr.begin(); it != descr.end(); it++){
            outfile << *it << " ";
        }
        outfile << std::endl;  
    }
    outfile.close();
}

int main( int argc, char** argv )
{
    if( argc != 6)
    {
        cout <<" Usage: model_file weights_file mode(cpu/gpu) [image/filename_list]filename outputfile " << endl;
        return -1;
    }
    if (strcmp(argv[3],"gpu") == 0 ){
        Caffe::set_mode(Caffe::GPU);
    }
    else if (strcmp(argv[3],"cpu") == 0 ){
        Caffe::set_mode(Caffe::CPU);
        cout << "cpu " << endl;
    }

    Predictor predictor(argv[1], argv[2]);


    Mat image = cv::imread(argv[4], CV_LOAD_IMAGE_COLOR);   // Read the file

    vector<string> fileList;
    if(!image.data)                              // Check for invalid input
    {
         fileList = parseImageNameFile(argv[4]);
         
    } else {
         fileList.push_back(argv[4]);
    }

    vector<vector<float> > descriptors = predictor.makePredictionList(fileList);

    saveResult(argv[5], fileList, descriptors);
}     
 /*    Caffe::set_mode(Caffe::GPU);

     string model_file = "/media/hpc2_storage/eustinova/bilinear/3parts_bilinear_patch_v2/cuhk03_detected_split1_bilinear_patch_5/protos/train_val_model_.prototxt" ;
     string weights_file = "/media/hpc2_storage/eustinova/bilinear/3parts_bilinear_patch_v2/cuhk03_labeled_split1_bilinear_patch_5/snapshots/train_iter_55000.caffemodel" ;


     vector<string> images_filenames;
     for (int i=0; i <200;i++){
     images_filenames.push_back("/home/eustinova/dataset/CUHK03/labeled/1/labeled_1_1_1.jpg");
       }

     Predictor predictor(model_file, weights_file);
     vector<vector<float> > descriptors = predictor.makePredictionList(images_filenames);
     
     vector<float> result =  descriptors[0];
     std::copy(result.begin(),result.end(), std::ostream_iterator<float>(std::cout<< " " ));
     for (vector<float>::iterator it= result.begin(); it != result.end();it++){
         cout << *it << endl;
     }
std::cout << "RES SIZE " << descriptors.size() << std::endl;
     cout << "final!" << endl;
     return 0;*/


/*
int main( int argc, char** argv )
{
    std::cout << "argc "<< argc << std::endl;
    if( argc != 5)
    {
     cout <<" Usage: model_file weights_file mode(cpu/gpu) image" << endl;
     return -1;
    }


// initialize net---------                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    // if (strcmp(argv[3],"gpu") == 0 ){
        std::cout  << "gpu" << std::endl;
       
   // }
   // else if (strcmp(argv[3],"cpu") == 0 ){
    //    std::cout << "cpu" << std::endl;
   //  
     //   Caffe::set_mode(Caffe::CPU);
   // }
     std::cout << "AAA1" << std::endl;
shared_ptr<Net<float> > caffe_net;
 Caffe::set_mode(Caffe::GPU);
     Caffe::Brew mo = Caffe::mode();
        std::cout << "Caffe::mode() == Caffe::CPU" << (mo == Caffe::CPU )  << std:: endl;
   // caffe::Net<float> caffe_net(argv[1], caffe::TEST);
  caffe_net.reset(new Net<float>(argv[1], caffe::TEST));
  caffe_net->CopyTrainedLayersFrom(argv[2]);

     std::cout << "AAA2" << std::endl;
  //  caffe_net.CopyTrainedLayersFrom(argv[2]);
    
//-----------------------
    std::cout << "AAA3" << std::endl;
    Mat image;
    image = cv::imread(argv[4], CV_LOAD_IMAGE_COLOR);   // Read the file
    
  //  vector<Mat> parts =  breakImage(image, 3, 10);
    if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    Blob<float>* input_layer = caffe_net->input_blobs()[0];
    input_layer->Reshape(1, 9, 60, 60);

    caffe_net->Reshape();
    std::vector<cv::Mat> input_channels;
    WrapInputLayer(caffe_net, &input_channels);

    vector<cv::Mat> images;
    images.push_back(image);
    Preprocess(images, &input_channels);

    std::cout << "AAA4" << std::endl;
    Caffe::set_mode(Caffe::GPU);
    Caffe::Brew m = Caffe::mode();
    std::cout << "2Caffe::mode() == Caffe::GPU" << (m == Caffe::GPU )  << std::endl;

    Blob<float>* output_layer = caffe_net->ForwardPrefilled()[0];
    std::cout << "AAA5" << std::endl;
    
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();

    std::cout << "AAA6" << std::endl;
    
    vector<float> result =  std::vector<float>(begin, end);
    std::copy(result.begin(),result.end(), std::ostream_iterator<float>(std::cout<< " " ));
    for (vector<float>::iterator it= result.begin(); it != result.end();it++){
        cout << *it << endl;
    }
    cout << "final!" << endl;
    return 0;
}
*/
/*Caffe::set_mode(Caffe::GPU);

  LOG(INFO) << "Using GPU";

    string a = "/media/hpc2_storage/eustinova/hieroglyphs/experiments/cuhk03_np_loss_hist_smooth_0.7_1e-4_grid_0.01_adam/protos/train_val_model_.prototxt";
    string aa = "/home/eustinova/caffe-caffe-0.14/models/bvlc_reference_caffenet/deploy.prototxt";
    string b = "/media/hpc2_storage/eustinova/hieroglyphs/experiments/cuhk03_np_loss_hist_smooth_0.7_1e-4_grid_0.01_adam/snapshots/train_iter_55000.caffemodel";

 
    caffe::Net<float> caffe_net(a, caffe::TEST);
 
    caffe_net.CopyTrainedLayersFrom(b);
*/ 

/*    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", parts[0] );                   // Show our image inside it.
 
    waitKey(0);                                          // Wait for a keystroke in the window
    imshow( "Display window", parts[1] );                   // Show our image inside it.
    waitKey(0);                                          // Wait for a keystroke in the window
 
    imshow( "Display window", parts[2] );                   // Show our image inside it.
    waitKey(0);                                          // Wait for a keystroke in the window
*/

 //imshow( "Display window", input_channels[0] );                   // Show our image inside it.
 //   waitKey(0);

