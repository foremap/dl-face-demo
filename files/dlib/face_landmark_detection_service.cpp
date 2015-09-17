// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.  
    


    This face detector is made using the classic Histogram of Oriented
    Gradients (HOG) feature combined with a linear classifier, an image pyramid,
    and sliding window detection scheme.  The pose estimator was created by
    using dlib's implementation of the paper:
        One Millisecond Face Alignment with an Ensemble of Regression Trees by
        Vahid Kazemi and Josephine Sullivan, CVPR 2014
    and was trained on the iBUG 300-W face landmark dataset.  

    Also, note that you can train your own models using dlib's machine learning
    tools.  See train_shape_predictor_ex.cpp to see an example.

    


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.  
*/


#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <zmq.hpp>
#include <iostream>
#include <time.h>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <vector>
#include <string>

using namespace dlib;
using namespace std;

string int2str(int &);
// ----------------------------------------------------------------------------------------
int getdir(string dir, std::vector<string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }

    files.clear();
    while ((dirp = readdir(dp)) != NULL) {
        if( strcmp(dirp->d_name, ".") != 0 && strcmp(dirp->d_name, "..") != 0)
            files.push_back(string(dirp->d_name));
    }
    closedir(dp);
    return 0;
}

int main(int argc, char** argv)
{  
    //  Prepare our context and socket
    zmq::context_t context (1);
    zmq::socket_t socket (context, ZMQ_REP);
    socket.bind("tcp://*:5566");

    try
    {
        // This example takes in a shape model file and then a list of images to
        // process.  We will take these filenames in as command line arguments.
        // Dlib comes with example images in the examples/faces folder so give
        // those as arguments to this program.
        if (argc == 1)
        {
            cout << "Call this program like this:" << endl;
            cout << "./face_landmark_detection_ex shape_predictor_68_face_landmarks.dat faces/*.jpg" << endl;
            cout << "\nYou can get the shape_predictor_68_face_landmarks.dat file from:\n";
            cout << "http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2" << endl;
            return 0;
        }
        clock_t tstart = clock(); 

        // We need a face detector.  We will use this to get bounding boxes for
        // each face in an image.
        frontal_face_detector detector = get_frontal_face_detector();
        // And we also need a shape_predictor.  This is the tool that will predict face
        // landmark positions given an image and face bounding box.  Here we are just
        // loading the model from the shape_predictor_68_face_landmarks.dat file you gave
        // as a command line argument.
        shape_predictor sp;
        deserialize(argv[1]) >> sp;
        cout << "(Init) took " << float( clock () - tstart ) /  CLOCKS_PER_SEC << " second(s)." << endl;

        string src_dir, dest_dir, imgae_file;
        src_dir = argv[2];
        dest_dir = argv[3];

        while(true)
        {
            zmq::message_t request;
            socket.recv(&request);
            imgae_file = string(static_cast<char*>(request.data()), request.size());
            cout << "recieve signal ready" << imgae_file << endl;

            tstart = clock();
            cout << "File name : " << imgae_file << endl;
            int lastindex = imgae_file.find_last_of("."); 
            string rawname = imgae_file.substr(0, lastindex);

            //image_window win, win_faces;
            // Loop over all the images provided on the command line.
            //cout << "processing image " << src_dir + '/' + imgae_file << endl;
            array2d<rgb_pixel> img;
            load_image(img, src_dir + '/' + imgae_file);
            // Make the image larger so we can detect small faces.
            pyramid_up(img); // ===> enlarge

            // Now tell the face detector to give us a list of bounding boxes
            // around all the faces in the image.
            std::vector<rectangle> dets = detector(img);
            //cout << "Number of faces detected: " << dets.size() << endl;
            cout << "(Detect) took " << float( clock () - tstart ) /  CLOCKS_PER_SEC << " second(s)." << endl;
            tstart = clock();

            // Now we will go ask the shape_predictor to tell us the pose of
            // each face we detected.
            if (dets.size() != 0)
            {
                std::vector<full_object_detection> shapes;
                for (unsigned long j = 0; j < dets.size(); ++j)
                {
                    full_object_detection shape = sp(img, dets[j]);
                    // cout << "number of parts: "<< shape.num_parts() << endl;
                    // cout << "pixel position of first part:  " << shape.part(0) << endl;
                    // cout << "pixel position of second part: " << shape.part(1) << endl;
                    // You get the idea, you can get all the face part locations if
                    // you want them.  Here we just store them in shapes so we can
                    // put them on the screen.
                    shapes.push_back(shape);
                }

                // Now let's view our face poses on the screen.
                // win.clear_overlay();
                // win.set_image(img);
                // win.add_overlay(render_face_detections(shapes));

                // We can also extract copies of each face that are cropped, rotated upright,
                // and scaled to a standard size as shown here:
                dlib::array<array2d<rgb_pixel> > face_chips;
                extract_image_chips(img, get_face_chip_details(shapes), face_chips);
                //win_faces.set_image(tile_images(face_chips));
                cout << "(2D-align) took " << float( clock () - tstart ) /  CLOCKS_PER_SEC << " second(s)." << endl;
                tstart = clock();
                if (face_chips.size() == 1) {
                    save_jpeg(face_chips[0], dest_dir + "/single/" + rawname + ".jpg");
                } else {
                    for (int idx = 0; idx < face_chips.size(); ++idx) {
                        save_jpeg(face_chips[idx], dest_dir + "/multi/" + rawname + '-' + int2str(idx) + ".jpg");
                    }
                }
                cout << "(Save file) took " << float( clock () - tstart ) /  CLOCKS_PER_SEC << " second(s)." << endl;
            } else {
                save_jpeg(img, dest_dir + "/others/" + rawname + ".jpg");
            }
            std::remove((src_dir + '/' + imgae_file).c_str());

            //  Send reply back to client
            zmq::message_t reply(4);
            memcpy((void *) reply.data(), "Done", 4);
            socket.send(reply);
        }
    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}

string int2str(int &i) {
  string s;
  stringstream ss(s);
  ss << i;
  return ss.str();
}
// ----------------------------------------------------------------------------------------

