package deepnetts.hotdog;

import deepnetts.net.ConvolutionalNetwork;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Map;
import javax.imageio.ImageIO;
import javax.visrec.ml.classification.ImageClassifier;
import javax.visrec.ri.ml.classification.ImageClassifierNetwork;

// image dir da bude prvi parametar u konstruktoru, i da ukinem index i labels file
// skaliraj slike prilikom prepoznavanja
// direktorijum sa slikama da bude prvi parametar , izbaci ucitavanje fajlova i labela - samo dir
// remove log4j from visrec, stavi dependency na ri 1.0.4 ane na 1.0.3

public class HotDotRecognition {
    
    public static void main(String[] args) throws IOException, ClassNotFoundException {
        
        // load image
        BufferedImage image = ImageIO.read(new File("hotdog-dataset/hot_dog/7896.jpg")).getSubimage(0, 0, 64, 64);

        // load trained convolutional network
        ConvolutionalNetwork convNet = ConvolutionalNetwork.load("HotDogClassifier.dnet", ConvolutionalNetwork.class);
        
        // instantiate image classifier using convolutional neural network
        ImageClassifier<BufferedImage> imageClassifier = new ImageClassifierNetwork(convNet);
        
        // classify image and get results
        Map<String, Float> results = imageClassifier.classify(image); // result is a map with image labels as keys and coresponding probability
                
        System.out.println(results.toString());        
    }
    
}
