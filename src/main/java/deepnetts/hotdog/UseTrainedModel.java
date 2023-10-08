package deepnetts.hotdog;

import deepnetts.net.ConvolutionalNetwork;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Map;
import javax.imageio.ImageIO;
import javax.visrec.ml.classification.ImageClassifier;
import javax.visrec.ri.ml.classification.ImageClassifierNetwork;

public class UseTrainedModel {
    
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
