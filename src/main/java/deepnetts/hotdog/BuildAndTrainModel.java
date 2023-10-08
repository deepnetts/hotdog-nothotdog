package deepnetts.hotdog;

import deepnetts.core.DeepNetts;
import deepnetts.data.ImageSet;
import deepnetts.net.ConvolutionalNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.net.layers.Filter;
import deepnetts.net.layers.Filters;
import deepnetts.net.loss.LossType;
import deepnetts.util.ImageResize;
import deepnetts.util.RandomGenerator;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.logging.Logger;
import javax.visrec.ml.eval.EvaluationMetrics;

/**
 * HotDog / NotHotDog image recognition.
 * This examples shows how to use convolutional neural network for binary classification of images.
 *
 * Data set contains X images of hot dogs and Non-Duke examples.
 * 
 * For the best performance and accuracy the recommended way to run this example is to use Deep Netts Pro, with Free Development License.
 * https://www.deepnetts.com/download
 *
 * Step-by-step guide for setting up Deep Netts is available at
 * https://www.deepnetts.com/getting-started
 * 
 * @see ConvolutionalNetwork
 * @see ImageSet
 * 
 */
public class BuildAndTrainModel {

    static final Logger LOGGER = Logger.getLogger(DeepNetts.class.getName());
 // https://github.com/deepnetts/hotdog-nothotdog
    public static void main(String[] args) throws FileNotFoundException, IOException, ClassNotFoundException {

        // path to directory with training images (each category has it's folder)
        String dataSetPath = "hotdog-dataset";
        
        // width and height of input image (all images will be scaled to this dimensions)
        int imageWidth = 64;
        int imageHeight = 64;
        
        // fix random generator to get repeatable training
        RandomGenerator.getDefault().initSeed(123); 
        
        // PREPARE DATASET - create image set and init preprocessing
        LOGGER.info("Loading images...");
        ImageSet imageSet = new ImageSet(imageWidth, imageHeight, dataSetPath);
        imageSet.setResizeStrategy(ImageResize.STRATCH);
        imageSet.setInvertImages(true);
        imageSet.zeroMean();

        // BUILD AND TRAIN THE MODEL
        // create a convolutional neural network arhitecture for binary image classification
        LOGGER.info("Creating a neural network...");
        ConvolutionalNetwork convNet = ConvolutionalNetwork.builder()
                .addInputLayer(imageWidth, imageHeight, 3)
                .addConvolutionalLayer(3, Filters.ofSize(3))
                .addMaxPoolingLayer(2, 2)
                .addFullyConnectedLayer(32)
                .addOutputLayer(1, ActivationType.SIGMOID)
                .hiddenActivationFunction(ActivationType.LEAKY_RELU)
                .lossFunction(LossType.CROSS_ENTROPY)
                .build();

        // set training options and run training
        BackpropagationTrainer trainer = convNet.getTrainer();
        trainer.setStopError(0.03f)  
               .setLearningRate(0.01f)
               .setStopAccuracy(0.97f)
               .setStopEpochs(300);
        trainer.train(imageSet); // run training

        
        // TEST / EVALUATE THE MODEL        
        EvaluationMetrics testResults = convNet.test(imageSet);
        System.out.println(testResults);    
        
        // SAVE THE MODEL - save trained neural network to file
        convNet.save("HotDogClassifier.dnet");
      
        // shutdown the thread pool
        DeepNetts.shutdown();
    }

}