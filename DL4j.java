package org.deeplearning4j.datapipelineexamples.formats.image;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import java.io.File;
import java.io.IOException;
import java.util.Random;
public class DL4JExample {
    public static void main(String[] args) throws Exception {
        int height = 32; // Increased height to match kernel size
        int width = 32; // Increased width to match kernel size
        int channels = 1;
        int rngSeed = 123;
        Random randNumGen = new Random(rngSeed);
        int batchSize = 128;
        int outputNum = 2; // Number of output classes (changed to 2)
        int epochs = 10;

        File trainData = new File("E:/dataset/train");
        File testData = new File("E:/dataset/test");

        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
        FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
        recordReader.initialize(train);
        RecordReaderDataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);

        dataIter.setPreProcessor(new ImagePreProcessingScaler(0, 1));

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .updater(new Nesterovs(0.006, 0.9))
                .list()
                .layer(0, new ConvolutionLayer.Builder(1, 1)
                        .nIn(channels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(1, 1)
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(4, new ConvolutionLayer.Builder(1, 1)
                        .stride(1, 1)
                        .nOut(100)
                        .activation(Activation.RELU)
                        .build())
                .layer(5, new ConvolutionLayer.Builder(1, 1)
                        .stride(1, 1)
                        .nOut(100)
                        .activation(Activation.RELU)
                        .build())
                .layer(6, new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(7, new DenseLayer.Builder()
                        .nOut(500)
                        .activation(Activation.RELU)
                        .build())
                .layer(8, new DenseLayer.Builder()
                        .nOut(250)
                        .activation(Activation.RELU)
                        .build())
                .layer(9, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(height, width, channels))
                .build();



        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10)); // Print score every 10 iterations

        for (int i = 0; i < epochs; i++) {
            model.fit(dataIter);
        }

        recordReader.reset();
        recordReader.initialize(test);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
        dataIter.setPreProcessor(new ImagePreProcessingScaler(0, 1));
        Evaluation eval = model.evaluate(dataIter);
        System.out.println(eval.stats());



        File locationToSave = new File("E:/dataset");
        try {
            ModelSerializer.writeModel(model, locationToSave, true);
            System.out.println("Model saved successfully.");
        } catch (IOException e) {
            System.out.println("Error saving model: " + e.getMessage());
        }


    }
}