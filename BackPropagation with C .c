#include <stdio.h> //Standart giriş-çıkış işlemleri, örnek printf fonksiyonu
#include <stdlib.h> //Rastgele sayı üretimi, bellek yönetimi fonksiyonları
#include <math.h> //Matematiksel fonksiyon kütüphanesi
#include <time.h> //Zaman kütüphanesi

#define numInputs 6 //Giriş Katmandaki Nöron Sayısı
#define numHiddenNodes 40 // Gizli Katmandaki Nöron Sayısı
#define numOutputs 1 //Çıkış Katmandaki Nöron sayısı
#define numTrainingSets 64 //Eğitim Veri Sayısı

double sigmoid(double x) { //sigmoid aktivasyon fonksiyonu
    return 1.0 / (1.0 + exp(-x));
}

double dSigmoid(double x) { //Sigmoid Türev Fonksiyonu, geri yayılım için
    return x * (1.0 - x);   //Geri yayılım sırasında ağırlık güncellemelerinde                       //kullanılır
}

double init_weight() { //Ağırlıkları rastgele başlatma fonksiyonu, 
                       //rand() fonksiyonu 0 ile RAND_MAX arasında tam sayı //üretir.
                       // *4-2 işlemi bu değer [-2,2] aralığında ölçekler.
    return ((double)rand() / RAND_MAX) * 4 - 2; // [-2, 2] aralığında başlat
}


void shuffle(int *array, size_t n) { //Elemanların sırasını rastgele karıştırmak.
    if (n > 1) {
        for (size_t i = 0; i < n - 1; i++) { //Amaç Eğitim veri setinin sırasını 
                                            //her epoch sırasında değiştirmek.
                                            //overfit engellemesi
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

int main(void) {
    srand(time(NULL));
    const double lr = 0.1; //Learning Rate

    double hiddenLayer[numHiddenNodes];
    double outputLayer[numOutputs];

    double hiddenLayerBias[numHiddenNodes];
    double outputLayerBias[numOutputs];

    double hiddenWeights[numInputs][numHiddenNodes];
    double outputWeights[numHiddenNodes][numOutputs];



// Training input
double training_inputs[64][6] = {
    {0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 1},
    {0, 0, 0, 0, 1, 0},
    {0, 0, 0, 0, 1, 1},
    {0, 0, 0, 1, 0, 0},
    {0, 0, 0, 1, 0, 1},
    {0, 0, 0, 1, 1, 0},
    {0, 0, 0, 1, 1, 1},
    {0, 0, 1, 0, 0, 0},
    {0, 0, 1, 0, 0, 1},
    {0, 0, 1, 0, 1, 0},
    {0, 0, 1, 0, 1, 1},
    {0, 0, 1, 1, 0, 0},
    {0, 0, 1, 1, 0, 1},
    {0, 0, 1, 1, 1, 0},
    {0, 0, 1, 1, 1, 1},
    {0, 1, 0, 0, 0, 0},
    {0, 1, 0, 0, 0, 1},
    {0, 1, 0, 0, 1, 0},
    {0, 1, 0, 0, 1, 1},
    {0, 1, 0, 1, 0, 0},
    {0, 1, 0, 1, 0, 1},
    {0, 1, 0, 1, 1, 0},
    {0, 1, 0, 1, 1, 1},
    {0, 1, 1, 0, 0, 0},
    {0, 1, 1, 0, 0, 1},
    {0, 1, 1, 0, 1, 0},
    {0, 1, 1, 0, 1, 1},
    {0, 1, 1, 1, 0, 0},
    {0, 1, 1, 1, 0, 1},
    {0, 1, 1, 1, 1, 0},
    {0, 1, 1, 1, 1, 1},
    {1, 0, 0, 0, 0, 0},
    {1, 0, 0, 0, 0, 1},
    {1, 0, 0, 0, 1, 0},
    {1, 0, 0, 0, 1, 1},
    {1, 0, 0, 1, 0, 0},
    {1, 0, 0, 1, 0, 1},
    {1, 0, 0, 1, 1, 0},
    {1, 0, 0, 1, 1, 1},
    {1, 0, 1, 0, 0, 0},
    {1, 0, 1, 0, 0, 1},
    {1, 0, 1, 0, 1, 0},
    {1, 0, 1, 0, 1, 1},
    {1, 0, 1, 1, 0, 0},
    {1, 0, 1, 1, 0, 1},
    {1, 0, 1, 1, 1, 0},
    {1, 0, 1, 1, 1, 1},
    {1, 1, 0, 0, 0, 0},
    {1, 1, 0, 0, 0, 1},
    {1, 1, 0, 0, 1, 0},
    {1, 1, 0, 0, 1, 1},
    {1, 1, 0, 1, 0, 0},
    {1, 1, 0, 1, 0, 1},
    {1, 1, 0, 1, 1, 0},
    {1, 1, 0, 1, 1, 1},
    {1, 1, 1, 0, 0, 0},
    {1, 1, 1, 0, 0, 1},
    {1, 1, 1, 0, 1, 0},
    {1, 1, 1, 0, 1, 1},
    {1, 1, 1, 1, 0, 0},
    {1, 1, 1, 1, 0, 1},
    {1, 1, 1, 1, 1, 0},
    {1, 1, 1, 1, 1, 1}
};

double training_outputs[64][1] = { //Training output
    {0}, {1}, {1}, {0}, {1}, {0}, {0}, {1},
    {1}, {0}, {0}, {1}, {0}, {1}, {1}, {0},
    {1}, {0}, {0}, {1}, {0}, {1}, {1}, {0},
    {0}, {1}, {1}, {0}, {1}, {0}, {0}, {1},
    {1}, {0}, {0}, {1}, {0}, {1}, {1}, {0},
    {0}, {1}, {1}, {0}, {1}, {0}, {0}, {1},
    {0}, {1}, {1}, {0}, {1}, {0}, {0}, {1},
    {1}, {0}, {0}, {1}, {0}, {1}, {1}, {0}
};

    //İnput ile hidden arasında olan her sinir ağırlığını -2,2 arasında değer //alır
    for (int i = 0; i < numInputs; i++)
        for (int j = 0; j < numHiddenNodes; j++)
            hiddenWeights[i][j] = init_weight();
    
    for (int i = 0; i < numHiddenNodes; i++) {
        hiddenLayerBias[i] = init_weight(); //Her bir nöron için bias değer                                     //rastgele sıralanır.
        for (int j = 0; j < numOutputs; j++)
            outputWeights[i][j] = init_weight();
    }

    for (int i = 0; i < numOutputs; i++)
        outputLayerBias[i] = init_weight(); //Çıkış nöronun bias değeri atanır

    int trainingSetOrder[numTrainingSets];
    for (int i = 0; i < numTrainingSets; i++) trainingSetOrder[i] = i;
    //Başlangıç eğitim seti sıralı olarak (0,1,2,3...63) yerleştirilir.

    int numberOfEpochs = 50000; //epoch sayısı
    for (int epoch = 0; epoch < numberOfEpochs; epoch++) {
        shuffle(trainingSetOrder, numTrainingSets); //Train setini her epoch'da                                          //karıştırır.
        for (int x = 0; x < numTrainingSets; x++) {
            int i = trainingSetOrder[x];//Karıştırılmış sıraya göre kaçıncı                              //eğitim örneğinin kullanılacağını                               //belirler.

            for (int j = 0; j < numHiddenNodes; j++) { //Her bir gizli katman                                           //nöronu için çalışır.
                double activation = hiddenLayerBias[j];//Aktivasyon değeri başta                                 //sadece bias ile başlatılır.
                for (int k = 0; k < numInputs; k++)
                    activation += training_inputs[i][k] * hiddenWeights[k][j];
                    //giriş değerleri ile bağlantılı ağırlıklar çarpılır ve //aktivasyona eklenir
                hiddenLayer[j] = sigmoid(activation);
                //Toplam aktivasyon sigmoid fonksiyonu ile sıkıştırılır (0–1 arası çıktı üretir).
            }

            for (int j = 0; j < numOutputs; j++) { //Çıkış Katmanı için çalışır
                double activation = outputLayerBias[j]; //Aktivasyon yine bias                                       //değerleriyle başlatılır.
                for (int k = 0; k < numHiddenNodes; k++)
                    activation += hiddenLayer[k] * outputWeights[k][j];
                    //Gizli katmandaki her nöronun çıktısı ile bağlantılı //ağırlığı çarpılır
                outputLayer[j] = sigmoid(activation); //Aktivasyon değeri güncellenir
            }

            double deltaOutput[numOutputs]; //Çıkış nöronu için çalışır
            for (int j = 0; j < numOutputs; j++) {
                double error = training_outputs[i][j] - outputLayer[j]; //Beklenen değer- ağın tahmini arasındaki hata
                deltaOutput[j] = error * dSigmoid(outputLayer[j]);
                //bu değer geri yayılımda kullanılıp, sigmoidin türevi ile çarpılarak hata katsayısı oluşturur.
            }
            //Geri yayılım sırasında gizli katmandaki her nöron için hata hesaplanır.
            double deltaHidden[numHiddenNodes];
            for (int j = 0; j < numHiddenNodes; j++) { 
                double error = 0.0;
                for (int k = 0; k < numOutputs; k++)
                    error += deltaOutput[k] * outputWeights[j][k]; //Çıkış nöronundan gelen hata, o nöronla bağlantılı olan ağırlıkla çarpılarak toplanır.
                deltaHidden[j] = error * dSigmoid(hiddenLayer[j]); //hata değeri sigmoid türevi ile güncellenir.
            }

            for (int j = 0; j < numOutputs; j++) {
                outputLayerBias[j] += deltaOutput[j] * lr;
                //Her çıkış nöronunun bias değeri, hata ve öğrenme oranı ile güncellenir.
                for (int k = 0; k < numHiddenNodes; k++)
                    outputWeights[k][j] += hiddenLayer[k] * deltaOutput[j] * lr;
                    //Ağırlık güncellemesi
            }

            for (int j = 0; j < numHiddenNodes; j++) {
                hiddenLayerBias[j] += deltaHidden[j] * lr; //Bias güncellemesi
                for (int k = 0; k < numInputs; k++)
                    hiddenWeights[k][j] += training_inputs[i][k] * deltaHidden[j] * lr; //Her girişten gelen sinyal, delta ile çarpılır ve ağırlık güncellemesinde kullanılır.
            
            }
        }
    }

    // Final Değerlendirme
    printf("\n--- Final Evaluation ---\n");
    for (int i = 0; i < numTrainingSets; i++) { //Gizli katman hesaplanması
        for (int j = 0; j < numHiddenNodes; j++) {
            double activation = hiddenLayerBias[j];
            for (int k = 0; k < numInputs; k++)
                activation += training_inputs[i][k] * hiddenWeights[k][j];
            hiddenLayer[j] = sigmoid(activation);
        }

        for (int j = 0; j < numOutputs; j++) { //Çıkış katmanın hesaplanması
            double activation = outputLayerBias[j];
            for (int k = 0; k < numHiddenNodes; k++)
                activation += hiddenLayer[k] * outputWeights[k][j];
            outputLayer[j] = sigmoid(activation);
        }

        printf("Input: "); //Sonuçlar
        for (int k = 0; k < numInputs; k++) printf("%.0f ", training_inputs[i][k]);
        printf("-> Predicted: %.4f | Expected: %.0f\n", outputLayer[0], training_outputs[i][0]);
    }

    return 0;
}