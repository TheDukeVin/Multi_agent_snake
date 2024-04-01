
#include "lstm.h"
#include "test.h"
// #include "PG.h"

// Run training sessions in parallel

// const int numThreads = 10;

// Supervised trainers[numThreads];
// thread* threads[numThreads];

// void runThread(int i){
//     trainers[i].fout = ofstream("session" + to_string(i) + ".out");
//     trainers[i].train();
// }

// void testGrad(){
//     GradientTest grad;
//     for(int i=0; i<100; i++){
//         grad.test();
//     }
// }

// void testSuper(){
//     for(int i=0; i<numThreads; i++){
//         threads[i] = new thread(runThread, i);
//     }
//     for(int i=0; i<numThreads; i++){
//         threads[i]->join();
//         cout<<trainers[i].finalLoss<<' '<<trainers[i].finalAcc<<'\n';
//     }

//     // Supervised super;
//     // super.train();
// }




// const int numThreads = 1;
// PG trainers[numThreads];

// void runThread(int i){
//     trainers[i].fileOut = "session" + to_string(i) + ".out";
//     if(i < 1){
//         // trainers[i].learnRate = 0.01;
//         // trainers[i].momentum = 0.7;
//     }
//     // else if(i < 20){
//     //     trainers[i].learnRate = 0.005;
//     //     trainers[i].momentum = 0.9;
//     // }
//     trainers[i].trainParallel(1000, 100000);
// }

// void runPG(){
//     thread* threads[numThreads];
//     for(int i=0; i<numThreads; i++){
//         threads[i] = new thread(runThread, i);
//     }
//     for(int i=0; i<numThreads; i++){
//         threads[i]->join();
//         cout<<trainers[i].finalReward<<'\n';
//     }
//     trainers[0].seq.paramStore.save("net.out");
// }




// void testPoker1(){
//     Hand H1;
//     H1.cards[0][0] = 1;
//     H1.cards[0][1] = 1;
//     H1.cards[0][2] = 1;
//     H1.cards[0][3] = 1;
//     H1.cards[0][12] = 1;
//     cout << H1.getStrength().toString() << '\n';
//     Hand H2;
//     H2.cards[0][0] = 1;
//     H2.cards[0][1] = 1;
//     H2.cards[0][2] = 1;
//     H2.cards[0][3] = 1;
//     H2.cards[0][12] = 1;
//     H2.cards[1][2] = 1;
//     H2.cards[1][3] = 1;
//     H2.cards[1][4] = 1;
//     H2.cards[1][5] = 1;
//     H2.cards[1][6] = 1;
//     cout << H2.getStrength().toString() << '\n';
//     Hand H3;
//     H3.cards[0][0] = 1;
//     H3.cards[0][1] = 1;
//     H3.cards[0][2] = 1;
//     H3.cards[0][3] = 1;
//     H3.cards[0][5] = 1;
//     cout << H3.getStrength().toString() << '\n';
//     Hand H4;
//     H4.cards[0][0] = 1;
//     H4.cards[1][0] = 1;
//     H4.cards[2][0] = 1;
//     H4.cards[3][0] = 1;
//     H4.cards[0][1] = 1;
//     cout << H4.getStrength().toString() << '\n';
//     Hand H5;
//     H5.cards[0][0] = 1;
//     H5.cards[1][0] = 1;
//     H5.cards[2][0] = 1;
//     H5.cards[3][0] = 1;
//     H5.cards[0][1] = 1;
//     H5.cards[1][1] = 1;
//     H5.cards[2][1] = 1;
//     H5.cards[3][1] = 1;
//     cout << H5.getStrength().toString() << '\n';
// }

// void testPoker2(){
//     Hand H;
//     vector<int> cards;
//     for(int i=0; i<52; i++){
//         cards.push_back(i);
//     }
//     auto rd = random_device{};
//     auto rng = default_random_engine{rd()};
//     shuffle(cards.begin(), cards.end(), rng);
//     for(int i=0; i<52; i++){
//         cout << cards[i] / numVals << ' ' << cards[i] % numVals << '\n';
//         H.addCard(cards[i]);
//         if(i >= 4){
//             cout<<H.getStrength().toString() << '\n';
//         }
//     }
//     cout<<'\n';
// }

int main(){
    srand(time(0));
    int start_time = time(0);

    // testGrad();

    // testSuper();

    // runPG();

    // trainers[0].fileOut = "session0.out";
    // ofstream fout(trainers[0].fileOut);
    // fout.close();
    // trainers[0].train();
    // for(int i=0; i<10; i++){
    //     trainers[0].rollOut(true);
    // }

    // testPoker2();

    // PG runner;
    // runner.seq.paramStore.load("net.out");
    // runner.fileOut = "games.out";
    // ofstream fout(runner.fileOut);
    // fout.close();
    // runner.rollOut(true);
    // cout << "Final reward: " << runner.rollOutRewardSum << '\n';

    // cout<<"TIME: "<<(time(0) - start_time)<<'\n';

    // PVTest tester;
    // tester.test();

    ModelTest tester;
    tester.test();
}