
#include "environment.h"

Hand::Hand(){
    for(int s=0; s<numSuits; s++){
        for(int v=0; v<numVals; v++){
            cards[s][v] = false;
        }
    }
}

string Hand::toString(){
    string str = "";
    for(int s=0; s<numSuits; s++){
        for(int v=0; v<numVals; v++){
            str += to_string(cards[s][v]);
        }
        str += "\n";
    }
    return str;
}

Strength Hand::getStrength(){
    int sf = straightFlushCheck();
    if(sf != -1){
        return Strength(STRAIGHT_FLUSH, sf);
    }
    getCounts();
    int q = quadCheck();
    if(q != -1){
        return Strength(QUAD, q);
    }
    int fh = fullCheck();
    if(fh != -1){
        return Strength(FULL_HOUSE, fh);
    }
    int f = flushCheck();
    if(f != -1){
        return Strength(FLUSH, f);
    }
    int s = straightCheck();
    if(s != -1){
        return Strength(STRAIGHT, s);
    }
    int t = tripCheck();
    if(t != -1){
        return Strength(TRIP, t);
    }
    vector<int> tp = twoCheck();
    if(tp.size() > 0){
        return Strength(TWO_PAIR, tp);
    }
    vector<int> p = pairCheck();
    if(p.size() > 0){
        return Strength(PAIR, p);
    }
    vector<int> h = highCheck();
    return Strength(HIGH, h);
}

void Hand::addCard(int cardID){
    cards[cardID / numVals][cardID % numVals] = true;
}

bool Hand::checkCard(int cardID){
    return cards[cardID / numVals][cardID % numVals];
}

void Hand::getCounts(){
    sortedCounts = vector<pair<int, int> >();
    for(int v=0; v<numVals; v++){
        counts[v] = 0;
        for(int s=0; s<numSuits; s++){
            counts[v] += cards[s][v];
        }
        sortedCounts.push_back(make_pair(counts[v], v));
    }
    sort(sortedCounts.rbegin(), sortedCounts.rend());
}

int Hand::straightFlushCheck(){
    int ans = -1;
    for(int s=0; s<numSuits; s++){
        int streak = 0;
        for(auto v : straight_order){
            if(cards[s][v]){
                streak += 1;
                if(streak >= 5){
                    ans = max(ans, v);
                }
            }
            else{
                streak = 0;
            }
        }
    }
    return ans;
}

int Hand::quadCheck(){
    if(sortedCounts[0].first == 4){
        return sortedCounts[0].second;
    }
    return -1;
}

int Hand::fullCheck(){
    if(sortedCounts[0].first == 3 && sortedCounts[1].first >= 2){
        return sortedCounts[0].second;
    }
    return -1;
}

int Hand::flushCheck(){
    int ans = -1;
    for(int s=0; s<numSuits; s++){
        int count = 0;
        for(int v=0; v<numVals; v++){
            if(cards[s][v]){
                count += 1;
                if(count >= 5){
                    ans = max(ans, v);
                }
            }
        }
    }
    return ans;
}

int Hand::straightCheck(){
    int ans = -1;
    int count = 0;
    for(auto v : straight_order){
        if(counts[v] > 0){
            count += 1;
            if(count >= 5){
                ans = max(ans, v);
            }
        }
        else{
            count = 0;
        }
    }
    return ans;
}

int Hand::tripCheck(){
    if(sortedCounts[0].first == 3){
        return sortedCounts[0].second;
    }
    return -1;
}

vector<int> Hand::twoCheck(){
    if(sortedCounts[0].first == 2 && sortedCounts[1].first == 2){
        return vector<int>{sortedCounts[0].second, sortedCounts[1].second, sortedCounts[2].second};
    }
    return vector<int>();
}

vector<int> Hand::pairCheck(){
    if(sortedCounts[0].first == 2){
        return vector<int>{sortedCounts[0].second, sortedCounts[1].second, sortedCounts[2].second, sortedCounts[3].second};
    }
    return vector<int>();
}

vector<int> Hand::highCheck(){
    vector<int> vals;
    for(int i=0; i<5; i++){
        vals.push_back(sortedCounts[i].second);
    }
    return vals;
}








// ENVIRONMENT DETAILS

Environment::Environment(){
    time = 0;
    endState = false;
    for(int i=0; i<deckSize; i++){
        cards.push_back(i);
    }
    auto rd = random_device{};
    auto rng = default_random_engine{rd()};
    shuffle(cards.begin(), cards.end(), rng);
    cardIndex = 0;
}

string Environment::toString(){
    return "Time: " + to_string(time) + "\n" + agentHand.toString() + '\n' + oppHand.toString() + '\n';
}

vector<int> Environment::validActions(){
    vector<int> actions;
    for(int s=0; s<numSuits; s++){
        for(int v=0; v<numVals; v++){
            if(!agentHand.cards[s][v] && !oppHand.cards[s][v]){
                actions.push_back(s*numVals + v);
            }
        }
    }
    return actions;
}

double Environment::makeAction(int* actions){
    double reward = 0;
    while(cardIndex < deckSize){
        int currCard = cards[cardIndex];
        cardIndex ++;
        if(actions[currCard] == 1){
            agentHand.addCard(currCard);
            break;
        }
        else{
            oppHand.addCard(currCard);
        }
    }
    if(cardIndex == deckSize){
        endState = true;
        return 0;
    }
    time ++;
    if(time == TIME_HORIZON){
        endState = true;
        while(cardIndex < 13){
            int currCard = cards[cardIndex];
            cardIndex ++;
            oppHand.addCard(currCard);
        }
        reward = agentHand.getStrength().compare(oppHand.getStrength());
    }
    return reward;
}

void Environment::inputObservations(Data* input){
    for(int i=0; i<deckSize; i++){
        input->data[i] = agentHand.checkCard(i);
        input->data[i + deckSize] = oppHand.checkCard(i);
    }
    cout << "Inputs: \n";
    for(int i=0; i<inputSize; i++){
        cout << input->data[i] << ' ';
    }
    cout<<'\n';
}