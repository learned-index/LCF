#include "cuckoofilter.h"
#include "constants.h"

#include "torch/torch.h"

#include <assert.h>
#include <math.h>

#include <iostream>
#include <vector>
#include <chrono>
#include <ctime>
#include <string>
#include <vector>
#include <fstream>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <sstream>

using cuckoofilter::CuckooFilter;
using namespace std;

#define rdm(a,b) (rand() % (b-a) + a)
#define e_ 2.718281828459045235

vector<string> read_file(string filename){
  ifstream fin;
  fin.open(filename.c_str());
  if(!fin){
      cout << filename << " file could not be opened\n";
      exit(0);
  }
  string line;
  vector<string> data;
  while(getline(fin, line))
      data.emplace_back(line);

  fin.close();
  return data;
}

vector<int> read_predict(string filename){
  ifstream fin;
  fin.open(filename.c_str());
  if(!fin){
      cout << filename << " file could not be opened\n";
      exit(0);
  }
  string line;
  vector<int> data;
  while(getline(fin, line)){
    data.emplace_back(floor(pow(e_,stof(line))));

  }

  fin.close();
  return data;
}

vector<int> read_rnn_predict(string filename){
  ifstream fin;
  fin.open(filename.c_str());
  if(!fin){
      cout << filename << " file could not be opened\n";
      exit(0);
  }
  string line;
  vector<int> data;
  while(getline(fin, line)){
    data.emplace_back(stoi(line));

  }

  fin.close();
  return data;
}



torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
struct Net : torch::nn::Module {
  Net() {
    fc1 = register_module("fc1", torch::nn::Linear(1, 200));
    fc2 = register_module("fc2", torch::nn::Linear(200, 100));
    fc3 = register_module("fc3", torch::nn::Linear(100, 1));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(fc1(x));
    x = torch::relu(fc2(x));
    x = fc3(x);
    return x;
  }
};



void query_using_model(unordered_set<string> universe,int W0, unordered_map<string, int>& predict, vector<string>& all_dataset, int range, vector<CuckooFilter<string,Constants::FP_SIZE>> &ck_list, int ck_size, int new_ck_cnt,ofstream& fout,int dataset_query_range){
  double maxsize = ck_size * (ck_list.size()-1) + new_ck_cnt;
  bool use_model = false;
  if(dataset_query_range > maxsize && range > maxsize){
    use_model = true;
  }
  
  vector<string> dataset;
  dataset.assign(all_dataset.begin(), all_dataset.begin()+dataset_query_range);

  // fout << universe.size() << endl;
  vector<string> positive;
  vector<string> negative;
  for(int i = dataset.size()-1; i >= ((int)dataset.size() - range < 0 ? 0 : (int)dataset.size() - range); --i){
    if(universe.count(dataset[i]) != 0){
      universe.erase(dataset[i]);
      positive.emplace_back((dataset[i]));
    }
  }
  // fout << universe.size() << endl;
  
  negative.assign(universe.begin(), universe.end());

  int fp = 0;
  int fn = 0;
  
  auto begin = chrono::steady_clock::now();
  for(auto& data : positive){
    bool flag = false;
    int idx = ck_list.size()-1;
    int r = range;
    while(r > 0  && idx != -1){
      if(idx == ck_list.size()-1){
        r -= new_ck_cnt;
      }else{
        r -= ck_size;
      }
      if(ck_list[idx].Contain(data) == cuckoofilter::Ok){
        flag = true;
        break;
      }
      --idx;
    }

    if(!flag){

      if(use_model){
        int position = 0;
        if(predict.count(data) != 0){
          // position += (double)all_dataset.size()/predict[data];
          position = predict[data];
        }else{
          position = (double)all_dataset.size()/1;
        }

        if(position <= range && position > W0){
          continue;
        }
      }

      ++fn;
    }
  }


  for(auto& data : negative){
    bool flag = false;
    int idx = ck_list.size()-1;
    int r = range;
    while(r > 0  && idx != -1){
      if(idx == ck_list.size()-1){
        r -= new_ck_cnt;
      }else{
        r -= ck_size;
      }
      if(ck_list[idx].Contain(data) == cuckoofilter::Ok){
        flag = true;
        break;
      }
      --idx;
    }

    if(flag == false && use_model){
      int position = 0;
      if(predict.count(data) != 0){
        // position += (double)all_dataset.size()/predict[data];
        position = predict[data];
      }else{
        position = (double)all_dataset.size()/1;
      }

      if(position <= range && position > W0){
        flag = true;
      }
    }


    if(flag){
      ++fp;
    }
  }

  auto end = chrono::steady_clock::now();
  cout << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << endl;


  fout << "pos "+to_string(dataset_query_range)+" FP : " + to_string(fp) + " FN : " + to_string(fn) << " negative size : " << negative.size() << " positive size : " << positive.size() << endl;
  fout << "FP" + to_string(fp/(double)negative.size()) + ",FN" + to_string(fn/(double)positive.size()) << endl;
  // fout << "dataset range " << dataset_query_range << " max size " << to_string(maxsize) << endl;
}





int main(int argc, char **argv) {


  // read file
  string filepath = "";
  vector<string> dataset = read_file(filepath);


  unordered_set<string> universe;
  for(auto& data : dataset){
    universe.emplace(data);
  }

  ofstream fout;
  fout.open("output.txt");

  unordered_map<string, int> baseline;
  unordered_map<string, vector<double>> data_map;

  size_t cf_count = 249036;
  int W0 = 5000000;

  vector<CuckooFilter<string,Constants::FP_SIZE> > ck_list;

  std::random_device random;
  vector<unsigned __int128> random_vector(10);
  for(int i = 0; i < 10; ++i){
    random_vector[i] = random();
  }


  int cnt = 0;

  int new_ck_cnt = 0;

  double query_position = 0.01;

  auto net = Net();
  auto optimizer = torch::optim::Adam(net.parameters(), 0.00005);
  auto loss_func = torch::nn::MSELoss();

  double total_insert_time = 0;

  bool signal_flag = true; // true -> build model  false -> use prediction

  unordered_map<string, int> predict;
  if(signal_flag == false){
    cout << " use prediction " << endl;
    vector<int> predict_val = read_predict("/prediction.txt");
    vector<string> element = read_file("/universe.txt");
 
    assert(predict_val.size() == element.size());
    for(int i = 0; i < element.size(); ++i){
      predict.insert({element[i], predict_val[i]});
    }
  }

  for(auto& data : dataset){

    if(cnt % cf_count == 0){
      
      CuckooFilter<string, Constants::FP_SIZE> filter(cf_count,random_vector);
      ck_list.emplace_back(filter);
      if(ck_list.size() > Constants::L){
        ck_list.erase(ck_list.begin());
      }
      cout << endl;
      new_ck_cnt = 0;


      if(cnt / cf_count == 50 && signal_flag == true){
        //train model
        vector<float> x_data;
        vector<float> y_data;
        for(auto& d : data_map){
          if(d.second[2] != 1){
            x_data.emplace_back(stof(d.first) / 2000000);
            y_data.emplace_back(d.second[0] / 2000000);
          }
        }
        cout << x_data.size() << endl;

        torch::Tensor train_x = torch::tensor(x_data);
        train_x = train_x.reshape({x_data.size(),1});
        torch::Tensor train_y = torch::tensor(y_data);
        train_y = train_y.reshape({y_data.size(),1});


        net.train();

        int epoch = 2000;
        for(int i = 0; i < epoch; ++i){
          auto prediction = net.forward(train_x);    

          auto loss = loss_func(prediction, train_y);    

          optimizer.zero_grad();   
          loss.backward();         
          optimizer.step();        

          if(i % 100 == 0)
              cout << "Epoch " << i << ", loss: " << loss.item<float>() << endl;
        }

        cout << endl;

        cout << "predict size " << predict.size() << endl;
        double max_pred = 0;
        for(auto& pair : data_map){
          net.eval();
          float x = stof(pair.first) / 2000000;
          auto x_tensor = torch::tensor(x);
          x_tensor = x_tensor.reshape({1,1});
          auto mean_fit = net.forward(x_tensor);
          float mean = mean_fit.item().toFloat() * 2000000;
          max_pred = std::max(max_pred, (double)mean);
          predict.insert({pair.first, ceil(mean)});
        }
        cout << "predict size " << predict.size() << endl;
        cout << "max predict " << max_pred<<endl;
      
      }

    }


    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    if(data_map.count(data) == 0){
      /* msg[0] 均值 
        msg[1] 方差
        msg[2] 记录时所有数据总数
        msg[3] 记录时样本个数
      */
      vector<double> msg = {(double)cnt, 0, (double)cnt, 1};
      data_map[data] = msg;
    }else{

      if(signal_flag == false){
        
        if(predict.count(data) != 0)
          data_map[data][0] = dataset.size() / (double)predict[data];
        else
          data_map[data][0] = dataset.size() / 2.0;
        data_map[data][1] = (data_map[data][3]*data_map[data][1] + pow(cnt-data_map[data][2] - data_map[data][0], 2)) / (data_map[data][3]+1.0);
        data_map[data][2] = cnt;
        data_map[data][3] += 1;
        
      }else if(cnt / cf_count < 50){
        // 前四个window记录训练数据
        data_map[data][0] = ((cnt-data_map[data][2])+data_map[data][0]*data_map[data][3]) / (data_map[data][3]+1.0);
        data_map[data][1] = (data_map[data][3]*data_map[data][1] + pow(cnt-data_map[data][2] - data_map[data][0], 2)) / (data_map[data][3]+1.0);
        data_map[data][2] = cnt;
        data_map[data][3] += 1;
      }else if(signal_flag == true){
        // use model
        net.eval();
        // float x = log(stof(data)+1);
        float x = stof(data) / 2000000;
        auto x_tensor = torch::tensor(x);
        x_tensor = x_tensor.reshape({1,1});
        auto mean_fit = net.forward(x_tensor);
        // float mean = pow(10,mean_fit.item().toFloat());
        float mean = mean_fit.item().toFloat() * 2000000;


        data_map[data][0] = ((cnt-data_map[data][2])+mean*data_map[data][3]) / (data_map[data][3]+1.0);
        data_map[data][1] = (data_map[data][3]*data_map[data][1] + pow(cnt-data_map[data][2] - mean, 2)) / (data_map[data][3]+1.0);
        data_map[data][2] = cnt;
        data_map[data][3] += 1;
      }

    }

    ++cnt;

    ++new_ck_cnt;
    if(data_map[data][3] == 1){
      ck_list[ck_list.size()-1].Add(data);
    }else if(ck_list[ck_list.size()-1].Contain(data) != cuckoofilter::Ok){
      int len = ck_list.size();
      int pos = ceil(data_map[data][0] / cf_count);
      int offset = ceil(pow(data_map[data][1],0.5) / cf_count);
      for(int idx = (0 > len - 1 - pos - offset ? 0 : len - 1 - pos - offset); 
          idx <= (len <= len - 1 - pos + offset ? len - 1 : len - 1 - pos + offset);
          ++idx){
            // delete copy of data in previous cuckoo filter
            if(ck_list[idx].Contain(data) == cuckoofilter::Ok){
              ck_list[idx].Delete(data);
            }
          }
      ck_list[len-1].Add(data);  
    }

    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    total_insert_time += chrono::duration_cast<chrono::milliseconds>(end - begin).count();


    if(query_position - cnt / (double)dataset.size() <= 1e-8){
      query_using_model(universe,W0,predict,dataset,5000000,ck_list,cf_count,new_ck_cnt,fout,cnt);

      query_position += 0.01;
    }

  }

  cout << ck_list.size() << endl;

  cout << endl;
  cout << " total insert time : " << total_insert_time << " ms " << endl;



  double predict_time = 0;
  net.eval();
  chrono::steady_clock::time_point begin = chrono::steady_clock::now();
  if(signal_flag == true){
    for(auto& pair : data_map){
      float x = stof(pair.first) / 2000000;
      auto x_tensor = torch::tensor(x);
      x_tensor = x_tensor.reshape({1,1});
      auto mean_fit = net.forward(x_tensor);
      float mean = mean_fit.item().toFloat() * 2000000;
    }
  }
  chrono::steady_clock::time_point end = chrono::steady_clock::now();
  predict_time += chrono::duration_cast<chrono::milliseconds>(end - begin).count();
  cout << " predict size " << data_map.size() << " predict time " << predict_time << " ms "<< endl;



  for(int i = 0; i < ck_list.size(); ++i){
    fout << "rebalance" << ck_list[i].Rebalance() << endl;
    fout << "rebalance" << ck_list[i].Rebalance() << endl;
    fout << ck_list[i].Info() << endl;
    fout << ck_list[i].Size() << endl;
  }

  fout << " new_ck_cnt " << new_ck_cnt << " ck list size " << ck_list.size() << endl;

  fout.close();

  return 0;
}