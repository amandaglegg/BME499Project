//
//  ContentView.swift
//  GoodHealth
//
//  Created by Derrick Ushko & Amanda Glegg on 2022-07-25.
//

import SwiftUI
import HealthKit

struct ContentView: View {
    
    private var healthStore: HealthStore?
    
    @State var restingHeartRates: [RestingHeartRate] = [RestingHeartRate]()
    @State var heartRateVariabilities: [HeartRateVariability] = [HeartRateVariability]()
    @State var respiratoryRates: [RespiratoryRate] = [RespiratoryRate]()
    
    @State var restingHeartRateValues: [Int] = [Int]()
    @State var heartRateVariabilityValues: [Int] = [Int]()
    @State var respiratoryRateValues: [Int] = [Int]()
    
    @State var restingHeartRateStatistics = Statistic(current: 0, average: 0, deviation: 0, zScore: 0, change: 0)
    @State var heartRateVariabilityStatistics = Statistic(current: 0, average: 0, deviation: 0, zScore: 0, change: 0)
    @State var respiratoryRateStatistics = Statistic(current: 0, average: 0, deviation: 0, zScore: 0, change: 0)
    
    @State private var file: ECGDocument = ECGDocument(sample: "")
    @State private var isExportingExerciseECG: Bool = false
    @State private var isExportingRestingECG: Bool = false
    
    init() {
        healthStore = HealthStore()
    }
    
    private func updateUIFromRestingHeartRate(_ restingHeartRateCollection: HKStatisticsCollection) {
        
        let startDate = Calendar.current.date(byAdding: .month, value: -1, to: Date())!
        let endDate = Calendar.current.date(byAdding: .day, value: -1, to: Date())!
        let beatsPerMinute: HKUnit = HKUnit.count().unitDivided(by: HKUnit.minute())
        
        restingHeartRateCollection.enumerateStatistics(from: startDate, to: endDate) { (statistics, stop) in
            
            let restingHeartRateValue = statistics.averageQuantity()?.doubleValue(for: beatsPerMinute)
            let restingHeartRateEntry = RestingHeartRate(value: Int(restingHeartRateValue ?? 0), date: statistics.startDate)

            restingHeartRateValues.append(Int(restingHeartRateValue ?? 0))
            restingHeartRates.append(restingHeartRateEntry)
            
        }
        
        restingHeartRateValues.reverse()
        restingHeartRates.reverse()
        
    }
    
    private func updateUIFromHeartRateVariability(_ heartRateVariabilityCollection: HKStatisticsCollection) {
        
        let startDate = Calendar.current.date(byAdding: .month, value: -1, to: Date())!
        let endDate = Calendar.current.date(byAdding: .day, value: -1, to: Date())!
        let variability: HKUnit = HKUnit.secondUnit(with: .milli)
        
        heartRateVariabilityCollection.enumerateStatistics(from: startDate, to: endDate) { (statistics, stop) in
            
            let heartRateVariabilityValue = statistics.averageQuantity()?.doubleValue(for: variability)
            let heartRateVariabilityEntry = HeartRateVariability(value: Int(heartRateVariabilityValue ?? 0), date: statistics.startDate)

            heartRateVariabilityValues.append(Int(heartRateVariabilityValue ?? 0))
            heartRateVariabilities.append(heartRateVariabilityEntry)
            
        }
        
        heartRateVariabilityValues.reverse()
        heartRateVariabilities.reverse()
        
    }
    
    private func updateUIFromRespiratoryRate(_ respiratoryRateCollection: HKStatisticsCollection) {
        
        let startDate = Calendar.current.date(byAdding: .month, value: -1, to: Date())!
        let endDate = Calendar.current.date(byAdding: .day, value: -1, to: Date())!
        let breathsPerMinute: HKUnit = HKUnit.count().unitDivided(by: HKUnit.minute())
        
        respiratoryRateCollection.enumerateStatistics(from: startDate, to: endDate) { (statistics, stop) in
            
            let respiratoryRateValue = statistics.averageQuantity()?.doubleValue(for: breathsPerMinute)
            let respiratoryRateEntry = RespiratoryRate(value: Int(respiratoryRateValue ?? 0), date: statistics.startDate)

            respiratoryRateValues.append(Int(respiratoryRateValue ?? 0))
            respiratoryRates.append(respiratoryRateEntry)
            
        }
        
        respiratoryRateValues.reverse()
        respiratoryRates.reverse()
        
    }
    
    func calculateStatistics(values: [Int]?) -> Statistic {
        
        var onlyZeros = false
        if values!.count > 0
        {
          for i in(0..<values!.count)
          {
              if values![i] == 0
            {
              onlyZeros = true
             }
              else
            {
              onlyZeros = false
             }

           }
        }
        
        if onlyZeros {
            
            let result: Statistic = Statistic(current: 0, average: 0, deviation: 0, zScore: 0, change: 0)
            
            return result
            
        }
        
        else {
            
            let filtered = values!.filter { $0 != 0 }
            let size = Double(filtered.count)
            let x = filtered[0]
           
            let mu = filtered.reduce(0.0) {
                return $0 + Double($1)/size
            }
            
            let sumOfSquaredAvgDiff = filtered.map { pow(Double($0) - mu, 2.0)}.reduce(0, {$0 + $1})
            
            let sigma = sqrt(sumOfSquaredAvgDiff / size)
            
            let z = (Double(x) - mu) / sigma
            
            let delta = (Double(x) - mu)
            
            let result: Statistic = Statistic(current: x, average: mu, deviation: sigma, zScore: z, change: delta)
            
            return result
            
        }
        
    }
    
    var body: some View {
        
        TabView {
            
            ScrollView(.vertical) {
                
                VStack(alignment: .leading, spacing: 25) {
                    
                    Text("Vitals")
                        .font(.system(size: 32, weight: .bold, design: .default))
                    
                    ViewRestingHeartRate(statistic: restingHeartRateStatistics)
                    ViewHeartRateVariability(statistic: heartRateVariabilityStatistics)
                    ViewRespiratoryRate(statistic: respiratoryRateStatistics)
                    
                    if (heartRateVariabilityStatistics.zScore > 1 &&
                        heartRateVariabilityStatistics.zScore < 0.5 &&
                        respiratoryRateStatistics.zScore > 1.5) {
                        ViewWarning()
                    }
                    
                    Text("About")
                        .font(.system(size: 32, weight: .bold, design: .default))
                    
                    ScrollView(.horizontal) {
                        HStack(alignment: .top, spacing: 20) {
                            AboutHeartRate()
                            AboutHeartRateVariability()
                            AboutRespiratoryRate()
                        }
                    }
                    
                    HStack (alignment: .center, spacing: 1){
                        Text("Learn more about health data at ")
                        Button("Cleveland Clinic") {
                            if let url = URL(string: "https://my.clevelandclinic.org/health") {
                                UIApplication.shared.open(url)
                            }
                        }
                        .font(.system(size: 14, weight: .bold))
                    }
                    .font(.system(size: 14, weight: .regular, design: .default))
                    .foregroundColor(.gray)
                    .padding()
                    
                }
                .padding()
            }
            
            .tabItem {
                Label("Health Trends", systemImage: "heart.text.square.fill")
            }
             
            ScrollView (.vertical){
                VStack (alignment: .leading, spacing: 20){
                    
                    
                    Text("Heart Disease \nPrediction")
                        .font(.system(size: 32, weight: .bold, design: .default))
                        .frame(alignment: .leading)
                    
                    
                    HStack (alignment: .top, spacing: 10){
                        Text("1.")
                        VStack (alignment: .leading){
                            Text("Using your Apple Watch, record your ''Resting ECG'' while seated")
                            HStack (spacing: 1){
                                Text("Learn how to record an ECG ")
                                Button("here") {
                                    if let url = URL(string: "https://support.apple.com/en-ca/HT208955") {
                                        UIApplication.shared.open(url)
                                    }
                                }
                                .font(.system(size: 18, weight: .bold, design: .default))
                            }
                            .foregroundColor(.gray)
                            .padding(.top, 5)
                        }
                    }

                    HStack (alignment: .top){
                        Text("2.")
                        Text("Jog on the spot for 1 minute")
                    }

                    HStack (alignment: .top){
                        Text("3.")
                        Text("Record your ''Exercise ECG''")
                    }
                    
                    HStack (alignment: .top){
                        Text("4.")
                        Text("Upload ECGs (press buttons)")
                    }
                    
                    HStack {
                        Spacer()
                        Button {
                            healthStore?.secondElectrocardiogram { secondVoltageCollection in
                                var csvString = "\("Lead I")\n"
                                for reading in secondVoltageCollection! {
                                    let microvoltage = Double(reading * 1000000)
                                    csvString = csvString.appending("\(microvoltage)\n")
                                }
                                isExportingRestingECG = true
                                file.sample = csvString
                            }
                            
                        } label: {
                            Text("Resting ECG")
                        }
                        
                        .contentShape(Rectangle())
                        .frame(width: 150, height: 65, alignment: .center)
                        .font(.system(size: 18, weight: .regular, design: .default))
                        .foregroundColor(.white)
                        .background(.pink)
                        .cornerRadius(10)
                        .fileExporter(
                              isPresented: $isExportingRestingECG,
                              document: file,
                              contentType: .plainText,
                              defaultFilename: "resting_ECG"
                          ) { result in
                              if case .success = result {
                                  print("Export completed.")
                              } else {
                                  print("Export failed.")
                              }
                          }
                        Spacer()
                    }
                    
                    HStack {
                        Spacer()
                        Button {
                            healthStore?.firstElectrocardiogram { firstVoltageCollection in
                                var csvString = "\("Lead I")\n"
                                for reading in firstVoltageCollection! {
                                    let microvoltage = Double(reading * 1000000)
                                    csvString = csvString.appending("\(microvoltage)\n")
                                }
                                isExportingExerciseECG = true
                                file.sample = csvString
                            }
                        } label: {
                            Text("Exercise ECG")
                        }
                        .contentShape(Rectangle())
                        .frame(width: 150, height: 65, alignment: .center)
                        .font(.system(size: 18, weight: .regular, design: .default))
                        .foregroundColor(.white)
                        .background(.pink)
                        .cornerRadius(10)
                        .fileExporter(
                              isPresented: $isExportingExerciseECG,
                              document: file,
                              contentType: .plainText,
                              defaultFilename: "exercise_ECG"
                          ) { result in
                              if case .success = result {
                                  print("Export completed.")
                              } else {
                                  print("Export failed.")
                              }
                          }
                        Spacer()
                    }
                    
                    HStack (alignment: .top) {
                        Text("5.")
                        Text("Head to our website to get your results")
                    }
                    
                    HStack {
                        Spacer()
                        Button {
                            if let url = URL(string: "https://studentweb.uvic.ca/~vhartman/Capstone%20Website/templates/Consent") {
                                UIApplication.shared.open(url)
                            }
                        } label: {
                            Text("Take me there!")

                        }
                        .contentShape(Rectangle())
                        .frame(width: 150, height: 65, alignment: .center)
                        .font(.system(size: 18, weight: .regular, design: .default))
                        .foregroundColor(.white)
                        .background(.pink)
                        .cornerRadius(10)
                        
                        Spacer()
                        
                    }
                    
                    HStack (spacing: 1){
                        Spacer()
                        Text("")
                        Button("Click here to learn about our project Health AI Monitoring System") {
                            if let url = URL(string: "https://studentweb.uvic.ca/~vhartman/Capstone%20Website/templates/") {
                                UIApplication.shared.open(url)
                            }
                        }
                        .font(.system(size: 18, weight: .regular, design: .default))
                        Spacer()
                    }
                    .foregroundColor(.gray)
                    .padding(.top, 5)
                }
                .padding(20)
            }

            
            .tabItem {
                Label("Heart Disease Predictions", systemImage: "waveform.path.ecg.rectangle.fill")
            }
            
        }
        
        .onAppear {
            
            if let healthStore = healthStore {
                
                healthStore.requestAuthorization { success in
                    
                    if success {
                        
                        healthStore.latestRestingHeartRate { restingHeartRateCollection in
                            
                            if let restingHeartRateCollection = restingHeartRateCollection {
                                
                                updateUIFromRestingHeartRate(restingHeartRateCollection)
                                
                                restingHeartRateStatistics = calculateStatistics(values: restingHeartRateValues)
                                
                                //print(restingHeartRateStatistics)
                                //print(restingHeartRateValues)
                                //print(restingHeartRates)
                            }
                            
                        }
                        
                        healthStore.latestHeartRateVariability { heartRateVariabilityCollection in
                            
                            if let heartRateVariabilityCollection = heartRateVariabilityCollection {
                                
                                updateUIFromHeartRateVariability(heartRateVariabilityCollection)
                                
                                heartRateVariabilityStatistics = calculateStatistics(values: heartRateVariabilityValues)
                                
                                //print(heartRateVariabilityStatistics)
                                //print(heartRateVariabilityValues)
                                //print(heartRateVariabilities)
                            }
                                                        
                        }
                        
                        healthStore.latestRespiratoryRate { respiratoryRateCollection in
                            
                            if let respiratoryRateCollection = respiratoryRateCollection {
                                
                                updateUIFromRespiratoryRate(respiratoryRateCollection)
                                                                
                                respiratoryRateStatistics = calculateStatistics(values: respiratoryRateValues)
                                
                                //print(respiratoryRateStatistics)
                            }
                        }
                    }
                }
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
