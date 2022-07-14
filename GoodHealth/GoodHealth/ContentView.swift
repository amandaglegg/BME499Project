//
//  ContentView.swift
//  GoodHealth
//
//  Created by Derrick Ushko on 2022-06-14.
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
        
        /* NavigationView {
            
            List {
                ViewRestingHeartRate(statistic: restingHeartRateStatistics)
                    .listRowInsets(EdgeInsets())
                ViewHeartRateVariability(statistic: heartRateVariabilityStatistics)
                    .listRowInsets(EdgeInsets())
                ViewRespiratoryRate(statistic: respiratoryRateStatistics)
                    .listRowInsets(EdgeInsets())
            }
            .navigationTitle("Vitals")
        } */
        
        /* ScrollView(.vertical) {
            VStack(alignment: .leading, spacing: 25) {
                Text("Vitals")
                    .font(.system(size: 32, weight: .bold, design: .default))
                ViewRestingHeartRate(statistic: restingHeartRateStatistics)
                ViewHeartRateVariability(statistic: heartRateVariabilityStatistics)
                ViewRespiratoryRate(statistic: respiratoryRateStatistics)
                Text("About")
                    .font(.system(size: 32, weight: .bold, design: .default))
                AboutHeartRate()
            }
            .padding()
        } */
        
        TabView {
            
            ScrollView(.vertical) {
                
                VStack(alignment: .leading, spacing: 25) {
                    Text("Vitals")
                        .font(.system(size: 32, weight: .bold, design: .default))
                    ViewRestingHeartRate(statistic: restingHeartRateStatistics)
                    ViewHeartRateVariability(statistic: heartRateVariabilityStatistics)
                    ViewRespiratoryRate(statistic: respiratoryRateStatistics)
                    Text("About")
                        .font(.system(size: 32, weight: .bold, design: .default))
                    
                    ScrollView(.horizontal) {
                        HStack(alignment: .top, spacing: 20) {
                            AboutHeartRate()
                            AboutHeartRateVariability()
                            AboutRespiratoryRate()
                        }
                    }
                    
                }
                .padding()
            }
            
            .tabItem {
                Label("Home", systemImage: "heart.text.square.fill")
            }
            
            VStack {
                
                Button {
                    healthStore?.firstElectrocardiogram { firstVoltageCollection in
                        print(firstVoltageCollection ?? 0)
                    }
                } label: {
                    Text("1st ECG")
                        .padding(20)
                }
                .contentShape(Rectangle())
                .font(.system(size: 18, weight: .regular, design: .default))
                .foregroundColor(.white)
                .background(.pink)
                .cornerRadius(10)
                
                Button {
                    healthStore?.firstElectrocardiogram { firstVoltageCollection in
                        print(firstVoltageCollection ?? 0)
                    }
                } label: {
                    Text("2nd ECG")
                        .padding(20)
                }
                .contentShape(Rectangle())
                .font(.system(size: 18, weight: .regular, design: .default))
                .foregroundColor(.white)
                .background(.pink)
                .cornerRadius(10)
            }
            
            .tabItem {
                Label("ECG", systemImage: "waveform.path.ecg.rectangle.fill")
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
                        
                        /* print("\n Hello \n")
                        
                        healthStore.firstElectrocardiogram { firstVoltageCollection in
                            
                            print(firstVoltageCollection ?? 0)
                            
                        }
                        
                        print("\n Hello again :) \n")
                        
                        healthStore.secondElectrocardiogram { secondVoltageCollection in
                            
                            print(secondVoltageCollection ?? 0)
                            
                        } */
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
