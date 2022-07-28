//
//  HealthStore.swift
//  GoodHealth
//
//  Created by Derrick Ushko on 2022-06-14.
//

import Foundation
import HealthKit

class HealthStore {
    
    var healthStore: HKHealthStore?
    var queryRHR: HKStatisticsCollectionQuery?
    var queryHRV: HKStatisticsCollectionQuery?
    var queryRR: HKStatisticsCollectionQuery?
    
    init() {
        if HKHealthStore.isHealthDataAvailable() {
            healthStore = HKHealthStore()
        }
    }
    
    func requestAuthorization(completion: @escaping (Bool) -> Void) {
        
        let readRHR = HKQuantityType.quantityType(forIdentifier: HKQuantityTypeIdentifier.restingHeartRate)!
        
        let readHRV = HKQuantityType.quantityType(forIdentifier: HKQuantityTypeIdentifier.heartRateVariabilitySDNN)!
        
        let readRR = HKQuantityType.quantityType(forIdentifier: HKQuantityTypeIdentifier.respiratoryRate)!
        
        let readECG = HKObjectType.electrocardiogramType()
        
        guard let healthStore = self.healthStore else {
            return completion(false)
        }
        
        healthStore.requestAuthorization(toShare: [], read: [readRHR, readRR, readHRV, readECG]) { (success, error) in
            completion(success)
        }
        
    }
    
    func latestRestingHeartRate(completion: @escaping (HKStatisticsCollection?) -> Void) {
        
        guard let restingHeartRateType = HKQuantityType.quantityType(forIdentifier: HKQuantityTypeIdentifier.restingHeartRate) else {
            return
        }
        
        let today = Date.now
        
        let oneMonthAgo = Calendar.current.date(byAdding: .day, value: -30, to: Date.now)!
        
        let dailyInterval = DateComponents(day: 1)
        
        let predicate = HKQuery.predicateForSamples(withStart: oneMonthAgo, end: today, options: .strictEndDate)
        
        queryRHR = HKStatisticsCollectionQuery(quantityType: restingHeartRateType, quantitySamplePredicate: predicate, options: .discreteAverage, anchorDate: today, intervalComponents: dailyInterval)
        
        queryRHR!.initialResultsHandler = { query, statisticsCollection, error in
            completion(statisticsCollection)
        }
        
        if let healthStore = healthStore, let queryRHR = self.queryRHR {
            healthStore.execute(queryRHR)
        }
        
    }
    
    func latestHeartRateVariability(completion: @escaping (HKStatisticsCollection?) -> Void) {
        
        guard let heartRateVariabilityType = HKQuantityType.quantityType(forIdentifier: HKQuantityTypeIdentifier.heartRateVariabilitySDNN) else {
            return
        }
        
        let today = Date.now
        
        let oneMonthAgo = Calendar.current.date(byAdding: .day, value: -30, to: Date.now)!
        
        let dailyInterval = DateComponents(day: 1)
        
        let predicate = HKQuery.predicateForSamples(withStart: oneMonthAgo, end: today, options: .strictEndDate)
        
        queryHRV = HKStatisticsCollectionQuery(quantityType: heartRateVariabilityType, quantitySamplePredicate: predicate, options: .discreteAverage, anchorDate: today, intervalComponents: dailyInterval)
        
        queryHRV!.initialResultsHandler = { query, statisticsCollection, error in
            completion(statisticsCollection)
        }
        
        if let healthStore = healthStore, let queryHRV = self.queryHRV {
            healthStore.execute(queryHRV)
        }
        
    }
    
    func latestRespiratoryRate(completion: @escaping (HKStatisticsCollection?) -> Void) {
        
        guard let respiratoryRateType = HKQuantityType.quantityType(forIdentifier: HKQuantityTypeIdentifier.respiratoryRate) else {
            return
        }
        
        let today = Date()
        
        let oneMonthAgo = Calendar.current.date(byAdding: .month, value: -1, to: Date())!
        
        let dailyInterval = DateComponents(day: 1)
        
        let predicate = HKQuery.predicateForSamples(withStart: oneMonthAgo, end: today, options: .strictStartDate)
        
        queryRHR = HKStatisticsCollectionQuery(quantityType: respiratoryRateType, quantitySamplePredicate: predicate, options: .discreteAverage, anchorDate: today, intervalComponents: dailyInterval)
        
        queryRHR!.initialResultsHandler = { query, statisticsCollection, error in
            completion(statisticsCollection)
        }
        
        if let healthStore = healthStore, let queryRHR = self.queryRHR {
            healthStore.execute(queryRHR)
        }
        
    }
    
    func firstElectrocardiogram(completion: @escaping ([Double]?) -> Void) {
        
        if #available(iOS 14.0, *) {
                let predicate = HKQuery.predicateForSamples(withStart: Date.distantPast,end: Date.distantFuture,options: .strictEndDate)
                let sortDescriptor = NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: false)
                let queryFirstECG = HKSampleQuery(sampleType: HKObjectType.electrocardiogramType(), predicate: predicate, limit: 0, sortDescriptors: [sortDescriptor]){ (query, samples, error) in
                    guard let samples = samples,
                          let firstSample = samples[0] as? HKElectrocardiogram else {
                        return
                    }
                    
                    print(firstSample)
                    
                    var firstSamples = [Double] ()
                    let queryFirstVoltage = HKElectrocardiogramQuery(firstSample) { (query, result) in
                        
                        switch result {
                        case .error(let error):
                            print("error: ", error)
                            
                        case .measurement(let value):
                            //print("value: ", value)
                            let sampleOne = value.quantity(for: .appleWatchSimilarToLeadI)!.doubleValue(for: HKUnit.volt())
                            firstSamples.append(sampleOne)
                            
                        case .done:
                            
                            // export ECG data here
                            
                            
                            completion(firstSamples)
                            print("done")
                        @unknown default:
                            fatalError()
                        }
                    }
                    
                    self.healthStore!.execute(queryFirstVoltage)
                }
                
            healthStore!.execute(queryFirstECG)
        } else {
            // Fallback on earlier versions
        }
    }
    
    func secondElectrocardiogram(completion: @escaping ([Double]?) -> Void) {
        
        if #available(iOS 14.0, *) {
                let predicate = HKQuery.predicateForSamples(withStart: Date.distantPast,end: Date.distantFuture,options: .strictEndDate)
                let sortDescriptor = NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: false)
                let querySecondECG = HKSampleQuery(sampleType: HKObjectType.electrocardiogramType(), predicate: predicate, limit: 0, sortDescriptors: [sortDescriptor]){ (query, samples, error) in
                    guard let samples = samples,
                          let secondSample = samples[1] as? HKElectrocardiogram else {
                        return
                    }
                    
                    print(secondSample)
                    
                    var secondSamples = [Double] ()
                    let querySecondVoltage = HKElectrocardiogramQuery(secondSample) { (query, result) in
                        
                        switch result {
                        case .error(let error):
                            print("error: ", error)
                            
                        case .measurement(let value):
                            //print("value: ", value)
                            let sampleTwo = value.quantity(for: .appleWatchSimilarToLeadI)!.doubleValue(for: HKUnit.volt())
                            secondSamples.append(sampleTwo)
                            
                        case .done:
                            completion(secondSamples)
                            print("done")
                        @unknown default:
                            fatalError()
                        }
                    }
                    
                    self.healthStore!.execute(querySecondVoltage)
                }
                
            healthStore!.execute(querySecondECG)
        } else {
            // Fallback on earlier versions
        }
    }
    
}
