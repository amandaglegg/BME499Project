//
//  Statistic.swift
//  GoodHealth
//
//  Created by Derrick Ushko on 2022-07-04.
//

import Foundation

struct Statistic: Codable {
    var current: Int
    var average: Double
    var deviation: Double
    var zScore: Double
    var change: Double
}
