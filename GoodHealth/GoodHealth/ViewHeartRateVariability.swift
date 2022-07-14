//
//  ViewHeartRateVariability.swift
//  GoodHealth
//
//  Created by Derrick Ushko on 2022-07-04.
//

import SwiftUI

let sampleHRV = Statistic(current: 28, average: 40.0, deviation: 7.6, zScore: -1.6, change: -12)

struct ViewHeartRateVariability: View {
    
    var statistic: Statistic
    
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack (alignment: .center) {
                Image(systemName: "bolt.heart.fill")
                Text("Heart Rate Variability")
            }
            .font(.system(size: 18, weight: .bold, design: .default))
            .foregroundColor(.purple)
            
            HStack (alignment: .bottom) {
                Text("\(statistic.current)")
                    .font(.system(size: 24, weight: .bold, design: .rounded))
                Text("ms")
                    .font(.system(size: 18, weight: .regular, design: .default))
                    .foregroundColor(.gray)
                Spacer()
                let roundedHRV = String(format: "%.1f", statistic.change)
                if (statistic.change > 0) {
                    Text("+" + roundedHRV)
                }
                else {
                    Text(roundedHRV)
                }
                Text("ms monthly")
                    .foregroundColor(.gray)
            }
            
            if (statistic.zScore < -0.5) {
                HStack {
                    Image(systemName: "exclamationmark.shield.fill")
                    Text("Variability is lower than normal")
                }
                .font(.system(size: 18, weight: .regular, design: .default))
                .foregroundColor(.red)
                .padding(.vertical, 5)
            }
        }
        //.padding()
    }
}

struct ViewHeartRateVariability_Previews: PreviewProvider {
    static var previews: some View {
        ViewHeartRateVariability(statistic: sampleHRV)
    }
}
