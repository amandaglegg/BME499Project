//
//  HighlightRow.swift
//  GoodHealth
//
//  Created by Derrick Ushko on 2022-07-04.
//

import SwiftUI

let sampleRHR = Statistic(current: 70, average: 64.3, deviation: 3.5, zScore: 1.2, change: -6)

struct ViewRestingHeartRate: View {
    
    var statistic: Statistic
    
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack (alignment: .center) {
                Image(systemName: "heart.fill")
                Text("Resting Heart Rate")
            }
            .font(.system(size: 18, weight: .bold, design: .default))
            .foregroundColor(.pink)
            
            HStack (alignment: .bottom) {
                Text("\(statistic.current)")
                    .font(.system(size: 24, weight: .bold, design: .rounded))
                Text("bpm")
                    .font(.system(size: 18, weight: .regular, design: .default))
                    .foregroundColor(.gray)
                Spacer()
                let roundedRHR = String(format: "%.0f", statistic.change)
                if (statistic.change > 0) {
                    Text("+" + roundedRHR)
                }
                else {
                    Text(roundedRHR)
                }
                Text("bpm monthly")
                    .foregroundColor(.gray)
            }
            
            if (statistic.zScore > 1) {
                HStack {
                    Image(systemName: "exclamationmark.shield.fill")
                    Text("Heart rate is higher than normal")
                }
                .font(.system(size: 18, weight: .regular, design: .default))
                .foregroundColor(.red)
                .padding(.vertical, 5)
            }
        }
        //.padding()
    }
    
}

struct ViewRestingHeartRate_Previews: PreviewProvider {
    static var previews: some View {
        ViewRestingHeartRate(statistic: sampleRHR)
    }
}
