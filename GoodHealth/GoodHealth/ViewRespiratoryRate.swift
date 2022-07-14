//
//  ViewRespiratoryRate.swift
//  GoodHealth
//
//  Created by Derrick Ushko on 2022-07-04.
//

import SwiftUI

let sampleRR = Statistic(current: 14, average: 13.5, deviation: 0.6, zScore: 0.8, change: 0)

struct ViewRespiratoryRate: View {
    
    var statistic: Statistic
    
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack  (alignment: .center) {
                Image(systemName: "lungs.fill")
                Text("Respiratory Rate")
            }
            .font(.system(size: 18, weight: .bold, design: .default))
            .foregroundColor(.blue)
            
            HStack (alignment: .bottom) {
                Text("\(statistic.current)")
                    .font(.system(size: 24, weight: .bold, design: .rounded))
                Text("breaths/min")
                    .font(.system(size: 18, weight: .regular, design: .default))
                    .foregroundColor(.gray)
                Spacer()
                let roundedRR = String(format: "%.1f", statistic.change)
                if (statistic.change > 0) {
                    Text("+" + roundedRR)
                }
                else {
                    Text(roundedRR)
                }
                Text("breaths monthly")
                    .foregroundColor(.gray)
            }
            
            if (statistic.zScore > 1.5) {
                HStack {
                    Image(systemName: "exclamationmark.shield.fill")
                    Text("Respiratory rate is higher than normal")
                }
                .font(.system(size: 18, weight: .regular, design: .default))
                .foregroundColor(.red)
                .padding(.vertical, 5)
            }
        }
        //.padding()
    }
}

struct ViewRespiratoryRate_Previews: PreviewProvider {
    static var previews: some View {
        ViewRespiratoryRate(statistic: sampleRR)
    }
}
