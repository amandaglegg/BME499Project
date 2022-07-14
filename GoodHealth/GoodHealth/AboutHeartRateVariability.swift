//
//  AboutHeartRateVariability.swift
//  GoodHealth
//
//  Created by Derrick Ushko on 2022-07-13.
//

import SwiftUI

struct AboutHeartRateVariability: View {
    
    var body: some View {
        
        VStack(alignment: .leading) {
            
            Image("Purple")
                .resizable()
                .frame(width: 220, height: 275)
            
            Text("Heart Rate Variability")
                .font(.system(size: 18, weight: .bold, design: .default))
                .padding(10)
            
            Text("Heart rate variability is where the amount of time between your heartbeats fluctuates slightly. Generally, low heart rate variability is considered a sign of current or future health problems.")
                .font(.system(size: 14, weight: .regular, design: .default))
                .padding(.leading, 10)
                .padding(.trailing, 10)
                .padding(.bottom, 15)
        }
        .frame(width: 220)
        .cornerRadius(30)
    }
}

struct AboutHeartRateVariability_Previews: PreviewProvider {
    static var previews: some View {
        AboutHeartRateVariability()
    }
}
