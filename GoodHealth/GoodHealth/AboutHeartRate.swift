//
//  AboutHeartRate.swift
//  GoodHealth
//
//  Created by Derrick Ushko on 2022-07-06.
//

import SwiftUI

struct AboutHeartRate: View {
    
    var body: some View {
        
        VStack(alignment: .leading) {
            
            Image("Pink")
                .resizable()
                .frame(width: 220, height: 275)
            
            Text("Resting Heart Rate")
                .font(.system(size: 18, weight: .bold, design: .default))
                .padding(10)
            
            Text("A normal resting heart rate for adults ranges from 60 to 100 beats per minute. Generally, a lower heart rate at rest implies more efficient heart function and better cardiovascular fitness.")
                .font(.system(size: 14, weight: .regular, design: .default))
                .padding(.leading, 10)
                .padding(.trailing, 10)
                .padding(.bottom, 15)
        }
        .frame(width: 220)
        .cornerRadius(30)
    }
}

struct AboutHeartRate_Previews: PreviewProvider {
    static var previews: some View {
        AboutHeartRate()
    }
}
