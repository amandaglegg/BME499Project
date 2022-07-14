//
//  AboutRespiratoryRate.swift
//  GoodHealth
//
//  Created by Derrick Ushko on 2022-07-13.
//

import SwiftUI

struct AboutRespiratoryRate: View {
    
    var body: some View {
        
        VStack(alignment: .leading) {
            
            Image("Blue")
                .resizable()
                .frame(width: 220, height: 275)
            
            Text("Respiratory Rate")
                .font(.system(size: 18, weight: .bold, design: .default))
                .padding(10)
            
            Text("A person's respiratory rate is the number of breaths you take per minute. The normal respiration rate for an adult at rest is 12 to 20 breaths per minute. A respiration rate under 12 or over 25 breaths per minute while resting is considered abnormal.")
                .font(.system(size: 14, weight: .regular, design: .default))
                .padding(.leading, 10)
                .padding(.trailing, 10)
                .padding(.bottom, 15)
        }
        .frame(width: 220)
        .cornerRadius(30)
    }
}

struct AboutRespiratoryRate_Previews: PreviewProvider {
    static var previews: some View {
        AboutRespiratoryRate()
    }
}
