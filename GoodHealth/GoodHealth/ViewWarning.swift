//
//  ViewWarning.swift
//  GoodHealth
//
//  Created by Derrick Ushko on 2022-07-13.
//

import SwiftUI

struct ViewWarning: View {
    var body: some View {
       
        VStack(alignment: .leading, spacing: 5) {
            
            HStack (alignment: .center) {
                Image(systemName: "exclamationmark.shield")
                Text("Warning")
            }
            .font(.system(size: 24, weight: .bold, design: .default))
            .foregroundColor(.white)
            .padding(.top, 10)
            .padding(.leading, 10)
            .padding(.trailing, 10)
            
            Text("Your vitals are trending abnormally high.")
                .foregroundColor(.white)
                .padding(.bottom, 10)
                .padding(.leading, 10)
                .padding(.trailing, 10)

            
            HStack (alignment: .center) {
                Image(systemName: "allergens")
                Text("Are you feeling sick ?")
            }
            .font(.system(size: 18, weight: .bold, design: .default))
            .foregroundColor(.white)
            .padding(.bottom, 10)
            .padding(.leading, 10)
            .padding(.trailing, 10)
            
            VStack (alignment: .leading, spacing: 5){
                
                Text("Recommendations:")
                    .bold()
                
                HStack (alignment: .center, spacing: 10) {
                    Text("Drink plenty of water")
                    Image(systemName: "drop.fill")
                }
                
                HStack (alignment: .center, spacing: 10) {
                    Text("Get enough sleep")
                    Image(systemName: "powersleep")
                    
                }
                
                HStack (alignment: .center, spacing: 10) {
                    Text("Consider medications")
                    Image(systemName: "pills.fill")
                    
                }
            }
            .font(.system(size: 18, weight: .regular, design: .default))
            .padding(.bottom, 10)
            .padding(.leading, 10)
            .padding(.trailing, 10)
            .foregroundColor(.white)

        }
        .background(.red)
        .cornerRadius(20)
    }
}

struct ViewWarning_Previews: PreviewProvider {
    static var previews: some View {
        ViewWarning()
    }
}
