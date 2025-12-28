/**
 * Test recommendations API directly
 */

const axios = require('axios');

const API_URL = 'http://localhost:3001/api';

async function testRecommendations() {
  console.log('🧪 Testing Recommendations API...\n');

  try {
    // 1. Login as test user (COLD_START)
    console.log('1️⃣ Logging in as test user (COLD_START)...');
    const loginResponse = await axios.post(`${API_URL}/auth/login`, {
      email: 'test@cofars.com',
      password: 'test123'
    });
    
    const token = loginResponse.data.access_token;
    console.log('✅ Login successful\n');

    // 2. Test context-aware recommendations with different contexts
    const contexts = [
      { timeSlot: 'morning', isWeekend: false, name: 'Morning Weekday' },
      { timeSlot: 'afternoon', isWeekend: false, name: 'Afternoon Weekday' },
      { timeSlot: 'evening', isWeekend: false, name: 'Evening Weekday' },
      { timeSlot: 'morning', isWeekend: true, name: 'Morning Weekend' },
      { timeSlot: 'evening', isWeekend: true, name: 'Evening Weekend' },
    ];

    for (const context of contexts) {
      console.log(`\n2️⃣ Testing ${context.name}...`);
      try {
        const response = await axios.get(`${API_URL}/recommendations/context-aware`, {
          headers: { Authorization: `Bearer ${token}` },
          params: {
            timeSlot: context.timeSlot,
            isWeekend: context.isWeekend,
            limit: 10
          }
        });

        const recommendations = response.data;
        console.log(`✅ Received ${recommendations.length} recommendations`);
        
        if (recommendations.length > 0) {
          console.log('   Top 3 products:');
          recommendations.slice(0, 3).forEach((product, index) => {
            console.log(`   ${index + 1}. ${product.name.substring(0, 50)}...`);
          });
        } else {
          console.log('   ⚠️  No recommendations returned!');
        }
      } catch (error) {
        console.log(`❌ Error: ${error.response?.data?.message || error.message}`);
      }
    }

    // 3. Test for-you recommendations
    console.log('\n\n3️⃣ Testing /recommendations/for-you...');
    try {
      const response = await axios.get(`${API_URL}/recommendations/for-you`, {
        headers: { Authorization: `Bearer ${token}` },
        params: { limit: 10 }
      });

      const recommendations = response.data;
      console.log(`✅ Received ${recommendations.length} recommendations`);
      
      if (recommendations.length > 0) {
        console.log('   Top 3 products:');
        recommendations.slice(0, 3).forEach((product, index) => {
          console.log(`   ${index + 1}. ${product.name.substring(0, 50)}...`);
        });
      }
    } catch (error) {
      console.log(`❌ Error: ${error.response?.data?.message || error.message}`);
    }

    // 4. Login as admin (POWER user) and test
    console.log('\n\n4️⃣ Logging in as admin user (POWER)...');
    const adminLoginResponse = await axios.post(`${API_URL}/auth/login`, {
      email: 'admin@cofars.com',
      password: 'admin123'
    });
    
    const adminToken = adminLoginResponse.data.access_token;
    console.log('✅ Login successful');

    console.log('\n5️⃣ Testing admin recommendations...');
    try {
      const response = await axios.get(`${API_URL}/recommendations/context-aware`, {
        headers: { Authorization: `Bearer ${adminToken}` },
        params: {
          timeSlot: 'morning',
          isWeekend: false,
          limit: 10
        }
      });

      const recommendations = response.data;
      console.log(`✅ Received ${recommendations.length} recommendations`);
      
      if (recommendations.length > 0) {
        console.log('   Top 3 products:');
        recommendations.slice(0, 3).forEach((product, index) => {
          console.log(`   ${index + 1}. ${product.name.substring(0, 50)}...`);
        });
      }
    } catch (error) {
      console.log(`❌ Error: ${error.response?.data?.message || error.message}`);
    }

  } catch (error) {
    console.error('❌ Test failed:', error.response?.data || error.message);
  }
}

testRecommendations();
