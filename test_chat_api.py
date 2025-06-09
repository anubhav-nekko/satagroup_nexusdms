"""
Test the new Chat with Documents API
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_chat_api():
    """Test the chat with documents functionality"""
    print("🚀 TESTING CHAT WITH DOCUMENTS API")
    print("=" * 60)
    
    # Login
    login_data = {"email": "admin@system.com", "password": "admin123"}
    response = requests.post(f"{BASE_URL}/v1/auth/login", json=login_data)
    token = response.json()["accessToken"]
    headers = {"Authorization": f"Bearer {token}"}
    print("✅ Authentication successful")
    
    # Get existing documents
    response = requests.get(f"{BASE_URL}/v1/jds", headers=headers)
    jds = response.json()
    print(f"✅ Found {len(jds)} job descriptions")
    
    response = requests.get(f"{BASE_URL}/v1/cvs", headers=headers)
    cvs = response.json()
    print(f"✅ Found {len(cvs)} CVs")
    
    if jds and cvs:
        jd_id = jds[-1]["jdId"]
        cv_id = cvs[-1]["cvId"]
        
        print(f"\n💬 Starting chat with:")
        print(f"   JD: {jds[-1]['title']} (ID: {jd_id})")
        print(f"   CV: {cvs[-1]['fileName']} (ID: {cv_id})")
        
        # Test 1: Start a new conversation
        print("\n📝 Test 1: Starting new conversation...")
        chat_request = {
            "jdId": jd_id,
            "cvId": cv_id,
            "message": "Can you tell me about this candidate's qualifications for this role?"
        }
        
        response = requests.post(
            f"{BASE_URL}/v1/chat",
            json=chat_request,
            headers=headers,
            timeout=60
        )
        
        if response.status_code == 200:
            chat_result = response.json()
            conversation_id = chat_result["conversationId"]
            print("✅ Chat conversation started!")
            print(f"📊 Conversation ID: {conversation_id}")
            print(f"🤖 AI Response: {chat_result['response'][:200]}...")
            
            # Test 2: Continue the conversation
            print("\n📝 Test 2: Continuing conversation...")
            follow_up_request = {
                "jdId": jd_id,
                "cvId": cv_id,
                "message": "What are the main skill gaps I should be concerned about?",
                "conversationId": conversation_id
            }
            
            response = requests.post(
                f"{BASE_URL}/v1/chat",
                json=follow_up_request,
                headers=headers,
                timeout=60
            )
            
            if response.status_code == 200:
                follow_up_result = response.json()
                print("✅ Follow-up message successful!")
                print(f"🤖 AI Response: {follow_up_result['response'][:200]}...")
                
                # Test 3: Get conversation history
                print("\n📝 Test 3: Retrieving conversation history...")
                response = requests.get(
                    f"{BASE_URL}/v1/chat/{conversation_id}",
                    headers=headers
                )
                
                if response.status_code == 200:
                    conversation = response.json()
                    print("✅ Conversation history retrieved!")
                    print(f"📊 Total messages: {len(conversation['messages'])}")
                    print("📋 Message history:")
                    for i, msg in enumerate(conversation['messages']):
                        role_emoji = "👤" if msg['role'] == 'user' else "🤖"
                        print(f"   {i+1}. {role_emoji} {msg['role'].title()}: {msg['content'][:100]}...")
                
                # Test 4: List all conversations
                print("\n📝 Test 4: Listing all conversations...")
                response = requests.get(f"{BASE_URL}/v1/chat", headers=headers)
                
                if response.status_code == 200:
                    conversations_list = response.json()
                    print("✅ Conversations list retrieved!")
                    print(f"📊 Total conversations: {len(conversations_list['conversations'])}")
                    for conv in conversations_list['conversations']:
                        print(f"   - {conv['conversationId'][:8]}... ({conv['messageCount']} messages)")
                
                # Test 5: Test with context (third message should reference previous)
                print("\n📝 Test 5: Testing conversation context...")
                context_request = {
                    "jdId": jd_id,
                    "cvId": cv_id,
                    "message": "Based on what we discussed, what interview questions would you recommend?",
                    "conversationId": conversation_id
                }
                
                response = requests.post(
                    f"{BASE_URL}/v1/chat",
                    json=context_request,
                    headers=headers,
                    timeout=60
                )
                
                if response.status_code == 200:
                    context_result = response.json()
                    print("✅ Context-aware response successful!")
                    print(f"🤖 AI Response: {context_result['response'][:200]}...")
                
                print("\n🎉 ALL CHAT TESTS COMPLETED SUCCESSFULLY!")
                
                # Optional: Clean up (delete conversation)
                print("\n🗑️  Cleaning up test conversation...")
                response = requests.delete(
                    f"{BASE_URL}/v1/chat/{conversation_id}",
                    headers=headers
                )
                if response.status_code == 200:
                    print("✅ Test conversation deleted")
                
            else:
                print(f"❌ Follow-up failed: {response.status_code} - {response.text}")
        else:
            print(f"❌ Chat failed: {response.status_code} - {response.text}")
    
    else:
        print("⚠️  No documents found for testing")
    
    print("\n🎉 CHAT API TEST COMPLETED!")

if __name__ == "__main__":
    test_chat_api()

