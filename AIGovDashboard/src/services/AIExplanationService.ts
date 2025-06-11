const BASE_URL = 'https://api-service-storage-backend-staging-dot-block-convey-p1.uc.r.appspot.com';

export interface AIExplanationResponse {
  answer: string;
  success: boolean;
}

export const getExplanation = async (question: string): Promise<AIExplanationResponse> => {
  try {
    console.log('Fetching explanation for question:', question); // Log the question for debugging
    
    // Create FormData object with 'prompt' instead of 'question' to match backend expectations
    const formData = new FormData();
    formData.append('prompt', question); // Using 'prompt' to match backend API
    // No file is being sent, so we don't append a file
    
    // Log what's actually in the FormData object
    console.log('Form data prompt:', formData.get('prompt'));
    
    const response = await fetch(`${BASE_URL}/chatbot/chat`, {
      method: 'POST',
      // Remove the Content-Type header - fetch will set it automatically with boundary
      body: formData,
    });

    if (!response.ok) {
      console.error('API response error:', response.status, response.statusText);
      throw new Error(`Failed to get explanation: ${response.status} ${response.statusText}`);
    }

    // Response is a string, not JSON
    const responseText = await response.text();
    console.log('Explanation response:', responseText);
    
    return {
      answer: responseText || 'No explanation available.',
      success: true
    };
  } catch (error) {
    console.error('Error fetching explanation:', error);
    return {
      answer: 'Sorry, we could not retrieve an explanation at this time.',
      success: false
    };
  }
}; 