document.addEventListener('DOMContentLoaded', () => {
    const sentenceInput = document.getElementById('sentenceInput');
    const predictButton = document.getElementById('predictButton');
    const predictionList = document.getElementById('predictionList');
    const suggestionsDiv = document.getElementById('suggestions');

    // Function to fetch predictions
    const fetchPredictions = async () => {
        const text = sentenceInput.value.trim();
        if (!text) {
            predictionList.innerHTML = '';
            suggestionsDiv.innerHTML = '';
            suggestionsDiv.style.display = 'none';
            return;
        }

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: text })
            });

            const data = await response.json();

            if (data.error) {
                predictionList.innerHTML = `<li style="color: red;">Error: ${data.error}</li>`;
                suggestionsDiv.innerHTML = '';
                suggestionsDiv.style.display = 'none';
                return;
            }

            // Display predictions in the list
            predictionList.innerHTML = '';
            if (data.prediction && data.prediction.length > 0) {
                data.prediction.forEach(word => {
                    const li = document.createElement('li');
                    li.textContent = word;
                    predictionList.appendChild(li);
                });
            } else {
                predictionList.innerHTML = '<li>No predictions found.</li>';
            }

            // Display suggestions for autocomplete
            suggestionsDiv.innerHTML = '';
            if (data.prediction && data.prediction.length > 0) {
                data.prediction.forEach(word => {
                    const suggestionItem = document.createElement('div');
                    suggestionItem.classList.add('suggestion-item');
                    suggestionItem.textContent = word;
                    suggestionItem.addEventListener('click', () => {
                        const words = sentenceInput.value.trim().split(/\s+/);
                        words.pop(); // Remove the last (potentially incomplete) word
                        words.push(word); // Add the suggested word
                        sentenceInput.value = words.join(' ') + ' '; // Add a space for the next word
                        suggestionsDiv.style.display = 'none';
                        sentenceInput.focus(); // Keep focus on the textarea
                        fetchPredictions(); // Fetch new predictions after adding a word
                    });
                    suggestionsDiv.appendChild(suggestionItem);
                });
                suggestionsDiv.style.display = 'block';
            } else {
                suggestionsDiv.style.display = 'none';
            }

        } catch (error) {
            console.error('Error fetching predictions:', error);
            predictionList.innerHTML = `<li style="color: red;">Failed to fetch predictions.</li>`;
            suggestionsDiv.innerHTML = '';
            suggestionsDiv.style.display = 'none';
        }
    };

    // Event listener for button click (for explicit prediction)
    predictButton.addEventListener('click', fetchPredictions);

    // Event listener for textarea input (for real-time suggestions)
    let debounceTimer;
    sentenceInput.addEventListener('input', () => {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(fetchPredictions, 300); // Debounce for 300ms
    });

    // Hide suggestions when clicking outside the input/suggestions
    document.addEventListener('click', (event) => {
        if (!sentenceInput.contains(event.target) && !suggestionsDiv.contains(event.target)) {
            suggestionsDiv.style.display = 'none';
        }
    });
});