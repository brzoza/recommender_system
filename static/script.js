document.getElementById('recommendationForm').addEventListener('submit', function(e) {
    e.preventDefault();
    const customerId = document.getElementById('customer_id').value;
    fetch(`/recommendations?customer_id=${customerId}`)
        .then(response => response.json())
        .then(data => {
            const resultsDiv = document.getElementById('results');
            if (data.error) {
                resultsDiv.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
            } else {
                resultsDiv.innerHTML = `
                    <h2>Recommendations for Customer ${data.customer_id}</h2>
                    <div class="card mt-3">
                        <div class="card-header">Optimal Time Between Purchases</div>
                        <div class="card-body">
                            <p>Optimal Days: ${data.optimal_days_between_purchases || 'No data available'}</p>
                        </div>
                    </div>
                    <div class="card mt-3">
                        <div class="card-header">Association Rules</div>
                        <div class="card-body">
                            <p>Products: ${data.recommendations.association_rules.join(', ')}</p>
                            <p>Probability: ${data.probabilities.association_rules}</p>
                        </div>
                    </div>
                    <div class="card mt-3">
                        <div class="card-header">RNN</div>
                        <div class="card-body">
                            <p>Products: ${data.recommendations.rnn.join(', ')}</p>
                            <p>Probability: ${data.probabilities.rnn}</p>
                        </div>
                    </div>
                    <div class="card mt-3">
                        <div class="card-header">Collaborative Filtering</div>
                        <div class="card-body">
                            <p>Products: ${data.recommendations.collaborative_filtering.join(', ')}</p>
                            <p>Probability: ${data.probabilities.collaborative_filtering}</p>
                        </div>
                    </div>
                `;
            }
        })
        .catch(error => {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = `<div class="alert alert-danger">An error occurred: ${error.message}</div>`;
        });
});
