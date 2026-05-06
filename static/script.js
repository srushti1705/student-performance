/* Dashboard JavaScript Functionality */

document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard initialized');
    
    // Auto-refresh data every 5 minutes
    setInterval(refreshDashboardData, 300000);
});

function refreshDashboardData() {
    console.log('Refreshing dashboard data...');
    fetch('/api/dashboard-stats')
        .then(response => response.json())
        .then(data => {
            console.log('Dashboard stats updated:', data);
            updateStatsDisplay(data);
        })
        .catch(error => console.error('Error refreshing data:', error));
}

function updateStatsDisplay(data) {
    // This would update the dashboard cards dynamically
    // Implementation depends on specific DOM structure
}

// Placeholder for additional dashboard functionality
console.log('Dashboard script loaded successfully');

        
        if (remaining < 0) {
            remainingHoursDisplay.style.color = '#d9534f';
            realtimeWarning.innerText = '⚠️ Total daily hours cannot exceed 24';
        } else {
            remainingHoursDisplay.style.color = '#41431B';
            realtimeWarning.innerText = '';
            
            if (study < 2) {
                realtimeWarning.innerText = '⚠️ Very low study hours detected';
            } else if (extracurricular > 4) {
                realtimeWarning.innerText = '⚠️ Too much extracurricular activity';
            }
        }
    }

    timeInputs.forEach(input => {
        input.addEventListener('input', updateRealTimeFeedback);
    });
    
    // Initial call
    updateRealTimeFeedback();


    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const formData = new FormData(form);
        const data = {
            study_hours: parseFloat(formData.get('study_hours')),
            extracurricular_hours: parseFloat(formData.get('extracurricular_hours')),
            sleep_hours: parseFloat(formData.get('sleep_hours')),
            social_hours: parseFloat(formData.get('social_hours')),
            physical_activity: parseFloat(formData.get('physical_activity')),
            stress_level: formData.get('stress_level')
        };

        // Validation: Check if total hours exceed 24
        const totalHours = data.study_hours + data.extracurricular_hours + 
                          data.sleep_hours + data.social_hours + data.physical_activity;
        
        if (totalHours > 24) {
            alert(`⚠️ Unrealistic Input!\n\nTotal daily hours: ${totalHours.toFixed(1)}\nA day only has 24 hours.\n\nPlease adjust your inputs.`);
            return;
        }

        // Validation: Check individual ranges are removed as user can freely adjust up to 24 hours total


        submitBtn.disabled = true;
        loader.style.display = 'inline-block';
        document.querySelector('.btn-text').innerText = 'Generating...';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (response.ok) {
                // Update UI
                document.getElementById('scoreValue').innerText = Math.round(result.score);
                document.getElementById('categoryLabel').innerText = result.category;

                const updateList = (id, items) => {
                    const el = document.getElementById(id);
                    el.innerHTML = '';
                    items.forEach(msg => {
                        const li = document.createElement('li');
                        li.innerText = msg;
                        el.appendChild(li);
                    });
                };

                updateList('explanationList', result.explanation);
                updateList('suggestionsList', result.suggestions);
                updateList('whatIfRequirements', result.what_if.requirements);
                
                // Display personalized recommendations
                const personalizedText = document.getElementById('personalizedText');
                personalizedText.innerText = result.personalized_intro;
                
                const recommendationsContainer = document.getElementById('personalizedRecommendations');
                recommendationsContainer.innerHTML = '';
                result.personalized_recommendations.forEach((rec, index) => {
                    const div = document.createElement('div');
                    div.className = 'recommendation-item';
                    div.innerHTML = `
                        <div class="recommendation-priority">${index + 1}</div>
                        <div class="recommendation-content">
                            <strong>${rec.action}</strong>
                            ${rec.reason}
                            <small>💡 ${rec.impact}</small>
                        </div>
                    `;
                    recommendationsContainer.appendChild(div);
                });
                
                // Simplified Chart: Bar Chart
                updateChart(result.chart_data);

                document.getElementById('formCard').classList.add('hidden');
                resultSection.classList.remove('hidden');
                window.scrollTo({ top: 0, behavior: 'smooth' });
            } else {
                alert('Error: ' + result.error);
            }
        } catch (error) {
            alert('Failed to connect to the server.');
        } finally {
            submitBtn.disabled = false;
            loader.style.display = 'none';
            document.querySelector('.btn-text').innerText = 'Generate Analysis';
        }
    });

    function updateChart(chartData) {
        const ctx = document.getElementById('performanceChart').getContext('2d');
        if (performanceChart) performanceChart.destroy();

        performanceChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: chartData.labels,
                datasets: [{
                    label: 'Current Level',
                    data: chartData.current,
                    backgroundColor: '#41431B',
                    borderRadius: 5
                }, {
                    label: 'Success Benchmark',
                    data: chartData.ideal,
                    backgroundColor: '#AEB784',
                    borderRadius: 5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { 
                        beginAtZero: true, 
                        max: 100,
                        grid: { display: false },
                        title: {
                            display: true,
                            text: 'Optimal Level Capacity (%)',
                            font: { weight: 'bold' }
                        },
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    },
                    x: { grid: { display: false } }
                },
                plugins: {
                    legend: { position: 'bottom' },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.dataset.label + ': ' + Math.round(context.parsed.y) + '%';
                            }
                        }
                    }
                }
            }
        });
    }

    resetBtn.addEventListener('click', () => {
        resultSection.classList.add('hidden');
        document.getElementById('formCard').classList.remove('hidden');
        form.reset();
        timeInputs.forEach(input => {
            previousValues[input.name] = parseFloat(input.value) || 0;
            input.nextElementSibling.innerText = input.value + ' hours';
        });
        updateRealTimeFeedback();
    });
});
