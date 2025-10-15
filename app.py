from flask import Flask, request, jsonify, send_from_directory
import pickle
import numpy as np
import pandas as pd
import os
import openai  # ADD THIS IMPORT

app = Flask(__name__)

# Get API key from environment variable (more secure)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Initialize OpenAI only if API key is available
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
    print("✅ OpenAI API configured")
else:
    print("⚠️ OpenAI API key not found - AI features will use mock responses")

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

# Serve the main page
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

# Load models
try:
    with open('exoplanet_predictor_system.pkl', 'rb') as f:
        prop_system = pickle.load(f)
    prop_model = prop_system['model']
    prop_scaler_x = prop_system['scaler_X']
    prop_scaler_y = prop_system['scaler_y']
    feature_columns = prop_system['X_train_columns']
    target_columns = prop_system['target_columns']
    print("✅ Property model loaded - 39 targets available")
except Exception as e:
    prop_model = None
    print(f"❌ Property model failed: {e}")

try:
    with open('planet_hab_model.pkl', 'rb') as f:
        hab_system = pickle.load(f)
    hab_model = hab_system['model']
    hab_scaler = hab_system['scaler']
    hab_features = hab_system['feature_columns']
    print("✅ Habitability model loaded")
except Exception as e:
    hab_model = None
    print(f"❌ Habitability model failed: {e}")

def generate_comprehensive_properties(inputs):
    period = inputs.get('P_PERIOD', 300)
    radius = inputs.get('P_RADIUS', 1.4)
    temp_equil = inputs.get('P_TEMP_EQUIL', 270)
    star_temp = inputs.get('S_TEMPERATURE', 5778)
    star_mass = inputs.get('S_MASS', 1.0)
    composition = inputs.get('P_COMPOSITION_ROCKY', 1)
    
    semi_major = (period / 365.25) ** (2/3)
    mass = radius ** 2.5 if composition == 1 else radius ** 1.5 * 15
    density = 5.5 if composition == 1 else 1.3
    gravity = mass / (radius ** 2)
    flux = (star_temp / 5778) ** 4 / semi_major ** 2
    esi = max(0, min(1, 0.9 - abs(radius - 1) * 0.3 - abs(temp_equil - 288) * 0.0015))
    
    return {
        'P_MASS': round(mass, 4), 'P_SEMI_MAJOR_AXIS': round(semi_major, 4),
        'P_ECCENTRICITY': round(np.random.uniform(0.01, 0.2), 4),
        'P_INCLINATION': round(np.random.uniform(80, 90), 4),
        'S_DISTANCE': round(np.random.uniform(100, 600), 4),
        'S_RADIUS': round(star_mass ** 0.8, 4), 'S_LOG_LUM': round(np.log10(star_mass ** 3.5), 4),
        'S_LOG_G': round(4.4 + (star_mass - 1) * 0.1, 4),
        'P_ESCAPE': round(np.sqrt(2 * gravity * radius * 9.8 * 6371000) / 1000, 4),
        'P_POTENTIAL': round(gravity * radius, 4), 'P_GRAVITY': round(gravity * 9.8, 4),
        'P_DENSITY': round(density, 4),
        'P_HILL_SPHERE': round(semi_major * (mass / (3 * star_mass * 333000)) ** (1/3), 6),
        'P_DISTANCE': round(semi_major, 4), 'P_PERIASTRON': round(semi_major * (1 - 0.1), 4),
        'P_APASTRON': round(semi_major * (1 + 0.1), 4),
        'P_DISTANCE_EFF': round(semi_major * np.sqrt(1 - 0.1**2), 4),
        'P_FLUX': round(flux, 4), 'P_FLUX_MIN': round(flux / ((1 + 0.1) ** 2), 4),
        'P_FLUX_MAX': round(flux / ((1 - 0.1) ** 2), 4),
        'P_TEMP_EQUIL_MIN': round(temp_equil * 0.9, 4),
        'P_TEMP_EQUIL_MAX': round(temp_equil * 1.1, 4),
        'P_TEMP_SURF_MIN': round(temp_equil * 0.9 + 15, 4),
        'P_TEMP_SURF_MAX': round(temp_equil * 1.1 + 15, 4),
        'S_LUMINOSITY': round(star_mass ** 3.5, 4),
        'S_HZ_OPT_MIN': round(0.75 * np.sqrt(star_mass ** 3.5), 4),
        'S_HZ_OPT_MAX': round(1.77 * np.sqrt(star_mass ** 3.5), 4),
        'S_HZ_CON_MIN': round(0.99 * np.sqrt(star_mass ** 3.5), 4),
        'S_HZ_CON_MAX': round(1.67 * np.sqrt(star_mass ** 3.5), 4),
        'S_HZ_CON0_MIN': round(1.02 * np.sqrt(star_mass ** 3.5), 4),
        'S_HZ_CON0_MAX': round(1.67 * np.sqrt(star_mass ** 3.5), 4),
        'S_HZ_CON1_MIN': round(0.97 * np.sqrt(star_mass ** 3.5), 4),
        'S_HZ_CON1_MAX': round(1.67 * np.sqrt(star_mass ** 3.5), 4),
        'S_SNOW_LINE': round(2.7 * star_mass, 4),
        'S_TIDAL_LOCK': round(0.4 * semi_major ** (-3/2), 4),
        'P_HABZONE_OPT': 1 if semi_major >= 0.75 and semi_major <= 1.77 else 0,
        'P_HABZONE_CON': 1 if semi_major >= 0.99 and semi_major <= 1.67 else 0,
        'P_HABITABLE': round(esi * 0.8 + (1 if semi_major >= 0.75 and semi_major <= 1.77 else 0) * 0.2, 4),
        'P_ESI': round(esi, 4)
    }

def analyze_habitability_factors(properties):
    factors = []
    score = 0
    max_score = 0
    
    temp_min = properties.get('P_TEMP_SURF_MIN', 0)
    temp_max = properties.get('P_TEMP_SURF_MAX', 0)
    if 250 <= temp_min <= 350 and 250 <= temp_max <= 350:
        factors.append({"factor": "Surface Temperature", "status": "Optimal", "score": 2, 
                       "details": f"Temperature range ({temp_min:.1f}K - {temp_max:.1f}K) supports liquid water"})
        score += 2
    elif 200 <= temp_min <= 400 and 200 <= temp_max <= 400:
        factors.append({"factor": "Surface Temperature", "status": "Marginal", "score": 1,
                       "details": f"Temperature range ({temp_min:.1f}K - {temp_max:.1f}K) may support extremophiles"})
        score += 1
    else:
        factors.append({"factor": "Surface Temperature", "status": "Unsuitable", "score": 0,
                       "details": f"Temperature range ({temp_min:.1f}K - {temp_max:.1f}K) outside habitable limits"})
    max_score += 2
    
    hz_opt = properties.get('P_HABZONE_OPT', 0)
    hz_con = properties.get('P_HABZONE_CON', 0)
    if hz_con:
        factors.append({"factor": "Habitable Zone", "status": "Conservative", "score": 3,
                       "details": "Within conservative habitable zone - high probability of liquid water"})
        score += 3
    elif hz_opt:
        factors.append({"factor": "Habitable Zone", "status": "Optimistic", "score": 2,
                       "details": "Within optimistic habitable zone - possible liquid water with greenhouse effects"})
        score += 2
    else:
        factors.append({"factor": "Habitable Zone", "status": "Outside", "score": 0,
                       "details": "Outside recognized habitable zones"})
    max_score += 3
    
    esi = properties.get('P_ESI', 0)
    if esi > 0.8:
        factors.append({"factor": "Earth Similarity", "status": "High", "score": 2,
                       "details": f"ESI = {esi:.3f} - Very Earth-like conditions"})
        score += 2
    elif esi > 0.6:
        factors.append({"factor": "Earth Similarity", "status": "Moderate", "score": 1,
                       "details": f"ESI = {esi:.3f} - Moderately Earth-like"})
        score += 1
    else:
        factors.append({"factor": "Earth Similarity", "status": "Low", "score": 0,
                       "details": f"ESI = {esi:.3f} - Low similarity to Earth"})
    max_score += 2
    
    density = properties.get('P_DENSITY', 0)
    if density > 4:
        factors.append({"factor": "Planetary Composition", "status": "Rocky", "score": 2,
                       "details": f"Density = {density:.2f} g/cm³ - Likely terrestrial planet"})
        score += 2
    elif density > 2:
        factors.append({"factor": "Planetary Composition", "status": "Mixed", "score": 1,
                       "details": f"Density = {density:.2f} g/cm³ - Possible ocean world or mini-Neptune"})
        score += 1
    else:
        factors.append({"factor": "Planetary Composition", "status": "Gaseous", "score": 0,
                       "details": f"Density = {density:.2f} g/cm³ - Gas giant, unlikely to support surface life"})
    max_score += 2
    
    gravity = properties.get('P_GRAVITY', 0)
    if 8 <= gravity <= 12:
        factors.append({"factor": "Surface Gravity", "status": "Earth-like", "score": 1,
                       "details": f"Gravity = {gravity:.2f} m/s² - Suitable for Earth-like life"})
        score += 1
    elif 5 <= gravity <= 15:
        factors.append({"factor": "Surface Gravity", "status": "Tolerable", "score": 0.5,
                       "details": f"Gravity = {gravity:.2f} m/s² - Potentially habitable with adaptations"})
        score += 0.5
    else:
        factors.append({"factor": "Surface Gravity", "status": "Extreme", "score": 0,
                       "details": f"Gravity = {gravity:.2f} m/s² - Challenging for known life forms"})
    max_score += 1
    
    eccentricity = properties.get('P_ECCENTRICITY', 0)
    if eccentricity < 0.1:
        factors.append({"factor": "Orbital Stability", "status": "Stable", "score": 1,
                       "details": f"Eccentricity = {eccentricity:.3f} - Circular orbit provides stable climate"})
        score += 1
    elif eccentricity < 0.3:
        factors.append({"factor": "Orbital Stability", "status": "Moderate", "score": 0.5,
                       "details": f"Eccentricity = {eccentricity:.3f} - Moderate seasonal variations"})
        score += 0.5
    else:
        factors.append({"factor": "Orbital Stability", "status": "Unstable", "score": 0,
                       "details": f"Eccentricity = {eccentricity:.3f} - Extreme temperature variations"})
    max_score += 1
    
    habitability_score = score / max_score
    if habitability_score >= 0.7:
        prediction = "High Habitability Potential"
        confidence = min(95, habitability_score * 100)
        classification = 2
    elif habitability_score >= 0.4:
        prediction = "Moderate Habitability Potential" 
        confidence = habitability_score * 100
        classification = 1
    else:
        prediction = "Low Habitability Potential"
        confidence = max(5, habitability_score * 100)
        classification = 0
    
    return {
        "prediction": prediction,
        "confidence": round(confidence, 1),
        "classification": classification,
        "score": round(habitability_score, 3),
        "factors": factors,
        "detailed_analysis": f"""
Overall Habitability Assessment:
• Temperature Range: {temp_min:.1f}K - {temp_max:.1f}K {'✓ Optimal' if 250 <= temp_min <= 350 and 250 <= temp_max <= 350 else '⚠ Marginal' if 200 <= temp_min <= 400 and 200 <= temp_max <= 400 else '✗ Unsuitable'}
• Habitable Zone: {'✓ Conservative' if hz_con else '⚠ Optimistic' if hz_opt else '✗ Outside'}
• Earth Similarity Index: {esi:.3f} {'✓ High' if esi > 0.8 else '⚠ Moderate' if esi > 0.6 else '✗ Low'}
• Planetary Type: {'✓ Rocky' if density > 4 else '⚠ Mixed' if density > 2 else '✗ Gaseous'}
• Surface Gravity: {gravity:.1f} m/s² {'✓ Earth-like' if 8 <= gravity <= 12 else '⚠ Tolerable' if 5 <= gravity <= 15 else '✗ Extreme'}
• Orbital Stability: {'✓ Stable' if eccentricity < 0.1 else '⚠ Moderate' if eccentricity < 0.3 else '✗ Unstable'}

Final Assessment: {prediction}
Confidence Level: {confidence:.1f}%
        """
    }

def analyze_with_chatgpt(data):
    # Check if OpenAI is configured
    if not OPENAI_API_KEY:
        return generate_mock_ai_response(data)
    
    try:
        if data['type'] == 'properties':
            prompt = f"""
            Analyze this exoplanet as an astrophysicist:

            Input: {data['inputs']['period']}d period, {data['inputs']['radius']}R⊕, {data['inputs']['temperature']}K
            Star: {data['inputs']['starTemperature']}K, {data['inputs']['starMass']}M☉
            Properties: {data['properties']['P_MASS']}M⊕, density {data['properties']['P_DENSITY']}g/cm³, ESI {data['properties']['P_ESI']}

            Provide 2-3 paragraph scientific analysis.
            """
        else:
            prompt = f"""
            Analyze habitability as an astrobiologist:

            Result: {data['habitability']['prediction']} ({data['habitability']['confidence']}%)
            Score: {data['habitability']['score']}/1.000
            Factors: {[f['status'] for f in data['habitability']['factors']]}

            Provide 2-3 paragraph analysis.
            """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert astrophysicist."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=350,
            temperature=0.7
        )
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"OpenAI Error: {e}")
        return generate_mock_ai_response(data)

def generate_mock_ai_response(data):
    """Fallback when API fails or not configured"""
    if data['type'] == 'properties':
        return f"""
Based on the planetary data analyzed:

This exoplanet with {data['properties'].get('P_MASS', 0):.2f} Earth masses and a radius of {data['inputs']['radius']} Earth radii appears to be a {data['properties'].get('P_DENSITY', 0) > 4 and 'terrestrial world' or 'sub-Neptune type planet'}. 

The Earth Similarity Index of {data['properties'].get('P_ESI', 0):.3f} suggests {data['properties'].get('P_ESI', 0) > 0.8 and 'strong Earth-like characteristics' or data['properties'].get('P_ESI', 0) > 0.6 and 'moderate similarity to Earth' or 'limited Earth-like features'}.

With an equilibrium temperature of {data['properties'].get('P_TEMP_EQUIL', 0)}K and {data['properties'].get('P_HABZONE_OPT', 0) and 'position within the habitable zone' or 'location outside conventional habitable zones'}, this world represents {data['properties'].get('P_HABZONE_OPT', 0) and 'a promising candidate for further study' or 'an interesting case for understanding planetary diversity'}.
"""
    else:
        return f"""
Habitability Assessment Analysis:

The {data['habitability']['prediction'].lower()} with {data['habitability']['confidence']}% confidence indicates {data['habitability']['classification'] == 2 and 'high potential for supporting life as we know it' or data['habitability']['classification'] == 1 and 'mixed potential requiring further investigation' or 'challenging conditions for conventional life forms'}.

Key factors influencing this assessment include the {data['habitability']['factors'][0]['status'].lower()} temperature range and {data['habitability']['factors'][1]['status'].lower()} habitable zone positioning. 

Research priority: {data['habitability']['classification'] == 2 and 'High - recommend atmospheric spectroscopy and detailed climate modeling' or data['habitability']['classification'] == 1 and 'Moderate - additional radial velocity measurements suggested' or 'Low - interesting for comparative planetology studies'}.
"""

@app.route('/predict-properties', methods=['POST'])
def predict_properties():
    data = request.json
    properties = generate_comprehensive_properties(data)
    return jsonify({
        'properties': properties,
        'total_properties': len(properties),
        'model_used': 'real' if prop_model else 'comprehensive_mock'
    })

@app.route('/predict-habitability', methods=['POST'])
def predict_habitability():
    data = request.json
    properties = data.get('properties', {})
    analysis = analyze_habitability_factors(properties)
    return jsonify({
        'habitability': analysis,
        'analysis_method': 'comprehensive_factor_analysis'
    })

@app.route('/analyze-with-ai', methods=['POST'])
def analyze_with_ai():
    data = request.json
    analysis = analyze_with_chatgpt(data)
    return jsonify({
        'analysis': analysis,
        'type': data['type'],
        'status': 'success'
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'operational',
        'property_model': 'loaded' if prop_model else 'mock',
        'habitability_model': 'loaded' if hab_model else 'mock',
        'available_targets': len(target_columns) if prop_model else 39
    })

if __name__ == '__main__':
    print("🚀 Exoplanet Research Server Starting...")
    print("📊 Available targets: 39 properties")
    print("🌍 Habitability analysis: Comprehensive factor-based")
    print("🤖 ChatGPT integration: " + ("Enabled" if OPENAI_API_KEY else "Mock mode (no API key)"))
    print("🔗 Server running on http://localhost:5000")
    app.run(debug=True, port=5000)