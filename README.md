# MUTUAL-FRIEND-RECOMMENDATION-IN-MSNs-USING-TWO-STAGE-DEEP-LEARNING-FRAMEWORK

This project leverages Django to facilitate administrative tasks for a web application aimed at analyzing and inferring friendships within mobile social networks. The provided manage.py script acts as the command-line utility for managing the project.

## Features
- Django-based backend for managing tasks.
- Includes utilities to start the server, run migrations, and handle other administrative tasks.
- Designed for use in mobile social network analysis.

## Project Setup

### Prerequisites
1. Python 3.8 or higher.
2. Django 3.2 or higher.
3. A virtual environment manager (e.g., venv or virtualenv).

### Installation
1. Clone the repository:
   bash
   git clone https://github.com/your-repo/friendship-inference.git
   cd friendship-inference
   

2. Create and activate a virtual environment:
   bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   

3. Install the required dependencies:
   bash
   pip install -r requirements.txt
   

4. Set up the database:
   bash
   python manage.py migrate
   

### Running the Application
1. Start the development server:
   bash
   python manage.py runserver
   

2. Open your browser and navigate to:
   
   http://127.0.0.1:8000/
   

## Project Structure
- manage.py: Command-line utility for administrative tasks.
- friendship_inference_in_mobile_social_networks/: Contains project-specific settings and application code.

## Troubleshooting
If you encounter an error like:

Couldn't import Django. Are you sure it's installed and available on your PYTHONPATH environment variable? Did you forget to activate a virtual environment?

Ensure that:
- Django is installed in your virtual environment.
- The virtual environment is activated.

## Contribution Guidelines
1. Fork the repository.
2. Create a feature branch:
   bash
   git checkout -b feature-name
   
3. Commit your changes:
   bash
   git commit -m "Add feature description"
   
4. Push to the branch:
   bash
   git push origin feature-name
   
5. Submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
Special thanks to the contributors and the open-source community for their support.
