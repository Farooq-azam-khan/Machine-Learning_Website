from flask_wtf import FlaskForm
from wtforms import FloatField, SubmitField
from wtforms.validators import DataRequired

class DataPredict(FlaskForm):
    pedal_length    = FloatField('Pedal Length', validators=[DataRequired()])
    pedal_width     = FloatField('Pedal Width', validators=[DataRequired()])
    sepal_length    = FloatField('Sepal Length', validators=[DataRequired()])
    sepal_width     = FloatField('Sepal Width', validators=[DataRequired()])
    submit = SubmitField("Predict")
