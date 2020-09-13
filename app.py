import time

import numpy
from flask import Flask, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy

from tablesAlgorithm import tables_arrangement_algorithm
from Table import Table

app = Flask(__name__)

# initialize database

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db = SQLAlchemy(app)

# initializing matrix represents the restaurant
colour_matrix = numpy.empty([10, 10], dtype="<U10")


# to add a new table:
# 1 type python
# 2 from app import db
# 3 db.create_all()
# 4exit()

# create a table for the database.
class Tables(db.Model):  # create model
    id = db.Column(db.Integer, primary_key=True)
    table_width = db.Column(db.String(200), nullable=False)
    table_length = db.Column(db.String(200), nullable=False)


# home page
@app.route('/')
def index():
    for i in range(10):
        for j in range(10):
            if i < 4 and j < 4:
                colour_matrix[i][j] = "light"
            else:
                colour_matrix[i][j] = "dark"

    return render_template('index.html')


@app.route('/AddTables', methods=['POST', 'GET'])
def add_tables():
    # in case the user submitted a new table
    if request.method == 'POST':

        # unpack the data which the user entered
        table_width = request.form['table_width']
        table_length = request.form['table_length']

        # add the new table to the database
        new_table = Tables(table_width=table_width, table_length=table_length)
        db.session.add(new_table)
        db.session.commit()

        # grab all content from table in order to show the user all tables
        tables = Tables.query.order_by(Tables.id).all()

        return render_template('AddTables.html', tables=tables)
    else:

        # grab all content from table in order to show the user all tables
        tables = Tables.query.order_by(Tables.id).all()  # grab all content from table
        return render_template('AddTables.html', tables=tables)


# let the user option to delete one of the tables which recognized by id
@app.route('/delete/<int:id_>')
def delete(id_):
    table_to_delete = Tables.query.get_or_404(id_)
    db.session.delete(table_to_delete)
    db.session.commit()
    return redirect('/AddTables')


@app.route('/ChooseDimensions', methods=['POST', 'GET'])
def choose_dimensions():
    # in case the user filled in and applied his restaurant size
    if request.method == 'POST':

        # unpack size
        restaurant_width = int(request.form['restaurant_width'])
        restaurant_length = int(request.form['restaurant_length'])

        # change the restaurant matrix according to the user's request
        for k in range(10):
            for m in range(10):
                if k < restaurant_length and m < restaurant_width:
                    colour_matrix[k][m] = "light"
                else:
                    colour_matrix[k][m] = "dark"

    return render_template('ChooseDimensions.html', colour_matrix=colour_matrix)


# update a square when user create his restaurant. each square has different id with 4 numbers
@app.route('/update/<id_>')
def update(id_):
    # decode the id of the requested square
    row = (id_[0] + id_[1])
    column = (id_[2] + id_[3])
    if row[0] == 0:
        row = row[1]
    if column[0] == 0:
        column = column[1]

    # change the colour of the requested square:
    if colour_matrix[int(row)][int(column)] == 'light':
        colour_matrix[int(row)][int(column)] = 'dark'
    else:
        colour_matrix[int(row)][int(column)] = 'light'

    # back to our  page
    return redirect('/ChooseDimensions')


# this function is called when the user finished to enter all variables.
@app.route('/result', methods=['POST', 'GET'])
def result():
    # grab tables fom database and prepare a list to sent to the algorithm
    all_tables = Tables.query.all()
    tables_for_algorithm = []
    k = 1
    for table in all_tables:
        new_table = Table(float(table.table_width), float(table.table_length), k)
        tables_for_algorithm.append(new_table)
        k = k + 1

    # call the algorithm and return result in case of succession and impossible otherwise
    best_distance = str(tables_arrangement_algorithm(tables_for_algorithm, colour_matrix))
    if best_distance == '-1':
        return render_template('Impossible.html')

    else:
        return render_template('result.html')


# called un case of failure
@app.route('/Impossible')
def impossible():
    return render_template('Impossible.html')


# this function ensure refreshing of the cache memory in every template call
@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


if __name__ == "__main__":
    app.run(debug=True)
