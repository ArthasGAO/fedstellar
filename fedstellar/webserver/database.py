import sqlite3
import hashlib
import datetime

user_db_file_location = "database_file/users.db"
note_db_file_location = "database_file/notes.db"
image_db_file_location = "database_file/images.db"
node_db_file_location = "database_file/nodes.db"
scenario_db_file_location = "database_file/scenarios.db"

"""
    User Management
"""


def list_users(all_info=False):
    _conn = sqlite3.connect(user_db_file_location)
    _c = _conn.cursor()
    result = _c.execute("SELECT * FROM users")
    result = result.fetchall()

    if not all_info:
        result = [user[0] for user in result]

    _conn.close()

    return result


def get_user_info(user):
    _conn = sqlite3.connect(user_db_file_location)
    _c = _conn.cursor()

    command = f"SELECT * FROM users WHERE user = '{user}'"
    _c.execute(command)
    result = _c.fetchone()

    _conn.commit()
    _conn.close()

    return result


def verify(user, password):
    _conn = sqlite3.connect(user_db_file_location)
    _c = _conn.cursor()

    _c.execute("SELECT password FROM users WHERE user = '" + user + "';")
    result = _c.fetchone()[0] == hashlib.sha256(password.encode()).hexdigest()

    _conn.close()

    return result


def delete_user_from_db(user):
    _conn = sqlite3.connect(user_db_file_location)
    _c = _conn.cursor()
    _c.execute("DELETE FROM users WHERE user = '" + user + "';")
    _conn.commit()
    _conn.close()

    # when we delete a user FROM database USERS, we also need to delete all his or her notes data FROM database NOTES
    _conn = sqlite3.connect(note_db_file_location)
    _c = _conn.cursor()
    _c.execute("DELETE FROM notes WHERE user = '" + user + "';")
    _conn.commit()
    _conn.close()

    # when we delete a user FROM database USERS, we also need to 
    # [1] delete all his or her images FROM image pool (done in app.py)
    # [2] delete all his or her images records FROM database IMAGES
    _conn = sqlite3.connect(image_db_file_location)
    _c = _conn.cursor()
    _c.execute("DELETE FROM images WHERE owner = '" + user + "';")
    _conn.commit()
    _conn.close()


def add_user(user, password, role):
    _conn = sqlite3.connect(user_db_file_location)
    _c = _conn.cursor()

    _c.execute("INSERT INTO users values(?, ?, ?)", (user.upper(), hashlib.sha256(password.encode()).hexdigest(), role))

    _conn.commit()
    _conn.close()


"""
    Notes Management
"""


def read_note_from_db(id):
    _conn = sqlite3.connect(note_db_file_location)
    _c = _conn.cursor()

    command = "SELECT note_id, timestamp, note FROM notes WHERE user = '" + id.upper() + "';"
    _c.execute(command)
    result = _c.fetchall()

    _conn.commit()
    _conn.close()

    return result


def match_user_id_with_note_id(note_id):
    # Given the note id, confirm if the current user is the owner of the note which is being operated.
    _conn = sqlite3.connect(note_db_file_location)
    _c = _conn.cursor()

    command = "SELECT user FROM notes WHERE note_id = '" + note_id + "';"
    _c.execute(command)
    result = _c.fetchone()[0]

    _conn.commit()
    _conn.close()

    return result


def write_note_into_db(id, note_to_write):
    _conn = sqlite3.connect(note_db_file_location)
    _c = _conn.cursor()

    current_timestamp = str(datetime.datetime.now())
    _c.execute("INSERT INTO notes values(?, ?, ?, ?)", (id.upper(), current_timestamp, note_to_write, hashlib.sha1((id.upper() + current_timestamp).encode()).hexdigest()))

    _conn.commit()
    _conn.close()


def delete_note_from_db(note_id):
    _conn = sqlite3.connect(note_db_file_location)
    _c = _conn.cursor()

    command = "DELETE FROM notes WHERE note_id = '" + note_id + "';"
    _c.execute(command)

    _conn.commit()
    _conn.close()


"""
    Image Management
"""


def image_upload_record(uid, owner, image_name, timestamp):
    _conn = sqlite3.connect(image_db_file_location)
    _c = _conn.cursor()

    _c.execute("INSERT INTO images VALUES (?, ?, ?, ?)", (uid, owner, image_name, timestamp))

    _conn.commit()
    _conn.close()


# get uid and name from imagen where uid = image_uid
# store the uid and name in a tuple
def get_image_file_name(image_uid):
    _conn = sqlite3.connect(image_db_file_location)
    _c = _conn.cursor()

    command = "SELECT uid, name FROM images WHERE uid = '" + image_uid + "';"
    _c.execute(command)
    result = _c.fetchone()

    _conn.commit()
    _conn.close()

    return result[0] + "-" + result[1]


def list_images_for_user(owner):
    _conn = sqlite3.connect(image_db_file_location)
    _c = _conn.cursor()

    command = "SELECT uid, timestamp, name FROM images WHERE owner = '{0}'".format(owner)
    _c.execute(command)
    result = _c.fetchall()

    _conn.commit()
    _conn.close()

    return result


def match_user_id_with_image_uid(image_uid):
    # Given the note id, confirm if the current user is the owner of the note which is being operated.
    _conn = sqlite3.connect(image_db_file_location)
    _c = _conn.cursor()

    command = "SELECT owner FROM images WHERE uid = '" + image_uid + "';"
    _c.execute(command)
    result = _c.fetchone()[0]
    print(result)

    _conn.commit()
    _conn.close()

    return result


def delete_image_from_db(image_uid):
    _conn = sqlite3.connect(image_db_file_location)
    _c = _conn.cursor()

    command = "DELETE FROM images WHERE uid = '" + image_uid + "';"
    _c.execute(command)

    _conn.commit()
    _conn.close()


"""
    Nodes Management
"""


def list_nodes(sort_by="idx"):
    # list all nodes in the database
    _conn = sqlite3.connect(node_db_file_location)
    _c = _conn.cursor()
    # Get all nodes and decently sort them by idx
    command = "SELECT * FROM nodes ORDER BY " + sort_by + ";"
    _c.execute(command)
    result = _c.fetchall()

    _conn.commit()
    _conn.close()

    return result


def list_nodes_by_scenario_name(scenario_name):
    # list all nodes in the database
    _conn = sqlite3.connect(node_db_file_location)
    _c = _conn.cursor()
    # Get all nodes and decently sort them by idx
    command = "SELECT * FROM nodes WHERE scenario = '" + scenario_name + "' ORDER BY idx;"
    _c.execute(command)
    result = _c.fetchall()

    _conn.commit()
    _conn.close()

    return result


def update_node_record(node_uid, idx, ip, port, role, neighbors, latitude, longitude, timestamp, federation, scenario):
    # Check if the node record with node_uid and scenario already exists in the database
    # If it does, update the record
    # If it does not, create a new record
    _conn = sqlite3.connect(node_db_file_location)
    _c = _conn.cursor()

    command = "SELECT * FROM nodes WHERE uid = '" + node_uid + "' AND scenario = '" + scenario + "';"
    _c.execute(command)
    result = _c.fetchone()
    print("Update Node Record Result:")
    print(result)
    if result is None:
        # Create a new record
        _c.execute("INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (node_uid, idx, ip, port, role, neighbors, latitude, longitude, timestamp, federation, scenario))
    else:
        # Update the record
        command = "UPDATE nodes SET idx = '" + idx + "', ip = '" + ip + "', port = '" + port + "', role = '" + role + "', neighbors = '" + neighbors + "', latitude = '" + latitude + "', longitude = '" + longitude + "', timestamp = '" + timestamp + "', federation = '" + federation + "' WHERE uid = '" + node_uid + "' AND scenario = '" + scenario + "';"
        _c.execute(command)

    _conn.commit()
    _conn.close()


def remove_all_nodes():
    _conn = sqlite3.connect(node_db_file_location)
    _c = _conn.cursor()

    command = "DELETE FROM nodes;"
    _c.execute(command)

    _conn.commit()
    _conn.close()


def remove_nodes_by_scenario_name(scenario_name):
    _conn = sqlite3.connect(node_db_file_location)
    _c = _conn.cursor()

    command = "DELETE FROM nodes WHERE scenario = '" + scenario_name + "';"
    _c.execute(command)

    _conn.commit()
    _conn.close()


"""
    Scenario Management
"""


def get_all_scenarios(sort_by="start_time"):
    _conn = sqlite3.connect(scenario_db_file_location)
    _c = _conn.cursor()
    command = "SELECT * FROM scenarios ORDER BY " + sort_by + ";"
    _c.execute(command)
    result = _c.fetchall()

    _conn.commit()
    _conn.close()

    return result


def scenario_update_record(scenario_name, start_time, end_time, title, description, status, network_subnet):
    _conn = sqlite3.connect(scenario_db_file_location)
    _c = _conn.cursor()

    command = "SELECT * FROM scenarios WHERE name = '" + scenario_name + "';"
    _c.execute(command)
    result = _c.fetchone()

    if result is None:
        # Create a new record
        _c.execute("INSERT INTO scenarios VALUES (?, ?, ?, ?, ?, ?, ?)", (scenario_name, start_time, end_time, title, description, status, network_subnet))
    else:
        # Update the record
        command = "UPDATE scenarios SET start_time = '" + start_time + "', end_time = '" + end_time + "', title = '" + title + "', description = '" + description + "', status = '" + status + "', network_subnet = '" + network_subnet + "' WHERE name = '" + scenario_name + "';"
        _c.execute(command)

    _conn.commit()
    _conn.close()


def scenario_set_all_status_to_finished():
    # Set all scenarios to finished and update the end_time to current time
    _conn = sqlite3.connect(scenario_db_file_location)
    _c = _conn.cursor()

    command = "UPDATE scenarios SET status = 'finished', end_time = '" + str(datetime.datetime.now()) + "';"
    _c.execute(command)

    _conn.commit()
    _conn.close()


def scenario_set_status_to_finished(scenario_name):
    _conn = sqlite3.connect(scenario_db_file_location)
    _c = _conn.cursor()

    command = "UPDATE scenarios SET status = 'finished', end_time = '" + str(datetime.datetime.now()) + "' WHERE name = '" + scenario_name + "';"
    _c.execute(command)

    _conn.commit()
    _conn.close()


def get_running_scenario():
    _conn = sqlite3.connect(scenario_db_file_location)
    _c = _conn.cursor()
    command = "SELECT * FROM scenarios WHERE status = 'running';"
    _c.execute(command)
    result = _c.fetchone()

    _conn.commit()
    _conn.close()

    return result


def get_scenario_by_name(scenario_name):
    _conn = sqlite3.connect(scenario_db_file_location)
    _c = _conn.cursor()
    command = "SELECT * FROM scenarios WHERE name = '" + scenario_name + "';"
    _c.execute(command)
    result = _c.fetchone()

    _conn.commit()
    _conn.close()

    return result


def remove_scenario_by_name(scenario_name):
    _conn = sqlite3.connect(scenario_db_file_location)
    _c = _conn.cursor()

    command = "DELETE FROM scenarios WHERE name = '" + scenario_name + "';"
    _c.execute(command)

    _conn.commit()
    _conn.close()


if __name__ == "__main__":
    print(list_users())
