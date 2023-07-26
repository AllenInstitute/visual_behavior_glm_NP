import yaml
import datetime
import numpy as np
import pandas as pd
from pymongo import MongoClient

class Database(object):
    '''
    utilities for connecting to MongoDB databases (mouseseeks or visual_behavior_data)

    parameter:
      database: defines database to connect to. Can be 'mouseseeks' or 'visual_behavior_data'
    '''

    def __init__(self, database, ):
        # get database ip/port info from a text file on the network
        db_info_filepath = \
            '//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/mongo_db_info.yml'
        with open(db_info_filepath, 'r') as stream:
            db_info = yaml.safe_load(stream)

        # connect to the client
        ip = db_info[database]['ip']
        port = db_info[database]['port']
        self.client = MongoClient('mongodb://{}:{}'.format(ip, port))

        # set each table as an attribute of the class (but not admin) and as an entry in a dictionary
        # this will provide flexibility in how the tables are called
        self.database = {}
        self.database_names = []
        databases = [db for db in self.client.list_database_names() if db != 'admin']
        for db in databases:
            self.database_names.append(db)
            self.database[db] = self.client[db]
            setattr(self, db, self.client[db])
        # make subscriptable
        self._db_names = {db: self.client[db] for db in databases}

    def __getitem__(self, item):
        # this allows databases to be accessed by name
        return self._db_names[item]

    def query(self, database, collection, query={}, return_as='dataframe'):
        '''
        Run a query on a collection in the database.
        The query should be formated as set of key/value pairs
        Sending an empty query will return the entire collection
        '''

        return pd.DataFrame(list(self.database[database][collection].find(query)))

    def close(self):
        '''
        close connection to client
        '''
        self.client.close()


def is_int(n):
    return isinstance(n, (int, np.integer))


def is_float(n):
    return isinstance(n, (float, np.float))


def is_uuid(n):
    return isinstance(n, uuid.UUID)


def is_bool(n):
    return isinstance(n, (bool, np.bool_))


def is_array(n):
    return isinstance(n, np.ndarray)


def simplify_type(x):
    if is_int(x):
        return int(x)
    elif is_bool(x):
        return int(x)
    elif is_float(x):
        return float(x)
    elif is_array(x):
        return [simplify_type(e) for e in x]
    else:
        return x


def simplify_entry(entry):
    '''
    entry is one document
    '''
    entry = {k: simplify_type(v) for k, v in entry.items()}
    return entry


def clean_and_timestamp(entry):
    '''make sure float and int types are basic python types (e.g., not np.float)'''
    entry = simplify_entry(entry)
    entry.update({'entry_time_utc': str(datetime.datetime.utcnow())})
    return entry


def update_or_create(collection, document, keys_to_check, force_write=False):
    '''
    updates a collection of the document exists
    inserts if it does not exist
    uses keys in `keys_to_check` to determine if document exists. 
    Other keys will be written, but not used for checking uniqueness
    '''
    if force_write:
        collection.insert_one(simplify_entry(document))
    else:
        query = {key: simplify_type(document[key]) for key in keys_to_check}
        if collection.find_one(query) is None:
            # insert a document if this experiment/cell doesn't already exist
            collection.insert_one(simplify_entry(document))
        else:
            # update a document if it does exist
            collection.update_one(query, {"$set": simplify_entry(document)})


