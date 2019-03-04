# -*- coding: utf-8 -*-
import os
import socket
import datetime
from copy import deepcopy
from bson.objectid import ObjectId
from itertools import groupby

log_metric_fn = None
mongo = None


"""some operators for mongodb."""


def init_mongo(args):
    from pymongo import MongoClient
    mongo_client = MongoClient(
        host=os.getenv('JOBMONITOR_METADATA_HOST'),
        port=int(os.getenv('JOBMONITOR_METADATA_PORT'))
    )
    global mongo
    mongo = getattr(mongo_client, os.getenv('JOBMONITOR_METADATA_DB'))
    # init the job status at mongodb.
    job_content = {
        'user': args.user,
        'project': args.project,
        'experiment': args.experiment,
        'config': get_clean_arguments(args),
        'job_id': args.timestamp,
        'rank_id': get_rank(args),
        'host': socket.gethostname(),
        'status': 'SCHEDULED',
        'schedule_time': datetime.datetime.utcnow(),
        'output_dir': get_checkpoint_dir(args),
        'is_cuda': args.on_cuda
    }
    mongo_job_id = mongo.job.insert_one(job_content)
    args.mongo_job_id = {'_id': ObjectId(str(mongo_job_id.inserted_id))}

    # set job to 'started' in MongoDB
    update_mongo_record(
        args.mongo_job_id,
        {'$set': {
            'status': 'RUNNING',
            'start_time': datetime.datetime.utcnow(),
        }}
    )


def announce_job_termination_to_mongo(args, last_time=None):
    global mongo
    end_time = datetime.datetime.utcnow()

    # update.
    mongo.job.update(
        args.mongo_job_id,
        {'$set': {
            'status': 'FINISHED', 'end_time': end_time,
            'lasted_time':
                (end_time -
                find_record_from_mongo(args.mongo_job_id)[0]['start_time']).seconds
        }}
    )


def update_mongo_record(mongo_job_id, content):
    global mongo
    mongo.job.update_one(mongo_job_id, content)


def find_mongo_record(mongo_job_id):
    global mongo
    return mongo.job.find_one(mongo_job_id)


def delete_mongo_record(mongo_job_id):
    global mongo
    return mongo.job.delete_one(mongo_job_id)


def delete_mongo_collection():
    global mongo
    mongo.job.drop()


def find_record_from_mongo(condition, projection=None):
    # some examples.
    # db.find(projection={"pmc_id": True})
    # db.find({"pmc_id": {"$ne": ""}})
    global mongo
    return [s for s in mongo.job.find(condition, projection=projection)]


def _get_non_duplicated_time(records):
    def merge_results(results):
        return sum(
            [(result['end_time'] - result['start_time']).seconds for result in results])

    def _organize_results_per_host(_records):
        _records = sorted(_records, key=lambda x: x['start_time'])
        first_record = _records[0]
        new_records = []

        for record in _records:
            if first_record['end_time'] >= record['start_time'] >= first_record['start_time']:
                first_record['start_time'] = min(first_record['start_time'], record['start_time'])
                first_record['end_time'] = max(first_record['end_time'], record['end_time'])
            elif first_record['end_time'] < record['start_time']:
                new_records.append(first_record)
                first_record = record
    
        new_records.append(first_record)
        return merge_results(new_records)

    # sort records.
    new_records = []
    records = sorted(records, key=lambda x: x['host'])

    for _, values in groupby(records, key=lambda x: x['host']):
        new_records += [_organize_results_per_host(list(values))]
    return sum(new_records)


def get_gpu_hours_from_mongo(year, month, day):
    # init client.
    from pymongo import MongoClient
    mongo_client = MongoClient(
        host=os.getenv('JOBMONITOR_METADATA_HOST'),
        port=int(os.getenv('JOBMONITOR_METADATA_PORT'))
    )

    mongo = getattr(mongo_client, os.getenv('JOBMONITOR_METADATA_DB'))

    # define the time range.
    end_time = datetime.datetime(year, month, day, 23, 59, 59)
    start_time = end_time - datetime.timedelta(days=7)

    # get all GPU hours.
    matched_records = [
        s for s in mongo.job.find({
            'is_cuda': True, 'status': 'FINISHED',
            'start_time': {'$gt': start_time, '$lt': end_time}
        })
    ]
    return 1.0 * _get_non_duplicated_time(matched_records) / 60 / 60


def get_clean_arguments(args):
    copy_args = deepcopy(args)

    if 'graph' in args:
        copy_args._graph = args.graph.__dict__
        copy_args.graph = None
    return copy_args.__dict__


def get_rank(args):
    return args.graph.rank if 'graph' in args else 'root'


def get_checkpoint_dir(args):
    return args.checkpoint_root if 'checkpoint_root' in args else ''



"""some operators for telegraf."""


def init_telegraf(args):
    from telegraf.client import TelegrafClient
    telegraf_client = TelegrafClient(
        host=os.getenv('JOBMONITOR_TELEGRAF_HOST'),
        port=int(os.getenv('JOBMONITOR_TELEGRAF_PORT')),
        tags={
            'host': socket.gethostname(),
            'user': args.user,
            'project': args.project,
            'experiment': args.experiment,
            'job_id': args.timestamp,
            'job_details': args.job_details,
            'job_info': args.job_info
        }
    )

    global log_metric_fn
    log_metric_fn = telegraf_client.metric


def log_metric(*args):
    return log_metric_fn(*args)


"""some operators for influxdb."""


def init_influxdb(db_name='jobmonitor'):
    from influxdb import InfluxDBClient
    influx_client = InfluxDBClient(
        host=os.getenv('JOBMONITOR_TIMESERIES_HOST'),
        database=db_name
    )
    return influx_client


def get_measurement(cli, measurement=None, tags={}):
    rs = cli.query('select * from {}'.format(measurement))
    return list(rs.get_points(measurement=measurement, tags=tags))
