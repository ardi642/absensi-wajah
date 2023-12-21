from pymilvus import connections, db, Collection, utility, CollectionSchema, FieldSchema, DataType
from dotenv import load_dotenv
import os

load_dotenv('.env')
db_name = os.environ.get("db_name")
collection_name = os.environ.get("collection_name") 
user = os.environ.get("user")
password = os.environ.get("password")
host = os.environ.get("host")
port = os.environ.get("port")

conn = connections.connect(host="127.0.0.1", port=19530)

databases = db.list_database()

connections.disconnect("default")

if db_name not in databases:
    db.create_database(db_name)

conn = connections.connect(
    alias="default",
    user=user,
    password=password,
    host=host,
    port=port,
    db_name=db_name
)

if not utility.has_collection(collection_name):
    id = FieldSchema(
    name="id",
    dtype=DataType.INT64,
    auto_id=True,
    is_primary=True,
    )
    identity = FieldSchema(
    name="identity",
    dtype=DataType.VARCHAR,
    max_length=200,
    # The default value will be used if this field is left empty during data inserts or upserts.
    # The data type of `default_value` must be the same as that specified in `dtype`.
    default_value="Unknown"
    )
    embedding = FieldSchema(
    name="embedding",
    dtype=DataType.FLOAT_VECTOR,
    dim=512
    # The default value will be used if this field is left empty during data inserts or upserts.
    # The data type of `default_value` must be the same as that specified in `dtype`.

    )

    schema = CollectionSchema(
    fields=[id, identity, embedding],
    description="Face Recognition",
    enable_dynamic_field=True
    )

    collection = Collection(
    name=collection_name,
    schema=schema,
    using='default',
    shards_num=2
    )

    index_params = {
    "metric_type":"COSINE",
    "index_type":"IVF_FLAT",
    "params":{"nlist":1024}
    }

    collection.create_index(
    field_name="embedding", 
    index_params=index_params
    )

connections.disconnect("default")