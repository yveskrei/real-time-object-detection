# Setting up Elastic Search - Image Retrieval
In order to set up elasticsearch to allow insertion and search upon images in low latency constrains, we need to configure the ElasticSearch cluster with care. We will be working with Kibana's `Dev Tools` for this setup, and commands will be provided for setting up all the necessary components.

## Index Lifecycle Policy
We set up a policy that decided where our index would live in the ElasticSearch cluster.<br>
Indexes will live in `HOT` phase for a period of `7 days` before moving into a `WARM` phase forever. This allows us to index large amounts of video and not compromise on user experience(latency during search).<br>
We use the following command:
```
PUT _ilm/policy/embeddings-phasing-policy
{
  "policy": {
    "phases": {
      "hot": {
        "min_age": "0ms",
        "actions": {
          "rollover": {
            "max_age": "7d",
            "max_primary_shard_size": "50gb"
          },
          "set_priority": { "priority": 100 }
        }
      },
      "warm": {
        "min_age": "7d",
        "actions": {
          "set_priority": { "priority": 50 },
          "shrink": {
            "number_of_shards": 1,
            "allow_write_after_shrink": false
          },
          "readonly": {}
        }
      }
    }
  }
}
```
See reference about ElasticSearch's lifecycle tiers - [Docs](https://www.elastic.co/docs/manage-data/lifecycle/data-tiers)

## Component Template
This serves as a template for vectors that we insert into our system. We define a tight schema with the relevant fields inside(metadata, i.e `channel_id`, `timestamp`), and a definition to our vector which we then insert into the the indexes.<br>
We use the following command:
```
PUT _component_template/embedding-live-object-template
{
  "template": {
    "settings": {
      "index.lifecycle.name": "embeddings-phasing-policy",
      "index.lifecycle.rollover_alias": "embeddings-live",
      "index.codec": "best_compression",
      "number_of_shards": 2,
      "number_of_replicas": 0
    },
    "mappings": {
      "properties": {
        "timestamp": {
          "type": "date",
          "format": "epoch_millis"
        },
        "channel_id": {
          "type": "keyword"
        },
        "embedding": {
          "type": "dense_vector",
          "dims": 768,
          "element_type": "bfloat16",
          "index": true,
          "index_options": {
            "type": "bbq_disk"
          },
          "similarity": "cosine"
        }
      }
    }
  }
}
```

## Index Template
This allows us to define a template for indexes(which include the components we defined in the template before, and the lifecycle policy). This essentially glues together the two components we previously defined.<br>
We use the following command:
```
PUT _index_template/embedding-live-template
{
  "template": {
    "settings": {
      "index": {
        "lifecycle": {
          "name": "embeddings-phasing-policy",
          "rollover_alias": "embeddings-live"
        },
        "codec": "best_compression",
        "number_of_shards": "2",
        "number_of_replicas": "0",
        "refresh_interval": "30s",
        "mode": "standard"
      }
    },
    "aliases": {
      "embeddings_search": {}
    }
  },
  "index_patterns": [
    "embeddings-live-*"
  ],
  "composed_of": [
    "embedding-live-object-template"
  ]
}
```

## Creating a starting index
We the create a single index in order to start the lifecycle automatically, and hand over the automation to ElasticSearch.<br>
We run the following command:
```
PUT embeddings-live-000001
{
  "aliases": {
    "embeddings-live": {
      "is_write_index": true
    }
  }
}

```

## Integrating to our system
After performing all the steps above, we have an index ready for data insertion/search. While creating all the components we defined an alias for our indexes, which we will use in our third party code and reference these indexes with. The defined alias in these steps is `embeddings-live`.
