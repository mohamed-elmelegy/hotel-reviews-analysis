from elasticsearch import Elasticsearch
from ssl import create_default_context
import configparser


class ElasticsearchService:
    def __init__(self) -> None:
        """
            Initialize Elasticsearch connection 
            using configs.ini file
        """
        try:
            # load global configs
            config = configparser.ConfigParser()
            config.read('../configs.ini')

            # load SSL certs.pem
            context = create_default_context(cafile=config['ELASTIC']['ssl_cert_path'])

            # Found in the 'Manage Deployment' page
            CLOUD_ID = config['ELASTIC']['cloud_id']

            # Create the client instance
            # cloud_id, domain name, or server's IP address
            self.client = Elasticsearch(
                cloud_id=CLOUD_ID,
                http_auth=(config['ELASTIC']['username'], config['ELASTIC']['password']),
                # scheme=config['ELASTIC']['scheme'],
                # port=config['ELASTIC']['port'],
                ssl_context=context,
            )
        except:
            raise "Something went wrong!"


    def get_head(self, index_name, size=10):
        """
            Retrieve all hotels info from Elasticsearch index
        """
        # refresh indicies
        self.client.indices.refresh(index=index_name)
        # return all docs
        result = self.client.search(
            index=index_name,
            query={
                'match_all': {}
            }, 
            size=size, 
            pretty=True
        )

        return result['hits']['hits']


    def get_doc(self, index_name, hotel_name=''):
        """
            Retrieve specific hotel info from Elasticsearch index
        """
        # refresh indicies
        self.client.indices.refresh(index=index_name)
        # return all docs
        result = self.client.search(
            index=index_name,
            query={
                'match': {
                    'name': hotel_name
                }
            }, 
            pretty=True
        )

        return result['hits']['hits'][0]['_source']
    