from langchain_core.documents import Document

chunk = Document(
    page_content=('제4조(갱신보장계약 제1회 보험료의 납입연체와 계약의 해 제) \uf000 계약자가 갱신전 보장계약의 보험료를 정상적으로 납입 하고, '
 '갱신보장계약의 제1회 보험료를 갱신일까지 납입하지 않아 보험료 납입이 연체 중인 경우에 회사는 14일(보험기 간이 1년 미만인 경우에는 '
 '7일) 이상의 기간을 납입최고(독 촉)기간(납입최고(독촉)기간의 마지막 날이 영업일이 아닌 때에는 최고(독촉)기간은 그 다음 날까지로 '
 '합니다)으로 정 하여 계약자(타인을 위한 보험계약의 경우 특정된 보험수익 자를 포함합니다)가 납입최고(독촉)기간 안에 보험료를 납 입하지'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 190},
 'term_type': 'special',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000651',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
