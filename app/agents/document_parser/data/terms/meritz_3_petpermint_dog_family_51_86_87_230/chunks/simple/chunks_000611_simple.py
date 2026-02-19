from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 회사는 제1항에 따라 위험이 감소된 경우에는 그 차액보 험료를 돌려 드리며, 위험이 증가된 경우에는 통지를 받은 날부터 '
 '1개월 내에 보험료의 증액을 청구하거나 계약을 해 지할 수 있습니다. \uf000 계약자 또는 피보험자는 주소 또는 연락처가 변경된 경 '
 '우에는 지체없이 이를 회사에 알려야 합니다. 다만, 계약자 가 알리지 않은 경우 회사가 알고 있는 최종의 주소 또는 연락처로 등기우편 등 '
 '우편물에 대한 기록이 남는 방법으로 회사가 알린 사항은 일반적으로 도달에 필요한 기간이 지난 때에는 계약자 또는 피보험자에게 도달한 '
 '것으로 봅니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 181},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000611',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
