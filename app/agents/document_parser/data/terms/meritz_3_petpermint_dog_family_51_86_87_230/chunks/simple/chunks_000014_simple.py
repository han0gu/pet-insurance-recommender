from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만,【별표 2(장해분류표)】에 장해판정시기를 별도로 정한 경우에는 그에 따릅니다. \uf000 제1항에 따라 장해지급률이 '
 '결정되었으나 그 이후 보장 을 받을 수 있는 기간(보장의 효력이 없어진 경우에는 보험 기간이 10년 이상인 보장은 상해 발생일부터 2년 '
 '이내로 하 고, 보험기간이 10년 미만인 보장은 상해 발생일부터 1년이 내)에 장해상태가 더 악화된 때에는 그 악화된 장해상태를 기준으로 '
 '장해지급률을 결정합니다'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 54},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000014',
              'chunk_char_len': 231,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
