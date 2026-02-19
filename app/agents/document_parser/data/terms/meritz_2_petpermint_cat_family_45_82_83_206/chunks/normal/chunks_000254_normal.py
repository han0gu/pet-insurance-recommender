from langchain_core.documents import Document

chunk = Document(
    page_content=('약이 연장된 경우 연장된 날 기준으로 매년 현재의 예정기 초율(적용이율, 적용위험률, 부가보험요율) 적용 및 반려동 물의 연령 증가 등의 '
 '사유로 보험요율이 변동될 수 있으며 이 때의 보험료는「보험료 및 해약환급금 산출방법서」에 따라 산출합니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 99},
 'term_type': 'special',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000254',
              'chunk_char_len': 136,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
