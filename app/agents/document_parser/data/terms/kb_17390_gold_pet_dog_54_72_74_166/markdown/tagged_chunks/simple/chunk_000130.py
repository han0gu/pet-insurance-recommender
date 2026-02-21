from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 제1항에 따라 위험이 증가하거나 감소되는 경우 납입보험료가 변경될 수 있으며,\n'
 '- 계약내용 변경시점 이후 잔여 보험기간의 보장을 위한 재원인 계약자적립액 등의\n'
 '- 64 -차이로 계약자가 추가로 납입하여야 할 (또는 반환받을) 금액이 발생할 수 있습니\n'
 '다.# \uf000 제1항에 따라 보험료 등의 감액 또는 증액시 환급금이 없거나 최초가입시 안내한| 만기(해약)환급금보다 | 적거나 '
 '많아질 수 있습니다. |\n'
 '| --- | --- |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000130',
              'chunk_char_len': 240,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
