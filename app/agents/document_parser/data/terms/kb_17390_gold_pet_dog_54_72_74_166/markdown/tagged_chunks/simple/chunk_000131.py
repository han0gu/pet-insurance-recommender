from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- |\n'
 '| 용 어 풀 이 감액 보험료, 보험금, 계약자적립액 등을 산정하는 기준이 되는 보험가입금액을 계 약시 선택한 금액보다 적은 금액으로 '
 '줄이는 것 (이에 따라 보험료, 보험금 및 | 용 어 풀 이 감액 보험료, 보험금, 계약자적립액 등을 산정하는 기준이 되는 보험가입금액을 '
 '계 약시 선택한 금액보다 적은 금액으로 줄이는 것 (이에 따라 보험료, 보험금 및 |\n'
 '| 적립액(해약환급금)도 | 줄어듭니다.) |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000131',
              'chunk_char_len': 239,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
