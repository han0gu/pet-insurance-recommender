from langchain_core.documents import Document

chunk = Document(
    page_content=('파산 등으로 예금을 지급할 수 없는 경우 해당 금융기관을 대신하여 해약환급금(또는 만기시 보험금)에 기타지급금을 합한 법 금액 및 '
 '사고보험금을 각각 보험계약자 1인당 최고 1억원까지 지급함으로써 예금 ㆍ 자를 보호하는 제도를 말합니다'),
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
 'indexing': {'chunk_id': 'chunk_000332',
              'chunk_char_len': 130,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
