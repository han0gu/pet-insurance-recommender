from langchain_core.documents import Document

chunk = Document(
    page_content=('- 곤란하여 의료기관에 입실하여 의사의 관리하에 치료에 전념하는 것을 말합니다.\n'
 '- \uf000 제1항의 "의료기관"이라 함은 의료법 제3조(의료기관) 제2항에서 정한 국내의 병\n'
 '- 원이나 의원 또는 국외의 의료관련법에서 정한 의료기관을 말합니다.\n'
 '# 제5조(보험금의 청구)\uf000 보험수익자는 다음의 서류를 제출하고 보험금을 청구하여야 합니다.128 KB 금쪽같은 '
 '펫보험(강아지)(무배당)(26.01)- 1. 청구서(회사 양식)\n'
 '- 2. 국가동물 등록한 경우에는 동물등록증 또는 등록번호'),
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
 'indexing': {'chunk_id': 'chunk_000739',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
