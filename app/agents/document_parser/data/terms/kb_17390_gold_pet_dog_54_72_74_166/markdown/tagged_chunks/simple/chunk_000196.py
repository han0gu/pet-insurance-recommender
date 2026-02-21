from langchain_core.documents import Document

chunk = Document(
    page_content=('- 이 아닌 경우에는 본인의 인감증명서, 본인서명사실확인서 또는 안전성과 신뢰\n'
 'KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 69- 69 -규정약관성이 확보된 전자적 수단을 활용한보험수익자 의사표시의 확인방법 '
 '포함)제41조(지정대리청구인에 의한 보험금의 청구)# 지정대리청구인은 회사가 정하는 방법에 따라 다음의 서류를 제출하고 보험금을 청구 '
 '보험금청구권, 만기환급금청구권, 보험료 반환청구권, 해약환급금 청구권 및 계약자- 하여야 합니다.\n'
 '- 1. 청구서(회사양식)\n'
 '- 2. 사고증명서'),
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
 'indexing': {'chunk_id': 'chunk_000196',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
