from langchain_core.documents import Document

chunk = Document(
    page_content=('# 험상품자료"에서 확인할 수 있습니다)제11조(주소변경통지)\n'
 '\uf000 계약자(보험수익자가 계약자와 다른 경우 보험수익자를 포함합니다)는 주소 또는연락처가 변경된 경우에는 지체없이 그 변경내용을 '
 '회사에 알려야 합니다.\uf000 제1항에서 정한대로 계약자 또는 보험수익자가 변경내용을 알리지 않은 경우에는\n'
 '계약자 또는 보험수익자가 회사에 알린 최종의 주소 또는 연락처로 등기우편 등 우\n'
 '편물에 대한 기록이 남는 방법으로 회사가 알린 사항은 일반적으로 도달에 필요한'),
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
 'indexing': {'chunk_id': 'chunk_000048',
              'chunk_char_len': 251,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
