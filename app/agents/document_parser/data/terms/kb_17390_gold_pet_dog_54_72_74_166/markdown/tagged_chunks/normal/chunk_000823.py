from langchain_core.documents import Document

chunk = Document(
    page_content=('을 직접적인 원인으로 보험계약에서 정한 보험금지급사유가 발생한 경우에는 회사\n'
 '는 보험금을 지급하지 않습니다.- 138 -- \uf000 제1항의 회사가 보험금을 지급하지 않는 기간(이하 "부담보 기간"이라 '
 '합니다)은\n'
 '- 특정질병의 상태에 따라 "1개월부터 5년" 또는 "보험계약의 보험기간 전체"(단,\n'
 '- 계약이 갱신 또는 재가입 계약인 경우 최초 계약일로부터 최종 갱신 또는 재가입\n'
 '- 계약의 종료일까지의 기간을 말하며, 이하 "보험계약의 보험기간 전체"라 합니다)'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000823',
              'chunk_char_len': 256,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
