from langchain_core.documents import Document

chunk = Document(
    page_content=('회사는 피보험자가 이 특별약관의 보험기간 중에 2대호흡계특정질환으로 진단확정된\n'
 '경우에는 아래에 정한 금액을 최초 1회에 한하여 2대호흡계특정질환진단비로 보험수| 지급합니다. 구 분 | 익자에게 | 익자에게 |\n'
 '| --- | --- | --- |\n'
 '| 지급합니다. 구 분 | 지 급 금 | 액 |\n'
 '| 2대호흡계특정질환진단비 | 보험계약일부터 1년미만 이 특별약관의 보험가입금액 50% | 보험계약일부터 1년이상 이 특별약관의 '
 '보험가입금액 100% |\n'
 '제2조(보험금 지급에 관한 세부규정)'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000384',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
