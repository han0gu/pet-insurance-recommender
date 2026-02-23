from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- | --- |\n'
 '| 구 분 | 보험계약일부터 1년미만 | 보험계약일부터 1년이상 |\n'
 '| 6대호흡계특정질환진단비 | 이 특별약관의 보험가입금액 50% | 이 특별약관의 보험가입금액 100% |\n'
 '제2조(보험금 지급에 관한\n'
 '\uf000 보험수익자와 회사가세부규정)제1조(보험금의 지급사유)의 보험금 지급사유에 대해 합의- 하지 못할 때는 보험수익자와 회사가 '
 '함께 제3자를 정하고 그 제3자의 의견에 따\n'
 '- 를 수 있습니다. 제3자는 의료법 제3조(의료기관)에 규정한 종합병원 소속 전문'),
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
 'indexing': {'chunk_id': 'chunk_000391',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
