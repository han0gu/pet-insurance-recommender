from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제1항에도 불구하고, "건강보험 행위 급여․비급여 목록 및 급여 상대가치점수" 개\n'
 '정으로 급여 판정이 변경되더라도 제1조(보험금의 지급사유)의 지급사유 발생 당\n'
 '특시의 "건강보험 행위 급여․비급여 목록 및 급여 상대가치점수"에 따라 이미 보험# 금 지급여부가 판단된 경우에는 이를 다시 판단하지 '
 '않습니다.별약관제4조(보험금의 청구)\uf000- 보험수익자는 다음의 서류를 제출하고 보험금을 청구하여야 합니다.\n'
 '- 1. 청구서(회사 양식)\n'
 '- 2. 사고증명서(진료비세부내역서("건강보험심사평가원 진료수가코드(EDI)" 필수 상해'),
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
 'indexing': {'chunk_id': 'chunk_000351',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
