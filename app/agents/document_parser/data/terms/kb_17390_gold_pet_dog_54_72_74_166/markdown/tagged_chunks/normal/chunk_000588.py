from langchain_core.documents import Document

chunk = Document(
    page_content=('·최대 보상한도액 = 30만원(항암약물치료 보상한도액 적용)\n'
 '\uf000 보험수익자와 회사가 제1조(보험금의 지급사유)의 보험금 지급사유에 대해 합의\n'
 '하지 못할 때는 보험수익자와 회사가 함께 제3자를 정하고 그 제3자의 의견에 따\n'
 '를 수 있습니다. 제3자는 동물병원 소속 수의사 중에 정하며, 보험금 지급사유\n'
 '판정에 드는 의료비용은 회사가 전액 부담합니다. 질# 제3조("반려동물주요치료"의 정의)\uf000 이 특별약관에 있어서 '
 '"반려동물주요치료"라 함은 국내에서 수의사가 보험증권에'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000588',
              'chunk_char_len': 261,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
