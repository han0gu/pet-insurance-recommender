from langchain_core.documents import Document

chunk = Document(
    page_content=('. 제4호 이외의 방사선을 쬐는 것 또는 방사능 오염으로 인한 손해<br>는 피보험자가 상법 제657조 제1항에 의해 보험사고의 발생을 '
 "회사에 알린 경우 약<br>KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 121</p><br><p id='167' "
 "data-category='paragraph' style='font-size:18px'>- 121 -</p><p id='168' "
 "data-category='paragraph' style='font-size:14px'>에는 제4조(보상하는 손해의 범위) 제1호 및 "
 '제2호 "다"목'),
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
 'indexing': {'chunk_id': 'chunk_001159',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
