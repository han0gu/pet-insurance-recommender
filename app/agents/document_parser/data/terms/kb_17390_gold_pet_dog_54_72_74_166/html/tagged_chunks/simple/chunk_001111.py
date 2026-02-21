from langchain_core.documents import Document

chunk = Document(
    page_content=('. 제3자는 동물병원 소속 수의사 중에 정하며, 보험금 지급사유 판정에 드는<br>의료비용은 회사가 전액 부담합니다.</p><h1 '
 "id='103' style='font-size:14px'>제3조(보험금을 지급하지 않는 사유)<br>회사는 아래의 사유로 인한 손해는 "
 "보상하지</h1><br><p id='104' data-category='paragraph' "
 "style='font-size:14px'>않습니다.</p><br><p id='105' data-category='paragraph' "
 "style='font-size:14px'>1"),
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
 'indexing': {'chunk_id': 'chunk_001111',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
