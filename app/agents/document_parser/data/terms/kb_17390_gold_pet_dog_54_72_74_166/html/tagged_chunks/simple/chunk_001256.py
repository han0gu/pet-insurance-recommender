from langchain_core.documents import Document

chunk = Document(
    page_content=('. 제3자는 의료법 제3조(의료기관)에 규정한 종합병원 소속 전문의 중<br>에 정하며, 보험금 지급사유 판정에 드는 의료비용은 회사가 '
 '전액 부담합니다.<br>\uf000 제1조(보험금의 지급사유)의 반려동물 위탁비용은 반려동물 위탁 시 수탁기관에<br>지불한 비용을 '
 "말하며 추가 식대, 용품 구매 등의 비용은 제외한 기본 비용에 한합</p><br><p id='76' "
 "data-category='list'></p><br><h1 id='77' style='font-size:14px'>니다.</h1><p "
 "id='78'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001256',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
