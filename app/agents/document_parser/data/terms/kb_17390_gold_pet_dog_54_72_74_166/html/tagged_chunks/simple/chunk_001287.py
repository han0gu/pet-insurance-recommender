from langchain_core.documents import Document

chunk = Document(
    page_content=('. 피보험자의 치매를 제외한 정신적 기능장해, 선천성 뇌질환 및 심신상실<br>2. 성병<br>3. 알코올 중독, 습관성 약품 또는 '
 "환각제의 복용 및 사용</p><br><p id='124' data-category='paragraph' "
 "style='font-size:14px'>\uf000 회사는 아래의 의료비로 보험금 지급사유가 발생한 때에는 보험금을 지급하지 "
 "않<br>습니다.</p><br><p id='125' data-category='list' style='font-size:14px'>1"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_001287',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
