from langchain_core.documents import Document

chunk = Document(
    page_content=('. 귓바퀴의 결손<br>1) ‘귓바퀴의 대부분이 결손된 때’라 함은 귓바퀴의 연골부가 1/2 이상 결<br>손된 경우를 '
 "말한다.<br>2) 귓바퀴의 연골부가 1/2 미만 결손이고 청력에 이상이 없으면 외모의 추</p><br><p id='192' "
 "data-category='list'></p><br><p id='193' data-category='paragraph' "
 "style='font-size:14px'>상(추한 모습)장해로만 평가한다.</p><p id='194' "
 "data-category='paragraph'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001501',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
