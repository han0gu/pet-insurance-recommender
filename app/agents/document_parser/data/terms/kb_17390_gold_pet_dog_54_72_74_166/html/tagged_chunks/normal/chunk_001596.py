from langchain_core.documents import Document

chunk = Document(
    page_content=(". 장해판정기준</h1><br><p id='102' data-category='list' style='font-size:16px'>1) "
 "골절부에</p><br><p id='103' data-category='paragraph' "
 "style='font-size:16px'>금속내고정물 등을 사용하였기 때문에 그것이 기능장해의 원인</p><br><p id='104' "
 "data-category='list' style='font-size:16px'>이 되는 때에는 그 내고정물 등이 제거된 후 장해를 "
 '평가한다'),
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
 'indexing': {'chunk_id': 'chunk_001596',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
