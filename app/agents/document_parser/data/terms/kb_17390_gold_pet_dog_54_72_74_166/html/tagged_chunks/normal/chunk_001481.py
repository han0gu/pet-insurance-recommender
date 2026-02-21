from langchain_core.documents import Document

chunk = Document(
    page_content=('. 장해판정기준<br>1) 시력장해의 경우 공인된 시력검사표에 따라 최소 3회 이상 측정한다.<br>2) ‘교정시력’이라 함은 '
 '안경(콘택트렌즈를 포함한 모든 종류의 시력 교<br>정수단)으로 교정한 원거리 최대교정시력을 말한다'),
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
 'indexing': {'chunk_id': 'chunk_001481',
              'chunk_char_len': 126,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
