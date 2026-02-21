from langchain_core.documents import Document

chunk = Document(
    page_content=(". 반려동물 위탁비용(반려인 질병입원 1일이상 180일한도)(실손)<br>(강아지)【갱신계약】 특</p><br><h1 id='107' "
 "style='font-size:20px'>(【갱신계약】은 자동갱신으로 운영합니다)</h1><br><p id='108' "
 "data-category='paragraph' style='font-size:14px'>별</p><br><p id='109' "
 "data-category='paragraph' style='font-size:14px'>약</p><br><p id='110'"),
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
 'indexing': {'chunk_id': 'chunk_001274',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
