from langchain_core.documents import Document

chunk = Document(
    page_content=("신체 각 관절에 대한<br>평균 운동가능영역을 기준으로 정상각도 및 측정방법 등을 따른다.</p><br><p id='88' "
 "data-category='paragraph' style='font-size:14px'>KB 금쪽같은 "
 "펫보험(강아지)(무배당)(26.01) 147</p><br><p id='89' data-category='paragraph' "
 "style='font-size:18px'>- 147 -</p><header id='90' style='font-size:16px'>나) "
 '관절기능장해를 표시할 경우 장해부위의 장해각도와'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_001582',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
