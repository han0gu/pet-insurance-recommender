from langchain_core.documents import Document

chunk = Document(
    page_content=('국내의 병원 및 의<br>성<br>원에서 행한 의료행위에 한합니다.<br>특<br>\uf000 제1항에도 불구하고, 보건복지부에서 '
 "고시하는「건강보험 행위 급여․비급여 목록 약</p><br><p id='282' data-category='paragraph' "
 "style='font-size:16px'>KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 83</p><br><p id='283' "
 "data-category='paragraph' style='font-size:14px'>반</p><p id='284'"),
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
 'indexing': {'chunk_id': 'chunk_000535',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
