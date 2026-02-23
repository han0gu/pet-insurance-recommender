from langchain_core.documents import Document

chunk = Document(
    page_content=('ㆍ<br>(손가락 하나마다)<br>4) 한 손의 5개 손가락 모두의 손가락뼈 일부를 잃었을 때 또는 뚜 규정<br>30<br>렷한 장해를 '
 '남긴 때<br>5) 한 손의 첫째 손가락의 손가락뼈 일부를 잃었을 때 또는 뚜렷<br>10<br>한 장해를 남긴 때<br>6) 한 손의 '
 '첫째 손가락 이외의 손가락의 손가락뼈 일부를 잃었을<br>5<br>때 또는 뚜렷한 장해를 남긴 때(손가락 하나마다)</p><br><p '
 "id='129' data-category='paragraph' style='font-size:14px'>표</p><p"),
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
 'indexing': {'chunk_id': 'chunk_001613',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
