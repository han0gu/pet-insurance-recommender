from langchain_core.documents import Document

chunk = Document(
    page_content=('style=\'font-size:14px\'>\uf000 이 특별약관에 있어서 "깁스(Cast)치료"라 함은 석고붕대 또는 '
 '섬유유리붕대<br>(Fiberglass Cast)를 병변이 있는 뼈, 관절부위의 둘레 모두에 착용시켜<br>(Circular Cast) '
 '감은 다음 굳어지게 하여 치료 효과를 가져 오는 치료법을 말합<br>니다'),
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
 'indexing': {'chunk_id': 'chunk_000747',
              'chunk_char_len': 178,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
