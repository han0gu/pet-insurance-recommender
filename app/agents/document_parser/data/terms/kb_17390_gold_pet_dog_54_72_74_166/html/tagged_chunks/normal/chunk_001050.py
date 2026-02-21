from langchain_core.documents import Document

chunk = Document(
    page_content=('. 과잉진료행위로 인한 비용<br>18. 아포퀠(Apoquel) 등의 JAK inhibitor(Janus kinase inhibitor) '
 '약물과 사이토<br>포인트(Cytopoint)(다만, 제2항 제18호는 "특정약물치료Ⅱ"를 받은 경우에는<br>제외합니다.)<br>19'),
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
 'indexing': {'chunk_id': 'chunk_001050',
              'chunk_char_len': 152,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
