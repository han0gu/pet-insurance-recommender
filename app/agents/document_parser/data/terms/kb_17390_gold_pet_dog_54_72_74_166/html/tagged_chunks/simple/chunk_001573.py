from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다발성늑골 기형의 경우 각각의 각(角) 변<br>형을 합산하지 않고 그 중 가장 높은 각(角) 변형을 기준으로 평가한다.</p><p '
 "id='77' data-category='paragraph' style='font-size:18px'>- 146 -</p><p "
 "id='78' data-category='list'></p><figure id='79'><img style='font-size:14px' "
 'alt="부 가 설 명 가슴뼈" data-coord="top-left:(128,108); bottom-right:(685,491)"'),
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
 'indexing': {'chunk_id': 'chunk_001573',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
