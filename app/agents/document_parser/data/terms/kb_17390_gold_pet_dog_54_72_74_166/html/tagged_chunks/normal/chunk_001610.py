from langchain_core.documents import Document

chunk = Document(
    page_content=("id='122' data-category='paragraph' style='font-size:16px'>다.</p><br><p "
 "id='123' data-category='list' style='font-size:16px'>지급률의 결정<br>1) 한 다리의 3대 "
 '관절 중 관절 하나에 기능장해가 생기고 다른 관절 하나 특별<br>에 기능장해가 발생한 경우 지급률은 각각 적용하여 합산한다'),
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
 'indexing': {'chunk_id': 'chunk_001610',
              'chunk_char_len': 216,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
