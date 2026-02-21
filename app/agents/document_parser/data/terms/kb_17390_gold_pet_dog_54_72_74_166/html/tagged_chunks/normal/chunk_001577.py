from langchain_core.documents import Document

chunk = Document(
    page_content=('한 팔의 3대 관절 중 관절 하나의 기능에 심한 장해를 남긴 때 20</td><td>보 통약</td></tr><tr><td>5) 한 팔의 '
 '3대 관절 중 관절 하나의 기능에 뚜렷한 장해를 남긴 때</td><td>관 10</td></tr><tr><td>6) 한 팔의 3대 관절 '
 '중 관절 하나의 기능에 약간의 장해를 남긴 때</td><td>5</td></tr><tr><td>7) 한 팔에 가관절이 남아 뚜렷한 장해를 '
 '남긴 때</td><td>20 특별</td></tr><tr><td>8) 한 팔에 가관절이 남아 약간의 장해를 남긴'),
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
 'indexing': {'chunk_id': 'chunk_001577',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
