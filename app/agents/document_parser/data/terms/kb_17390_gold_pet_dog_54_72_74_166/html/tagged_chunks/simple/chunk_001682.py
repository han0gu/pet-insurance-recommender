from langchain_core.documents import Document

chunk = Document(
    page_content=('세안, 양치, 샤워, 목욕 등 모든 개인위생 관리시 타 인의 지속적인 도움이 필요한 '
 '상태</td><td>10%</td></tr><tr><td>2) 세안, 양치시 부분적인 도움 하에 혼자서 가능하나 목 목욕 욕이나 샤워시 '
 '타인의 도움이 필요한 상태</td><td>5%</td></tr><tr><td>3) 세안, 양치와 같은 개인위생관리를 독립적으로 시행 '
 '가능하나 목욕이나 샤워시 부분적으로 타인의 도움이 필요한 상태</td><td>3%</td></tr><tr><td rowspan="3">옷 '
 '입고</td><td>1) 상·하의 의복 착탈시'),
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
 'indexing': {'chunk_id': 'chunk_001682',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
