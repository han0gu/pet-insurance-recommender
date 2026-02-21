from langchain_core.documents import Document

chunk = Document(
    page_content=('또는 전자<br>적 의사표시(통신판매계약의 경우 통신수단)를 통해 확인하고, 자동갱신 의사가<br>확인되는 경우 갱신 전 보장특약의 '
 '갱신일에 갱신일 현재의 약관 등으로 갱신합니<br>다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001377',
              'chunk_char_len': 103,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
