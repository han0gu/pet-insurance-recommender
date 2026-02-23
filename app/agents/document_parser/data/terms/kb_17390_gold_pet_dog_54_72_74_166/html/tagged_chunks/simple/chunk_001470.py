from langchain_core.documents import Document

chunk = Document(
    page_content=('또는 질병의 진단확정일부터 1년 이내)에 장해상태가 더 악<br>화된 때에는 그 악화된 장해상태를 기준으로 장해지급률을 '
 "결정한다.</p><br><h1 id='159' style='font-size:14px'>2"),
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
 'indexing': {'chunk_id': 'chunk_001470',
              'chunk_char_len': 117,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
