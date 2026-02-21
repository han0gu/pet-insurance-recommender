from langchain_core.documents import Document

chunk = Document(
    page_content=('분</td><td>3세</td><td>6세</td><td>9세</td><td>12세</td><td>… 비 '
 '고</td><td></td></tr><tr><td>특약보험료 최초계약 보험료표</td><td>5,000원 '
 '5,000원</td><td>6,200원 6,500원</td><td>7,600원 8,000원</td><td>12,500원 '
 '10,000원</td><td>… ….</td><td></td></tr><tr><td>첫 번째 갱신계약'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001383',
              'chunk_char_len': 241,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
