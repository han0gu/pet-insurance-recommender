from langchain_core.documents import Document

chunk = Document(
    page_content=('style=\'font-size:14px\'>"창상봉합술(급여)"이라 함은 상해의 직접결과로써, "창상</p><br><p '
 'id=\'83\' data-category=\'list\' style=\'font-size:14px\'>봉합술" 치료를 받은 경우를 '
 '말합니다'),
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
 'indexing': {'chunk_id': 'chunk_000583',
              'chunk_char_len': 138,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
