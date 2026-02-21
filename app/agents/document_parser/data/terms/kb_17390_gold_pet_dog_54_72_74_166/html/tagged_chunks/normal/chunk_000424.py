from langchain_core.documents import Document

chunk = Document(
    page_content=(". 하지란 고관절 이하 대퇴부, 하퇴부, 족부를 의미하며, 둔부, 서혜부, 복부 등</p><br><p id='100' "
 "data-category='paragraph' style='font-size:16px'>KB 금쪽같은 "
 "펫보험(강아지)(무배당)(26.01) 77</p><br><p id='101' data-category='paragraph' "
 "style='font-size:18px'>- 77 -</p><br><p id='102' data-category='paragraph' "
 "style='font-size:14px'>관</p><p"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_000424',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
