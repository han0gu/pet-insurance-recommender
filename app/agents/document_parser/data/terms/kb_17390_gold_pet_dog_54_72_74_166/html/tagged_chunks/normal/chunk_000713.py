from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다만, 제4조(특별약관의 소멸)에 따라</p><br><p id='19' data-category='list'></p><br><h1 "
 "id='20' style='font-size:14px'>제5조(준용규정)</h1><p id='21' "
 "data-category='paragraph' style='font-size:14px'>94 KB 금쪽같은 "
 "펫보험(강아지)(무배당)(26.01)</p><br><p id='22' data-category='paragraph' "
 "style='font-size:14px'>이 특별약관에서 정하지 않은 사항은"),
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
 'indexing': {'chunk_id': 'chunk_000713',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
