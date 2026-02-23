from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>특</p><p id='148' data-category='paragraph' "
 "style='font-size:16px'>KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 119</p><br><p id='149' "
 "data-category='paragraph' style='font-size:20px'>- 119 -</p><p id='150' "
 "data-category='paragraph' style='font-size:14px'>다.<br>\uf000 반려동물(강아지) "
 '일반조항에서 정하지 않은 사항은'),
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
 'indexing': {'chunk_id': 'chunk_001134',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
