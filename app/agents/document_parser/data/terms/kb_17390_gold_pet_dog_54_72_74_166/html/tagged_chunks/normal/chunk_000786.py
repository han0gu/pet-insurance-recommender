from langchain_core.documents import Document

chunk = Document(
    page_content=('. 피보험자 본인의 가족관계등록상 또는 주민등록상에 기재된 배우자(이하 "배우<br>약</p><br><p id=\'150\' '
 "data-category='paragraph' style='font-size:14px'>병</p><p id='151' "
 "data-category='paragraph' style='font-size:14px'>제</p><p id='152' "
 "data-category='paragraph' style='font-size:16px'>KB 금쪽같은 "
 "펫보험(강아지)(무배당)(26.01) 101</p><br><p id='153'"),
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
 'indexing': {'chunk_id': 'chunk_000786',
              'chunk_char_len': 300,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
