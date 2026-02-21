from langchain_core.documents import Document

chunk = Document(
    page_content=('않았으나, 마음<br>이나 정신의 장애로 인하여 사물을 변별할 능력이나 의사를 결정할 능력이 미<br>약한 사람을 '
 "말합니다.</p><br><p id='4' data-category='paragraph' style='font-size:14px'>64 "
 "KB 금쪽같은 펫보험(강아지)(무배당)(26.01)</p><br><p id='5' data-category='paragraph' "
 "style='font-size:14px'>제22조(계약내용의 변경 등)</p><br><p id='6' "
 "data-category='paragraph'"),
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
 'indexing': {'chunk_id': 'chunk_000191',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
