from langchain_core.documents import Document

chunk = Document(
    page_content=("합니다.</p><br><p id='106' data-category='paragraph' "
 "style='font-size:14px'>제5조(준용규정)</p><br><p id='107' "
 "data-category='paragraph' style='font-size:14px'>\uf000 이</p><br><p id='108' "
 "data-category='paragraph' style='font-size:14px'>특별약관에서 정하지 않은 사항에 대하여는 "
 "전환대상계약 약관, 소득세법 등</p><p id='109'"),
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
 'indexing': {'chunk_id': 'chunk_001437',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
