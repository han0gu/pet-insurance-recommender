from langchain_core.documents import Document

chunk = Document(
    page_content=(". 분쟁조정 신청<br>3. 수사기관의 조사<br>4. 해외에서 발생한 보험사고에 대한 조사</p><br><p id='62' "
 "data-category='paragraph' style='font-size:16px'>- 56 -</p><p id='63' "
 "data-category='list' style='font-size:16px'>5"),
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
 'indexing': {'chunk_id': 'chunk_000049',
              'chunk_char_len': 185,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
