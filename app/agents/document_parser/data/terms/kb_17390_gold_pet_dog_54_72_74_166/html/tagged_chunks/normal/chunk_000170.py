from langchain_core.documents import Document

chunk = Document(
    page_content=(". 회사가 제1항에 따라 제공하여야 할 약관 및 계약자 보관용 청약서를 계약자가</p><br><p id='210' "
 "data-category='list' style='font-size:16px'>청약할 때 계약자에게 전달하지 않았거나 약관의 중요한 "
 '내용을 설명하지 않은<br>경우<br>2'),
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
 'indexing': {'chunk_id': 'chunk_000170',
              'chunk_char_len': 160,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
