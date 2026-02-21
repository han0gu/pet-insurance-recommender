from langchain_core.documents import Document

chunk = Document(
    page_content=("다르게 알리거나 알리지 않아 발생하는 불이익은 계약자가 부담합니다.</p><br><p id='61' "
 "data-category='list'></p><br><p id='62' data-category='paragraph' "
 "style='font-size:14px'>제5조(준용규정)</p><br><p id='63' data-category='paragraph' "
 "style='font-size:14px'>이 특별약관에서 정하지 않은 사항은 보통약관 및 해당 특별약관을 따릅니다.</p><p "
 "id='64'"),
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
 'indexing': {'chunk_id': 'chunk_001403',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
