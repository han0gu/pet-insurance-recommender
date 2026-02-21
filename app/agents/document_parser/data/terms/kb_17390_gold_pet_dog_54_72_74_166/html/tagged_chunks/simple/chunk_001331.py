from langchain_core.documents import Document

chunk = Document(
    page_content=("특별약관에서 정하지 않은 사항은 보험계약을 따릅니다.</p><p id='196' data-category='paragraph' "
 "style='font-size:18px'>2.</p><br><p id='197' data-category='paragraph' "
 "style='font-size:18px'>선지급서비스</p><h1 id='198' "
 "style='font-size:14px'>제1조(적용대상)</h1><br><p id='199' data-category='list' "
 "style='font-size:14px'>\uf000 계약자와 동일한"),
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
 'indexing': {'chunk_id': 'chunk_001331',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
